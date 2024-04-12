"""
Yet another attempt to get project based workflows right.
"""

import abc
import codecs, dill
from dataclasses import dataclass
from copy import copy, deepcopy
from hashlib import md5
from typing import (
        List as TList,
        Optional
)
from logging import getLogger, INFO
from functools import lru_cache

from pyiron_base import DataContainer
from pyiron_base.storage.has_stored_traits import HasStoredTraits
from pyiron_base.interfaces.has_hdf import HasHDF
from traitlets import (
    default,
    validate,
    TraitError,
    Any,
    Unicode,
    Union,
    Instance,
    Tuple,
    Float,
    List,
    Bool,
    Callable
)
from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage
import pyiron_contrib.jobfactories
from pyiron_contrib.repair import HandyMan

import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)
from tqdm.auto import tqdm

from .util import get_table, symlink_project

class RunAgain(Exception):
    """
    Thrown when a workflow wants to be called again at some unspecified point in the future.  Usually because it's not
    finished yet with some calculations.
    """
    pass

@lru_cache
def closest_common_ancestor(cls1, cls2):
    mro1 = cls1.__mro__
    mro2 = cls2.__mro__
    common_ancestor = None
    for c1 in mro1:
        for c2 in mro2:
            if c1 is c2:
                common_ancestor = c1
                break
        if common_ancestor:
            break
    return common_ancestor

class InputOutputBase(HasStoredTraits):
    def take(self, other: "Input"):
        """
        Copy common input parameters.

        Searches for a common ancestor class for this object and other and then
        copies all traitlets that are defined their from other to this object.

        This allows base classes of higher order project flows to copy inputs
        between the sub project flows without explicitly having to know the
        input parameters.
        """
        ancestor = closest_common_ancestor(type(self), type(other))
        if ancestor is object:
            raise ValueError("Both inputs must share an ancestor!")
        for name in ancestor.class_trait_names():
            try:
                value = getattr(other, name)
                if value is not None:
                    setattr(self, name, deepcopy(value))
            except AttributeError: # name is not set on other
                pass
        return self

class Input(InputOutputBase):
    pass

class SymlinkInput(Input):
    symlink = Bool(default_value=True, help='Whether to symlink the project or not')

class Output(InputOutputBase):
    results = Instance(pd.DataFrame, allow_none=True)

@dataclass
class WorkflowProjectConfig:
    delete_existing_job: bool
    finished_threshold: float
    broken_threshold: float

class ProjectFlow(HasHDF, abc.ABC):

    Input = Input
    Output = Output

    def __init__(self, project=None, name=None):
        self._project = None
        self._input = None
        self._output = None

        if project is not None and name is not None:
            self._project = project.create_group(name)
            if 'input' in self.project.data:
                self.input = self.project.data.input
            if 'output' in self.project.data:
                self._output = self.project.data.output
        elif project is not None or name is not None:
            raise ValueError('`project` and `name` must be None or not together!')

    @property
    def project(self):
        return self._project

    def attach(self, project, name):
        """
        Open a new flow in the given project with name and copy input.
        """
        cls = type(self)
        clsname = cls.__name__
        n = project.create_group(name).data.get('projectflow', clsname)
        if clsname != n:
            raise ValueError(f'Project already owned by another flow of type: {n}!')
        flow = cls(project, name)
        if flow.input.read_only: flow.input.unlock()
        # s = flow.input._storage = self.input.storage.copy()
        # for k, v in s.items():
        for k, v in self.input.trait_values().items():
            setattr(flow.input, k, v) # without this traitlets doesn't realize it has values
        return flow

    def detach(self):
        self._project = None
        self._input = None
        self._output = None

    def sync(self):
        if self.project is None:
            raise ValueError('Flow not attached to any project!')
        self.project.data.projectflow = self.__class__.__qualname__
        self.project.data.input = self.input
        self.project.data.output = self.output
        self.project.data.write()


    @property
    def input(self) -> Input:
        if self._input is None:
            self._input = self.Input()
        return self._input

    @input.setter
    def input(self, value: Input):
        if not isinstance(value, self.Input):
            raise TypeError(
                f'Input must be of type {self.Input.__name__}, not {value.__class__.__name__}!'
            )
        self._input = value

    @property
    def intermediates(self):
        if self.project is not None:
            return self.project.data.create_group('intermediates')
        else:
            raise ValueError('flow must be attached to a project first!')

    @property
    def output(self) -> Output:
        if self._output is None:
            self._output = self.Output()
        if not isinstance(self._output, self.Output):
            # sometimes I can be a doofus and forget to change the cls.Output to the output class I actually want and
            # only realize it once I've run some calculations; this sidesteps this by wrapping the output in the correct
            # type on loading
            warnings.warn(f"output is type {type(self._output)} but should be {self.Output}, wrapping it!")
            new = self.Output()
            self._output = new.take(self._output)
        return self._output

    @abc.abstractmethod
    def _run(
        self,
        delete_existing_job=False,
        delete_aborted_job=True
    ):
        pass

    def run(
        self,
        delete_existing_job=False,
        delete_aborted_job=True
    ):
        self.input.lock()
        self.sync()
        if self.input.symlink:
            symlink_project(self.project)
        self._run(
            delete_existing_job=delete_existing_job,
            delete_aborted_job=delete_aborted_job
        )

    @abc.abstractmethod
    def _analyze(
        self,
        delete_existing_job=False
    ) -> pd.DataFrame:
        pass

    def analyze(
        self,
        delete_existing_job=False,
    ):
        if self.output.read_only: self.output.unlock()
        self.output.results = self._analyze(delete_existing_job=delete_existing_job)
        self.output.lock()
        self.sync()
        return self.output.results

    def _to_hdf(self, hdf):
        hdf['projectflow'] = self.__class__.__name__
        hdf['input'] = self.input
        hdf['output'] = self.output

    def _from_hdf(self, hdf, version=None):
        self.input = hdf['input'].to_object()
        self._output = hdf['output'].to_object()

    # Supports training data workflows
    # this should be much better integrated with run as it would be useful also
    # for the verification flows (not in this repository yet), but
    # I'd like to play it safe and not modify both at the same time.

    def considered_empty(self):
        return len(self.project.job_table()) == 0

    def considered_finished(self, threshold=0.9):
        status = self.project.get_jobs_status().to_dict()
        total = sum(status.values())
        if total == 0: return False
        if status.get('submitted', 0) > 0: return False
        if status.get('running', 0) > 0: return False
        return status.get('finished', 0) / total > threshold

    def considered_broken(self, threshold=0.05):
        status = self.project.get_jobs_status().to_dict()
        total = sum(status.values())
        if total == 0: return False
        running = status.get('running', 0) \
                + status.get('submitted', 0)
        broken  = status.get('aborted', 0) \
                + status.get('warning', 0) \
                + status.get('not_converged', 0)
        # as long jobs are running anyway fix everything we can
        if running > 0 and broken > 0: return True
        # if nothing is running anymore only fix if it exceeds the threshold
        return broken / total > threshold

    def check(
            self, config: WorkflowProjectConfig,
            if_new=lambda: None,
            if_finished=lambda: None,
            number_of_jobs: Optional[int] = None,
    ):
        """
        Check if this workflow has run before, has broken jobs or has finished.

        `if_new` and `if_finished` are called appriopriately.  If any broken jobs are detected, they are
        tried to be repaired with HandyMan.
        """
        logger = getLogger()
        logger.setLevel(INFO)

        if self.considered_empty() or config.delete_existing_job:
            logger.info('empty project, running from scratch')
            if_new(self)
            raise RunAgain('starting new workflow')

        if number_of_jobs is not None and len(self.project.job_table()) < number_of_jobs:
            logger.info('project has less than advertised jobs; run from new again')
            if_new(self)
            raise RunAgain('starting some new calculations')

        if self.considered_broken(threshold=config.broken_threshold):
            logger.info('found aborted jobs; calling the handy man!')
            hm = HandyMan(suppress_fix_errors=False)
            cs = hm.fix_project(self.project)
            logger.info('repair stats:')
            for tool, l in cs.fixing.items():
                logger.info('\t%s: %i', tool, len(l))
            logger.info('unfixable: %i', len(cs.hopeless))
            if len(cs.fixing) > 0:
                raise RunAgain('running repairs')
            logger.info('either all jobs unfixable or failing; moving on')

        if self.considered_finished(threshold=config.finished_threshold):
            logger.info('finished')
            return if_finished(self)

        logger.info('still running; try later')
        raise RunAgain("Still Running!")

def _to_pickle(value):
    return codecs.encode(dill.dumps(value), "base64").decode()


def _from_pickle(value):
    return dill.loads(codecs.decode(value.encode(), "base64"))

class FunctionContainer(HasHDF):
    __slots__ = ('func',)

    def __init__(self, func=None):
        self.func = func
        self.__doc__ = func.__doc__

    def _to_hdf(self, hdf):
        hdf['dill'] = _to_pickle(self.func)

    def _from_hdf(self, hdf, version=None):
        self.func = _from_pickle(hdf['dill'])

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class StructureInput(SymlinkInput, Input):
    symlink = Bool(default_value=False, help='Whether to symlink the project or not')

    # This is what it really should be, but the way traitlets implement type checking conflicts with importlib.reload,
    # which makes prototyping very cumbersome
    # job = Instance(pyiron_contrib.jobfactories.JobFactory)
    job = Any(help='Must be a JobFactory from pyiron_contrib.jobfactories')
    structures = Instance(StructureStorage, args=())
    output_structures = Bool(default_value=False, help="populate output.structures with results of each calculation")
    table_setup = Callable(default_value=None, allow_none=True)
    hash_job_names = Bool(default_value=False)

    @default('job')
    def get_job_default(self):
        return pyiron_contrib.jobfactories.MlipFactory()

    @validate('table_setup')
    def validate_table_setup(self, proposal):
        val = proposal['value']
        if not isinstance(val, FunctionContainer):
            return FunctionContainer(val)
        else:
            return val

class StructureOutput(Output):
    structures = Instance(StructureStorage, args=())

def get_user_arrays(storage: StructureStorage) -> TList[str]:
    """
    Extract a list of user defined arrays, that are not used by the structure storage internally.
    """
    return [
            k for k in storage.list_arrays()
                if k not in ['identifier', 'start_index', 'length', 'symbols', 'positions', 'cell', 'pbc', 'spins'] \
                    and storage.has_array(k)['per'] == "chunk"
    ]

class StructureProjectFlow(ProjectFlow):
    """
    Runs a given job factory for a given structure set.

    Modify the output of analyze by providing input.table_setup.
    """

    Input = StructureInput
    Output = StructureOutput

    def _run(
            self,
            delete_existing_job=False,
            delete_aborted_job=True
    ):
        job = self.input.job.copy()
        job.project = self.project.create_group('runs')
        # sounds weird, but if symlink==True, our own project will get symlinked.  We want to symlink at least our
        # sub project though, that keeps the actual calculations.  This will keep the pyiron table out of the symlink
        # but in the back up
        if not self.input.symlink:
            symlink_project(job.project)
        for i, structure in tqdm(enumerate(self.input.structures.iter_structures()),
                                 total=len(self.input.structures)):
            def modify(job):
                job['user/structure'] = self.input.structures['identifier', i]
                for k in get_user_arrays(self.input.structures):
                    job[f'user/{k}'] = self.input.structures[k, i]
                return job
            if self.input.hash_job_names:
                # pyiron job names have stupid restrictions, side step them
                # with hashes
                name = 'H' + md5(str(self.input.structures['identifier', i]).encode('utf8')).hexdigest()
            else:
                name = f'structure_{i}'
            job.run(
                name=name,
                modify=modify,
                structure=structure,
                delete_existing_job=delete_existing_job,
                delete_aborted_job=delete_aborted_job,
            )

    def _analyze(self, delete_existing_job=False):
        hamilton = self.input.job.hamilton
        if 'calc_static' in self.input.job.storage.methods:
            self.input.output_structures = False
        if self.input.output_structures:
            if delete_existing_job:
                self.output.structures = StructureStorage()
            for j in self.project.iter_jobs(hamilton=hamilton, status='finished'):
                identifier = j['user/structure']
                if identifier not in self.output.structures['identifier']:
                    self.output.structures.add_structure(
                            j.get_structure(-1),
                            identifier=identifier
                    )

        table_setup = getattr(self.input, 'table_setup', None)
        def add(tab):
            tab.analysis_project = self.project['runs']
            tab.db_filter_function = lambda df: df.hamilton==hamilton
            tab.add['structure'] = lambda j: j['user/structure']
            if table_setup is not None:
                table_setup(tab)
        df = get_table(
                self.project,
                'structure_table',
                add=add,
                delete_existing_job=delete_existing_job
        ).get_dataframe()
        meta = pd.DataFrame({
            k: list(self.input.structures[k]) for k in get_user_arrays(self.input.structures)
        })
        if len(df) > 0 and len(meta) > 0:
            meta['structure'] = self.input.structures['identifier']
            df = df.merge(meta, on='structure')
        return df
