from argparse import ArgumentParser, RawDescriptionHelpFormatter, FileType
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from itertools import combinations_with_replacement
from typing import Union, Optional, Mapping
import logging

from pyiron_base import Project
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure
from pyiron_potentialfit.atomistics.job.trainingcontainer import TrainingContainer

from pyiron_contrib.jobfactories import VaspFactory

from structuretoolkit import get_neighbors

from .projectflow import (
        StructureProjectFlow,
        WorkflowProjectConfig,
        RunAgain
)
from .util import DistanceFilter,fast_forward

from tqdm.auto import tqdm
import numpy as np

class TrainingDataFlow(StructureProjectFlow):

    def run(self, delete_existing_job=False, delete_aborted_job=True):
        self.symlink=True
        super().run(
            delete_existing_job=delete_existing_job,
            delete_aborted_job=delete_aborted_job
        )

    def collect(
            self,
            name: str = 'Results',
            num_neighbors: Optional[int] = None,
            delete_existing_job: bool = False,
    ) -> TrainingContainer:
        """
        Collect results in a new TrainingContainer.

        Args:
            name (str):
                job name of the new container
            num_neighbors (int):
                calculate nearest neighbors directly on the collected structures
            delete_existing_job (bool):
                recreate TrainingContainer if it exists already

        Returns:
            TrainingContainer: container with all structures + efs
        """
        train = self.project.create.job.TrainingContainer(
                name,
                delete_existing_job=delete_existing_job,
                delete_aborted_job=True
        )
        if train.status.finished: return train

        hamilton = self.input.job.hamilton
        for j in self.project.iter_jobs(
                recursive=True, status='finished', hamilton=hamilton,
                convert_to_object=False
        ):
            train.include_job(j)
        if num_neighbors is not None:
            train.input.save_neighbors = True
            train.input.num_neighbors = num_neighbors
            train.server.queue = 'cmti'
            train.server.cores = 1
            train.server.max_memory = 1e-1 * train.number_of_structures
        train.run()
        return train

@dataclass
class CalculationConfig:
    encut: float
    # if float interpreted as k mesh spacing
    kpoints: Optional[Union[int, float]]
    incar: dict

    workflow: WorkflowProjectConfig

    cores: int = 40
    run_time: float = 2 * 60 * 60 # seconds
    queue: str = "cmti"

    def configure_dft_job(self, job: Union[VaspFactory, "GenericDFTJob"]):
        job.set_encut(self.encut)
        kpoints = self.kpoints
        if isinstance(kpoints, int):
            job.set_kpoints([self.kpoints]*3)
        elif isinstance(kpoints, float):
            job.set_kpoints(k_mesh_spacing=kpoints)
        elif kpoints is None:
            pass
        else:
            assert False, f"kpoints must be float/int/None, not {kpoints}"

        job.server.queue = self.queue
        job.server.cores = self.cores
        job.server.run_time = self.run_time
        return job

    def configure_vasp_job(self, job: VaspFactory):
        if isinstance(self.kpoints, float):
            self.incar['KSPACING'] = self.kpoints
        if 'KSPACING' in self.incar:
            self.kpoints = None
            self.incar['KGAMMA'] = '.TRUE.'
        job = self.configure_dft_job(job)
        job.incar['NCORE'] = max(20, self.cores)
        # j.incar['LCHARG'] = '.FALSE.'
        job.incar['EDIFF'] = 1e-6
        job.incar['ALGO'] = 'Normal'
        for k, v in self.incar.items():
            job.incar[k] = v
        job.set_eddrmm_handling(status="ignore")
        job.enable_nband_hack({
            "Al": 2, # = 3/2 + 1/2 VASP default
            "H": 2
        })
        # job.server.queue = 'cmti'
        # job.server.cores = 40
        # job.server.run_time = 60 * 60 * 2
        return job

    def get_job(self):
        job = VaspFactory()
        job = self.configure_vasp_job(job)
        return job


def run_container(pr, cont, config: CalculationConfig):
    logger = logging.getLogger('calculations')
    logger.info('running DFT on %s in project %s', cont.name, pr.path)

    train = TrainingDataFlow(pr, cont.name)
    def if_new(train):
        dfilter = DistanceFilter()
        filtered_cont = cont._container.sample(
                lambda f, i: dfilter(f.get_structure(i))
        )

        if train.input.read_only:
            train.input.unlock()
        train.input.structures = filtered_cont

        train.input.job = config.get_job()
        train.run(delete_existing_job=config.workflow.delete_existing_job)

    def if_finished(train):
        results = pr.create_group('containers').load(cont.name)
        if results is None:
            results = train.collect(
                    num_neighbors=50
            )
            results.copy_to(
                    pr.create_group('containers'),
                    new_job_name=train.project.name
            )
        return results

    train.check(config.workflow, if_new, if_finished,
            number_of_jobs=train.input.structures.number_of_structures
    )

def deduplicate(cont, replace=True):
    D = defaultdict(list)
    DD = []
    for i in tqdm(range(len(cont._container))):
        if i in DD: continue
        N = cont._container['length', i]
        J = i + 1 + np.where(cont._container['length'][i+1:] == N)[0]
        Ji = np.where(np.isclose(cont._container['energy'][J], cont._container['energy', i], rtol=1e-3))[0]
        J = J[Ji]
        for j in J:
            same = np.allclose(cont._container['positions', i], cont._container['positions', j])
            same &= np.all(cont._container['symbols', i] == cont._container['symbols', j])
            same &= np.allclose(cont._container['cell', i], cont._container['cell', j])
            same &= np.allclose(cont._container['pbc', i], cont._container['pbc', j])
            if same:
                D[i].append(j)
                DD.append(j)
    dcont = cont.sample(cont.name + 'Unique', lambda f, i: i not in DD, run=False)
    if replace:
        cont.remove()
        dcont.rename(cont.name)
    return dcont

def combine(pr, containers, name='Everything',
            min_dist=None, force_cap=None,
            check_duplicates=True,
            delete_existing_job=False):
    every = pr.create.job.TrainingContainer(
            name,
            delete_existing_job=delete_existing_job
    )
    if not every.status.initialized: return every
    for cont in containers:
        if cont.status.initialized: continue
        df = cont.to_pandas()
        df['name'] = df.name.map(lambda s: cont.name + '_' + s)
        if force_cap is not None:
            I = df.forces.map(lambda f: np.linalg.norm(f, axis=-1).max() < force_cap)
            print(cont.name, sum(I), df.shape[0])
            df = df.loc[I]
        if min_dist is not None:
            if isinstance(min_dist, dict):
                def element_wise_dist(a):
                    pair = defaultdict(lambda: np.inf)
                    n = a.get_neighbors(num_neighbors=25, cutoff_radius=5, mode='ragged')
                    for i, (I, D) in enumerate(zip(n.indices, n.distances)):
                        for j, d in zip(I, D):
                            ei, ej = sorted((a.symbols[i], a.symbols[j]))
                            pair[ei, ej] = min(d, pair[ei, ej])
                    return pair
                def larger_than_min_dist(a):
                    pair = element_wise_dist(a)
                    for ei, ej in combinations_with_replacement(a.get_species_symbols(), 2):
                        ei, ej = sorted((ei, ej))
                        if pair[ei, ej] < min_dist[ei] + min_dist[ej]:
                            return False
                    return True
            else:
                def larger_than_min_dist(a):
                    # d = a.get_neighbors(1, cutoff_radius=2*min_dist).distances
                    d = get_neighbors(a, num_neighbors=1).distances
                    return np.prod(d.shape) == 0 or d.min() > min_dist
            I = df.atoms.map(larger_than_min_dist)
            print(cont.name, sum(I), df.shape[0])
            df = df.loc[I]
        every.include_dataset(df)
    if check_duplicates:
        every.save()
        every.status.finished = True
        every = deduplicate(every, replace=True)
    every.input.save_neighbors=True
    every.input.num_neighbors = 150
    every.server.queue = 'cmti'
    every.server.cores = 1
    every.server.run_time = 60 * 60
    every.run()
    return every

epilog="""
Run high throughput, high convergence DFT on a set of structures.

After you have created structures with potfit.structures you can use this
module to run DFT on them. This program assumes that you run it from the same
directory as potfit.structures, or use -p,-s accordingly.  This program is also
designed to be called multiple times and will restart broken or not converged
calculations.  This may take a few iterations.  After all calculations are
finished or cannot be further repaired, the results will be aggregated in
`TrainingContainers`.  One for each --container that you pass and a --final one
that collects every structure.

We lay out the project like this,

`root/structures/containers`:             where we read the --containers from
`root/calculations`:                      work exlusively in this project
`root/calculations/containers`:           will ultimately contain the final training data
`root/calculations/{container_names}`:    working projects for each of the passed containers

where `root` is the programs working directory and `container_names` what you passed as --containers.

The default k point and encut values are converged for the Mg/Al/Ca system.  Do your own testing, check
potfit.convergence how to go about it.
"""

if __name__ == '__main__':
    parser = ArgumentParser(
            description='Creating training data for an MLIP',
            epilog=epilog,
            formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
            '-e', '--encut', type=float, default=550,
            help='Plane wave energy cut off'
    )
    kgroup = parser.add_mutually_exclusive_group()
    kgroup.add_argument(
            '-k', '--kpoints', type=int, dest='kpoints',
            help='k point sampling (k mesh spacing if given as float'
    )
    kgroup.add_argument(
            '-m', '--kspacing', type=float, default=0.1, dest='kpoints',
            help='k point sampling (k mesh spacing if given as float'
    )
    parser.add_argument(
            '--incar', type=FileType('r'), default=None,
            help='INCAR to apply on top'
    )
    parser.add_argument(
            '-p', '--project', default='calculations',
            help='project to work in'
    )
    parser.add_argument(
            '-s', '--structures-project', default='structures',
            help='project that contains the structure containers'
    )
    parser.add_argument(
            '-c', '--containers', type=str, nargs='+',
            default=(
                'Crystals', 'CrystalsVolMinTrace', 'CrystalsAllMinTrace',
                'CrystalsVolMinAllMinRandom'
            ),
            help='names of structure containers to use'
    )
    parser.add_argument(
            '-f', '--final', default='Everything',
            help='Name of the container that collects all results'
    )
    parser.add_argument(
            '-d', '--min-dist', type=float, default=None,
            help='Smallest nearest neighbor distance to allow'
    )
    parser.add_argument(
            '--cores', type=int, default=CalculationConfig.cores,
            help='Number of cores for each DFT run'
    )
    parser.add_argument(
            '-r', '--run-time', type=float, default=CalculationConfig.run_time,
            help='Run time limit for each DFT run in seconds'
    )
    parser.add_argument(
            '--fast-forward', type=int, default=None,
            help='Automatically go to the next step after sleeping for this many seconds'
    )
    workflow_group = parser.add_argument_group('Workflow')
    workflow_group.add_argument(
            '--delete-existing-job', action='store_true',
            help='delete existing calculations'
    )
    workflow_group.add_argument(
            '--broken-threshold', type=float, default=0.05,
            help='Maximum amount of aborted/not_converged/warning jobs to accept; '
                 'if above try to repair/converge jobs'
    )

    args = parser.parse_args()

    incar = {}
    if args.incar is not None:
        incar = {k.strip(): v.strip() for k, v in map(lambda x: x.split('='), args.incar)}

    conf = CalculationConfig(
            encut=args.encut,
            kpoints=args.kpoints,
            incar=incar,
            cores=args.cores,
            run_time=args.run_time,
            workflow=WorkflowProjectConfig(
                delete_existing_job=args.delete_existing_job,
                broken_threshold=args.broken_threshold,
                finished_threshold=0.9
            )
    )

    pr = Project(args.project)
    pr.data.config = asdict(conf)
    pr.data.write()
    structures_pr = Project(args.structures_project)

    logger = logging.getLogger('calculations')

    done = True
    for cname in args.containers:
        logger.info('Running %s', cname)
        cont = structures_pr['containers'].load(cname)
        try:
            run_container(pr, cont, conf)
        except RunAgain:
            done = False

    if done:
        logger.info('collecting all calculations in %s', args.final)
        every = combine(pr['containers'], map(pr['containers'].load, args.containers),
                        name=args.final, min_dist=args.min_dist,
                        delete_existing_job=args.delete_existing_job)
    elif args.fast_forward is not None:
        fast_forward(args.fast_forward, __spec__)
