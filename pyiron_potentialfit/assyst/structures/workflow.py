from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Union, Optional
import time
from io import StringIO
from logging import getLogger
from pprint import pprint
from itertools import count
import warnings
from math import inf

from ..util import ServerConfig, DistanceFilter
from ..vasp import VaspConfig, Kspacing
from .random import rattle
from .minimize import minimize
from .spg import spg
from ..projectflow import RunAgain, WorkflowProjectConfig

from pyiron_base import Project


class State(Enum):
    """
    The current state of the structure generation.
    """

    SPG = "spg"
    VOLMIN = "volmin"
    ALLMIN = "allmin"
    RANDOM = "random"
    FINISHED = "finished"


@dataclass
class TrainingDataConfig:
    def __post_init__(self):
        if self.stoichiometry is None:
            self.stoichiometry = list(range(1, self.max_atoms + 1))
        if isinstance(self.vasp, dict):
            self.vasp = VaspConfig(**self.vasp)
        if isinstance(self.server, dict):
            self.server = ServerConfig(**self.server)
        if isinstance(self.workflow, dict):
            self.workflow = WorkflowProjectConfig(**self.workflow)
        self.delete_existing_job = bool(self.delete_existing_job)
        # FIXME: Backwards compat
        if self.delete_existing_job:
            self.workflow.delete_existing_job = True

    elements: List[str]
    name: str
    max_atoms: int = 10
    stoichiometry: List[int] = None
    trace: int = 3
    rattle_disp: float = 0.5
    rattle_strain: float = 0.05
    rattle_repetitions: int = 4
    stretch_hydro: float = 0.8
    stretch_shear: float = 0.2
    stretch_repetitions: int = 4
    # can be either a single cutoff distance or a dictionary mapping chemical
    # symbols to min *radii*; you need to half the value if you go from using a
    # float to a dict
    min_dist: float | dict[str, float] = None
    # deprecated; use workflow config
    delete_existing_job: bool = False

    vasp: VaspConfig = field(
        default_factory=lambda: VaspConfig(encut=None, kmesh=Kspacing(0.5))
    )
    server: ServerConfig = field(
        default_factory=lambda: ServerConfig(cores=10, run_time=5 * 60, queue="cmti")
    )
    workflow: WorkflowProjectConfig = field(
        default_factory=lambda: WorkflowProjectConfig(
            delete_existing_job=False,
            broken_threshold=0.1,
            finished_threshold=0.9,
        )
    )

    def get_distance_filter(self):
        match self.min_dist:
            case float():
                return DistanceFilter({el: self.min_dist / 2 for el in self.elements})
            case dict():
                return DistanceFilter(self.min_dist)
            case _:
                assert (
                    False
                ), f"min_dist cannot by of type {type(self.min_dist)}: {self.min_dist}!"


def create_structure_set(
    pr: Project,
    state: Union[str, State],
    conf: TrainingDataConfig,
    fast_forward: bool = False,
) -> State:
    """
    Create a new structure set for training.

    Args:
        pr (Project): project to work in
        state (str, :class:`.State`): the current state
        conf (:class:`.TrainingDataConfig`): the configuration for the structure set

    Returns:
        :class:`.State`: the current state the generation is in; if this is not :attr:`.State.FINISHED` you should call
        this function again after some time until it is.
    """
    state = State(state)
    logger = getLogger("structures")
    buf = StringIO("")
    pprint(asdict(conf), stream=buf)
    logger.info("config: %s", buf.getvalue())
    logger.info("current state: %s", state.value)
    if state == State.SPG:
        spg(
            pr.create_group("containers"),
            conf.elements,
            conf.max_atoms,
            conf.stoichiometry,
            name=conf.name,
            min_dist=conf.min_dist,
            delete_existing_job=conf.workflow.delete_existing_job,
        )
        state = State.VOLMIN
        if not fast_forward:
            return state
    if state == State.VOLMIN:
        try:
            cont = pr["containers"].load(f"{conf.name}")
            if cont is None:
                state = State.SPG
                logger.error("failed to load previous structure container, backtracing...")
                return create_structure_set(pr, state, conf, fast_forward)
            minimize(
                pr,
                cont,
                "volume",
                conf.trace,
                conf.min_dist,
                conf.vasp,
                conf.server,
                conf.workflow,
            )
        except RunAgain:
            return state
        state = State.ALLMIN
        if not fast_forward:
            return state
    if state == State.ALLMIN:
        try:
            cont = pr["containers"].load(f"{conf.name}VolMin")
            if cont is None:
                state = State.VOLMIN
                logger.error("failed to load previous structure container, backtracing...")
                return create_structure_set(pr, state, conf, fast_forward)
            minimize(
                pr,
                cont,
                "all",
                conf.trace,
                conf.min_dist,
                conf.vasp,
                conf.server,
                conf.workflow,
            )
        except RunAgain:
            return state
        state = State.RANDOM
        if not fast_forward:
            return state
    if state == State.RANDOM:
        cont = pr["containers"].load(f"{conf.name}VolMinAllMin")
        if cont is None:
            state = State.ALLMIN
            logger.error("failed to load previous structure container, backtracing...")
            return create_structure_set(pr, state, conf, fast_forward)
        rattle(
            pr["containers"],
            cont,
            conf.rattle_disp,
            conf.rattle_strain,
            conf.rattle_repetitions,
            conf.stretch_hydro,
            conf.stretch_shear,
            conf.stretch_repetitions,
            filterf=conf.get_distance_filter(),
            delete_existing_job=conf.workflow.delete_existing_job,
        )
        state = State.FINISHED
    return state


def run(
    pr: Project, config: TrainingDataConfig, tries: Optional[int] = 10, wait: int = 60
):
    """
    Create structure set.

    Repeatedly calls :func:`.create_structure_set` until it finishes.
    Saves the current state in `pr.data` and resumes from last known state if called previously on the same project.

    Args:
        pr (Project): project to work in
        config (TrainingDataConfig): parameters for structure creation
        tries (int): how often to call :func:`.create_structure_set` on all containers; if None try indefinitely
        wait (int): how long to wait in between calls to :func:`.create_structure_set`
    """
    state = State(pr.data.get("state", "spg"))
    pr.data.config = asdict(config)
    pr.data.write()
    if tries is None:
        counter = count()
        tries = inf
    else:
        counter = range(tries)
    for i in counter:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'KSPACING' found in INCAR, no KPOINTS file written",
            )
            state = create_structure_set(pr, state, config, fast_forward=True)
        pr.data.state = state.value
        pr.data.write()
        if state == State.FINISHED:
            break
        if i + 1 < tries:
            time.sleep(wait)
    if state != State.FINISHED:
        warnings.warn(
            "Structure creation is not finished! Call this function again later!"
        )
    return state.value


def export_structures(pr, export, ending, format):
    os.makedirs(export, exist_ok=True)
    for cont in pr["containers"].iter_jobs(hamilton="StructureContainer"):
        dir_path = os.path.join(export, cont.name)
        os.makedirs(dir_path, exist_ok=True)
        for i, s in enumerate(cont.iter_structures()):
            s.write(
                os.path.join(dir_path, cont._container["identifier", i]) + "." + ending,
                format=format,
            )
