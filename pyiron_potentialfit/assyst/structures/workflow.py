from enum import Enum
from dataclasses import dataclass, field
from typing import List, Union

from ..util import ServerConfig
from ..vasp import VaspConfig, Kspacing
from .random import rattle
from .minimize import minimize
from .spg import spg

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
    min_dist: float = None
    delete_existing_job: bool = False

    vasp: VaspConfig = field(default_factory=lambda: VaspConfig(encut=None, kmesh=Kspacing(0.5)))
    server: ServerConfig = field(default_factory=lambda: ServerConfig(cores=10, run_time=5*60, queue='cmti'))

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
            delete_existing_job=conf.delete_existing_job,
        )
        state = State.VOLMIN
        if not fast_forward:
            return state
    if state == State.VOLMIN:
        try:
            cont = pr["containers"].load(f"{conf.name}")
            minimize(
                pr,
                cont,
                "volume",
                conf.trace,
                conf.min_dist,
                conf.vasp,
                conf.server,
                delete_existing_job=conf.delete_existing_job,
            )
        except RunAgain:
            return state
        state = State.ALLMIN
        if not fast_forward:
            return state
    if state == State.ALLMIN:
        try:
            cont = pr["containers"].load(f"{conf.name}VolMin")
            minimize(
                pr,
                cont,
                "all",
                conf.trace,
                conf.min_dist,
                conf.vasp,
                conf.server,
                delete_existing_job=conf.delete_existing_job,
            )
        except RunAgain:
            return state
        state = State.RANDOM
        if not fast_forward:
            return state
    if state == State.RANDOM:
        cont = pr["containers"].load(f"{conf.name}VolMinAllMin")
        rattle(
            pr["containers"],
            cont,
            conf.rattle_disp,
            conf.rattle_strain,
            conf.rattle_repetitions,
            conf.stretch_hydro,
            conf.stretch_shear,
            conf.stretch_repetitions,
            conf.min_dist,
            delete_existing_job=conf.delete_existing_job,
        )
        state = State.FINISHED
    return state

def run(pr, config):
    state = State.SPG
    while (state := create_structure_set(pr, state, conf, True)) != State.FINISHED:
        time.sleep(60)

def export_structures(pr, export, ending, format):
    os.makedirs(export, exist_ok=True)
    for cont in pr["containers"].iter_jobs(hamilton="StructureContainer"):
        dir_path = os.path.join(export, cont.name)
        os.makedirs(dir_path, exist_ok=True)
        for i, s in enumerate(cont.iter_structures()):
            s.write(os.path.join(dir_path, cont._container["identifier", i]) + "." + ending, format=format)
