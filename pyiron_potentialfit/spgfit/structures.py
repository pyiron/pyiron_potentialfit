from dataclasses import dataclass, asdict
from enum import Enum
from io import StringIO
from pprint import pprint
from logging import getLogger, INFO

getLogger().setLevel(INFO)
logger = getLogger("structures")
from typing import Iterable, Optional, Tuple, List
from random import choices
from itertools import product

from pyiron_base import Project
from pyiron_atomistics.atomistics.job.structurecontainer import StructureContainer
from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics import ase_to_pyiron
from pyiron_potentialfit.atomistics.job.trainingcontainer import TrainingContainer
from typing import List, Union, Tuple
from tqdm.auto import tqdm


import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .projectflow import (
    ProjectFlow,
    StructureProjectFlow,
    Input,
    Output,
    RunAgain,
    WorkflowProjectConfig,
)
from .util import RCORE, DistanceFilter, fast_forward, ServerConfig
from .vasp import Kpoints, Kspacing, VaspConfig

from pyiron_contrib.jobfactories import VaspFactory
from pyiron_contrib.repair import HandyMan


def shake(displacement=0.1):
    """
    Return a function that randomly displaces atoms in structures.

    Args:
        displacement (float): standard deviation of atomic displacement
    """

    def mod(structure):
        structure.positions += np.random.normal(
            scale=displacement, size=structure.positions.shape
        )
        return structure

    return mod


def stretch(hydro: float = 0.05, shear: float = 0.005):
    """
    Return a function that strains structures.

    Random strains are drawn from a uniform distribution within the positive
    and negative limits given.

    Args:
        hydro (float): Maximum strain along normal axes
        shear (float): Maximum strain along shear axes
    """

    def mod(structure):
        E = shear * (2 * np.random.rand(3, 3) - 1)
        E = 0.5 * (E + E.T)  # symmetrize
        np.fill_diagonal(E, hydro * (2 * np.random.rand(3) - 1))
        structure.apply_strain(E)
        return structure

    return mod


def transmute(elems, conc, mask=None):
    if len(elems) != 2:
        raise ValueError("elems must contain two elements")

    def mod(structure):
        if mask is not None:
            indices = np.argwhere(mask(structure)).T[0]
        else:
            indices = range(len(structure))
        indices = np.array(indices)
        np.random.shuffle(indices)
        c = int(len(indices) * conc)
        for i in indices[:c]:
            structure[i] = elems[0]
        for i in indices[c:]:
            structure[i] = elems[1]
        return structure

    return mod


def fill_container(
    source: HasStructure,
    sink: StructureContainer,
    repetitions: int = 4,
    combine: int = 1,
    modifiers=((0.5, shake()), (0.5, stretch())),
    min_dist=None,
):
    """
    Fill a container with new structures.

    Iterates over all structures in `source`
    """
    ps, mods = zip(*modifiers)
    for structure in tqdm(source.iter_structures(), total=source.number_of_structures):
        for _ in range(repetitions):
            for i in range(10):
                s = structure.copy()
                for mod in choices(mods, weights=ps, k=combine):
                    s = mod(s)
                if min_dist is None:
                    sink.append(s)
                    break
                else:
                    sd = s.repeat(2)
                    sd.pbc = [False, False, False]
                    dist = sd.get_neighbors(
                        num_neighbors=1, cutoff_radius=2 * min_dist
                    ).distances
                    if (dist > min_dist).all():
                        sink.append(s)
                        break
            else:
                print(
                    "WARN: Tried 10 times to find a structures, but "
                    "min_dist is never satisfied."
                )
    return sink


def extract_structure(jobpath, frame):
    ii = jobpath["output/generic/indices"]
    if ii is not None:
        indices = ii[frame]
    else:
        indices = jobpath["input/structure/indices"]
    cell = jobpath["output/generic/cells"][frame]
    positions = jobpath["output/generic/positions"][frame]
    if len(indices) == len(jobpath["input/structure/indices"]):
        structure = jobpath["input/structure"].to_object()
        structure.positions[:] = positions
        structure.cell.array[:] = cell
        structure.indices[:] = indices
    else:
        structure = Atoms(
            species=jobpath["input/structure/species"],
            indices=indices,
            positions=positions,
            cell=cell,
            pbc=jobpath["input/structure/cell/pbc"],
        )
    return structure


from traitlets import Instance, Float, Bool, Int, CaselessStrEnum, Dict


class MinimizeVaspInput(Input):

    symlink = Bool(default_value=False, help="Whether to symlink the project or not")

    structures = Instance(StructureStorage, args=())
    degrees_of_freedom = CaselessStrEnum(
        values=["volume", "cell", "all", "internal"], default_value="volume"
    )

    encut = Float(default_value=None, allow_none=True)
    kspacing = Float(default_value=0.5)
    use_symmetry = Bool(default_value=False)

    vasp_config = Dict(default_value={})

    cores = Int(default_value=10)
    run_time = Float(default_value=1 * 60 * 60)


class MinimizeVaspFlow(ProjectFlow):

    Input = MinimizeVaspInput

    def _run(self, delete_existing_job=False, delete_aborted_job=True):
        sflow = StructureProjectFlow()

        # ugly little dance to avoid having to implement HDF for dataclasses correctly
        vasp_config = VaspConfig(**self.input.vasp_config)

        vasp = VaspFactory()
        # AlH specific hack, VaspFactory ignores this for other structures automatically
        vasp.enable_nband_hack({"Al": 2, "H": 2})  # = 3/2 + 1/2 VASP default
        vasp.incar["KSPACING"] = self.input.kspacing
        vasp.incar["EDIFF"] = 1e-6
        if not self.input.use_symmetry:
            vasp.incar["ISYM"] = 0
        if self.input.encut is not None:
            vasp.set_encut(self.input.encut)
        vasp.cores = self.input.cores
        vasp.run_time = self.input.run_time
        vasp.queue = "cmti"

        if self.input.degrees_of_freedom == "volume":
            vasp.minimize_volume()
        elif self.input.degrees_of_freedom == "all":
            vasp.minimize_all()
        elif self.input.degrees_of_freedom == "internal":
            vasp.minimize_internal()
        else:
            assert (
                False
            ), f"DoF cannot be {self.input.degrees_of_freedom}, traitlets broken?"

        sflow.input.job = vasp
        if vasp_config.magmoms is not None:
            def apply_magmom(structure):
                return structure.copy().set_initial_magnetic_moments(
                        [vasp_config.magmoms.get(sym, 0.0) for sym in structure.symbols]
                )
            sflow.input.structures = self.input.structures.transform_structures(apply_magmom)
        else:
            sflow.input.structures = self.input.structures.copy()
        sflow.input.table_setup = lambda tab: tab

        sflow.attach(self.project, "structures").run()

    def _analyze(self, delete_existing_job=False):
        # TODO: move both collects here and store structures in a custom Output
        pass

    def collect(
        self,
        name,
        number_of_structures=1,
        min_dist=None,
        accept_not_converged=False,
        delete_existing_job=False,
    ) -> StructureContainer:
        cont = self.project.create.job.StructureContainer(
            name, delete_existing_job=delete_existing_job, delete_aborted_job=True
        )
        if cont.status.finished:
            return cont

        if accept_not_converged:
            df = self.project.job_table(hamilton="Vasp")
            df = df.query("status.isin(['finished', 'not_converged', 'warning'])")
            jobs = map(self.project.inspect, tqdm(df.id))
        else:
            jobs = self.project.iter_jobs(
                hamilton="Vasp", status="finished", convert_to_object=False
            )

        for j in jobs:
            N = len(j["output/generic/steps"])
            stride = max(N // number_of_structures, 1)
            for i in range(1, N + 1, stride)[:number_of_structures]:
                s = extract_structure(j, -i)
                if (
                    min_dist is not None
                    and np.prod(s.get_neighbors(cutoff_radius=min_dist).distances.shape)
                    > 0
                ):
                    continue
                if number_of_structures == 1:
                    name = j.name
                else:
                    name = f"{j.name}_step_{i}"
                cont.add_structure(s, identifier=name, job_id=j.id, step=i)
        cont.run()
        return cont

    def collect_training(
        self, name, min_dist=None, delete_existing_job=False
    ) -> TrainingContainer:
        cont = self.project.create.job.TrainingContainer(
            name, delete_existing_job=delete_existing_job, delete_aborted_job=True
        )
        if not cont.status.initialized:
            return cont

        cont.input.save_neighbors = False

        jobs = self.project.iter_jobs(
            hamilton="Vasp", status="finished", convert_to_object=False
        )

        for j in jobs:
            s = extract_structure(j, -1)
            if (
                min_dist is not None
                and np.prod(s.get_neighbors(cutoff_radius=min_dist).distances.shape) > 0
            ):
                continue
            cont.include_job(j)
        cont.run()
        return cont


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

    vasp: VaspConfig = VaspConfig(encut=None, kmesh=Kspacing(0.5))
    server: ServerConfig = ServerConfig(cores=10, run_time=5*60, queue='cmti')


class State(Enum):
    """
    The current state of the structure generation.
    """

    SPG = "spg"
    VOLMIN = "volmin"
    ALLMIN = "allmin"
    RANDOM = "random"
    FINISHED = "finished"


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


from pyxtal import pyxtal
from pyxtal.msg import Comp_CompatibilityError, VolumeError
from pyxtal.tolerance import Tol_matrix


def _pyxtal(
    group: Union[int, List[int]],
    species: Tuple[str],
    num_ions: Tuple[int],
    dim=3,
    repeat=1,
    storage=None,
    allow_exceptions=True,
    checker=lambda _: True,
    **kwargs,
) -> Union[Atoms, StructureStorage]:
    """
    Generate random crystal structures with PyXtal.

    `group` must be between 1 and the largest possible value for the given dimensionality:
        dim=3 => 1 - 230 (space groups)
        dim=2 => 1 -  80 (layer groups)
        dim=1 => 1 -  75 (rod groups)
        dim=0 => 1 -  58 (point groups)

    When `group` is passed as a list of integers or `repeat>1`, generate multiple structures and return them in a :class:`.StructureStorage`.

    Args:
        group (list of int, or int): the symmetry group to generate or a list of them
        species (tuple of str): which species to include, defines the stoichiometry together with `num_ions`
        num_ions (tuple of int): how many of each species to include, defines the stoichiometry together with `species`
        dim (int): dimensionality of the symmetry group, 0 is point groups, 1 is rod groups, 2 is layer groups and 3 is space groups
        repeat (int): how many random structures to generate
        storage (:class:`.StructureStorage`, optional): when generating multiple structures, add them to this instead of creating a new storage
        allow_exceptions (bool): when generating multiple structures, silence errors when the requested stoichiometry and symmetry group are incompatible
        **kwargs: passed to `pyxtal.pyxtal` function verbatim

    Returns:
        :class:`~.Atoms`: the generated structure, if repeat==1 and only one symmetry group is requested
        :class:`.StructureStorage`: a storage of all generated structure, if repeat>1 or multiple symmetry groups are requested

    Raises:
        ValueError: if stoichiometry and symmetry group are incompatible and allow_exceptions==False or only one structure is requested
    """
    logger = getLogger("structures")

    def generate(group):
        s = pyxtal()
        factor = 1
        for _ in range(5):
            try:
                s.from_random(
                    dim=dim, group=group, species=species, numIons=num_ions, **kwargs
                )
                s = ase_to_pyiron(s.to_ase())
                s.center_coordinates_in_unit_cell()
                return s
            except RuntimeError as err:
                if err.args[0] == "long time to generate structure, check inputs":
                    logger.warn(
                        f"pyxtal complained: {err.args} {factor} {dim} {group} {species} {num_ions}"
                    )
                if not err.args[0].startswith("Volume"):
                    raise
            except VolumeError:
                pass
            except:
                raise
            factor *= 1.5
        else:
            raise RuntimeException(
                "Failed to generate structure, aborted after factor: {factor}!"
            )

    # return a single structure
    if repeat == 1 and isinstance(group, int):
        return generate(group)
    else:
        if storage is None:
            storage = StructureStorage()
        if isinstance(group, int):
            group = [group]
        failed_groups = []
        for g in tqdm(group, desc="Spacegroups"):
            for i in range(repeat):
                stoich = "".join(f"{s}{n}" for s, n in zip(species, num_ions))
                try:
                    for _ in range(5):
                        s = generate(g)
                        if checker(s):
                            break
                    else:
                        logger.warn("Check failed 5 times in a row, skipping!")
                        continue
                except (Comp_CompatibilityError, RuntimeError) as e:
                    if allow_exceptions:
                        # This exception indicates that the stoichiometry is generally incompatible with the chosen group
                        # so we can just skip it
                        failed_groups.append(g)
                        break
                    else:
                        raise ValueError(
                            f"Symmetry group {g} incompatible with stoichiometry {stoich}!"
                        ) from None
                # some structures come out with really weird cell shapes, especially with low number of atoms
                # get the primitive cell as per spglib to "normalize" that a bit
                # at the same time we do *not* want to reduce the size of the cells, because having a few larger super
                # cells will allow us to sample their displacements a bit more
                ps = s.get_symmetry().get_primitive_cell()
                if len(ps) == len(s):
                    s = ps
                storage.add_structure(
                    s, identifier=f"{stoich}_{g}_{i}", symmetry=g, repeat=i
                )
        if len(failed_groups) > 0:
            stoich = "".join(f"{s}{n}" for s, n in zip(species, num_ions))
            logger.warning(
                f'Groups [{", ".join(map(str,failed_groups))}] could not be generated with stoichiometry {stoich}!'
            )
        return storage


def spg(
    pr,
    elements,
    max_atoms,
    stoichiometry,
    name="Crystals",
    min_dist=None,
    delete_existing_job=False,
):
    logger = getLogger("structures")
    logger.info("Creating new structures for %s <= %i", elements, max_atoms)
    store = pr.create.job.StructureContainer(
        name, delete_existing_job=delete_existing_job
    )
    if store.status.finished:
        return store

    if min_dist is not None:
        tm = Tol_matrix.from_single_value(min_dist)
    else:
        # function is called radii, but source code suggest it is actually used
        # to check the *distance* between to atom pairs, so we multiply by two
        # here (because the pair distance is made up from two radii)
        tm = Tol_matrix.from_radii([2 * r for r in RCORE.values()])
    stoichs = [
        ni
        for ni in product(stoichiometry, repeat=len(elements))
        if sum(ni) <= max_atoms
    ]
    if len(stoichs) == 0:
        logger.critical(
            f"No valid stoichiometries for {elements}, {stoichiometry} <= {max_atoms}!"
        )
    for num_ions in (bar := tqdm(stoichs)):
        if sum(num_ions) == 0:
            continue
        stoich = "".join(f"{s}{n}" for s, n in zip(elements, num_ions))
        bar.set_description(f"Stoichiometry {stoich}")

        def check_cell_shape(structure):
            # Want to avoid structures that are very long but narrow
            # vecs = np.linalg.norm(structure.cell.array, axis=-1)
            vecs = structure.cell.lengths()
            return vecs.max() / vecs.min() < 6

        # very few structures with low distances seem to slip through pyxtals checks, so double check here
        if min_dist is None:
            distance_filter = DistanceFilter()
        else:
            distance_filter = DistanceFilter({e: min_dist/2 for e in elements})
        el, ni = zip(*((el, ni) for el, ni in zip(elements, num_ions) if ni > 0))
        # missing checker support
        # pr.create.structure.pyxtal(
        _pyxtal(
            range(1, 230 + 1),
            species=el,
            num_ions=ni,
            storage=store,
            checker=lambda s: check_cell_shape(s) and distance_filter(s),
            factor=1.5,
            tm=tm,
        )
    store["user/num_atoms"] = stoichiometry
    store.run()
    return store


def minimize(
    pr,
    cont: StructureContainer,
    degrees_of_freedom,
    trace,
    min_dist,
    vasp: VaspConfig,
    server: ServerConfig,
    delete_existing_job=False,
):
    logger = getLogger("structures")
    logger.info("Minimizing structures: %s -> %s", cont.name, degrees_of_freedom)
    n = {"volume": "VolMin", "all": "AllMin", "cell": "CellMin", "internal": "IntMin"}[
        degrees_of_freedom
    ]
    minf = MinimizeVaspFlow(pr, f"{cont.name}{n}")

    def if_new(flow):
        logger.info("starting from scratch")
        if flow.input.read_only:
            flow.input.unlock()
        flow.input.structures = cont._container.copy()
        # FIXME: join together
        flow.input.vasp_config = asdict(vasp)
        flow.input.kspacing = vasp.kmesh.kspacing
        flow.input.degrees_of_freedom = degrees_of_freedom
        flow.input.cores = server.cores
        flow.input.run_time = server.run_time
        flow.run(delete_existing_job=delete_existing_job)
        raise RunAgain("Just starting!")

    def if_finished(flow):
        logger.info("collecting structures")
        if trace > 1:
            cont = pr["containers"].load(f"{flow.project.name}Trace")
            if cont is None:
                cont = flow.collect(
                    "Trace",
                    min_dist=min_dist,
                    number_of_structures=trace,
                    accept_not_converged=True,
                )
                cont.copy_to(
                    pr.create_group("containers"),
                    new_job_name=f"{flow.project.name}Trace",
                )
        cont = pr["containers"].load(flow.project.name)
        if cont is None:
            cont = flow.collect(
                "Final",
                min_dist=min_dist,
                number_of_structures=1,
                accept_not_converged=True,
            )
            cont.copy_to(pr.create_group("containers"), new_job_name=flow.project.name)
        return cont

    config = WorkflowProjectConfig(
        delete_existing_job=delete_existing_job,
        broken_threshold=0.1,
        finished_threshold=0.9,
    )
    return minf.check(config, if_new, if_finished)


def rattle(
    pr,
    cont: StructureContainer,
    rattle_disp,
    rattle_strain,
    rattle_repetitions,
    stretch_hydro,
    stretch_shear,
    stretch_repetitions,
    min_dist,
    delete_existing_job=False,
):
    logger = getLogger("structures")
    logger.info("Creating rattle/stretch structures")

    rand = pr.create.job.StructureContainer(
        f"{cont.name}Random", delete_existing_job=delete_existing_job
    )
    if rand.status.initialized:
        N = 0
        fill_container(
            cont._container.sample(lambda f, i: f["length", i] > 1),
            rand,
            repetitions=rattle_repetitions,
            combine=2,
            modifiers=(
                (1, shake(rattle_disp)),
                (1, stretch(rattle_strain))
            ),
            min_dist=min_dist,
        )
        logger.info("added %i rattle structures", rand.number_of_structures - N)
        N = rand.number_of_structures
        fill_container(
            cont._container,
            rand,
            repetitions=stretch_repetitions,
            combine=1,
            modifiers=(
                (0.7, stretch(hydro=stretch_hydro, shear=0.05)),
                (0.3, stretch(hydro=0.05, shear=stretch_shear)),
            ),
            min_dist=min_dist,
        )
        logger.info("added %i stretch structures", rand.number_of_structures - N)
        rand.run()
    return rand


epilog = """
Follows the systematic approach reported in this paper[1].  Execution is
organized in four steps:

    1. Generation of random structures with pyxtal. (5min to 1h, depending on #elements)
    2. Volume minimization at low convergence. (<30s walltime per structure, total depends on cluster load)
    3. Full minimization at low convergence. (<2min walltime per structure, total depends on cluster load)
    4. Application of random strains, shears and rattling of atoms (at most a few min)

Each step is built from the structures obtained in the previous one.
Minimization is done at default plane wave cutoff and KSPACING=0.5.

We lay out the project like this,

`root/structures`:                      work exlusively in this project
`root/structures/containers`:           will ultimately contain the structures ready for high convergence calculations
`root/structures/{name}VolMin`:         working project for the volume minimization
`root/structures/{name}VolMinAllmin`:   working project for the full minimization

where `root` is the programs working directory and `name` what you passed as --name.
The container project will contain a `StructureStorage` job for each of the stages.  The minimization stages write two
containers, one for the final structures and one that includes intermediate steps of the minimization (with postifx
`Trace`).  The former is used as input for the next stage, the latter is intended to be used for high convergence
calculations with potfit.calculations.

The program is designed to be run multiple times, each time checking whether it can advance to the next stage.  To have to
program loop until it reaches the final state, use `--fast-forward`.  After the first invocation the program caches the
input and you may run it without additional flags (but it doesn't hurt to do it either).  DFT calculations that abort or
fail to converge during the minimization steps are automatically restarted.  The program will log to stdout its current
stage and what it does inside each.

For example you can create a simple data set for Al with

> python -m potfit.structures --elements Al --stoichiometry 4 8 --name Al4and8 --fast-forward

For alloys usage is similar, but you can also specify --max-atoms to limit the total size of your system, e.g.

> python -m potfit.structures --elements Fe Al --stoichiometry 2 4 8 --max-atoms 10

will include Fe2Al8 and Fe8Al2, but not Fe4Al8.

You should adjust `--min-dist` for each system under study.  Structures that are strongly repulsive can strongly affect
fitting results for a whole structure set.

[1]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.104103
"""


def main():
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="'KSPACING' found in INCAR, no KPOINTS file written",
    )

    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    parser = ArgumentParser(
        description="Create structures for training data.",
        epilog=epilog,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="Crystals",
        help="Base name of the structure containers",
    )
    parser.add_argument(
        "-e",
        "--elements",
        type=str,
        nargs="+",
        help="Chemical element to create structures for",
    )
    parser.add_argument(
        "-m",
        "--max-atoms",
        type=int,
        default=TrainingDataConfig.max_atoms,
        help="Maximum number of atoms per structure",
    )
    parser.add_argument(
        "-o",
        "--stoichiometry",
        type=int,
        nargs="*",
        help="number of atoms of each element to combine",
    )
    parser.add_argument(
        "-t",
        "--trace",
        type=int,
        default=TrainingDataConfig.trace,
        help="Number of structures to include from each minimization step",
    )
    parser.add_argument(
        "-r",
        "--rattle",
        type=float,
        nargs=2,
        default=(TrainingDataConfig.rattle_disp, TrainingDataConfig.rattle_strain),
        help="The displacement (in A) and tri-axial strain for the Rattle set",
    )
    parser.add_argument(
        "--rattle-repetitions",
        type=int,
        default=TrainingDataConfig.rattle_repetitions,
        help="How many rattled structures to generate from each minimized structure",
    )
    parser.add_argument(
        "-s",
        "--stretch",
        type=float,
        nargs=2,
        default=(TrainingDataConfig.stretch_hydro, TrainingDataConfig.stretch_shear),
        help="The tri-axial and shear strains for the Stretch set",
    )
    parser.add_argument(
        "--stretch-repetitions",
        type=int,
        default=TrainingDataConfig.stretch_repetitions,
        help="How many stretched structures to generate from each minimized structure",
    )
    parser.add_argument(
        "-d",
        "--min-dist",
        type=float,
        default=None,
        help="Smallest nearest neighbor distance to allow; default is taken from RCORE value of Vasp PBE pseudopotential files",
    )
    parser.add_argument(
        "-p", "--project", default="structures", help="project to work in"
    )
    parser.add_argument(
        "--fast-forward",
        type=int,
        default=None,
        help="Automatically go to the next step after sleeping for this many seconds",
    )
    parser.add_argument(
        "--delete-existing-job",
        action="store_true",
        help="Retry the current step from scratch",
    )
    parser.add_argument(
        "--export", type=str,
        help="Optionally specify a directory where to dump POSCAR files with the generated structures after everything "
             "is finished."
    )
    parser.add_argument(
        "--magmom", nargs=2, action="append", default=[],
        help="Initial magnetic moments as `element symbol`, followed by the collinear magnetic moment; all atoms of "
             "the same element will be initialized the same; may be given multiple times, once per element"
    )

    parser.add_argument(
        "--cores", type=int, default=10,
        help="Number of cores to use per job during minimizations"
    )

    args = parser.parse_args()

    pr = Project(args.project)
    conf = vars(args).copy()
    conf["rattle_disp"], conf["rattle_strain"] = conf["rattle"]
    conf["stretch_hydro"], conf["stretch_shear"] = conf["stretch"]
    del conf["rattle"]
    del conf["stretch"]
    del conf["project"]
    del conf["fast_forward"]
    del conf["magmom"]
    del conf["cores"]
    conf = {k: v for k, v in conf.items() if v is not None}
    conf = TrainingDataConfig(**conf)

    for (el, mm) in args.magmom:
        mm = float(mm)
        if el not in args.elements:
            raise ValueError(f"Elements for magmoms, must be also given via -e, not {el}!")
        conf.vasp.magmoms[el] = mm

    conf.server.cores = args.cores

    state = pr.data.get("state", "spg")
    # old_conf = pr.data.get('config', {})
    # for k, v in old_conf.items():
    #     setattr(conf, k, v)
    pr.data.config = asdict(conf)
    pr.data.write()

    state = create_structure_set(pr, state, conf, args.fast_forward is not None).value
    pr.data.state = state
    pr.data.write()
    if state != "finished" and args.fast_forward is not None:
        fast_forward(args.fast_forward, __spec__)
    if state == "finished" and args.export is not None:
        os.makedirs(args.export, exist_ok=True)
        for cont in pr["containers"].iter_jobs(hamilton="StructureContainer"):
            dir_path = os.path.join(args.export, cont.name)
            os.makedirs(dir_path, exist_ok=True)
            for i, s in enumerate(cont.iter_structures()):
                s.write(os.path.join(dir_path, cont._container["identifier", i]) + ".POSCAR", format="vasp")


if __name__ == "__main__":
    main()
