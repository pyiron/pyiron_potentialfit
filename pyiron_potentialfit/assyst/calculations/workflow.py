from collections import defaultdict
from dataclasses import dataclass, field, asdict
from itertools import combinations_with_replacement
from typing import Optional, Iterable
import logging
import warnings
import time

from ..projectflow import StructureProjectFlow, WorkflowProjectConfig, RunAgain
from ..util import DistanceFilter, ServerConfig
from ..vasp import VaspConfig

from pyiron_base import Project
from pyiron_potentialfit.atomistics.job.trainingcontainer import TrainingContainer
from pyiron_contrib.jobfactories import VaspFactory
from structuretoolkit import get_neighbors


from tqdm.auto import tqdm
import numpy as np


class TrainingDataFlow(StructureProjectFlow):

    def run(self, delete_existing_job=False, delete_aborted_job=True):
        self.symlink = True
        super().run(
            delete_existing_job=delete_existing_job,
            delete_aborted_job=delete_aborted_job,
        )

    def collect(
        self,
        name: str = "Results",
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
            name, delete_existing_job=delete_existing_job, delete_aborted_job=True
        )
        if train.status.finished:
            return train

        hamilton = self.input.job.hamilton
        for j in self.project.iter_jobs(
            recursive=True,
            status="finished",
            hamilton=hamilton,
            convert_to_object=False,
        ):
            train.include_job(j)
        if num_neighbors is not None:
            train.input.save_neighbors = True
            train.input.num_neighbors = num_neighbors
            train.server.queue = "cmti"
            train.server.cores = 1
            train.server.max_memory = 1e-1 * train.number_of_structures
        train.run()
        return train


@dataclass
class CalculationConfig:
    vasp: VaspConfig
    server: ServerConfig
    workflow: WorkflowProjectConfig

    min_dist: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.vasp, dict):
            self.vasp = VaspConfig(**self.vasp)
        if isinstance(self.server, dict):
            self.server = ServerConfig(**self.server)
        if isinstance(self.workflow, dict):
            self.workflow = WorkflowProjectConfig(**self.workflow)

    def get_job(self):
        job = VaspFactory()
        job.enable_nband_hack({"Al": 2, "H": 2})  # = 3/2 + 1/2 VASP default
        job.set_eddrmm_handling(status="ignore")
        self.vasp.incar.setdefault("NCORE", min(5, self.server.cores))
        self.vasp.incar.setdefault("EDIFF", 1e-8)
        self.vasp.incar.setdefault("PREC", "Accurate")
        self.vasp.incar.setdefault("ALGO", "Fast")
        self.vasp.configure_vasp_job(job)
        self.server.configure_server_on_job(job)
        return job


def run_container(pr: Project, cont: "StructureContainer", config: CalculationConfig):
    """
    Run DFT on all structures in a container.

    Args:
        pr (Project): all DFT jobs are created in a sub project named by name of the given container
        cont (StructureContainer): structures that should be run
        config (:class:`.CalculationConfig`): parameters for the DFT calculations
    """
    logger = logging.getLogger("calculations")
    logger.info("running DFT on %s in project %s", cont.name, pr.path)

    train = TrainingDataFlow(pr, cont.name)

    def if_new(train):
        if config.min_dist is not None:
            if isinstance(config.min_dist, float):
                dfilter = DistanceFilter(
                    {el: config.min_dist / 2 for el in cont._container.get_elements()}
                )
            elif isinstance(config.min_dist, dict):
                dfilter = DistanceFilter(config.min_dist)
            else:
                assert False, f"min_dist cannot by of type {type(config.min_dist)}: {config.min_dist}!"
        else:
            dfilter = DistanceFilter()
        filtered_cont = cont._container.sample(lambda f, i: dfilter(f.get_structure(i)))

        if train.input.read_only:
            train.input.unlock()
        if config.vasp.magmoms is not None and len(config.vasp.magmoms) > 0:

            def apply_magmom(structure):
                if not structure.has("initial_magmoms"):
                    structure.set_initial_magnetic_moments(
                        [config.vasp.magmoms.get(sym, 0.0) for sym in structure.symbols]
                    )
                return structure

            filtered_cont = filtered_cont.transform_structures(
                apply_magmom
            ).collect_structures()
        train.input.structures = filtered_cont

        train.input.job = config.get_job()
        train.run(delete_existing_job=config.workflow.delete_existing_job)

    def if_finished(train):
        results = pr.create_group("containers").load(cont.name)
        if results is None:
            results = train.collect(num_neighbors=50)
            results.copy_to(
                pr.create_group("containers"), new_job_name=train.project.name
            )
        return results

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="'KSPACING' found in INCAR, no KPOINTS file written",
        )
        return train.check(
            config.workflow,
            if_new,
            if_finished,
            number_of_jobs=train.input.structures.number_of_structures,
        )


def run(
    pr: Project,
    config: CalculationConfig,
    *containers: "StructureContainer",
    tries: int = 10,
    wait: float = 60
):
    """
    Run high quality DFT on all structures in `containers`.

    Calls :func:`.run_container` underneath, which will create the initial DFT jobs and then attempt to converge and
    repair any issues.  Once the structures in a container are deemed converged, the results are packed into a
    :class:`.TrainingContainer` created in `pr['containers']`.

    Args:
        pr (Project): where to run it; will create a sub project named after every container given
        config (CalculationConfig): DFT/pyiron parameters
        *containers (StructureContainer): structures to run DFT on
        tries (int): how often to call :func:`.run_container` on all containers
        wait (float): how long to wait in between
    """
    if tries <= 0:
        raise ValueError("tries must be an positive integer!")
    pr.data.config = asdict(config)
    pr.data.write()
    if tries is None:
        counter = count()
        tries = np.inf
    else:
        counter = range(tries)
    for i in counter:
        retry = False
        for cont in containers:
            try:
                run_container(pr, cont, config)
            except RunAgain:
                retry = True
        if not retry:
            break
        if i + 1 < tries:
            time.sleep(wait)
    if retry:
        warnings.warn(
            "Structure creation is not finished! Call this function again later!"
        )


def deduplicate(cont, replace=True):
    D = defaultdict(list)
    DD = []
    for i in tqdm(range(len(cont._container))):
        if i in DD:
            continue
        N = cont._container["length", i]
        J = i + 1 + np.where(cont._container["length"][i + 1 :] == N)[0]
        Ji = np.where(
            np.isclose(
                cont._container["energy"][J], cont._container["energy", i], rtol=1e-3
            )
        )[0]
        J = J[Ji]
        for j in J:
            same = np.allclose(
                cont._container["positions", i], cont._container["positions", j]
            )
            same &= np.all(
                cont._container["symbols", i] == cont._container["symbols", j]
            )
            same &= np.allclose(cont._container["cell", i], cont._container["cell", j])
            same &= np.allclose(cont._container["pbc", i], cont._container["pbc", j])
            if same:
                D[i].append(j)
                DD.append(j)
    dcont = cont.sample(cont.name + "Unique", lambda f, i: i not in DD, run=False)
    if replace:
        cont.remove()
        dcont.rename(cont.name)
    return dcont


def combine(
    pr: Project,
    containers: Iterable[TrainingContainer],
    name="Everything",
    min_dist=None,
    force_cap=None,
    energy_cap=None,
    check_duplicates=True,
    delete_existing_job=False,
) -> TrainingContainer:
    """
    Combine a list of training containers into a new container.

    Args:
        pr (Project): where to put the new container
        containers (iterable of TrainingContainer): containers to combine
        min_dist (float or dict of str to float, optional): if given, filter structures that are have atoms than this;
                if a dict it specifies the minimal allowed radii of each element
        force_cap (float): filter structures that have atomic forces larger than this value
        check_duplicates (bool): discard duplicated structures; some care has been taken to optimize this, but it can be
                costly for large datasets
        delete_existing_job (bool): combine containers again, even if `pr[name]` exists already

    Returns:
        :class:`.TrainingContainer`: contained with the combined training data
    """
    every = pr.create.job.TrainingContainer(
        name, delete_existing_job=delete_existing_job
    )
    if not every.status.initialized:
        return every
    for cont in containers:
        if cont.status.initialized:
            continue
        df = cont.to_pandas()
        df["name"] = df.name.map(lambda s: cont.name + "_" + s)
        if energy_cap is not None:
            I = df.energy / df.number_of_atoms <= energy_cap
            df = df.loc[I]
        if force_cap is not None:
            I = df.forces.map(lambda f: np.linalg.norm(f, axis=-1).max() < force_cap)
            df = df.loc[I]
        if min_dist is not None:
            if isinstance(min_dist, dict):

                def element_wise_dist(a):
                    pair = defaultdict(lambda: np.inf)
                    n = a.get_neighbors(
                        num_neighbors=25, cutoff_radius=5, mode="ragged"
                    )
                    for i, (I, D) in enumerate(zip(n.indices, n.distances)):
                        for j, d in zip(I, D):
                            ei, ej = sorted((a.symbols[i], a.symbols[j]))
                            pair[ei, ej] = min(d, pair[ei, ej])
                    return pair

                def larger_than_min_dist(a):
                    pair = element_wise_dist(a)
                    for ei, ej in combinations_with_replacement(
                        a.get_species_symbols(), 2
                    ):
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
            df = df.loc[I]
        every.include_dataset(df)
    if check_duplicates:
        every.save()
        every.status.finished = True
        every = deduplicate(every, replace=True)
    every.input.save_neighbors = True
    every.input.num_neighbors = 150
    every.server.queue = "cmti"
    every.server.cores = 1
    every.server.run_time = 60 * 60
    every.run()
    return every
