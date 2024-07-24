from dataclasses import asdict
from logging import getLogger

from ..util import ServerConfig
from ..vasp import VaspConfig
from ..projectflow import (
    ProjectFlow,
    StructureProjectFlow,
    Input,
    RunAgain,
    WorkflowProjectConfig,
)

from pyiron_potentialfit.atomistics.job.trainingcontainer import TrainingContainer
from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage
from pyiron_atomistics.atomistics.job.structurecontainer import StructureContainer
from pyiron_contrib.jobfactories import VaspFactory

from traitlets import Instance, Float, Bool, Int, CaselessStrEnum, Dict

class MinimizeVaspInput(Input):

    symlink = Bool(default_value=False, help="Whether to symlink the project or not")

    structures = Instance(StructureStorage, args=())
    degrees_of_freedom = CaselessStrEnum(
        values=["volume", "cell", "all", "internal"], default_value="volume"
    )

    # encut = Float(default_value=None, allow_none=True)
    kspacing = Float(default_value=0.5)
    # use_symmetry = Bool(default_value=False)

    vasp_config = Dict(default_value={})
    server_config = Dict(default_value={})

    # cores = Int(default_value=10)
    # run_time = Float(default_value=1 * 60 * 60)


class MinimizeVaspFlow(ProjectFlow):

    Input = MinimizeVaspInput

    def _run(self, delete_existing_job=False, delete_aborted_job=True):
        sflow = StructureProjectFlow()

        # ugly little dance to avoid having to implement HDF for dataclasses correctly
        vasp_config = VaspConfig(**self.input.vasp_config)
        server_config = ServerConfig(**self.input.server_config)

        vasp = VaspFactory()
        # AlH specific hack, VaspFactory ignores this for other structures automatically
        vasp.enable_nband_hack({"Al": 2, "H": 2})  # = 3/2 + 1/2 VASP default
        vasp_config.configure_vasp_job(vasp)
        server_config.configure_server_on_job(vasp)

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
                structure.set_initial_magnetic_moments(
                        [vasp_config.magmoms.get(sym, 0.0) for sym in structure.symbols]
                )
                return structure
            sflow.input.structures = self.input.structures.transform_structures(apply_magmom)
        else:
            sflow.input.structures = self.input.structures.copy()
        sflow.input.table_setup = lambda tab: tab

        sflow.attach(self.project, "structures").run()

    def _analyze(self, delete_existing_job=False):
        # TODO: move both collects here and store structures in a custom Output
        pass

    @staticmethod
    def _extract_structure(jobpath, frame):
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
                pbc=jobpath["input/stguctuge/cell/pbc"],
            )
        return structure

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
                s = self._extract_structure(j, -i)
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
            s = self._extract_structure(j, -1)
            if (
                min_dist is not None
                and np.prod(s.get_neighbors(cutoff_radius=min_dist).distances.shape) > 0
            ):
                continue
            cont.include_job(j)
        cont.run()
        return cont


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

    vasp.incar.setdefault("ISYM", 0)
    vasp.incar.setdefault("EDIFF", 1e-6)
    if server.queue is None:
        server.queue = "cmti"

    def if_new(flow):
        logger.info("starting from scratch")
        if flow.input.read_only:
            flow.input.unlock()
        flow.input.structures = cont._container.copy()
        flow.input.vasp_config = asdict(vasp)
        flow.input.degrees_of_freedom = degrees_of_freedom
        flow.input.server_config = asdict(server)
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
    return minf.check(config, if_new, if_finished,
                      number_of_jobs=cont.number_of_structures)
