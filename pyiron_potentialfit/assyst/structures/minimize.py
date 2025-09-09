from dataclasses import asdict
from logging import getLogger

import numpy as np

from ..util import ServerConfig
from ..vasp import VaspConfig
from ..projectflow import (
    ProjectFlow,
    StructureProjectFlow,
    Input,
    Output,
    RunAgain,
    WorkflowProjectConfig,
)
from ..jobfactories import VaspFactory

from pyiron_potentialfit.atomistics.job.trainingcontainer import (
    TrainingContainer,
    TrainingStorage,
)
from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage
from pyiron_atomistics.atomistics.job.structurecontainer import StructureContainer

from traitlets import Instance, Float, Bool, Int, CaselessStrEnum, Dict


class MinimizeVaspInput(Input):

    symlink = Bool(default_value=False, help="Whether to symlink the project or not")

    structures = Instance(StructureStorage, args=())
    degrees_of_freedom = CaselessStrEnum(
        values=["volume", "cell", "all", "internal"], default_value="volume"
    )

    vasp_config = Dict(default_value={})
    server_config = Dict(default_value={})

    # old flags; by now fully absorbed into vasp/server config, but keep around in case backwards compat is a pain
    # kspacing = Float(default_value=0.5)
    # encut = Float(default_value=None, allow_none=True)
    # use_symmetry = Bool(default_value=False)
    # cores = Int(default_value=10)
    # run_time = Float(default_value=1 * 60 * 60)

    number_of_structures = Int(default_value=1)
    min_dist = Float(default_value=None, allow_none=True)
    accept_not_converged = Int(default_value=False)


class MinimizeOutput(Output):
    final_structures = Instance(TrainingStorage, args=())
    trace_structures = Instance(TrainingStorage, args=())


class MinimizeVaspFlow(ProjectFlow):

    Input = MinimizeVaspInput
    Output = MinimizeOutput

    def _run(self, delete_existing_job=False, delete_aborted_job=True):
        sflow = StructureProjectFlow()

        # ugly little dance to avoid having to implement HDF for dataclasses correctly
        vasp_config = VaspConfig(**self.input.vasp_config)
        server_config = ServerConfig(**self.input.server_config)

        vasp = VaspFactory()
        # AlH specific hack, VaspFactory ignores this for other structures automatically
        vasp.enable_nband_hack({"Al": 2, "H": 2})  # = 3/2 + 1/2 VASP default

        if self.input.degrees_of_freedom == "volume":
            ediffg = vasp_config.incar.get("EDIFFG", 10 * vasp_config.incar["EDIFF"])
            if ediffg < 0:
                # user tries to set force tolerance which won't work for volume minimization!
                del vasp_config.incar["EDIFFG"]
            vasp.minimize_volume()
        elif self.input.degrees_of_freedom == "all":
            vasp.minimize_all()
        elif self.input.degrees_of_freedom == "internal":
            vasp.minimize_internal()
        else:
            assert (
                False
            ), f"DoF cannot be {self.input.degrees_of_freedom}, traitlets broken?"

        server_config.configure_server_on_job(vasp)
        vasp_config.configure_vasp_job(vasp)
        sflow.input.job = vasp
        if vasp_config.magmoms is not None and len(vasp_config.magmoms) > 0:

            def apply_magmom(structure):
                if not structure.has("initial_magmoms"):
                    structure.set_initial_magnetic_moments(
                        [vasp_config.magmoms.get(sym, 0.0) for sym in structure.symbols]
                    )
                return structure

            sflow.input.structures = self.input.structures.transform_structures(
                apply_magmom
            ).collect_structures()
        else:
            sflow.input.structures = self.input.structures.copy()
        sflow.input.table_setup = lambda tab: tab

        sflow.attach(self.project, "structures").run()

    def _analyze(self, delete_existing_job=False):
        if (
            self.output.final_structures.number_of_structures > 0
            and not delete_existing_job
        ):
            return
        ok_status = ["finished"]
        if self.input.accept_not_converged:
            ok_status += ["not_converged", "warning"]
        for j in self.project.iter_jobs(hamilton="Vasp", convert_to_object=False):
            if j.status not in ok_status:
                continue

            N = len(j["output/generic/steps"])
            stride = max(N // self.input.number_of_structures, 1)
            for i in range(1, N + 1, stride)[: self.input.number_of_structures]:
                s = self._extract_structure(j, -i)
                if (
                    self.input.min_dist is not None
                    and np.prod(
                        s.get_neighbors(
                            cutoff_radius=self.input.min_dist
                        ).distances.shape
                    )
                    > 0
                ):
                    continue
                name = j.content["user/structure"]
                if self.input.number_of_structures > 1:
                    name = f"{name}_step_{i}"
                self.output.trace_structures.include_job(
                    j, iteration_step=-i, identifier=name
                )
                if i == 1:
                    self.output.final_structures.include_job(
                        j, iteration_step=-i, identifier=name
                    )

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


def minimize(
    pr,
    cont: StructureContainer,
    degrees_of_freedom,
    trace,
    min_dist,
    vasp: VaspConfig,
    server: ServerConfig,
    workflow: WorkflowProjectConfig,
):
    logger = getLogger("structures")
    logger.info("Minimizing structures: %s -> %s", cont.name, degrees_of_freedom)
    n = {"volume": "VolMin", "all": "AllMin", "cell": "CellMin", "internal": "IntMin"}[
        degrees_of_freedom
    ]
    minf = MinimizeVaspFlow(pr, f"{cont.name}{n}")

    vasp.incar.setdefault("ISYM", 0)
    vasp.incar.setdefault("IBRION", 2)
    vasp.incar.setdefault("POTIM", 0.1)
    vasp.incar.setdefault("EDIFF", 1e-6)

    def if_new(flow):
        logger.info("starting from scratch")
        if flow.input.read_only:
            flow.input.unlock()
        flow.input.structures = cont._container.copy()
        flow.input.vasp_config = asdict(vasp)
        flow.input.degrees_of_freedom = degrees_of_freedom
        flow.input.server_config = asdict(server)
        # tricky: I kind of do not want to filter here
        # if a dict it's a dict of atomic radii, MinimizeVaspInput can only
        # understand scalars for now, so take twice the smallest radius
        # if isinstance(min_dist, dict):
        #     flow.input.min_dist = 2 * min(min_dist.values())
        # else:
        #     flow.input.min_dist = min_dist
        flow.run(delete_existing_job=workflow.delete_existing_job)
        raise RunAgain("Just starting!")

    def if_finished(flow):
        logger.info("collecting structures")
        flow.analyze()
        if trace > 1:
            cont = flow.project.load("Trace")
            if cont is None:
                cont = flow.project.create.job.StructureContainer("Trace")
                for i, s in enumerate(flow.output.trace_structures.iter_structures()):
                    cont.add_structure(
                        s, identifier=flow.output.trace_structures["identifier", i]
                    )
                cont.run()
                cont.copy_to(
                    pr.create_group("containers"),
                    new_job_name=f"{flow.project.name}Trace",
                )
        cont = flow.project.load("Final")
        if cont is None:
            cont = flow.project.create.job.StructureContainer("Final")
            for i, s in enumerate(flow.output.final_structures.iter_structures()):
                cont.add_structure(
                    s, identifier=flow.output.final_structures["identifier", i]
                )
            cont.run()
            cont.copy_to(pr["containers"], new_job_name=flow.project.name)
        return cont

    return minf.check(
        workflow, if_new, if_finished, number_of_jobs=cont.number_of_structures
    )
