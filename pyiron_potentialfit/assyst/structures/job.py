from dataclasses import asdict

from pyiron_base import PythonTemplateJob

from .workflow import export_structures, run, TrainingDataConfig

class AssystStructures(PythonTemplateJob):
    """
    Create structure set with ASSYST.
    """
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input.update(asdict(TrainingDataConfig(
                elements=["Mg"],
                name="Mg10"
        )))

    def run_static(self):
        self.status.running = True
        run(self.child_project, TrainingDataConfig(**self.input))
        self.status.collect = True
        for cont in self.child_project["containers"].iter_jobs(hamilton="StructureContainer"):
            # proper but loses identifiers
            # self.output[cont.name] = cont.collect_structures()
            self.output[cont.name] = cont._container.copy()
        self.to_hdf()
        self.status.finished = True

    def export_structures(self, path, ending="POSCAR", format="vasp"):
        """
        Export structure set to files for external use.

        Args:
            path (str): directory to write files to
            ending (str): file ending
            format (str): file format to use (follows ASE names)
        """
        export_structures(self.child_project, path, ending, format)
