__version__ = "0.1"
__all__ = []


from pyiron_atomistics import Project
from pyiron_base import JOB_CLASS_DICT


# Make classes available for new pyiron version
JOB_CLASS_DICT.update(
    {
        "TrainingContainer": "pyiron_potentialfit.atomistics.job.trainingcontainer",
        "RandomDisMaster": "pyiron_potentialfit.mlip.masters",
        "RandomMDMaster": "pyiron_potentialfit.mlip.masters",
        "RunnerFit": "pyiron_potentialfit.runner.job",
        "MlipSelect": "pyiron_potentialfit.mlip.mlipselect",
        "Mlip": "pyiron_potentialfit.mlip.mlip",
        "LammpsMlip": "pyiron_potentialfit.mlip.lammps",
        "MlipJob": "pyiron_potentialfit.mlip.mlipjob",
        "Atomicrex": "pyiron_potentialfit.atomicrex.atomicrex_job",
        "StructureMasterInt": "pyiron_potentialfit.atomistics.job.structurelistmasterinteractive",
        "MlipDescriptors": "pyiron_potentialfit.mlip.mlipdescriptors",
        "PacemakerJob": "pyiron_potentialfit.pacemaker.job",
        "MeamFit": "pyiron_potentialfit.meamfit.meamfit",
        "FitsnapJob": "pyiron_potentialfit.fitsnap.job",
    }
)


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
