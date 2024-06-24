from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations_with_replacement
import warnings
import os.path
import os

import numpy as np

from pyiron_base import Project, GenericParameters, GenericJob

# Extracted from Vasp PBE POTCARs (default valency)
# (implicitly sorted by atomic number, important!)
RCORE = {
    # RCORE as per POTCAR * bohr to angtrom factor
    "H": 1.100000 * 0.5291773,
    "He": 1.100000 * 0.5291773,
    "Li": 2.050000 * 0.5291773,
    "Be": 1.900000 * 0.5291773,
    "B": 1.700000 * 0.5291773,
    "C": 1.500000 * 0.5291773,
    "N": 1.500000 * 0.5291773,
    "O": 1.520000 * 0.5291773,
    "F": 1.520000 * 0.5291773,
    "Ne": 1.700000 * 0.5291773,
    "Na": 2.200000 * 0.5291773,
    "Mg": 2.000000 * 0.5291773,
    "Al": 1.900000 * 0.5291773,
    "Si": 1.900000 * 0.5291773,
    "P": 1.900000 * 0.5291773,
    "S": 1.900000 * 0.5291773,
    "Cl": 1.900000 * 0.5291773,
    "Ar": 1.900000 * 0.5291773,
    "K": 2.300000 * 0.5291773,
    "Ca": 2.300000 * 0.5291773,
    "Sc": 2.500000 * 0.5291773,
    "Ti": 2.800000 * 0.5291773,
    "V": 2.700000 * 0.5291773,
    "Cr": 2.500000 * 0.5291773,
    "Mn": 2.300000 * 0.5291773,
    "Fe": 2.300000 * 0.5291773,
    "Co": 2.300000 * 0.5291773,
    "Ni": 2.300000 * 0.5291773,
    "Cu": 2.300000 * 0.5291773,
    "Zn": 2.300000 * 0.5291773,
    "Ga": 2.600000 * 0.5291773,
    "Ge": 2.300000 * 0.5291773,
    "As": 2.100000 * 0.5291773,
    "Se": 2.100000 * 0.5291773,
    "Br": 2.100000 * 0.5291773,
    "Kr": 2.300000 * 0.5291773,
    "Rb": 2.500000 * 0.5291773,
    "Sr": 2.500000 * 0.5291773,
    "Y": 2.800000 * 0.5291773,
    "Zr": 3.000000 * 0.5291773,
    "Nb": 2.400000 * 0.5291773,
    "Mo": 2.750000 * 0.5291773,
    "Tc": 2.800000 * 0.5291773,
    "Ru": 2.700000 * 0.5291773,
    "Rh": 2.700000 * 0.5291773,
    "Pd": 2.600000 * 0.5291773,
    "Ag": 2.500000 * 0.5291773,
    "Cd": 2.300000 * 0.5291773,
    "In": 3.100000 * 0.5291773,
    "Sn": 3.000000 * 0.5291773,
    "Sb": 2.300000 * 0.5291773,
    "Te": 2.300000 * 0.5291773,
    "I": 2.300000 * 0.5291773,
    "Xe": 2.500000 * 0.5291773,
    "Cs": 2.500000 * 0.5291773,
    "Ba": 2.800000 * 0.5291773,
    "La": 2.800000 * 0.5291773,
    "Hf": 3.000000 * 0.5291773,
    "Ta": 2.900000 * 0.5291773,
    "W": 2.750000 * 0.5291773,
    "Re": 2.700000 * 0.5291773,
    "Os": 2.700000 * 0.5291773,
    "Ir": 2.600000 * 0.5291773,
    "Pt": 2.600000 * 0.5291773,
    "Au": 2.500000 * 0.5291773,
    "Hg": 2.500000 * 0.5291773,
    "Tl": 3.200000 * 0.5291773,
    "Pb": 3.100000 * 0.5291773,
    "Bi": 3.000000 * 0.5291773,
}


class DistanceFilter:
    def __init__(self, radii: dict[str, float] | None = None):
        if radii is None:
            radii = {}
        self._radii = RCORE.copy()
        self._radii.update(radii)

    @staticmethod
    def _element_wise_dist(structure):
        pair = defaultdict(lambda: np.inf)
        n = structure.get_neighbors(num_neighbors=25, cutoff_radius=5, mode="ragged")
        for i, (I, D) in enumerate(zip(n.indices, n.distances)):
            for j, d in zip(I, D):
                ei, ej = sorted((structure.symbols[i], structure.symbols[j]))
                pair[ei, ej] = min(d, pair[ei, ej])
        return pair

    def __call__(self, structure):
        """
        Return True if structure statifies minimum distance criteria.
        """
        pair = self._element_wise_dist(structure)
        for ei, ej in combinations_with_replacement(structure.get_species_symbols(), 2):
            ei, ej = sorted((ei, ej))
            if pair[ei, ej] < self._radii[ei] + self._radii[ej]:
                return False
        return True


def get_table(pr, table_name, add=None, delete_existing_job=False):
    if table_name in pr.list_nodes() and not delete_existing_job:
        tab = pr.load(table_name)
        tab.update_table()
        return tab
    else:
        if add is None:
            raise ValueError("add cannot be None on first run!")
        tab = pr.create_table(table_name, delete_existing_job=delete_existing_job)
        add(tab)
        tab.run()
        return tab


def read_generic_parameters(hdf, key):
    gp = GenericParameters(table_name="data_dict")
    gp.from_hdf(hdf)
    return gp[key]


def fast_forward(timeout, spec):
    print("Fast forward in", timeout, "...")
    # restarting the script instead of looping in python means I can edit the files while they run.  *Nothing*
    # could possibly go wrong and it's convenient!
    import time, sys, os

    time.sleep(timeout)
    os.execl(sys.executable, sys.executable, "-m", spec.name, *sys.argv[1:])
    ## Failed iterations for the historic record
    # os.execl(sys.executable, sys.argv[0], '-m', __spec__.name, *sys.argv[1:])
    # won't work because modules don't have the exec bit set
    # os.execl(*sys.argv)
    # won't work because argument will be (mis-) interpreted by the python executable itself
    # os.execlp('python', *sys.argv)


SYMLINK_TARGET = os.getenv("SPGFIT_SYMLINK_TARGET", default="/cmmc/ptmp")


def symlink_project(pr: Project):
    if SYMLINK_TARGET.lower() == "false":
        return
    if not os.path.isdir(SYMLINK_TARGET):
        warnings.warn(
            f"Configured symlink path '{SYMLINK_TARGET}' does not exist. Likely because you are not running on the "
            "MPIE cluster.  You can configure it by setting the SPGFIT_SYMLINK_TARGET environment variable or "
            "overwriting pyiron_potentialfit.spgfit.util.SYMLINK_TARGET before running any SpgFit functions. "
            "You can disable this feature by setting it to 'False'."
        )
        return
    target_dir = pr.project_path
    if target_dir[-1] == "/":
        target_dir = target_dir[:-1]
    pr.symlink(os.path.join(SYMLINK_TARGET, os.path.dirname(target_dir)))


# Generic config objects for all tools

@dataclass
class ServerConfig:
    """Stores computational parameters and applies them to pyiron jobs."""

    cores: int = 40
    run_time: float = 2 * 60 * 60  # seconds
    queue: str = "cmti"

    def configure_server_on_job(self, job: GenericJob) -> GenericJob:
        """
        Apply the stored parameter to the given job.

        Args:
            job (:class:`pyiron_base.jobs.job.GenericJob`): job to configure

        Returns:
            same job as given
        """
        job.server.queue = self.queue
        job.server.cores = self.cores
        job.server.run_time = self.run_time
        return job
