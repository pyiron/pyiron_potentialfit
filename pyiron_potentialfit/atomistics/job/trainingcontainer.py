# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Store structures together with energies and forces for potential fitting applications.

Basic usage:

>>> pr = Project("training")
>>> container = pr.create.job.TrainingContainer("small_structures")

Let's make a structure and invent some forces

>>> structure = pr.create.structure.ase_bulk("Fe")
>>> forces = numpy.array([-1, 1, -1])
>>> container.add_structure(structure, energy=-1.234, forces=forces, identifier="Fe_bcc")

If you have a lot of precomputed structures you may also add them in bulk from a pandas DataFrame

>>> df = pandas.DataFrame({ "name": "Fe_bcc", "atoms": structure, "energy": -1.234, "forces": forces })
>>> container.include_dataset(df)

You can retrieve the full database with :method:`~.TrainingContainer.to_pandas()` like this

>>> container.to_pandas()
name    atoms   energy  forces  number_of_atoms
Fe_bcc  ...
"""

from typing import Callable, Dict, Any, Optional
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure
from pyiron_atomistics.atomistics.structure.structurestorage import (
    StructureStorage,
    StructurePlots,
)
from pyiron_atomistics.atomistics.structure.neighbors import NeighborsTrajectory
from pyiron_base import GenericJob, DataContainer
from pyiron_snippets.deprecate import deprecate


class TrainingContainer(GenericJob, HasStructure):
    """
    Stores ASE structures with energies and forces.
    """

    def __init__(self, project, job_name):
        super().__init__(project=project, job_name=job_name)
        self.__name__ = "TrainingContainer"
        self.__hdf_version__ = "0.3.0"
        self._container = TrainingStorage()

        self.input = DataContainer(
            {"save_neighbors": True, "num_neighbors": 12}, table_name="parameters"
        )

    def include_job(self, job, iteration_step=-1, identifier=None):
        """
        Add structure, energy and forces from job.

        Args:
            job (:class:`.AtomisticGenericJob`): job to take structure from
            iteration_step (int, optional): if job has multiple steps, this
            selects which to add
            identifier (str): name of the extracted structure in the container; default is the name of the job
        """
        self._container.include_job(job, iteration_step, identifier=identifier)

    def include_structure(self, structure, energy=None, name=None, **properties):
        """
        Add new structure to structure list and save energy and forces with it.

        For consistency with the rest of pyiron, energy should be in units of eV
        and forces in eV/A, but no conversion is performed.

        Args:
            structure_or_job (:class:`~.Atoms`): structure to add
            energy (float): energy of the whole structure
            forces (Nx3 array of float, optional): per atom forces, where N is
                the number of atoms in the structure
            stress (6 array of float, optional): per structure stresses in voigt
                notation
            name (str, optional): name describing the structure
        """
        self._container.include_structure(
            structure, name=name, energy=energy, **properties
        )

    def add_structure(
        self, structure, energy, forces=None, stress=None, identifier=None, **arrays
    ):
        """
        Add new structure to structure list and save energy and forces with it.

        For consistency with the rest of pyiron, energy should be in units of eV and forces in eV/A, but no conversion
        is performed.

        Args:
            structure_or_job (:class:`~.Atoms`): structure to add
            energy (float): energy of the whole structure
            forces (Nx3 array of float, optional): per atom forces, where N is the number of atoms in the structure
            stress (6 array of float, optional): per structure stresses in voigt notation
            name (str, optional): name describing the structure
        """
        self._container.add_structure(
            structure,
            energy,
            identifier=identifier,
            forces=forces,
            stress=stress,
            **arrays,
        )

    def include_dataset(self, dataset):
        """
        Add a pandas DataFrame to the saved structures.

        The dataframe should have the following columns:
            - name: human readable name of the structure
            - atoms(:class:`ase.Atoms`): the atomic structure
            - energy(float): energy of the whole structure
            - forces (Nx3 array of float): per atom forces, where N is the number of atoms in the structure
            - stress (6 array of float): per structure stress in voigt notation
        """
        self._container.include_dataset(dataset)

    def _get_structure(self, frame=-1, wrap_atoms=True):
        return self._container.get_structure(frame=frame, wrap_atoms=wrap_atoms)

    def _number_of_structures(self):
        return self._container.number_of_structures

    def get_neighbors(self, num_neighbors=None):
        """
        Calculate and add neighbor information in each structure.

        If input.save_neighbors is True the data is automatically added to the internal storage and will be saved
        together with the normal structure data.

        Args:
            num_neighbors (int, optional): Number of neighbors to collect, if not given use value from input

        Returns:
            NeighborsTrajectory: neighbor information
        """
        if num_neighbors is None:
            num_neighbors = self.input.num_neighbors
        n = NeighborsTrajectory(
            has_structure=self,
            store=self._container if self.input.save_neighbors else None,
            num_neighbors=num_neighbors,
        )
        n.compute_neighbors()
        return n

    def get_elements(self):
        """
        Return a list of chemical elements in the training set.

        Returns:
            :class:`list`: list of unique elements in the training set as strings of their standard abbreviations
        """
        return self._container.get_elements()

    def to_pandas(self):
        """
        Export list of structure to pandas table for external fitting codes.

        The table contains the following columns:
            - 'name': human-readable name of the structure
            - 'ase_atoms': the structure as a :class:`.Atoms` object
            - 'energy': the energy of the full structure
            - 'forces': the per atom forces as a :class:`numpy.ndarray`, shape Nx3
            - 'stress': the per structure stress as a :class:`numpy.ndarray`, shape 6
            - 'number_of_atoms': the number of atoms in the structure, N

        Returns:
            :class:`pandas.DataFrame`: collected structures
        """
        return self._container.to_pandas()

    def to_list(self, filter_function=None):
        """
        Returns the data as lists of pyiron structures, energies, forces, and the number of atoms

        Args:
            filter_function (function): Function applied to the dataset (which is a pandas DataFrame) to filter it

        Returns:
            tuple: list of structures, energies, forces, and the number of atoms
        """
        return self._container.to_list(filter_function)

    def write_input(self):
        pass

    def collect_output(self):
        pass

    def run_static(self):
        self.status.running = True
        if self.input.save_neighbors:
            self.get_neighbors()
            self.to_hdf()
        self.status.finished = True

    def run_if_interactive(self):
        if self.input.save_neighbors:
            self.get_neighbors()
        self.to_hdf()
        self.status.finished = True

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(self.project_hdf5)
        self._container.to_hdf(self.project_hdf5, "structures")

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        hdf_version = self.project_hdf5.get("HDF_VERSION", "0.1.0")
        if hdf_version == "0.1.0":
            table = pd.read_hdf(
                self.project_hdf5.file_name, self.name + "/output/structure_table"
            )
            self.include_dataset(table)
        else:
            self._container = TrainingStorage()
            self._container.from_hdf(self.project_hdf5, "structures")
            if hdf_version == "0.3.0":
                self.input.from_hdf(self.project_hdf5, "parameters")

    def sample(
        self,
        name: str,
        selector: Callable[[StructureStorage, int], bool],
        delete_existing_job: bool = False,
        run: bool = True,
    ) -> "TrainingContainer":
        """
        Create a new TrainingContainer with structures filtered by selector.

        `self` must have status `finished`.  `selector` is passed the underlying :class:`StructureStorage` of this
        container and the index of the structure and return a boolean whether to include the structure in the new
        container or not.  By default the new container is saved and run.

        Args:
            name (str): name of the new TrainingContainer
            selector (Callable[[StructureStorage, int], bool]): callable that selects structure to include
            delete_existing_job (bool): if job with name exist, remove it first
            run (bool): if True, immediately run and save the job.

        Returns:
            :class:`.TrainingContainer`: new container with selected structures

        Raises:
            ValueError: if a job with the given `name` already exists.
        """
        if not self.status.finished:
            raise ValueError(f"Job must be finished, not '{self.status}'!")
        cont = self.project.create.job.TrainingContainer(
            name, delete_existing_job=delete_existing_job
        )
        if not cont.status.initialized:
            raise ValueError(f"Job '{name}' already exists with status: {cont.status}!")
        cont._container = self._container.sample(selector)
        if run:
            cont.run()
        return cont

    @property
    def plot(self):
        """
        :class:`.TrainingPlots`: plotting interface
        """
        return self._container.plot

    def iter(self, *arrays, wrap_atoms=True):
        """
        Iterate over all structures in this object and all arrays that are defined

        Args:
            wrap_atoms (bool): True if the atoms are to be wrapped back into the unit cell; passed to
                               :meth:`.get_structure()`
            *arrays (str): name of arrays that should be iterated over

        Yields:
            :class:`pyiron_atomistics.atomistitcs.structure.atoms.Atoms`, arrays: every structure attached to the object and queried arrays
        """
        yield from self._container.iter(*arrays, wrap_atoms=wrap_atoms)

    def subtract_reference(self, reference_energies: dict[str, float]):
        """
        Subtract Atomic energies from structure energies.

        Must be called before the job is run!
        The original energy is kept in a new array "energy_uncorrected".

        Args:
            reference_energies (dict): a dictionary from element symbols to the energy of their isolated atoms.
        """
        if not self.status.initialized:
            raise ValueError(
                f"Must be called before job is run, not in state: '{self.status}'!"
            )
        self._container.subtract_reference(reference_energies)


class TrainingPlots(StructurePlots):
    """
    Simple interface to plot various properties of the structures inside the given :class:`.TrainingContainer`.
    """

    def energy_volume(self, crystal_systems=False):
        """
        Plot volume vs. energy.

        Volume and energy are normalized per atom before plotting.

        Args:
            crystal_systems (bool): if True, plot & label structures of different crystal systems separately.

        Returns:
            DataFrame: contains atomic energy and volumes in the columns 'E' and 'V'; if `crystal_systems` is given,
                       also contain space groups and crystal systems of each structure
        """

        N = self._store.get_array("length")
        E = self._store.get_array("energy") / N
        C = self._store.get_array("cell")
        V = np.linalg.det(C) / N

        df = pd.DataFrame({"V": V, "E": E})

        if crystal_systems:
            spg = self._calc_spacegroups()
            df = df.join(spg)
            for cs, dd in df.groupby("crystal_system"):
                plt.scatter(dd.V, dd.E, label=cs)
            plt.legend()
        else:
            plt.scatter(df.V, df.E)
        plt.xlabel(r"Atomic Volume [$\AA^3$]")
        plt.ylabel(r"Atomic Energy [eV]")

        return df

    def energy_distance(self, num_neighbors=None):
        """
        Plot energy vs minimum nearest neighbor distance in the structure.

        Args:
            num_neighbors (int): maximum number of neighbors to caluclate, when 'distances' are not defined in storage
                                 the default is the value from the previous call or 36
        """

        N = self._store.get_array("length")
        E = self._store.get_array("energy") / N
        neigh = self._calc_neighbors(num_neighbors=num_neighbors)
        D = np.array(
            [d.min() for d in np.split(neigh["distances"][:, 0], N.cumsum())[:-1]]
        )
        plt.scatter(D, E, marker=".")
        plt.xlabel(r"Closest Nearest Neighbor Distance [$\AA$]")
        plt.ylabel("Atomic Energy [ev/Atom]")

    def forces(self, axis: Optional[int] = None):
        """
        Plot a histogram of all forces.

        Args:
            axis (int, optional): plot only forces along this axis, if not given plot all forces
        """
        f = self._store.get_array("forces")
        if axis is not None:
            f = f[:, axis]
        else:
            f = f.ravel()
        plt.hist(f, bins=20)
        plt.xlabel(r"Force [eV/$\mathrm{\AA}$]")


_HDF_KEYS = {
    "energy": "output/generic/energy_pot",
    "forces": "output/generic/forces",
    "stress": "output/generic/pressures",
    "indices": "output/generic/indices",
    "species": "input/structure/species",
    "cell": "output/generic/cells",
    "positions": "output/generic/positions",
    "pbc": "input/structure/cell/pbc",
    "spins": "output/generic/dft/final_magmoms",
}

_JOB_HDF_OVERLAY_KEYS = {
    "Vasp": {
        # For DFT one should use the smeared energy to obtain values
        # consistent with the forces, but the default energy_pot of DFT
        # jobs is the energy extrapolated to zero smearing
        "energy": "output/generic/dft/energy_free",
        # HACK: VASP work-around, current contents of pressures are meaningless, correct values are in
        # output/generic/stresses
        "stress": "output/generic/stresses",
    }
}


class TrainingStorage(StructureStorage):
    def __init__(self):
        super().__init__()
        self.add_array("energy", dtype=np.float64, per="chunk", fill=np.nan)
        self._table_cache = None
        self.to_pandas()

    def to_pandas(self):
        """
        Export list of structure to pandas table for external fitting codes.

        The table contains the following columns:
            - 'name': human-readable name of the structure
            - 'ase_atoms': the structure as a :class:`.Atoms` object
            - 'energy': the energy of the full structure
            - 'forces': the per atom forces as a :class:`numpy.ndarray`, shape Nx3
            - 'stress': the per structure stress as a :class:`numpy.ndarray`, shape 6
            - 'number_of_atoms': the number of atoms in the structure, N

        Returns:
            :class:`pandas.DataFrame`: collected structures
        """
        if self._table_cache is None or len(self._table_cache) != len(self):
            self._table_cache = pd.DataFrame(
                {
                    "name": [self.get_array("identifier", i) for i in range(len(self))],
                    "atoms": [self.get_structure(i) for i in range(len(self))],
                    "energy": [self.get_array("energy", i) for i in range(len(self))],
                }
            )
            if self.has_array("forces"):
                self._table_cache["forces"] = [
                    self.get_array("forces", i) for i in range(len(self))
                ]
            if self.has_array("stress"):
                self._table_cache["stress"] = [
                    self.get_array("stress", i) for i in range(len(self))
                ]
            self._table_cache["number_of_atoms"] = [
                len(s) for s in self._table_cache.atoms
            ]
        return self._table_cache

    def include_job(
        self,
        job,
        iteration_step=-1,
        hdf_keys=None,
        identifier=None,
    ):
        """
        Add structure, energy, forces and pressures from an inspected or loaded job.

        The job must be an atomistic job.

        Forces and stresses are only added if present in the output.

        The locations in the job HDF5 file from which the training data is read
        can be customized by passing a dictionary as `hdf_keys`, where the
        values must be paths inside the job HDF5 file.  The available keys are
        given below together with their default values.

        * `energy`: `output/generic/energy_pot`
        * `forces`: `output/generic/forces`
        * `stress`: `output/generic/pressures`
        * `indices`: `output/generic/indices`
        * `cell`: `output/generic/cells`
        * `positions`: `output/generic/positions`
        * `species`: `input/structure/species`
        * `pbc`: `input/structure/cell/pbc`
        * `spins`: `output/generic/dft/final_magmoms`

        Other keys are ignored.  All entries except `pbc` and `species` must
        lead to arrays that can be indexed by `iteration_step`.

        For `Vasp` jobs the defaults are changed like this

        * `energy`: `output/generic/dft/energy_free`
        * `stress`: `output/generic/stresses`

        to ensure that by default energies, forces and stresses are consistent.

        Args:
            job (:class:`.JobPath`, :class:`.AtomisticGenericJob`): job (path) to take structure from
            iteration_step (int, optional): if job has multiple steps, this selects which to add
            hdf_keys (dict of str): customize where values are read from the
                                    job HDF5 file
            identifier (str): name of the extracted structure in the storage; default is the name of the job
        """

        hdf_keys = _HDF_KEYS.copy()
        hdf_keys.update(_JOB_HDF_OVERLAY_KEYS.get(job.content["NAME"], {}))

        kwargs = {
            "energy": job.content[hdf_keys["energy"]][iteration_step],
        }
        try:
            kwargs["forces"] = job.content[hdf_keys["forces"]][iteration_step]
        except KeyError:
            pass

        try:
            pp = job.content[hdf_keys["stress"]]
            if len(pp) > 0:
                stress = np.asarray(pp[iteration_step])
                if stress.shape == (3, 3):
                    stress = np.array(
                        [
                            stress[0, 0],
                            stress[1, 1],
                            stress[2, 2],
                            stress[1, 2],
                            stress[0, 2],
                            stress[0, 1],
                        ]
                    )
                kwargs["stress"] = stress
        except KeyError:
            pass

        try:
            indices = job.content[hdf_keys["indices"]][iteration_step]
        except KeyError:
            # not all jobs save indices again in the output, but all atomistic
            # jobs do it in the input
            indices = job.content["input/structure/indices"]
        species = np.asarray(job.content[hdf_keys["species"]])
        cell = job.content[hdf_keys["cell"]][iteration_step]
        positions = job.content[hdf_keys["positions"]][iteration_step]
        pbc = job.content[hdf_keys["pbc"]]
        try:
            kwargs["spins"] = np.array(job.content[hdf_keys["spins"]][iteration_step])
        except (KeyError, IndexError):
            pass

        if "forces" in kwargs and not self.has_array("forces"):
            self.add_array(
                "forces", shape=(3,), dtype=np.float64, per="element", fill=np.nan
            )
        if "stress" in kwargs and not self.has_array("stress"):
            # save stress in voigt notation
            self.add_array(
                "stress", shape=(6,), dtype=np.float64, per="chunk", fill=np.nan
            )

        self.add_chunk(
            len(indices),
            identifier=identifier or job.name,
            symbols=species[indices],
            positions=positions,
            cell=[cell],
            pbc=[pbc],
            **kwargs,
        )

    @deprecate("Use add_structure instead")
    def include_structure(self, structure, energy, name=None, **properties):
        """
        Add new structure to structure list and save energy and forces with it.

        For consistency with the rest of pyiron, energy should be in units of eV and forces in eV/A, but no conversion
        is performed.

        Args:
            structure_or_job (:class:`~.Atoms`): structure to add
            energy (float): energy of the whole structure
            forces (Nx3 array of float, optional): per atom forces, where N is the number of atoms in the structure
            stress (6 array of float, optional): per structure stresses in voigt notation
            name (str, optional): name describing the structure
        """
        self.add_structure(structure, identifier=name, energy=energy, **properties)

    def add_structure(
        self, structure: Atoms, energy, identifier=None, **arrays
    ) -> None:
        if "forces" in arrays and not self.has_array("forces"):
            self.add_array(
                "forces", shape=(3,), dtype=np.float64, per="element", fill=np.nan
            )
        if "stress" in arrays and not self.has_array("stress"):
            # save stress in voigt notation
            self.add_array(
                "stress", shape=(6,), dtype=np.float64, per="chunk", fill=np.nan
            )
        super().add_structure(structure, identifier=identifier, energy=energy, **arrays)

    def include_dataset(self, dataset):
        """
        Add a pandas DataFrame to the saved structures.

        The dataframe should have the following columns:
            - name: human readable name of the structure
            - atoms(:class:`ase.Atoms`): the atomic structure
            - energy(float): energy of the whole structure
            - forces (Nx3 array of float): per atom forces, where N is the number of atoms in the structure
            - charges (Nx3 array of floats):
            - stress (6 array of float): per structure stress in voigt notation
        """
        if (
            "name" not in dataset.columns
            or "atoms" not in dataset.columns
            or "energy" not in dataset.columns
        ):
            raise ValueError(
                "At least columns 'name', 'atoms' and 'energy' must be present in dataset!"
            )
        for row in dataset.itertuples(index=False):
            kwargs = {}
            if hasattr(row, "forces"):
                kwargs["forces"] = row.forces
            if hasattr(row, "stress"):
                kwargs["stress"] = row.stress
            self.add_structure(
                row.atoms, energy=row.energy, identifier=row.name, **kwargs
            )

    def to_list(self, filter_function=None):
        """
        Returns the data as lists of pyiron structures, energies, forces, and the number of atoms

        Args:
            filter_function (function): Function applied to the dataset (which is a pandas DataFrame) to filter it

        Returns:
            tuple: list of structures, energies, forces, and the number of atoms
        """
        data_table = self.to_pandas()
        if filter_function is not None:
            data_table = filter_function(data_table)
        structure_list = data_table.atoms.to_list()
        energy_list = data_table.energy.to_list()
        if "forces" not in data_table.columns:
            raise ValueError("no forces defined in storage; call to_dict() instead.")
        force_list = data_table.forces.to_list()
        num_atoms_list = data_table.number_of_atoms.to_list()

        return (structure_list, energy_list, force_list, num_atoms_list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary of all structures and training properties."""
        dict_arrays = {}

        # Get structure information.
        dict_arrays["structure"] = list(self.iter_structures())

        # Some arrays are only for internal usage or structure information that
        # was already saved in dict['structure'].
        internal_arrays = [
            "start_index",
            "length",
            "cell",
            "pbc",
            "positions",
            "symbols",
        ]
        for array in self.list_arrays():
            # Skip internal arrays.
            if array in internal_arrays:
                continue

            dict_arrays[array] = self.get_array_ragged(array)
        return dict_arrays

    def iter(self, *arrays, wrap_atoms=True):
        """
        Iterate over all structures in this object and all arrays that are defined

        Args:
            wrap_atoms (bool): True if the atoms are to be wrapped back into the unit cell; passed to
                               :meth:`.get_structure()`
            *arrays (str): name of arrays that should be iterated over

        Yields:
            :class:`pyiron_atomistics.atomistitcs.structure.atoms.Atoms`, arrays: every structure attached to the object and queried arrays
        """
        array_vals = (self.get_array_ragged(a) for a in arrays)
        yield from zip(self.iter_structures(), *array_vals)

    @property
    def plot(self):
        """
        :class:`.TrainingPlots`: plotting interface
        """
        if self._plots is None:
            self._plots = TrainingPlots(self)
        return self._plots

    def subtract_reference(self, reference_energies: dict[str, float]):
        """
        Subtract Atomic energies from structure energies.

        The original energy is kept in a new array "energy_uncorrected".

        Args:
            reference_energies (dict): a dictionary from element symbols to the energy of their isolated atoms.
        """
        elements = self.get_elements()
        if not set(reference_energies).issuperset(elements):
            raise ValueError(
                f"Must specify reference energies for all present elements: {elements}!"
            )
        counts = list(map(Counter, self.get_array_ragged("symbols")))
        bias = np.array(
            [sum(reference_energies[e] * c[e] for e in elements) for c in counts]
        )
        self._per_chunk_arrays["energy_uncorrected"] = self._per_chunk_arrays[
            "energy"
        ].copy()
        self._per_chunk_arrays["energy"][: self.num_chunks] -= bias
