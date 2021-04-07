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
>>> container.include_structure(structure, energy=-1.234, forces=forces, name="Fe_bcc")

If you have a lot of precomputed structures you may also add them in bulk from a pandas DataFrame

>>> df = pandas.DataFrame({ "name": "Fe_bcc", "atoms": structure, "energy": -1.234, "forces": forces })
>>> container.include_dataset(df)

You can retrieve the full database with :method:`~.TrainingContainer.to_pandas()` like this

>>> container.to_pandas()
name    atoms   energy  forces  number_of_atoms
Fe_bcc  ...
"""

from warnings import catch_warnings

import pandas as pd
from pyiron_atomistics import pyiron_to_ase, ase_to_pyiron
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_base import GenericJob


class TrainingContainer(GenericJob):
    """
    Stores ASE structures with energies and forces.
    """

    def __init__(self, project, job_name):
        super().__init__(project=project, job_name=job_name)
        self.__name__ = "TrainingContainer"
        self._table = pd.DataFrame({
            "name": [],
            "atoms": [],
            "energy": [],
            "forces": [],
            "number_of_atoms": []
        })

    def include_job(self, job, iteration_step=-1):
        """
        Add structure, energy and forces from job.

        Args:
            job (:class:`.AtomisticGenericJob`): job to take structure from
            iteration_step (int, optional): if job has multiple steps, this selects which to add
        """
        self.include_structure(job.get_structure(iteration_step=iteration_step),
                               energy=job.output.energy_pot[iteration_step],
                               forces=job.output.forces[iteration_step],
                               name=job.name)

    def include_structure(self, structure, energy, forces=None, name=None):
        """
        Add new structure to structure list and save energy and forces with it.

        For consistency with the rest of pyiron, energy should be in units of eV and forces in eV/A, but no conversion
        is performed.

        Args:
            structure_or_job (:class:`~.Atoms`, :class:`ase.Atoms`): if :class:`~.Atoms` convert to :class:`ase.Atoms`
            energy (float): energy of the whole structure
            forces (Nx3 array of float, optional): per atom forces, where N is the number of atoms in the structure
            name (str, optional): name describing the structure
        """
        if isinstance(structure, Atoms):
            structure = pyiron_to_ase(structure)
        self._table = self._table.append(
                {"name": name, "atoms": structure, "energy": energy, "forces": forces,
                 "number_of_atoms": len(structure)},
                ignore_index=True)

    def include_dataset(self, dataset):
        """
        Add a pandas DataFrame to the saved structures.

        The dataframe should have the following columns:
            - name: human readable name of the structure
            - atoms(:class:`ase.Atoms`): the atomic structure
            - energy(float): energy of the whole structure
            - forces (Nx3 array of float): per atom forces, where N is the number of atoms in the structure
        """
        self._table = self._table.append(dataset, ignore_index=True)

    def get_structure(self, iteration_step=-1):
        """
        Returns a structure from the training set.

        Args:
            iteration_step (int, optional): index of the structure in training set

        Returns:
            :class:`.Atoms`: pyiron structure
        """
        return ase_to_pyiron(self._table.atoms[iteration_step])

    def get_elements(self):
        """
        Return a list of chemical elements in the training set.

        Returns:
            :class:`list`: list of unique elements in the training set as strings of their standard abbreviations
        """
        elements = set()
        for s in self._table.atoms:
            elements.update(s.get_chemical_symbols())
        return list(elements)

    def to_pandas(self):
        """
        Export list of structure to pandas table for external fitting codes.

        The table contains the following columns:
            - 'name': human-readable name of the structure
            - 'ase_atoms': the structure as a :class:`.Atoms` object
            - 'energy': the energy of the full structure
            - 'forces': the per atom forces as a :class:`numpy.ndarray`, shape Nx3
            - 'number_of_atoms': the number of atoms in the structure, N

        Returns:
            :class:`pandas.DataFrame`: collected structures
        """
        return self._table

    def to_list(self, filter_function=None):
        """
        Returns the data as lists of pyiron structures, energies, forces, and the number of atoms

        Args:
            filter_function (function): Function applied to the dataset (which is a pandas DataFrame) to filter it

        Returns:
            tuple: list of structures, energies, forces, and the number of atoms
        """
        if filter_function is None:
            data_table = self._table
        else:
            data_table = filter_function(self._table)
        structure_list = data_table.atoms.apply(ase_to_pyiron).to_list()
        energy_list = data_table.energy.to_list()
        force_list = data_table.forces.to_list()
        num_atoms_list = data_table.number_of_atoms.to_list()
        return structure_list, energy_list, force_list, num_atoms_list

    def write_input(self):
        pass

    def collect_output(self):
        pass

    def run_static(self):
        self.status.finished = True

    def run_if_interactive(self):
        self.to_hdf()
        self.status.finished = True

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        with catch_warnings():
            self._table.to_hdf(self.project_hdf5.file_name, self.name + "/output/structure_table")

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        self._table = pd.read_hdf(self.project_hdf5.file_name, self.name + "/output/structure_table")
