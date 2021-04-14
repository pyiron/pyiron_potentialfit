"""
Pyiron interface to atomicrex
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom
import posixpath
import os
import subprocess

import numpy as np
import pandas as pd

from pyiron_base import GenericJob, Settings, PyironFactory, Executable

from pyiron_contrib.atomistics.atomicrex.general_input import GeneralARInput, AlgorithmFactory
from pyiron_contrib.atomistics.atomicrex.structure_list import ARStructureList
from pyiron_contrib.atomistics.atomicrex.potential_factory import ARPotFactory, AbstractPotential
from pyiron_contrib.atomistics.atomicrex.output import Output
from pyiron_contrib.atomistics.atomicrex.function_factory import FunctionFactory

s = Settings()

## Class defined for future addition of other codes
## Not sure which functionality (if any) can be extracted yet, but a similar pattern is followed in other pyiron modules
class PotentialFittingBase(GenericJob):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)


class Atomicrex(PotentialFittingBase):
    """Class to set up and run atomicrex jobs
    """    
    def __init__(self, project, job_name):
        super().__init__(project, job_name)

        self.__name__ = "atomicrex"
        self.__version__ = (
            None
        )
        self._executable_activate(enforce=True)

        s.publication_add(self.publication)
        self.input = GeneralARInput()
        self.potential = None
        self.structures = ARStructureList()
        ## temprorary set working directory manually before full pyiron integration
        self._working_directory = None
        self.output = Output()
        self.factories = Factories()
        

    def plot_final_potential(self):
        """
        Plot the fitted potential.
        Returns the matplotlib objects to change the look of the plot.

        Returns:
            [matplotlib figure, axis]: [description]
        """        
        return self.potential.plot_final_potential(self)


    def to_hdf(self, hdf=None, group_name=None):
        """Internal function to store the job in hdf5 format
        """        
        super().to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(hdf=self.project_hdf5)
        self.potential.to_hdf(hdf=self.project_hdf5)
        self.structures.to_hdf(hdf=self.project_hdf5)
        self.output.to_hdf(hdf=self.project_hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        """Internal function to reload the job object from hdf5
        """        
        super().from_hdf(hdf=hdf, group_name=group_name)
        self.input.from_hdf(hdf=self.project_hdf5)
        self.potential = self.project_hdf5["potential"].to_object()
        self.structures.from_hdf(hdf=self.project_hdf5)
        self.output.from_hdf(hdf=self.project_hdf5)

    @property
    def publication(self):
        return {
            "atomicrex": {
                "atomicrex": {
                    "title": "Atomicrex-a general purpose tool for the construction of atomic interaction models",
                    "journal": "Modelling and Simulation in Materials Science and Engineering",
                    "volume": "25",
                    "number": "5",
                    "year": "2017",
                    "issn": "0965-0393",
                    "doi": "10.1088/1361-651X/aa6ecf",
                    "url": "https://doi.org/10.1088%2F1361-651x%2Faa6ecf",
                    "author": ["Alexander Stukowski", "Erik Fransson", "Markus Mock", "Paul Erhart"],
                }
            }
        }

    def collect_output(self, cwd=None):
        """Internal function that parses the output of an atomicrex job

        Args:
            cwd (str, optional): Working directory. Defaults to None.
        """        
        #self.input.from_hdf(self._hdf5)
        if cwd is None:
            cwd = self.working_directory
        filepath = f"{cwd}/error.out"
        
        finished_triggered = False
        params_triggered = False
        structures_triggered = False

        # Allocate numpy arrays for iterations and residual
        # I assume this is better than appending to a list if many iterations are done
        residuals = np.zeros(self.input.fit_algorithm.max_iter)
       
        # Since every step is written out in atomicrex arange can be used.
        # Needs to be adapted when atomicrex output is changed to write only every xth step.
        # Unsinged 32 bit int should be enough or this will overflow anyway in most cases.
        iterations = np.arange(start=1, stop=self.input.fit_algorithm.max_iter+1, dtype=np.uintc)
        iter_index = 0

        with open(filepath, "r") as f:
            final_parameter_lines = []
            final_property_lines = []

            for l in f:
                if l.startswith("ERROR"):
                    self.status.aborted=True
                    self.output.error = l
                
                elif not finished_triggered:
                    if l.startswith("Iterations"):
                        l = l.split()
                        finished_triggered = True
                        self.output.residual = residuals[0:iter_index]
                        self.output.iterations = iterations[0:iter_index]
                    else:
                        l = l.split()
                        try:
                            if l[1] == "iter=":
                                residuals[iter_index] = float(l[-1])
                                iter_index  += 1
                        except IndexError:
                            continue
                
                else: # if finished_triggered
                    if l.startswith("Potential parameters"):
                        # Get the number of dofs
                        n_fit_dofs = int(l.split("=")[1][:-3])
                        params_triggered = True
                
                    elif params_triggered:
                        if not l.startswith("---"):
                            final_parameter_lines.append(l)
                        else:
                            # Collecting lines with final parameters finished, hand over to the potential class
                            self.potential._parse_final_parameters(final_parameter_lines)
                            params_triggered = False
                
                    elif l.startswith("Computing"):
                        structures_triggered = True
                     
                    elif structures_triggered:
                        if not l.startswith("---"):
                            final_property_lines.append(l)
                        else:
                            # Collecting structure information finished, hand over structures class
                            self.structures._parse_final_properties(final_property_lines)
                            structures_triggered = False

        self.to_hdf()

    def convergence_check(self):
        """Internal function, TODO
        """        
        return

    def write_input(self, directory=None):
        """Internal function that writes input files

        Args:
            directory ([string], optional): Working directory. Defaults to None.
        """        
        if directory is None:
            directory = self.working_directory
        self.input._write_xml_file(directory = directory)
        self.potential.write_xml_file(directory = directory)
        self.structures.write_xml_file(directory = directory)

    def _executable_activate(self, enforce=False):
        """
        Internal function that sets up and Executable() object 
        and finds executables available in pyiron resources/atomicrex/bin

        Args:
            enforce (bool, optional): [description]. Defaults to False.
        """        
        if self._executable is None or enforce:
            if len(self.__module__.split(".")) > 1:
                self._executable = Executable(
                    codename=self.__name__,
                    module=self.__module__.split(".")[-2],
                    path_binary_codes=s.resource_paths,
                )
            else:
                self._executable = Executable(
                    codename=self.__name__, path_binary_codes=s.resource_paths
                )

    # Leftover of the potentials workshop.
    # Maybe this property will be used in unified interface
    # to several fitting codes in the future
    # instead of the potential_as_pd_df function
    @property
    def lammps_potential(self):
        pot = self.potential_as_pd_df()
        return pot


    def potential_as_pd_df(self):
        """
        Return the fitted potential as a pandas dataframe,
        which can be used for lammps calculations.
        """        
        return self.potential._potential_as_pd_df(job=self)


class Factories:
    """
    Provides conventient acces to other factory classes.
    Functionality to set up an atomicrex job can be found here.
    """    
    def __init__(self):
        self.potentials = ARPotFactory()
        self.functions = FunctionFactory()
        self.algorithms = AlgorithmFactory()
