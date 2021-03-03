# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
"""

from pyiron_base import GenericJob, Executable, Settings

import numpy as np

__author__ = "Marvin Poul"
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - " \
                "Computational Materials Design (CM) Department"
__version__ = "0.1"
__maintainer__ = "Marvin Poul"
__email__ = "poul@mpie.de"
__status__ = "development"
__date__ = "March 3, 2021"

s = Settings()

AngstromToBohr = 1.88972612
ElectronVoltToHartree = 0.03674932218


class RunnerFit(GenericJob):
    def __init__(self, project, job_name):
        super().__init__(project, job_name) 
        self._training_ids = []

        s.publication_add(self.publication)

    @property
    def publication(self):
        return {
            "runner": [
                {
                    "title": "First Principles Neural Network Potentials for Reactive Simulations of Large Molecular and Condensed Systems",
                    "journal": "Angewandte Chemie International Edition",
                    "volume": "56",
                    "number": "42",
                    "year": "2017",
                    "issn": "1521-3773",
                    "doi": "10.1002/anie.201703114",
                    "url": "https://doi.org/10.1002/anie.201703114",
                    "author": ["Jörg Behler"],
                },
                {
                    "title": "Constructing high‐dimensional neural network potentials: A tutorial review",
                    "journal": "International Journal of Quantum Chemistry",
                    "volume": "115",
                    "number": "16",
                    "year": "2015",
                    "issn": "1097-461X",
                    "doi": "10.1002/qua.24890",
                    "url": "https://doi.org/10.1002/qua.24890",
                    "author": ["Jörg Behler"],
                },
                {
                    "title": "Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces",
                    "journal": "Physical Review Letters",
                    "volume": "98",
                    "number": "14",
                    "year": "2007",
                    "issn": "1079-7114",
                    "doi": "10.1103/PhysRevLett.98.146401",
                    "url": "https://doi.org/10.1103/PhysRevLett.98.146401",
                    "author": ["Jörg Behler", "Michelle Parrinello"],
                },
            ]
        }

    def add_job_to_fitting(self, job):
        self._training_ids.append(job.id)

    def write_input(self):
        with open("input.data", "w") as f:
            for id in self._training_ids:
                container = self.project.load(id)
                for atoms, energy, forces, _ in zip(*container.to_list()):
                    f.write("begin\n")
                    c = np.array(atoms.cell) * AngstromToBohr
                    f.write(f"lattice {c[0][0]} {c[0][1]} {c[0][2]}\n")
                    f.write(f"lattice {c[1][0]} {c[1][1]} {c[1][2]}\n")
                    f.write(f"lattice {c[2][0]} {c[2][1]} {c[2][2]}\n")
                    p = atoms.positions * AngstromToBohr
                    ff = forces * ElectronVoltToHartree / AngstromToBohr
                    for i in range(len(atoms)):
                        f.write(f"atom {p[i, 0]} {p[i, 0]} {p[i, 0]}")
                        f.write(f" {atoms.elements[i].Abbreviation} 0.0 0.0")
                        f.write(f" {ff[i, 0]} {ff[i, 0]} {ff[i, 0]}\n")
                    f.write(f"energy {energy * ElectronVoltToHartree}\n")
                    f.write("charge 0.0\nend\n")

        with open("input.nn", "w") as f:
            f.write(input_template)

    def collect_output(self):
        pass

    # To link to the executable from the notebook
    def _executable_activate(self, enforce=False):     
        if self._executable is None or enforce:
            self._executable = Executable(
                codename="runner", module="runner", path_binary_codes=s.resource_paths
            )

    # def to_hdf(self, hdf=None, group_name=None): 
    #     super().to_hdf(
    #         hdf=hdf,
    #         group_name=group_name
    #    )

    # def from_hdf(self, hdf=None, group_name=None): 
    #     super().from_hdf(
    #         hdf=hdf,
    #        group_name=group_name
    #    )


input_template = """ 
### ####################################################################################################################
### This is the input file for the RuNNer tutorial (POTENTIALS WORKSHOP 2021-03-10) 
### This input file is intended for release version 1.2
### RuNNer is hosted at www.gitlab.com. The most recent version can only be found in this repository.
### For access please contact Prof. Jörg Behler, joerg.behler@uni-goettingen.de
###
### ####################################################################################################################
### General remarks: 
### - commands can be switched off by using the # character at the BEGINNING of the line
### - the input file can be structured by blank lines and comment lines
### - the order of the keywords is arbitrary
### - if keywords are missing, default values will be used and written to runner.out
### - if mandatory keywords or keyword options are missing, RuNNer will stop with an error message 
###
########################################################################################################################
########################################################################################################################
### The following keywords just represent a subset of the keywords offered by RuNNer
########################################################################################################################
########################################################################################################################

########################################################################################################################
### general keywords
########################################################################################################################
nn_type_short 1                           # 1=Behler-Parrinello
runner_mode 1                             # 1=calculate symmetry functions, 2=fitting mode, 3=predicition mode
number_of_elements 1                      # number of elements
elements Cu                               # specification of elements
random_seed 10                            # integer seed for random number generator                         
random_number_type 6                      # 6 recommended       

########################################################################################################################
### NN structure of the short-range NN  
########################################################################################################################
use_short_nn                              # use NN for short range interactions    
global_hidden_layers_short 2              # number of hidden layers               
global_nodes_short 15 15                  # number of nodes in hidden layers     
global_activation_short t t l             # activation functions  (t = hyperbolic tangent, l = linear)              

########################################################################################################################
### symmetry function generation ( mode 1): 
########################################################################################################################
test_fraction 0.10000                     # threshold for splitting between fitting and test set 

########################################################################################################################
### symmetry function definitions (all modes): 
########################################################################################################################
cutoff_type 1
symfunction_short Cu  2 Cu     0.000000      0.000000     12.000000
symfunction_short Cu  2 Cu     0.006000      0.000000     12.000000
symfunction_short Cu  2 Cu     0.016000      0.000000     12.000000
symfunction_short Cu  2 Cu     0.040000      0.000000     12.000000
symfunction_short Cu  2 Cu     0.109000      0.000000     12.000000

symfunction_short Cu  3 Cu Cu     0.00000       1.000000      1.000000     12.000000
symfunction_short Cu  3 Cu Cu     0.00000       1.000000      2.000000     12.000000
symfunction_short Cu  3 Cu Cu     0.00000       1.000000      4.000000     12.000000
symfunction_short Cu  3 Cu Cu     0.00000       1.000000     16.000000     12.000000
symfunction_short Cu  3 Cu Cu     0.00000      -1.000000      1.000000     12.000000
symfunction_short Cu  3 Cu Cu     0.00000      -1.000000      2.000000     12.000000
symfunction_short Cu  3 Cu Cu     0.00000      -1.000000      4.000000     12.000000
symfunction_short Cu  3 Cu Cu     0.00000      -1.000000     16.000000     12.000000

########################################################################################################################
### fitting (mode 2):general inputs for short range AND electrostatic part:
########################################################################################################################
epochs 20                                 # number of epochs                                     
fitting_unit eV                           # unit for error output in mode 2 (eV or Ha)
precondition_weights                      # optional precondition initial weights 

########################################################################################################################
### fitting options ( mode 2): short range part only:
########################################################################################################################
short_energy_error_threshold 0.10000      # threshold of adaptive Kalman filter short E         
short_force_error_threshold 1.00000       # threshold of adaptive Kalman filter short F        
kalman_lambda_short 0.98000               # Kalman parameter short E/F, do not change                        
kalman_nue_short 0.99870                  # Kalman parameter short E/F, do not change                      
use_short_forces                          # use forces for fitting                         
repeated_energy_update                    # optional: repeat energy update for each force update   
mix_all_points                            # do not change                    
scale_symmetry_functions                  # optional
center_symmetry_functions                 # optional 
short_force_fraction 0.01                 #
force_update_scaling -1.0                 #  

########################################################################################################################
### output options for mode 2 (fitting):  
########################################################################################################################
write_trainpoints                         # write trainpoints.out and testpoints.out files      
write_trainforces                         # write trainforces.out and testforces.out files    

########################################################################################################################
### output options for mode 3 (prediction):  
########################################################################################################################
calculate_forces                          # calculate forces    
calculate_stress                          # calculate stress tensor 
"""
