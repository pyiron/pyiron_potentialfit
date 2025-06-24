# coding: utf-8
# Copyright (c) ICAMS, Ruhr University Bochum, 2022

## Executable required: $pyiron/resources/pacemaker/bin/run_pacemaker_tf_cpu.sh AND  run_pacemaker_tf.sh

import logging
from typing import List, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import ruamel.yaml as yaml
import re
import scipy.optimize as so

from shutil import copyfile

from pyiron_base import (
    GenericJob,
    GenericParameters,
    state,
    Executable,
    FlattenedStorage,
)

from pyiron_potentialfit.atomistics.job.trainingcontainer import (
    TrainingStorage,
    TrainingContainer,
)
from pyiron_potentialfit.ml.potentialfit import PotentialFit

from pyiron_atomistics.atomistics.structure.atoms import (
    Atoms as pyironAtoms,
    ase_to_pyiron,
)
from ase.atoms import Atoms as aseAtoms

s = state.settings


class PacemakerJob(GenericJob, PotentialFit):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.__name__ = "Pacemaker2022"
        self.__version__ = "0.2"

        self._train_job_id_list = []
        self._test_job_id_list = []
        self._yace_files_list = []
        self._compress_by_default = True
        self.input = GenericParameters(table_name="input")
        self._cutoff = 7.0
        self.input["cutoff"] = self._cutoff
        self.input["metadata"] = {"comment": "pyiron-generated fitting job"}

        # data_config
        self.input["data"] = {}
        # potential_config
        self.input["potential"] = {
            "elements": [],
            "bonds": {
                "ALL": {
                    "radbase": "SBessel",
                    "rcut": self._cutoff,
                    "dcut": 0.01,
                    "radparameters": [5.25],
                }
            },
            "embeddings": {
                "ALL": {
                    "fs_parameters": [1, 1, 1, 0.5],
                    "ndensity": 2,
                    "npot": "FinnisSinclairShiftedScaled",
                }
            },
            "functions": {
                "ALL": {
                    "nradmax_by_orders": [15, 3, 2, 1],
                    "lmax_by_orders": [0, 3, 2, 1],
                }
            },
        }

        # fit_config
        self.input["fit"] = {
            "loss": {
                "L1_coeffs": 1e-8,
                "L2_coeffs": 1e-8,
                "kappa": 0.3,
                "w0_rad": 0,
                "w1_rad": 0,
                "w2_rad": 0,
            },
            "maxiter": 1000,
            "optimizer": "BFGS",
            "fit_cycles": 1,
        }
        self.input["backend"] = {
            "batch_size": 100,
            "display_step": 50,
            "evaluator": "tensorpot",
        }  # backend_config

        self.structure_data = None
        self.testing_structure_data = None
        self._executable = None
        self._executable_activate()

        state.publications.add(self.publication)

    @property
    def elements(self):
        return self.input["potential"].get("elements")

    @elements.setter
    def elements(self, val):
        self.input["potential"]["elements"] = val

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, val):
        self._cutoff = val
        self.input["cutoff"] = self._cutoff
        self.input["potential"]["bonds"]["ALL"]["rcut"] = self._cutoff

    @property
    def publication(self):
        return {
            "pacemaker": [
                {
                    "title": "Efficient parametrization of the atomic cluster expansion",
                    "journal": "Physical Review Materials",
                    "volume": "6",
                    "number": "1",
                    "year": "2022",
                    "doi": "10.1103/PhysRevMaterials.6.013804",
                    "url": "https://doi.org/10.1103/PhysRevMaterials.6.013804",
                    "author": [
                        "Anton Bochkarev",
                        "Yury Lysogorskiy",
                        "Sarath Menon",
                        "Minaam Qamar",
                        "Matous Mrovec",
                        "Ralf Drautz",
                    ],
                },
                {
                    "title": "Performant implementation of the atomic cluster expansion (PACE) and application to copper and silicon",
                    "journal": "npj Computational Materials",
                    "volume": "7",
                    "number": "1",
                    "year": "2021",
                    "doi": "10.1038/s41524-021-00559-9",
                    "url": "https://doi.org/10.1038/s41524-021-00559-9",
                    "author": [
                        "Yury Lysogorskiy",
                        "Cas van der Oord",
                        "Anton Bochkarev",
                        "Sarath Menon",
                        "Matteo Rinaldi",
                        "Thomas Hammerschmidt",
                        "Matous Mrovec",
                        "Aidan Thompson",
                        "Gábor Csányi",
                        "Christoph Ortner",
                        "Ralf Drautz",
                    ],
                },
                {
                    "title": "Atomic cluster expansion for accurate and transferable interatomic potentials",
                    "journal": "Physical Review B",
                    "volume": "99",
                    "year": "2019",
                    "doi": "10.1103/PhysRevB.99.014104",
                    "url": "https://doi.org/10.1103/PhysRevB.99.014104",
                    "author": ["Ralf Drautz"],
                },
            ]
        }

    def _save_structure_dataframe_pckl_gzip(self, df, dataset_type: str = "training"):
        if "NUMBER_OF_ATOMS" not in df.columns and "number_of_atoms" in df.columns:
            df.rename(columns={"number_of_atoms": "NUMBER_OF_ATOMS"}, inplace=True)
        df["NUMBER_OF_ATOMS"] = df["NUMBER_OF_ATOMS"].astype(int)

        # TODO: reference energy subtraction ?
        if "energy_corrected" not in df.columns and "energy" in df.columns:
            df.rename(columns={"energy": "energy_corrected"}, inplace=True)

        if "atoms" in df.columns:
            # check if this is pyironAtoms  -> aseAtoms
            at = df.iloc[0]["atoms"]
            if isinstance(at, pyironAtoms):
                df["ase_atoms"] = df["atoms"].map(lambda s: s.to_ase())
                df.drop(columns=["atoms"], inplace=True)
            else:
                assert isinstance(
                    at, aseAtoms
                ), "'atoms' column is not a valid ASE Atoms object"
                df.rename(columns={"atoms": "ase_atom"}, inplace=True)
        elif "ase_atoms" not in df.columns:
            raise ValueError(
                "DataFrame should contain 'atoms' (pyiron Atoms) or 'ase_atoms' (ASE atoms) columns"
            )

        if "stress" in df.columns:
            df.drop(columns=["stress"], inplace=True)

        if "pbc" not in df.columns:
            df["pbc"] = df["ase_atoms"].map(lambda atoms: np.all(atoms.pbc))

        # Use dataset_type to differentiate the filename
        data_file_name = os.path.join(
            self.working_directory, f"df_{dataset_type}.pckl.gzip"
        )
        logging.info(
            f"Saving {dataset_type} structures dataframe into {data_file_name} with pickle protocol = 4, compression = gzip"
        )
        df.to_pickle(data_file_name, compression="gzip", protocol=4)
        return data_file_name

    def write_input(self):
        # prepare training datafile
        if self._train_job_id_list and self.structure_data is None:
            train_df = self.create_training_dataframe(self._train_job_id_list)
            self.structure_data = train_df

        if isinstance(self.structure_data, pd.DataFrame):
            logging.info("structure_data is pandas.DataFrame")
            data_file_name = self._save_structure_dataframe_pckl_gzip(
                df=self.structure_data
            )
            self.input["data"]["filename"] = data_file_name

            # Automatically determine the list of elements
            elements_set = set()
            for at in self.structure_data["ase_atoms"]:
                elements_set.update(at.get_chemical_symbols())
            elements = sorted(elements_set)
            print("Set automatically determined list of elements: {}".format(elements))
            self.elements = elements

        elif isinstance(self.structure_data, str):  # filename
            if os.path.isfile(self.structure_data):
                logging.info("structure_data is valid file path")
                self.input["data"]["filename"] = self.structure_data
            else:
                raise ValueError(
                    "Provided structure_data filename ({}) doesn't exists".format(
                        self.structure_data
                    )
                )
        elif hasattr(
            self.structure_data, "get_pandas"
        ):  # duck-typing check for TrainingContainer
            logging.info("structure_data is TrainingContainer")
            df = self.structure_data.to_pandas()
            data_file_name = self._save_structure_dataframe_pckl_gzip(df=df)
            self.input["data"]["filename"] = data_file_name
        elif self.structure_data is None:
            raise ValueError(
                "`structure_data` is none, but should be pd.DataFrame, TrainingContainer or valid pickle.gzip filename"
            )

        # prepare testing datafile
        if self._test_job_id_list and self.testing_structure_data is None:
            test_df = self.create_training_dataframe(self._test_job_id_list)
            self.testing_structure_data = test_df

        # Save testing data file if testing_structure_data is a DataFrame
        if isinstance(self.testing_structure_data, pd.DataFrame):
            logging.info("testing_structure_data is a pandas DataFrame")
            test_file_name = self._save_structure_dataframe_pckl_gzip(
                df=self.testing_structure_data, dataset_type="testing"
            )
            self.input["data"]["test_filename"] = test_file_name

        metadata_dict = self.input["metadata"]
        metadata_dict["pyiron_job_id"] = str(self.job_id)

        input_yaml_dict = {
            "cutoff": self.input["cutoff"],
            "metadata": metadata_dict,
            "potential": self.input["potential"],
            "data": self.input["data"],
            "fit": self.input["fit"],
            "backend": self.input["backend"],
        }

        if isinstance(self.input["potential"], str):
            pot_file_name = self.input["potential"]
            if os.path.isfile(pot_file_name):
                logging.info("Input potential is filename")
                pot_basename = os.path.basename(pot_file_name)
                copyfile(
                    pot_file_name, os.path.join(self.working_directory, pot_basename)
                )
                input_yaml_dict["potential"] = pot_basename
            else:
                raise ValueError(
                    "Provided potential filename ({}) doesn't exists".format(
                        self.input["potential"]
                    )
                )

        with open(os.path.join(self.working_directory, "input.yaml"), "w") as f:
            yaml.YAML(typ="unsafe", pure=True).dump(input_yaml_dict, f)

    def _analyse_log(self, logfile="metrics.txt"):
        metrics_filename = os.path.join(self.working_directory, logfile)

        metrics_df = pd.read_csv(metrics_filename, sep="\s+")
        res_dict = metrics_df.to_dict(orient="list")
        return res_dict

    def _calculate_loss_time(self, df, message_filter, reference_time):
        df_filtered = df[message_filter]
        df_filtered.reset_index(drop=True, inplace=True)
        # Checking for duplicated iterations is important for testing dataset
        df_filtered = df_filtered[
            df_filtered["iteration"] != df_filtered["iteration"].shift()
        ]
        df_filtered["time"] = (
            (df_filtered["Timestamp"] - reference_time).dt.total_seconds().astype(int)
        )
        return df_filtered["time"].tolist()

    def _parse_time(self, logfile="log.txt"):
        # Read and parse the log file
        log_file_path = os.path.join(self.working_directory, logfile)
        with open(log_file_path, "r") as file:
            log_file = file.read()

        # Regex pattern for log entries
        log_pattern = r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})(?: (?P<level>[IWD]) -)? (?P<message>(?:.*?\n)+?(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}|\Z))"
        parsed_logs = re.findall(log_pattern, log_file, re.DOTALL)

        # Convert logs to DataFrame
        df = pd.DataFrame(parsed_logs, columns=["Timestamp", "Level", "Message"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S,%f")
        df["iteration"] = df["Message"].str.extract(r"\nIteration:\s+#(\d+)")

        # Calculate training loss times
        training_filter = df["Message"].str.contains(r"Loss: \d") & ~df[
            "Message"
        ].str.contains("TEST") | df["Message"].str.contains("FIT STATS")
        training_loss_time = self._calculate_loss_time(
            df, training_filter, df.loc[training_filter, "Timestamp"].iloc[0]
        )

        # Calculate testing loss times
        if self.project_hdf5["output/log_test/loss"]:
            testing_filter = df["Message"].str.contains(r"TEST STATS|TEST Cycle last")
            # Problem caused here when dataframe is empty
            testing_loss_time = self._calculate_loss_time(
                df, testing_filter, df.loc[testing_filter, "Timestamp"].iloc[0]
            )
        else:
            testing_loss_time = []

        # Calculate intermediate fitting times for ladder scheme
        intermediate_filter = df["Message"].str.startswith(
            "Intermediate potential saved in interim_potential_lad"
        )
        ladder_steps_time = self._calculate_loss_time(
            df, intermediate_filter, df.loc[training_filter, "Timestamp"].iloc[0]
        )

        return training_loss_time, testing_loss_time, ladder_steps_time

    def _convert_interim_ladder_potentials(self):
        from pyace import ACEBBasisSet

        for filename in sorted(self.files.list()):
            if "interim_potential_ladder" in filename and (
                filename.endswith(".yaml") or filename.endswith(".yml")
            ):
                filename_without_yaml = filename.replace(".yaml", "").replace(
                    ".yml", ""
                )
                filename_with_yace = filename_without_yaml + ".yace"
                self._yace_files_list.append(filename_with_yace)

                # Converting Yaml to Yace
                logging.info(f"{filename}: converting to {filename_with_yace}")
                bbasis = ACEBBasisSet(self.working_directory + "/" + filename)
                cbasis = bbasis.to_ACECTildeBasisSet()
                cbasis.save_yaml(self.working_directory + "/" + filename_with_yace)

    def collect_output(self):
        final_potential_filename_yaml = self.get_final_potential_filename()
        with open(final_potential_filename_yaml, "r") as f:
            yaml_lines = f.readlines()
        final_potential_yaml_string = "".join(yaml_lines)

        with open(self.get_final_potential_filename_ace(), "r") as f:
            ace_lines = f.readlines()
        final_potential_yace_string = "".join(ace_lines)

        with open(self.get_final_potential_filename_ace(), "r") as f:
            yace_data = yaml.YAML(typ="unsafe", pure=True).load(f)

        elements_name = yace_data["elements"]

        with self.project_hdf5.open("output/potential") as h5out:
            h5out["yaml"] = final_potential_yaml_string
            h5out["yace"] = final_potential_yace_string
            h5out["elements_name"] = elements_name

        # Convert all intermediate yaml files into intermediate yace files
        self._convert_interim_ladder_potentials()

        # Reads metrics.txt file by default for training data
        log_res_dict = self._analyse_log()
        with self.project_hdf5.open("output/log") as h5out:
            for key, arr in log_res_dict.items():
                h5out[key] = arr
        # Reads test_metric.txt for testing data
        log_res_dict = self._analyse_log(logfile="test_metrics.txt")
        with self.project_hdf5.open("output/log_test") as h5out:
            for key, arr in log_res_dict.items():
                h5out[key] = arr
        # If ladder scheme, parse the ladder_metrics txt files
        try:
            log_res_dict = self._analyse_log(logfile="ladder_metrics.txt")
            with self.project_hdf5.open("output/ladder") as h5out:
                for key, arr in log_res_dict.items():
                    h5out[key] = arr
            # For testing data
            log_res_dict = self._analyse_log(logfile="test_ladder_metrics.txt")
            with self.project_hdf5.open("output/test_ladder") as h5out:
                for key, arr in log_res_dict.items():
                    h5out[key] = arr
        except:
            logging.info("Single-shot scheme was used, no ladder data parsed")

        # parse the time from the log.txt file
        training_loss_time, testing_loss_time, ladder_steps_time = self._parse_time()
        self.project_hdf5["output/log/loss_time"] = training_loss_time
        self.project_hdf5["output/log_test/loss_time"] = testing_loss_time
        if "ladder" in self.project_hdf5["output"].keys():
            self.project_hdf5["output/ladder/ladder_time"] = ladder_steps_time

        # training data
        training_data_fname = os.path.join(
            self.working_directory, "fitting_data_info.pckl.gzip"
        )
        df = pd.read_pickle(training_data_fname, compression="gzip")
        df["atoms"] = df.ase_atoms.map(ase_to_pyiron)
        training_data_ts = TrainingStorage()
        for _, r in df.iterrows():
            training_data_ts.add_structure(
                r.atoms,
                energy=r.energy_corrected,
                forces=r.forces,
                identifier=r["name"],
            )

        data_list = [["train_pred.pckl.gzip", "training"]]

        # Add testing data to the list if available
        if self._test_job_id_list:
            data_list.append(["test_pred.pckl.gzip", "testing"])
            testing_data_fname = os.path.join(
                self.working_directory, "test_data_info.pckl.gzip"
            )

            # Load testing data, map atoms, and populate TrainingStorage
            test_df = pd.read_pickle(testing_data_fname, compression="gzip")
            test_df["atoms"] = test_df.ase_atoms.map(ase_to_pyiron)
            testing_data_ts = TrainingStorage()
            for _, row in test_df.iterrows():
                testing_data_ts.add_structure(
                    row.atoms,
                    energy=row.energy_corrected,
                    forces=row.forces,
                    identifier=row["name"],
                )

        # predicted data
        for file_name, name in data_list:
            predicted_fname = os.path.join(self.working_directory, file_name)
            df = pd.read_pickle(predicted_fname, compression="gzip")
            predicted_data_fs = FlattenedStorage()
            predicted_data_fs.add_array(
                "energy", dtype=np.float64, shape=(), per="chunk"
            )
            predicted_data_fs.add_array(
                "energy_true", dtype=np.float64, shape=(), per="chunk"
            )

            predicted_data_fs.add_array(
                "number_of_atoms", dtype=np.int64, shape=(), per="chunk"
            )

            predicted_data_fs.add_array(
                "forces", dtype=np.float64, shape=(3,), per="element"
            )
            predicted_data_fs.add_array(
                "forces_true", dtype=np.float64, shape=(3,), per="element"
            )
            for i, r in df.iterrows():
                identifier = r["name"] if "name" in r else str(i)
                predicted_data_fs.add_chunk(
                    r["NUMBER_OF_ATOMS"],
                    identifier=identifier,
                    energy=r.energy_pred,
                    forces=r.forces_pred,
                    energy_true=r.energy_corrected,
                    forces_true=r.forces,
                    number_of_atoms=r.NUMBER_OF_ATOMS,
                    energy_per_atom=r.energy_pred / r.NUMBER_OF_ATOMS,
                    energy_per_atom_true=r.energy_corrected / r.NUMBER_OF_ATOMS,
                )

            with self.project_hdf5.open("output") as hdf5_output:
                if name == "training":
                    training_data_ts.to_hdf(hdf=hdf5_output, group_name="training_data")
                    predicted_data_fs.to_hdf(
                        hdf=hdf5_output, group_name="predicted_data"
                    )
                elif name == "testing":
                    testing_data_ts.to_hdf(hdf=hdf5_output, group_name="testing_data")
                    predicted_data_fs.to_hdf(
                        hdf=hdf5_output, group_name="testing_predicted_data"
                    )

    def get_lammps_potential(self, pot_index: int = None):
        elements_name = self.project_hdf5["output/potential/elements_name"]
        elem = " ".join(elements_name)
        if pot_index is None:
            pot_file_name = self.get_final_potential_filename_ace()
        elif isinstance(pot_index, int):
            filename_with_yace = f"interim_potential_ladder_step_{pot_index}.yace"
            if filename_with_yace in self._yace_files_list:
                pot_file_name = self.working_directory + "/" + filename_with_yace
            else:
                raise ValueError(
                    f"File {filename_with_yace } not found in {self._yace_files_list}"
                )

        pot_dict = {
            "Config": [
                [
                    "pair_style pace\n",
                    "pair_coeff  * * {} {}\n".format(pot_file_name, elem),
                ]
            ],
            "Filename": [""],
            "Model": ["ACE"],
            "Name": [self.job_name],
            "Species": [elements_name],
        }

        ace_potential = pd.DataFrame(pot_dict)

        return ace_potential

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as h5in:
            self.input.to_hdf(h5in)
        self.project_hdf5["input/training_job_ids"] = self._train_job_id_list
        self.project_hdf5["input/testing_job_ids"] = self._test_job_id_list

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as h5in:
            self.input.from_hdf(h5in)
        if "training_job_ids" in self.project_hdf5["input"].list_nodes():
            self._train_job_id_list = self.project_hdf5["input/training_job_ids"]
        if "testing_job_ids" in self.project_hdf5["input"].list_nodes():
            self._test_job_id_list = self.project_hdf5["input/testing_job_ids"]

    def get_final_potential_filename(self):
        return os.path.join(self.working_directory, "output_potential.yaml")

    def get_final_potential_filename_ace(self):
        return os.path.join(self.working_directory, "output_potential.yace")

    def get_current_potential_filename(self):
        return os.path.join(self.working_directory, "interim_potential_0.yaml")

    # To link to the executable from the notebook
    def _executable_activate(self, enforce=False, codename="pacemaker"):
        if self._executable is None or enforce:
            self._executable = Executable(
                codename=codename,
                module="pacemaker",
                path_binary_codes=state.settings.resource_paths,
            )

    def _add_training_data(self, container: TrainingContainer) -> None:
        self.add_job_to_fitting(container.id)

    def add_job_to_fitting(self, job_id, *args, **kwargs):
        self._train_job_id_list.append(job_id)

    def _get_training_data(self) -> TrainingStorage:
        return self.project_hdf5["output/training_data"].to_object()

    def _get_predicted_data(self) -> FlattenedStorage:
        return self.project_hdf5["output/predicted_data"].to_object()

    def _add_testing_data(self, container: TrainingContainer) -> None:
        self.add_test_job_to_fitting(container.id)

    def add_test_job_to_fitting(self, job_id, *args, **kwargs):
        self._test_job_id_list.append(job_id)

    def _get_testing_data(self) -> TrainingStorage:
        return self.project_hdf5["output/testing_data"].to_object()

    def _get_testing_predicted_data(self) -> FlattenedStorage:
        return self.project_hdf5["output/testing_predicted_data"].to_object()

    # copied/adapted from mlip.py
    def create_training_dataframe(
        self, _train_job_id_list: List = None
    ) -> pd.DataFrame:
        if _train_job_id_list is None:
            _train_job_id_list = self._train_job_id_list
        df_list = []
        for job_id in _train_job_id_list:
            ham = self.project.inspect(job_id)
            if ham.__name__ == "TrainingContainer":
                job = ham.to_object()
                data_df = job.to_pandas()
                df_list.append(data_df)
            else:
                raise NotImplementedError(
                    "Currently only TrainingContainer is supported"
                )

        total_training_df = pd.concat(df_list, axis=0)
        total_training_df.reset_index(drop=True, inplace=True)

        return total_training_df

    def check_inner_cutoffs(self):
        """
        Plot the pair interactions around the inner cutoff radius.
        """
        if not self.status.finished:
            raise ValueError("Status must be finished to check inner cutoffs!")
        import pyace
        from pyiron_snippets.logger import logger

        # pyace resets our log level by accident, circumvent that here
        level = logger.getEffectiveLevel()
        ace = pyace.PyACECalculator(str(self.files.output_potential_yace))
        logger.setLevel(level)

        pot = yaml.YAML(typ="safe").load(self.content["output/potential/yaml"])
        cuts = {}
        for sp in pot["species"]:
            elems = tuple(sorted(sp["speciesblock"].split()))
            if "r_in" not in sp:
                continue
            if len(elems) == 1:
                elems = (elems[0],) * 2
            cuts[elems] = sp["r_in"]

        df = []
        for elems, ri in cuts.items():
            r = np.linspace(0.9, 1.1) * ri
            e = []
            for rr in r:
                s = self.project.create.structure.atoms(
                    elems, positions=[[0] * 3, [rr, 0, 0]], cell=[50] * 3
                )
                s.calc = ace
                e.append(s.get_potential_energy())
            df.append({"pair": "-".join(elems), "r": r, "e": e})

        df = pd.DataFrame(df).explode(["r", "e"]).infer_objects()
        sns.lineplot(data=df, x="r", y="e", hue="pair")
        plt.xlabel(r"Distance [$\AA$]")
        plt.ylabel(r"Energy [eV]")
        return df

    def compress(self, files_to_compress=None):
        """
        Compress the output files of a job object.

        Args:
            files_to_compress (list): A list of files to compress (optional)
        """
        if files_to_compress is None:
            files_to_compress = [
                f for f in self.files.list() if not f.endswith(".yace")
            ]
        super().compress(files_to_compress=files_to_compress)

    def fix_inner(
        self,
        potential: str = "output_potential.yaml",
        overwrite: bool = False,
        optimize_repulsion: bool = True,
        inner_cutoff_type: Literal["zbl", "distance"] = None,
        delta_in: float = 0.1,
        plot: bool = True,
        log: bool = True,
    ):
        """
        Try to optimize the inner cutoff functions for repulsion.

        Original observation: with the core-repulsion: auto setting pacemaker simply activates the
        inner cutoff where data stops.  For ZBL this often leads to non monotonous behavior in the
        region where the switching between the learning radial basis and ZBL happens.  Especially
        when delta_in is small large forces can be observed in this region.  This causes problems
        for relaxations from difficult starting points (interstitials, gamma surfaces), but also
        with stability during MD.

        Using energies and forces predicted for a dimer, try to make the onset of the core-repulsion as smooth as possible.
        The inner cutoff radius is moved to the inflection point of the dimer curve of bare ACE potential.
        If the potential intrinsically reproduces the repulsion well already, this can fail.

        A corresponding .yace file is automatically generated.
        Parameters `inner_cutoff_type` and `delta_in` are exactly as documented in pacemaker.

        .. warning::
            Use at your own risk!  Double check dimer and energy volume curves.
            This may be a quick fix for you potential if you have not included enough close distance data, but it also may not!

        Args:
            potential (str): path or name of potential to fix, must be in .yaml format
            overwrite (bool): if True, overwrite given potential, otherwise add postfix '_inner' to `potential`
            optimize_repulsion (bool): PACE offers some parameters to tune the core repulsion; use them to make the transition region as smooth as possible;
                for 'zbl' just match the absolute value at the inner cutoff;
                for 'distance' try to match both absolute value and gradient at the inner cutoff, this can be unstable;
                for 'density' this code is not tested and will likely fail
            inner_cutoff_type (str): which core repulsion to use, if None use whichever was specified at fit time.
            delta_in (float): range over which the core repulsion is turned on
            plot (bool): plot old and fixed dimer curves
            log (bool): print updated core-repulstion parameters
        """
        import pyace

        self.decompress()
        if potential in self.files.list():
            potential = self.files[potential]
        potfile = str(potential)

        ace = pyace.PyACECalculator(potfile)
        bbs = ace.basis.to_BBasisConfiguration()

        def ace_dimer(ace, elems, rr):
            if isinstance(rr, Iterable):
                return np.array(list(zip(*[ace_dimer(ace, elems, ri) for ri in rr])))
            s = self.project.create.structure.atoms(
                elems, positions=[[0] * 3, [rr, 0, 0]], cell=[50] * 3
            )
            s.calc = ace
            return s.get_potential_energy(), s.get_forces()[1, 0]

        def ace_dimer_e(ace, elems, rr):
            if isinstance(rr, Iterable):
                return np.array([ace_dimer_e(ace, elems, ri) for ri in rr])
            return ace_dimer(ace, elems, rr)[0]

        def ace_dimer_f(ace, elems, rr):
            if isinstance(rr, Iterable):
                return np.array([ace_dimer_f(ace, elems, ri) for ri in rr])
            return ace_dimer(ace, elems, rr)[1]

        def find_stitching_point(ace, elems, r0):
            # current cutoff in attractive region still!
            if ace_dimer_f(ace, elems, r0) < 0:
                # find minimum of potential then move a bit to the left
                r0 = so.fmin(lambda r: ace_dimer_e(ace, elems, r), x0=r0, disp=False)[0]
                r0 -= 0.1
            return so.fmin(
                lambda r: -(ace_dimer_f(ace, elems, r)) if r > 0 else np.inf,
                x0=r0,
                disp=False,
            )[0]

        def get_elems(block):
            elems = block.block_name.split()
            if len(elems) == 1:
                elems *= 2
            return elems

        # any modification we do to the El1-El2 block must be made for El2-El1
        # block as well to keep radial basis consistent
        def set_block_values(elems, **kwargs):
            for block in bbs.funcspecs_blocks:
                if set(get_elems(block)) != set(elems):
                    continue
                for k, v in kwargs.items():
                    setattr(block, k, v)

        # valid BBasis requires to set the cutoff_type for all blocks at the same
        # time
        for i, block in enumerate(bbs.funcspecs_blocks):
            elems = get_elems(block)
            if len(elems) == 2 and inner_cutoff_type is not None:
                set_block_values(elems, inner_cutoff_type=inner_cutoff_type)

        visited = set()
        for block in bbs.funcspecs_blocks:
            elems = get_elems(block)
            if len(elems) > 2:
                continue
            if tuple(sorted(elems)) in visited:
                continue
            visited.add(tuple(sorted(elems)))

            # save old inner cutoff and set to zero to have access to unadultered
            # potential below
            r_in = block.r_in
            set_block_values(elems, r_in=0.0)

            ace0 = pyace.PyACECalculator(bbs)
            rr = find_stitching_point(ace0, elems, r_in)

            if plot:
                plt.figure()
                plt.title(elems)

            if optimize_repulsion:
                # energies and forces we are trying to match
                r_range = np.linspace(rr - delta_in / 2, rr + delta_in / 2)
                e0, f0 = ace_dimer(ace0, elems, r_range)

                def step(crp):
                    if len(crp) == 2:
                        k, l = crp
                    else:
                        k, l = crp, 1
                    set_block_values(
                        elems,
                        core_rep_parameters=[np.exp(k), l],
                        r_in=rr,
                        delta_in=delta_in,
                    )
                    acei = pyace.PyACECalculator(bbs)
                    e, f = ace_dimer(acei, elems, r_range)
                    return abs(e - e0).mean()

                if block.inner_cutoff_type == "zbl":
                    # make sure repulsion function is active way before the region were are searching in
                    set_block_values(elems, r_in=rr + 2 + block.delta_in)
                    set_block_values(elems, core_rep_parameters=[1, 1])
                    ei = ace_dimer_e(pyace.PyACECalculator(bbs), elems, rr)
                    e0 = ace_dimer_e(ace0, elems, rr)
                    scale = (e0 / ei).mean()
                    set_block_values(
                        elems,
                        core_rep_parameters=[scale, 1],
                        r_in=rr,
                        delta_in=delta_in,
                    )
                    if plot:
                        plt.scatter(
                            rr, e0, marker="o", label="stitching point", zorder=10
                        )
                else:
                    ret = so.minimize(
                        step, x0=[1, 1], bounds=((-np.inf, np.inf), (0, np.inf))
                    )
                    set_block_values(
                        elems,
                        core_rep_parameters=[np.exp(ret.x[0]), ret.x[1]],
                        r_in=rr,
                        delta_in=delta_in,
                    )
                    if plot:
                        plt.scatter(r_range, e0, marker="v", label="stitching points")
                if log:
                    print(elems, rr, block.core_rep_parameters)
            else:
                set_block_values(
                    elems, r_in=rr, delta_in=delta_in, core_rep_parameters=[1, 1]
                )
                e0, f0 = ace_dimer(ace0, elems, rr)
                if plot:
                    plt.scatter(rr, e0, marker="v", label="stitching point")

            r = np.linspace(np.clip(0.9 * (rr - delta_in), 0.1, None), rr * 1.2, 100)
            acef = pyace.PyACECalculator(bbs)
            if plot:
                plt.plot(r, ace_dimer_e(ace, elems, r), "-", label="original repulsion")
                plt.plot(r, ace_dimer_e(ace0, elems, r), "--", label="no repulsion")
                plt.plot(r, ace_dimer_e(acef, elems, r), "-", label="updated repulsion")
                plt.legend()

        if overwrite:
            copyfile(potfile, f"{potfile}.backup")
            bbs.save(potfile)
            # convert to yace format
            pyace.ACEBBasisSet(bbs).to_ACECTildeBasisSet().save_yaml(
                os.path.splitext(potfile)[0] + ".yace"
            )
        else:
            ofile = os.path.splitext(potfile)[0] + "_inner.yaml"
            bbs.save(ofile)
            # convert to yace format
            pyace.ACEBBasisSet(bbs).to_ACECTildeBasisSet().save_yaml(
                os.path.splitext(ofile)[0] + ".yace"
            )
        self.compress()
