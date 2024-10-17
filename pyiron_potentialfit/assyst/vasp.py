"""Helper classes to run vasp jobs."""

import abc
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Tuple

from pyiron_base import GenericJob


@dataclass
class KMeshSpec(abc.ABC):
    @abc.abstractmethod
    def configure(self, job):
        pass


@dataclass(slots=True)
class Kpoints(KMeshSpec):
    kpoints: Union[int, Tuple[int, int, int]]

    def configure(self, job):
        job.set_kpoints(self.kpoints)


@dataclass(slots=True)
class Kspacing(KMeshSpec):
    kspacing: float

    def configure(self, job):
        try:
            job.input.incar["KSPACING"] = self.kspacing
        except AttributeError:
            # called on a VaspFactory instead of a Vasp job
            job.incar["KSPACING"] = self.kspacing
        # job.set_kpoints(k_mesh_spacing=self.kspacing)


@dataclass
class VaspConfig:
    encut: Optional[float] = None
    # if float interpreted as k mesh spacing
    kmesh: Optional[Union[int, float, KMeshSpec]] = None
    empty_states: Optional[float] = None
    incar: dict = field(default_factory=dict)

    # pyiron executable version name
    version: Optional[str] = None

    # element to initial collinear magmom
    magmoms: Optional[Dict[str, float]] = field(default_factory=dict)
    # element to POTCAR path
    potcars: Optional[Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        match self.kmesh:
            case int(points) | {"kpoints": points}:
                self.kmesh = Kpoints(points)
            case float(spacing) | {"kspacing": spacing}:
                self.kmesh = Kspacing(spacing)
            case Kspacing(_) | Kpoints(_) | None:
                pass
            case val:
                assert False, f"Invalid value {val}!"

    def configure_vasp_job(self, job):
        if self.encut is not None:
            job.set_encut(self.encut)
        if self.kmesh is not None:
            self.kmesh.configure(job)
        for element, path in self.potcars.items():
            job.potential[element] = path
        if (
            self.magmoms is not None
            and self.magmoms != {}
            and "LORBIT" not in self.incar
        ):
            self.incar["LORBIT"] = 10
        if self.empty_states is not None:
            job.set_empty_states(self.empty_states)
        for k, v in self.incar.items():
            try:
                job.input.incar[k] = v
            except AttributeError:
                # called on a VaspFactory instead of a Vasp job
                job.incar[k] = v
        if self.version is not None:
            if isinstance(job, GenericJob):
                job.version = self.version
            else:
                # called on a VaspFactory instead of a Vasp job
                job.attr.version = self.version
