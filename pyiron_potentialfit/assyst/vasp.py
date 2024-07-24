"""Helper classes to run vasp jobs."""
import abc
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Tuple

@dataclass
class KMeshSpec(abc.ABC):
    @abc.abstractmethod
    def configure(self, job):
        pass

@dataclass(slots=True)
class Kpoints(KMeshSpec):
    kpoints: Union[int,Tuple[int, int, int]]
    def configure(self, job):
        job.set_kpoints(self.kpoints)

@dataclass(slots=True)
class Kspacing(KMeshSpec):
    kspacing: float
    def configure(self, job):
        try:
            job.input.incar['KSPACING'] = self.kspacing
        except AttributeError:
            # called on a VaspFactory instead of a Vasp job
            job.incar['KSPACING'] = self.kspacing
        # job.set_kpoints(k_mesh_spacing=self.kspacing)

@dataclass
class VaspConfig:
    encut: Optional[float] = None
    # if float interpreted as k mesh spacing
    kmesh: Optional[Union[int, float, KMeshSpec]] = None
    incar: dict = field(default_factory=dict)

    # pyiron executable version name
    version: Optional[str] = None

    # element to initial collinear magmom
    magmoms: Optional[Dict[str, float]] = None
    # element to POTCAR path
    potcars: Optional[Dict[str, str]] = None

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
