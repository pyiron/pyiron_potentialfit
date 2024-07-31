from logging import getLogger
from random import choices

from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage
from pyiron_atomistics.atomistics.job.structurecontainer import StructureContainer
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure

import numpy as np
from tqdm.auto import tqdm

def shake(displacement=0.1):
    """
    Return a function that randomly displaces atoms in structures.

    Args:
        displacement (float): standard deviation of atomic displacement
    """

    def mod(structure):
        structure.positions += np.random.normal(
            scale=displacement, size=structure.positions.shape
        )
        return structure

    return mod


def stretch(hydro: float = 0.05, shear: float = 0.005):
    """
    Return a function that strains structures.

    Random strains are drawn from a uniform distribution within the positive
    and negative limits given.

    Args:
        hydro (float): Maximum strain along normal axes
        shear (float): Maximum strain along shear axes
    """

    def mod(structure):
        E = shear * (2 * np.random.rand(3, 3) - 1)
        E = 0.5 * (E + E.T)  # symmetrize
        np.fill_diagonal(E, hydro * (2 * np.random.rand(3) - 1))
        structure.apply_strain(E)
        return structure

    return mod


def fill_container(
    source: HasStructure,
    sink: StructureContainer,
    repetitions: int = 4,
    combine: int = 1,
    modifiers=((0.5, shake()), (0.5, stretch())),
    min_dist=None,
):
    """
    Fill a container with new structures.

    Iterates over all structures in `source`
    """
    ps, mods = zip(*modifiers)
    for structure in tqdm(source.iter_structures(), total=source.number_of_structures):
        for _ in range(repetitions):
            for i in range(10):
                s = structure.copy()
                for mod in choices(mods, weights=ps, k=combine):
                    s = mod(s)
                if min_dist is None:
                    sink.append(s)
                    break
                else:
                    # one weird aspect ratios, the neighbor searching code can allocate huge structures,
                    # because it explicitly repeats the structure to create ghost atoms
                    # since we only care about the presence of short distances between atoms and not the
                    # real neighbor information, simply double the structure to make sure we see all bonds 
                    # and turn off PBC
                    sd = s.repeat(2)
                    sd.pbc = [False, False, False]
                    dist = sd.get_neighbors(
                        num_neighbors=1, cutoff_radius=2 * min_dist
                    ).distances
                    if (dist > min_dist).all():
                        sink.append(s)
                        break
            else:
                print(
                    "WARN: Tried 10 times to find a structures, but "
                    "min_dist is never satisfied."
                )
    return sink


def rattle(
    pr,
    cont: StructureContainer,
    rattle_disp,
    rattle_strain,
    rattle_repetitions,
    stretch_hydro,
    stretch_shear,
    stretch_repetitions,
    min_dist,
    delete_existing_job=False,
):
    logger = getLogger("structures")
    logger.info("Creating rattle/stretch structures")

    rand = pr.create.job.StructureContainer(
        f"{cont.name}Random", delete_existing_job=delete_existing_job
    )
    if rand.status.initialized:
        N = 0
        fill_container(
            cont._container.sample(lambda f, i: f["length", i] > 1),
            rand,
            repetitions=rattle_repetitions,
            combine=2,
            modifiers=(
                (1, shake(rattle_disp)),
                (1, stretch(rattle_strain))
            ),
            min_dist=min_dist,
        )
        logger.info("added %i rattle structures", rand.number_of_structures - N)
        N = rand.number_of_structures
        fill_container(
            cont._container,
            rand,
            repetitions=stretch_repetitions,
            combine=1,
            modifiers=(
                (0.7, stretch(hydro=stretch_hydro, shear=0.05)),
                (0.3, stretch(hydro=0.05, shear=stretch_shear)),
            ),
            min_dist=min_dist,
        )
        logger.info("added %i stretch structures", rand.number_of_structures - N)
        rand.run()
    return rand
