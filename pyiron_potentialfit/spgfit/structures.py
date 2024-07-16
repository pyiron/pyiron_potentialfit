from dataclasses import dataclass, asdict
from enum import Enum
from io import StringIO
from pprint import pprint
from logging import getLogger, INFO

getLogger().setLevel(INFO)
logger = getLogger("structures")

from pyiron_base import Project
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics import ase_to_pyiron
from tqdm.auto import tqdm


import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from ..assyst.util import fast_forward
from ..assyst.vasp import Kpoints, Kspacing, VaspConfig

from ..assyst.structures.workflow import (
        export_structures,
        TrainingDataConfig, State,
        create_structure_set
)



def run(pr: Project, config: TrainingDataConfig, wait_time=60):
    """
    Create a new structure set.

    Simply calls :func:`.create_structure_set` in a loop until finished.

    Args:
        pr (Project): project to run the calculations in
        config (TrainingDataConfig): parameters for structure set
        wait_time (int): how many seconds to sleep between calling :func:`.create_structure_set`.
    """
    state = State.SPG
    while (state := create_structure_set(pr, state, conf, fast_forward=True)) != State.FINISHED:
        time.sleep(wait_time)

epilog = """
Follows the systematic approach reported in this paper[1].  Execution is
organized in four steps:

    1. Generation of random structures with pyxtal. (5min to 1h, depending on #elements)
    2. Volume minimization at low convergence. (<30s walltime per structure, total depends on cluster load)
    3. Full minimization at low convergence. (<2min walltime per structure, total depends on cluster load)
    4. Application of random strains, shears and rattling of atoms (at most a few min)

Each step is built from the structures obtained in the previous one.
Minimization is done at default plane wave cutoff and KSPACING=0.5.

We lay out the project like this,

`root/structures`:                      work exlusively in this project
`root/structures/containers`:           will ultimately contain the structures ready for high convergence calculations
`root/structures/{name}VolMin`:         working project for the volume minimization
`root/structures/{name}VolMinAllmin`:   working project for the full minimization

where `root` is the programs working directory and `name` what you passed as --name.
The container project will contain a `StructureStorage` job for each of the stages.  The minimization stages write two
containers, one for the final structures and one that includes intermediate steps of the minimization (with postifx
`Trace`).  The former is used as input for the next stage, the latter is intended to be used for high convergence
calculations with potfit.calculations.

The program is designed to be run multiple times, each time checking whether it can advance to the next stage.  To have to
program loop until it reaches the final state, use `--fast-forward`.  After the first invocation the program caches the
input and you may run it without additional flags (but it doesn't hurt to do it either).  DFT calculations that abort or
fail to converge during the minimization steps are automatically restarted.  The program will log to stdout its current
stage and what it does inside each.

For example you can create a simple data set for Al with

> python -m potfit.structures --elements Al --stoichiometry 4 8 --name Al4and8 --fast-forward

For alloys usage is similar, but you can also specify --max-atoms to limit the total size of your system, e.g.

> python -m potfit.structures --elements Fe Al --stoichiometry 2 4 8 --max-atoms 10

will include Fe2Al8 and Fe8Al2, but not Fe4Al8.

You should adjust `--min-dist` for each system under study.  Structures that are strongly repulsive can strongly affect
fitting results for a whole structure set.

[1]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.104103
"""


def main():
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="'KSPACING' found in INCAR, no KPOINTS file written",
    )

    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    parser = ArgumentParser(
        description="Create structures for training data.",
        epilog=epilog,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="Crystals",
        help="Base name of the structure containers",
    )
    parser.add_argument(
        "-e",
        "--elements",
        type=str,
        nargs="+",
        help="Chemical element to create structures for",
    )
    parser.add_argument(
        "-m",
        "--max-atoms",
        type=int,
        default=TrainingDataConfig.max_atoms,
        help="Maximum number of atoms per structure",
    )
    parser.add_argument(
        "-o",
        "--stoichiometry",
        type=int,
        nargs="*",
        help="number of atoms of each element to combine",
    )
    parser.add_argument(
        "-t",
        "--trace",
        type=int,
        default=TrainingDataConfig.trace,
        help="Number of structures to include from each minimization step",
    )
    parser.add_argument(
        "-r",
        "--rattle",
        type=float,
        nargs=2,
        default=(TrainingDataConfig.rattle_disp, TrainingDataConfig.rattle_strain),
        help="The displacement (in A) and tri-axial strain for the Rattle set",
    )
    parser.add_argument(
        "--rattle-repetitions",
        type=int,
        default=TrainingDataConfig.rattle_repetitions,
        help="How many rattled structures to generate from each minimized structure",
    )
    parser.add_argument(
        "-s",
        "--stretch",
        type=float,
        nargs=2,
        default=(TrainingDataConfig.stretch_hydro, TrainingDataConfig.stretch_shear),
        help="The tri-axial and shear strains for the Stretch set",
    )
    parser.add_argument(
        "--stretch-repetitions",
        type=int,
        default=TrainingDataConfig.stretch_repetitions,
        help="How many stretched structures to generate from each minimized structure",
    )
    parser.add_argument(
        "-d",
        "--min-dist",
        type=float,
        default=None,
        help="Smallest nearest neighbor distance to allow; default is taken from RCORE value of Vasp PBE pseudopotential files",
    )
    parser.add_argument(
        "-p", "--project", default="structures", help="project to work in"
    )
    parser.add_argument(
        "--fast-forward",
        type=int,
        default=None,
        help="Automatically go to the next step after sleeping for this many seconds",
    )
    parser.add_argument(
        "--delete-existing-job",
        action="store_true",
        help="Retry the current step from scratch",
    )
    parser.add_argument(
        "--export", type=str,
        help="Optionally specify a directory where to dump POSCAR files with the generated structures after everything "
             "is finished."
    )
    parser.add_argument(
        "--magmom", nargs=2, action="append", default=[],
        help="Initial magnetic moments as `element symbol`, followed by the collinear magnetic moment; all atoms of "
             "the same element will be initialized the same; may be given multiple times, once per element"
    )
    parser.add_argument(
        "--potcar", nargs=2, action="append", default=[], type=dict,
        help="use these POTCARs instead of the default ones; must be given as "
             "TYPE PATH pairs, but may be given multiple times, one for each "
             "element."
    )

    parser.add_argument(
        "--cores", type=int, default=10,
        help="Number of cores to use per job during minimizations"
    )

    args = parser.parse_args()

    pr = Project(args.project)
    conf = vars(args).copy()
    conf["rattle_disp"], conf["rattle_strain"] = conf["rattle"]
    conf["stretch_hydro"], conf["stretch_shear"] = conf["stretch"]
    del conf["rattle"]
    del conf["stretch"]
    del conf["project"]
    del conf["fast_forward"]
    del conf["magmom"]
    del conf["cores"]
    conf = {k: v for k, v in conf.items() if v is not None}
    conf = TrainingDataConfig(**conf)

    for (el, mm) in args.magmom:
        mm = float(mm)
        if el not in args.elements:
            raise ValueError(f"Elements for magmoms, must be also given via -e, not {el}!")
        conf.vasp.magmoms[el] = mm
    if args.potcar is not {}:
        conf.vasp.potcars = args.potcar

    conf.server.cores = args.cores

    state = pr.data.get("state", "spg")
    # old_conf = pr.data.get('config', {})
    # for k, v in old_conf.items():
    #     setattr(conf, k, v)
    pr.data.config = asdict(conf)
    pr.data.write()

    state = create_structure_set(pr, state, conf, args.fast_forward is not None).value
    pr.data.state = state
    pr.data.write()
    if state != "finished" and args.fast_forward is not None:
        fast_forward(args.fast_forward, __spec__)
    if state == "finished" and args.export is not None:
        export_structures(pr, args.export)


if __name__ == "__main__":
    main()
