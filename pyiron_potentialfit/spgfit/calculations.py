from argparse import ArgumentParser, RawDescriptionHelpFormatter, FileType
from dataclasses import asdict
import logging

from pyiron_base import Project

from ..assyst.projectflow import RunAgain
from ..assyst.calculations.workflow import CalculationConfig, run_container, combine
from ..assyst.util import fast_forward, ServerConfig
from ..assyst.vasp import VaspConfig

epilog = """
Run high throughput, high convergence DFT on a set of structures.

After you have created structures with potfit.structures you can use this
module to run DFT on them. This program assumes that you run it from the same
directory as potfit.structures, or use -p,-s accordingly.  This program is also
designed to be called multiple times and will restart broken or not converged
calculations.  This may take a few iterations.  After all calculations are
finished or cannot be further repaired, the results will be aggregated in
`TrainingContainers`.  One for each --container that you pass and a --final one
that collects every structure.

We lay out the project like this,

`root/structures/containers`:         where we read the --containers from
`root/training`:                      work exlusively in this project
`root/training/containers`:           will ultimately contain the final training data
`root/training/{container_names}`:    working projects for each of the passed containers

where `root` is the programs working directory and `container_names` what you passed as --containers.

The default k point and encut values are converged for the Mg/Al/Ca system.  Do your own testing, check
potfit.convergence how to go about it.
"""


def main():
    parser = ArgumentParser(
        description="Creating training data for an MLIP",
        epilog=epilog,
        formatter_class=RawDescriptionHelpFormatter,
    )

    dft_group = parser.add_argument_group("DFT")
    dft_group.add_argument(
        "-e", "--encut", type=float, default=550, help="Plane wave energy cut off"
    )
    kgroup = dft_group.add_mutually_exclusive_group()
    kgroup.add_argument(
        "-k",
        "--kpoints",
        type=int,
        dest="kpoints",
        help="k point sampling (k mesh spacing if given as float",
    )
    kgroup.add_argument(
        "-m",
        "--kspacing",
        type=float,
        default=0.1,
        dest="kpoints",
        help="k point sampling (k mesh spacing if given as float",
    )
    dft_group.add_argument(
        "--incar", type=FileType("r"), default=None, help="INCAR to apply on top"
    )

    parser.add_argument(
        "-p", "--project", default="training", help="project to work in"
    )
    parser.add_argument(
        "-s",
        "--structures-project",
        default="structures",
        help="project that contains the structure containers",
    )
    parser.add_argument(
        "-c",
        "--containers",
        type=str,
        nargs="+",
        default=(
            "Crystals",
            "CrystalsVolMinTrace",
            "CrystalsAllMinTrace",
            "CrystalsVolMinAllMinRandom",
        ),
        help="names of structure containers to use",
    )
    parser.add_argument(
        "-f",
        "--final",
        default="Everything",
        help="Name of the container that collects all results",
    )
    parser.add_argument(
        "-d",
        "--min-dist",
        type=float,
        default=None,
        help="Smallest nearest neighbor distance to allow",
    )

    server_group = parser.add_argument_group("Server")
    server_group.add_argument(
        "--queue", default=ServerConfig.queue, help="Pyiron queue to submit jobs to"
    )
    server_group.add_argument(
        "--cores",
        type=int,
        default=ServerConfig.cores,
        help="Number of cores for each DFT run",
    )
    server_group.add_argument(
        "-r",
        "--run-time",
        type=float,
        default=ServerConfig.run_time,
        help="Run time limit for each DFT run in seconds",
    )

    workflow_group = parser.add_argument_group("Workflow")
    workflow_group.add_argument(
        "--delete-existing-job",
        action="store_true",
        help="delete existing calculations",
    )
    workflow_group.add_argument(
        "--broken-threshold",
        type=float,
        default=0.05,
        help="Maximum amount of aborted/not_converged/warning jobs to accept; "
        "if above try to repair/converge jobs",
    )
    workflow_group.add_argument(
        "--fast-forward",
        type=int,
        default=None,
        help="Automatically go to the next step after sleeping for this many seconds",
    )

    args = parser.parse_args()

    incar = {}
    if args.incar is not None:
        incar = {
            k.strip(): v.strip() for k, v in map(lambda x: x.split("="), args.incar)
        }

    conf = CalculationConfig(
        vasp=VaspConfig(
            encut=args.encut,
            kmesh=args.kpoints,
            incar=incar,
        ),
        server=ServerConfig(
            cores=args.cores,
            run_time=args.run_time,
        ),
        workflow=WorkflowProjectConfig(
            delete_existing_job=args.delete_existing_job,
            broken_threshold=args.broken_threshold,
            finished_threshold=0.9,
        ),
        min_dist=args.min_dist,
    )

    pr = Project(args.project)
    pr.data.config = asdict(conf)
    pr.data.write()
    structures_pr = Project(args.structures_project)

    logger = logging.getLogger("assyst.calculations")

    done = True
    for cname in args.containers:
        logger.info("Running %s", cname)
        cont = structures_pr["containers"].load(cname)
        try:
            run_container(pr, cont, conf)
        except RunAgain:
            done = False

    if done:
        logger.info("collecting all calculations in %s", args.final)
        every = combine(
            pr["containers"],
            map(pr["containers"].load, args.containers),
            name=args.final,
            min_dist=args.min_dist,
            delete_existing_job=args.delete_existing_job,
        )
    elif args.fast_forward is not None:
        fast_forward(args.fast_forward, __spec__)


if __name__ == "__main__":
    main()
