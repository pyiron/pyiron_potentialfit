import os
import os.path
from functools import wraps
from inspect import signature

from pyiron_base import Project
from pyiron_potentialfit.atomistics.job.trainingcontainer import TrainingContainer
from typing import Iterable, Union

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..assyst.util import get_table, read_generic_parameters, get_potential_properties


def broadcast(*names):
    """
    Broadcast a function over arguments given as iterables.

    If multiple arguments are given, the function is broadcasted over the outer
    product of all (iterable) variables.

    >>> @broadcast('iter1', 'iter2')
    ... def my_call(iter1, iter2, scalar):
    ...     print(iter1, iter2, scalar)
    >>> my_call('a', 'b', 3)
    'a' 'b' 3
    >>> my_call(['a1', 'a2'], 'b', 3)
    'a1' 'b' 3
    'a2' 'b' 3
    >>> my_call(['a1', 'a2'], ['b1', 'b2'], 3)
    'a1' 'b1' 3
    'a2' 'b1' 3
    'a1' 'b2' 3
    'a2' 'b2' 3
    >>> my_call('a', 'b', [3, 4])
    'a' 'b' [3, 4]

    Args:
        *names (str): names of function arguments to wrap
    """

    def wrapper(func):
        @wraps(func)
        def f(*args, **kwargs):
            boundargs = signature(func).bind(*args, **kwargs)
            for name, value in boundargs.arguments.items():
                if name not in names:
                    continue
                if isinstance(value, Iterable) and not isinstance(value, str):
                    for v in value:
                        boundargs.arguments[name] = v
                        ret = f(*boundargs.args, **boundargs.kwargs)
                    return ret
            # unwrap all Fixed arguments first
            args = (a if not isinstance(a, Fixed) else a.value for a in boundargs.args)
            kwargs = {
                k: v if not isinstance(v, Fixed) else v.value
                for k, v in boundargs.kwargs.items()
            }
            # fall through all scalar call
            func(*args, **kwargs)

        f.Fixed = Fixed
        return f

    return wrapper


class Fixed:
    def __init__(self, v):
        self.v = v

    @property
    def value(self):
        return self.v


broadcast.Fixed = Fixed


def _guess_iterations(level):
    """
    Guess the number of BGFS iterations mlip does internally.

    Numbers chosen by fair dice roll,... err, analysis of some 1k mlip fit done
    on the cmti cluster of MPI SusMat.  Neither the cut off nor the number of
    stuctures/atoms seems to impact this, so fit a line and rounded up.
    """
    return 50 + 200 * level


# The time mlip takes to take a single BFGS optimization step scales like
#       pre * atoms * rmax * e^(exp * level)
# where atoms is the number of atoms in the training set,
# rmax is the outer cutoff, and level is the potential level.
# for the cmti cluster of MPI SusMat I estimated the parameters from jobs as
# below
MTP_RUNTIME_PARAMETERS = {
    "cmti": {"pre": 1.5e-5, "exp": 1 / 4},
}


def _guess_runtime(queue, atoms, rmax, level):
    """
    Estimate the amount of seconds an mlip job will take on known hardware
    """
    iterations = _guess_iterations(level)
    if queue not in MTP_RUNTIME_PARAMETERS:
        print(
            f"WARNING: no performance data for queue {queue} available. "
            "Will use data from cmti cluster and double estimate."
        )
        # just use the iterations as a shortcut to increase total estimate
        iterations *= 2
        queue = "cmti"
    pre = MTP_RUNTIME_PARAMETERS[queue]["pre"]
    exp = MTP_RUNTIME_PARAMETERS[queue]["exp"]
    return pre * iterations * atoms * rmax * np.exp(exp * level)


@broadcast("rmin", "rmax", "level")
def fit(
    fit_pr: Project,
    training_containers: Union[TrainingContainer, Iterable[TrainingContainer]],
    rmin: Union[float, Iterable[float]],
    rmax: Union[float, Iterable[float]],
    level: Union[int, Iterable[int]],
    iterations: int = None,
    energy_weight: float = None,
    force_weight: float = None,
    stress_weight: float = None,
    refit: bool = False,
    delete_existing_job=False,
    queue: str = "cmti",
    cores: int = 40,
) -> Project:
    """
    Fit a potential to the given structures.

    If `rmin`, `rmax` or `level` are iterables the function is broadcasted
    over them. `training_containers` is *not* broadcasted, instead all given
    training containers are combined.

    Args:
        fit_pr (Project): project that contains the fitting jobs
        training_containers (TrainingContainer, Iterable[TrainingContainer]): container that keeps all the structures
        rmin (float, Iterable[float]): lower cut off of the potential
        rmax (float, Iterable[float]): upper cut off of the potential
        level (int, Iterable[int]): level of the potential
        energy_weight (float): relative weight of energy error in cost function
        force_weight (float): relative weight of force error in cost function
        stress_weight (float): relative weight of stress error in cost function
        refit (bool): if True and the fit to be created already exists, start a
                      refit
        delete_existing_job (float): remove old job before creating new one
        queue (str): name of pyiron queue to submit jobs to
        cores (int): how many cores to use for each fit (beware: modified in a
                     non-trivial way depending on potential level and number of training
                     structures)
    """
    if isinstance(training_containers, TrainingContainer):
        training_containers = (training_containers,)
    else:
        training_containers = tuple(training_containers)
    train_name = "_".join(train.name for train in training_containers)
    train_number_of_structures = sum(
        train.number_of_structures for train in training_containers
    )
    pr = fit_pr.create_group(train_name)

    name = [f"MTP{level:02}", round(rmin, 2), round(rmax, 2)]
    if iterations is not None:
        name += ["I", iterations]
    if energy_weight is not None:
        name += ["E", energy_weight]
    if force_weight is not None:
        name += ["F", force_weight]
    if stress_weight is not None:
        name += ["S", stress_weight]

    # supply default after we make the job name, so that user supplied values
    # always force a unique job name and job tables are not littered by random
    # numbers
    if iterations is None:
        iterations = _guess_iterations(level)

    j = pr.create.job.Mlip(
        name, delete_existing_job=delete_existing_job, delete_aborted_job=True
    )
    j["user/level"] = level
    j["user/rmax"] = rmax
    j["user/rmin"] = rmin
    if iterations is not None:
        j["user/iterations"] = iterations
    if energy_weight is not None:
        j["user/energy_weight"] = energy_weight
    if force_weight is not None:
        j["user/force_weight"] = force_weight
    if stress_weight is not None:
        j["user/stress_weight"] = stress_weight

    runtime_guess = _guess_runtime(
        queue,
        sum(tc._container.num_elements for tc in training_containers),
        rmax,
        level,
    )
    if j.status.finished and refit and not j.name.endswith("_restart"):
        j = j.restart()
        j["user/refit"] = True
        j.server.queue = queue
        j.server.cores = cores
        if 16 < level < 24:
            j.server.cores *= 2
        else:
            j.server.cores *= 4
        j.server.cores *= int(np.ceil(train_number_of_structures / 20_000))
        j.server.cores = min(j.server.cores, 160)
        j.server.run_time = runtime_guess / j.server.cores
        j.run()
        return pr
    if not j.status.initialized:
        return pr

    j.input["potential"] = level
    j.input["min_dist"] = rmin
    j.input["max_dist"] = rmax
    for train in training_containers:
        j.add_training_data(train)
    if iterations is not None:
        j.input["iteration"] = iterations
    if energy_weight is not None:
        j.input["energy-weight"] = energy_weight
    if force_weight is not None:
        j.input["force-weight"] = force_weight
    if stress_weight is not None:
        j.input["stress-weight"] = stress_weight
    j.server.queue = queue
    j.server.queue = queue
    j.server.cores = cores
    if 16 < level < 24:
        j.server.cores *= 2
    else:
        j.server.cores *= 4
    j.server.cores *= int(np.ceil(train_number_of_structures / 20_000))
    j.server.cores = min(j.server.cores, 160)
    j.server.run_time = runtime_guess / j.server.cores
    j.run()

    return pr


from typing import Union, Iterable, Literal


def _guess_iterations_ace(*args):
    return 1000


def _guess_runtime_ace(queue, number_of_atoms, rmax, number_of_functions_per_element):
    return 24 * 60 * 60


@broadcast(
    "rmax", "number_of_functions_per_element", "iterations", "embedding", "ladder", "weighting"
)
def fit_ace(
    fit_pr: Project,
    training_containers: Union[TrainingContainer, Iterable[TrainingContainer]],
    rmax: Union[float, Iterable[float]],
    number_of_functions_per_element: Union[int, Iterable[int]],
    rmin: float | None = None,
    iterations: int | Iterable[int] = None,
    embedding: Iterable[Literal["linear", "sqrt"] | tuple[float]] = "sqrt",
    ladder: bool | tuple[int, float] = False,
    kappa=0.5,
    weighting: Literal["convex_hull"] | None = None,
    radial_smoothness: tuple[float, float, float] = None,
    delete_existing_job=False,
    queue: str = "s_cmmg",
    cores: int = 256,
    runtime: int = 24 * 60 * 60,
    seed: int | None = None,
) -> Project:
    if isinstance(training_containers, TrainingContainer):
        training_containers = (training_containers,)
    else:
        training_containers = tuple(training_containers)
    train_name = "_".join(train.name for train in training_containers)
    pr = fit_pr.create_group(train_name)

    name = [f"ACE{int(number_of_functions_per_element):04}", round(rmax, 2)]
    if rmin is not None:
        name.insert(1, round(rmin, 2))
    if isinstance(embedding, str):
        name += ["E", embedding]
    else:
        name += ["E", *map(str, embedding)]
    if weighting:
        # name += ["W", weighting]
        name += ["W", {"convex_hull": "ch"}.get(weighting, weighting)]
    if ladder:
        name += ["L"]
        if ladder is not True:
            name += list(ladder)
        else:
            ladder = (10, 0.02)
    if iterations is not None:
        name += ["I", iterations]
    if seed is not None:
        name += ["S", seed]

    # supply default after we make the job name, so that user supplied values always force a unique job name and
    # job tables are not littered by random numbers
    if iterations is None:
        iterations = _guess_iterations_ace(number_of_functions_per_element)

    j = pr.create.job.PacemakerJob(
        name, delete_existing_job=delete_existing_job, delete_aborted_job=True
    )
    if not j.status.initialized:
        return pr

    j["user/number_of_functions_per_element"] = number_of_functions_per_element
    j["user/rmax"] = rmax
    j["user/kappa"] = kappa
    j["user/iterations"] = iterations
    if ladder:
        j["user/ladder"] = ladder

    for c in training_containers:
        j.add_training_data(c)

    if seed is None:
        seed = int.from_bytes(os.urandom(4))
    j.input["data"]["seed"] = seed
    j.input["fit"]["maxiter"] = iterations
    j.input["fit"]["loss"]["kappa"] = kappa
    if radial_smoothness is not None:
        if len(radial_smoothness) != 3:
            raise ValueError("radial_smoothness must be a triple!")
        j.input["fit"]["loss"]["w0_rad"] = radial_smoothness[0]
        j.input["fit"]["loss"]["w1_rad"] = radial_smoothness[1]
        j.input["fit"]["loss"]["w2_rad"] = radial_smoothness[2]
    if ladder:
        j.input["fit"]["ladder_step"] = list(ladder)
    if weighting:
        j.input["fit"]["weighting"] = {
            "type": "EnergyBasedWeightingPolicy",
            "energy": weighting,
        }
    j.input["cutoff"] = rmax
    j.input["potential"]["functions"][
        "number_of_functions_per_element"
    ] = number_of_functions_per_element
    j.input["potential"]["functions"]["UNARY"] = {
        "nradmax_by_orders": [15, 6, 4, 3, 2, 2],
        "lmax_by_orders": [0, 3, 3, 2, 2, 1],
    }
    j.input["potential"]["functions"]["BINARY"] = {
        "nradmax_by_orders": [15, 6, 3, 2, 2, 1],
        "lmax_by_orders": [0, 3, 2, 1, 1, 0],
    }
    j.input["potential"]["functions"]["TERNARY"] = {
        "nradmax_by_orders": [
            15,
            6,
            2,
            2,
            1,
        ],
        "lmax_by_orders": [
            0,
            3,
            1,
            1,
            0,
        ],
    }
    match embedding:
        case "linear":
            j.input["potential"]["embeddings"]["ALL"]["fs_parameters"] = [1, 1]
        case "sqrt":
            j.input["potential"]["embeddings"]["ALL"]["fs_parameters"] = [1, 1, 1, 0.5]
        case _:
            j.input["potential"]["embeddings"]["ALL"]["fs_parameters"] = list(embedding)
    j.input["potential"]["embeddings"]["ALL"]["ndensity"] = int(
        len(j.input["potential"]["embeddings"]["ALL"]["fs_parameters"]) // 2
    )
    j.input["potential"]["bonds"]["ALL"]["rcut"] = rmax
    if rmin is not None:
        j.input["potential"]["bonds"]["ALL"]["r_in"] = rmin
        # this is the delta_in applied with repulsion=auto, let's just use that
        # here as well
        j.input["potential"]["bonds"]["ALL"]["delta_in"] = 0.1
    else:
        j.input["fit"]["repulsion"] = "auto"
    j.input["potential"]["bonds"]["ALL"]["inner_cutoff_type"] = "zbl"
    j.input["backend"]["batch_size"] = 10_000

    if runtime is None:
        runtime = (
            _guess_runtime_ace(
                queue,
                sum(tc._container.num_elements for tc in training_containers),
                rmax,
                number_of_functions_per_element,
            )
            / j.server.cores
        )
        runtime = 24 * 60 * 60  # FIXME:workout scaling

    j.server.queue = queue
    j.server.cores = cores
    j.server.run_time = runtime
    j.run()

    return pr


def plot(fit_job, legend=True):
    train = fit_job["input/training_data"].to_object()
    efs = fit_job["output/training_efs"].to_object()
    N = train.get_array("length")

    plt.subplot(241, title="Energy per Atom", aspect=1)
    plt.scatter(
        efs.get_array("energy").ravel() / N,
        train.get_array("energy").ravel() / N,
        marker=".",
        label=fit_job.name,
    )
    plt.xlabel("True [eV]")
    plt.ylabel("Predicted [eV]")
    if legend:
        plt.legend()
    plt.subplot(245)
    ediff = efs.get_array("energy").ravel() / N - train.get_array("energy").ravel() / N
    plt.hist(ediff, bins=50, log=True, label=f"{ediff.std():.05}")
    plt.xlabel(r"$E$ [eV]")
    if legend:
        plt.legend()

    train_forces = train.get_array("forces")

    plt.subplot(242, title="Force", aspect=1)
    plt.scatter(
        efs.get_array("forces").ravel(),
        train_forces.ravel(),
        marker=".",
        label=fit_job.name,
    )
    plt.xlabel("True [eV/A]")
    plt.ylabel("Predicted [eV/A]")
    if legend:
        plt.legend()
    plt.subplot(246)
    fdiff = efs.get_array("forces").ravel() - train_forces.ravel()
    plt.hist(fdiff, bins=50, log=True, label=f"{fdiff.std():.05}")
    plt.xlabel(r"$F$ [eV/$\AA$]")
    if legend:
        plt.legend()

    train_stress = train.get_array("stress")

    train_hydrostatic = train_stress[:, :3].ravel()
    train_shear = train_stress[:, 3:].ravel()
    pred_hydrostatic = efs.get_array("stress")[:, :3].ravel()
    pred_shear = efs.get_array("stress")[:, 3:].ravel()

    plt.subplot(243, title="Hydrostatic Pressures", aspect=1)
    plt.scatter(train_hydrostatic, pred_hydrostatic, marker=".", label=fit_job.name)
    plt.xlabel("True [eV]")
    plt.ylabel("Predicted [eV]")
    if legend:
        plt.legend()
    plt.subplot(247)
    sdiff = pred_hydrostatic - train_hydrostatic
    plt.hist(sdiff, bins=50, log=True, label=f"{sdiff.std():.05}")
    plt.xlabel("$\Delta \sigma V$ [eV]")
    if legend:
        plt.legend()

    plt.subplot(244, title="Shear Pressures", aspect=1)
    plt.scatter(train_shear, pred_shear, marker=".", label=fit_job.name)
    plt.xlabel("True [eV]")
    plt.ylabel("Predicted [eV]")
    if legend:
        plt.legend()
    plt.subplot(248)
    sdiff = pred_shear - train_shear
    plt.hist(sdiff, bins=50, log=True, label=f"{sdiff.std():.05}")
    plt.xlabel("$\Delta \sigma V$ [eV]")
    if legend:
        plt.legend()


def _get_training_data(j):
    inpt = j["input/training_data"]
    if inpt is None:
        # Pacemaker layout
        inpt = j["output/training_data"]
    return inpt.to_object()


def _get_predicted_data(j):
    pred = j["output/training_efs"]
    if pred is None:
        # Pacemaker layout
        pred = j["output/predicted_data"]
    return pred.to_object()


def energy_rmse(j):
    inpt = _get_training_data(j)
    N = inpt.get_array("length")
    train = np.squeeze(inpt.get_array("energy")) / N
    pred = np.squeeze(_get_predicted_data(j).get_array("energy")) / N
    return np.sqrt(np.mean((train - pred) ** 2))


def energy_mae(j):
    inpt = _get_training_data(j)
    N = inpt.get_array("length")
    train = np.squeeze(inpt.get_array("energy")) / N
    pred = np.squeeze(_get_predicted_data(j).get_array("energy")) / N
    return np.abs(train - pred).mean()


def energy_max(j):
    inpt = _get_training_data(j)
    N = inpt.get_array("length")
    train = np.squeeze(inpt.get_array("energy")) / N
    pred = np.squeeze(_get_predicted_data(j).get_array("energy")) / N
    return np.abs(train - pred).max()


def force_rmse(j):
    inpt = _get_training_data(j)
    train = np.squeeze(inpt.get_array("forces"))
    pred = np.squeeze(_get_predicted_data(j).get_array("forces"))
    return np.sqrt(np.mean(np.linalg.norm(train - pred, axis=-1) ** 2))


def force_mae(j):
    inpt = _get_training_data(j)
    N = inpt.get_array("length")
    train = np.squeeze(inpt.get_array("forces"))
    pred = np.squeeze(_get_predicted_data(j).get_array("forces"))
    return np.mean(np.linalg.norm(train - pred, axis=-1))


def force_max(j):
    inpt = _get_training_data(j)
    N = inpt.get_array("length")
    train = np.squeeze(inpt.get_array("forces"))
    pred = np.squeeze(_get_predicted_data(j).get_array("forces"))
    return np.abs(train - pred).max()


def stress_hydro_rmse(j):
    if j.__name__ != "Mlip":
        return pd.NA
    train = np.squeeze(_get_training_data(j).get_array("stress"))
    pred = np.squeeze(_get_predicted_data(j).get_array("stress"))
    return np.sqrt(np.mean((train[:, :3] - pred[:, :3]) ** 2))


def stress_hydro_mae(j):
    if j.__name__ != "Mlip":
        return None
    train = np.squeeze(_get_training_data(j).get_array("stress"))
    pred = np.squeeze(_get_predicted_data(j).get_array("stress"))
    return np.abs(train[:, :3] - pred[:, :3]).mean()


def stress_hydro_max(j):
    if j.__name__ != "Mlip":
        return pd.NA
    train = np.squeeze(_get_training_data(j).get_array("stress"))
    pred = np.squeeze(_get_predicted_data(j).get_array("stress"))
    return np.abs(train[:, :3] - pred[:, :3]).max()


def stress_shear_rmse(j):
    if j.__name__ != "Mlip":
        return pd.NA
    train = np.squeeze(_get_training_data(j).get_array("stress"))
    pred = np.squeeze(_get_predicted_data(j).get_array("stress"))
    return np.sqrt(np.mean((train[:, 3:] - pred[:, 3:]) ** 2))


def stress_shear_mae(j):
    if j.__name__ != "Mlip":
        return pd.NA
    train = np.squeeze(_get_training_data(j).get_array("stress"))
    pred = np.squeeze(_get_predicted_data(j).get_array("stress"))
    return np.abs(train[:, 3:] - pred[:, 3:]).mean()


def stress_shear_max(j):
    if j.__name__ != "Mlip":
        return pd.NA
    train = np.squeeze(_get_training_data(j).get_array("stress"))
    pred = np.squeeze(_get_predicted_data(j).get_array("stress"))
    return np.abs(train[:, 3:] - pred[:, 3:]).max()


def energy_spread(j):
    train = _get_training_data(j)
    return np.ptp(train.get_array("energy").ravel() / train.get_array("length"))


def analyze(fit_pr: Project, delete_existing_job=False):
    """
    Collect error metrics on a project of fits.

    Return MAE and RMSE of (per atom) energies, (magnitude) forces and (axial and shear) stresses for each fitted
    potential together with the energy spread observed in the training data.

    The returned table is in "melted" or long form, i.e. each potential will have multiple rows associated with it, one
    for each quantity and error metric.  This is for ease of plotting in :func:`.plot_error_vs_level()` and
    :func:`.plot_error_vs_rmax()`.  You can obtain an easier to read representation for a single potential by pivoting
    it back, like so

    >>> df = analyze(...)
    >>> df.query('job_id==...').pivot(index='quantity', columns='metric', values='error')

    where you replace the dots in the last line by the numerical job id of the fit or even dropping the query

    >>> df.pivot(index=['job_id', 'quantity'], columns='metric', values='error')

    .. warning:: Beware!

        Because the stress errors are separated into diagonal and off-diagonal contributions, they are defined on a
        component level, which means they are **not** rotationally invariant!  Because our data generally do not have a
        preferred orientation this effect may or may not average out, but the technically correct thing to do would be
        to calculate the Frobenius norm of the stress tensor difference and average that.  For now this remains to do.

    Args:
        fit_pr (Project): project with Mlip jobs inside
        delete_existing_job (bool): if True recreate pyiron table from scratch

    Returns:
        pandas.DataFrame: return data frame with error metrics
    """

    def add(tab):
        tab.analysis_project = fit_pr
        tab.db_filter_function = lambda df: df.hamilton.isin(["Mlip", "Pacemaker2022"])
        tab.add._user_function_dict["potprops"] = get_potential_properties
        tab.add["energy_spread"] = energy_spread
        tab.add["energy_rmse"] = energy_rmse
        tab.add["energy_mae"] = energy_mae
        tab.add["energy_max"] = energy_max
        tab.add["force_rmse"] = force_rmse
        tab.add["force_mae"] = force_mae
        tab.add["force_max"] = force_max
        tab.add["stress_hydro_rmse"] = stress_hydro_rmse
        tab.add["stress_hydro_mae"] = stress_hydro_mae
        tab.add["stress_hydro_max"] = stress_hydro_max
        tab.add["stress_shear_rmse"] = stress_shear_rmse
        tab.add["stress_shear_mae"] = stress_shear_mae
        tab.add["stress_shear_max"] = stress_shear_max

    tab = get_table(
        fit_pr, "level_error_table", add, delete_existing_job=delete_existing_job
    )
    df = tab.get_dataframe()
    if len(df) == 0:
        return df
    # get_potential_properties adds the job id as well, so drop one
    df.drop("id", axis="columns", inplace=True)

    # Melt table to have easy seaborn plotting later
    errors = [
        "energy_spread",
        "energy_rmse",
        "energy_mae",
        "energy_max",
        "force_rmse",
        "force_mae",
        "force_max",
        "stress_hydro_rmse",
        "stress_hydro_mae",
        "stress_hydro_max",
        "stress_shear_rmse",
        "stress_shear_mae",
        "stress_shear_max",
    ]
    errors = set(errors).intersection(df.columns)
    cols = set(df.columns).difference(errors)
    df = df.melt(id_vars=cols, value_vars=errors, value_name="error")
    vv = df.variable.str.rsplit("_", expand=True, n=1)
    vv.columns = ["quantity", "metric"]
    df = pd.concat([df.drop("variable", axis="columns"), vv], axis="columns")
    df.metric = df.metric.str.upper()
    # stress errors not defined on ACE, drop entries
    df = df.query("model != 'ACE' or not quantity.str.startswith('stress')")

    return df


def plot_error_vs_level(df, logy=True, **kwargs):
    return sns.relplot(
        data=df.query("metric!='SPREAD'"),
        kind="line",
        markers=True,
        x="level",
        y="error",
        col="quantity",
        row="metric",
        hue="rmax",
        style="training",
        facet_kws={"sharey": "col"},
        **kwargs,
    ).set(yscale="log" if logy else "linear")


def plot_error_vs_rmax(df, logy=True, **kwargs):
    return sns.relplot(
        data=df.query("metric!='SPREAD'"),
        kind="line",
        markers=True,
        x="rmax",
        y="error",
        col="quantity",
        row="metric",
        hue="level",
        style="training",
        facet_kws={"sharey": "col"},
        **kwargs,
    ).set(yscale="log" if logy else "linear")


epilog = """
We lay out the project like this,

`root/training/containers`:     where we read the --containers from
`root/fits`:                    work exclusively in this project
`root/fits/{container_names}`:  working projects for each of the passed containers

where `root` is the programs working directory and `container_names` what you passed as --containers.  For each given
container a potential will be fit for all the combinations of --level, --rmin and --rmax in the respective project
folder.

If you use --refit and the program finds an existing potential for the given parameters and training set, it will
restart the fit from the first solution.  This sometimes leads to better training accuracy.  But sometimes also not.  Do
your own testing.
"""


def main():
    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    parser = ArgumentParser(
        description="Fit MTPs to training data.",
        epilog=epilog,
        formatter_class=RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-l", "--level", type=int, nargs="+", help="The level of the potentials"
    )
    parser.add_argument(
        "-a", "--rmax", type=float, nargs="+", help="Outer cutoff for the potentials"
    )
    parser.add_argument(
        "-i",
        "--rmin",
        type=float,
        nargs="+",
        help="Inner cutoff for the data to consider",
    )

    parser.add_argument("-p", "--project", default="fits", help="project to work in")
    parser.add_argument(
        "--training-project",
        default="training",
        help="project that contains the training containers",
    )
    parser.add_argument(
        "-c",
        "--containers",
        type=str,
        nargs="+",
        default=("Everything",),
        help="names of training containers to use",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Fit again to same data, if potential with given parameters already exists",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Maximum number of minimizer iterations during fitting",
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
    )
    parser.add_argument(
        "--force-weight",
        type=float,
    )
    parser.add_argument(
        "--stress-weight",
        type=float,
    )
    parser.add_argument(
        "--delete-existing-job",
        action="store_true",
    )

    args = parser.parse_args()

    fit_pr = Project(args.project)
    train_pr = fit_pr[".."].create_group(args.training_project)
    for contname in args.containers:
        cont = train_pr.load(contname)
        fit(
            fit_pr,
            cont,
            rmin=args.rmin,
            rmax=args.rmax,
            level=args.level,
            iterations=args.iterations,
            refit=args.refit,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            stress_weight=args.stress_weight,
            delete_existing_job=args.delete_existing_job,
        )


if __name__ == "__main__":
    main()
