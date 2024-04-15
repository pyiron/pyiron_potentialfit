import os.path
from functools import wraps
from inspect import signature

from pyiron_base import Project
from pyiron_potentialfit.atomistics.job.trainingcontainer import TrainingContainer
from typing import Iterable, Union

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm

from .util import get_table, read_generic_parameters

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
                if name not in names: continue
                if isinstance(value, Iterable):
                    for v in value:
                        boundargs.arguments[name] = v
                        ret = f(*boundargs.args, **boundargs.kwargs)
                    return ret
            # fall through all scalar call
            func(*boundargs.args, **boundargs.kwargs)
        return f
    return wrapper


@broadcast('min_dist', 'max_dist', 'level')
def fit(fit_pr: Project, train: TrainingContainer,
        min_dist: Union[float, Iterable[float]],
        max_dist: Union[float, Iterable[float]],
        level: Union[int, Iterable[int]],
        iterations: int=5000,
        energy_weight: float=None, force_weight: float=None, stress_weight: float=None,
        refit: bool=True, delete_existing_job=False
) -> Project:
    """
    Fit a potential to the given structures.

    If min_dist, max_dist or level are iterables the function is broadcasted
    over them.

    Args:
        fit_pr (Project): project that contains the fitting jobs
        train (TrainingContainer): container that keeps all the structures
        min_dist (float, Iterable[float]): lower cut off of the potential
        max_dist (float, Iterable[float]): upper cut off of the potential
        level (int, Iterable[int]): level of the potential
        energy_weight (float): relative weight of energy error in cost function
        force_weight (float): relative weight of force error in cost function
        stress_weight (float): relative weight of stress error in cost function
        refit (bool): if True and the fit to be created already exists, start a
                      refit
        delete_existing_job (float): remove old job before creating new one
    """
    pr = fit_pr.create_group(train.name)

    name = [f'MTP{level:02}', round(min_dist, 2), round(max_dist, 2)]
    if iterations is not None:
        name += ['I', iterations]
    if energy_weight is not None:
        name += ['E', energy_weight]
    if force_weight is not None:
        name += ['F', force_weight]
    if stress_weight is not None:
        name += ['S', stress_weight]

    j = pr.create.job.Mlip(
            name,
            delete_existing_job=delete_existing_job,
            delete_aborted_job=True
    )
    j['user/level'] = level
    j['user/rmax'] = max_dist
    j['user/rmin'] = min_dist
    if iterations is not None:
        j['user/iterations'] = iterations
    if energy_weight is not None:
        j['user/energy_weight'] = energy_weight
    if force_weight is not None:
        j['user/force_weight'] = force_weight
    if stress_weight is not None:
        j['user/stress_weight'] = stress_weight
    if j.status.finished and refit and not j.name.endswith("_restart"):
        j = j.restart()
        j['user/restart'] = True
        j.server.queue = 'cmti'
        if level < 16:
            j.server.cores =  40
        if level < 24:
            j.server.cores =  80
        else:
            j.server.cores = 120
        j.server.run_time = 0.5 * level * train.number_of_structures
        j.run()
        return pr
    if not j.status.initialized: return pr

    j.add_job_to_fitting(train.id, 0, train.number_of_structures - 1, 1)
    j.input['potential'] = level
    j.input['min_dist'] = min_dist
    j.input['max_dist'] = max_dist
    if iterations is not None:
        j.input['iteration'] = iterations
    if energy_weight is not None:
        j.input['energy-weight'] = energy_weight
    if force_weight is not None:
        j.input['force-weight'] = force_weight
    if stress_weight is not None:
        j.input['stress-weight'] = stress_weight
    j.server.queue = 'cmti'
    if level < 16:
        j.server.cores =  20
    if level < 24:
        j.server.cores =  40
    else:
        j.server.cores =  80
    j.server.cores *= int(np.ceil(train.number_of_structures / 20_000))
    j.server.run_time = 0.5 * level * train.number_of_structures
    j.server.cores = min(j.server.cores, 8 * 40)
    j.run()

    return pr

def plot(fit_job, legend=True):
    train = fit_job['input/training_data'].to_object()
    efs = fit_job['output/training_efs'].to_object()
    N = train.get_array('length')

    plt.subplot(241, title="Energy per Atom", aspect=1)
    plt.scatter(efs.get_array('energy').ravel()/N, train.get_array('energy').ravel()/N,
                marker='.', label=fit_job.name)
    plt.xlabel("True [eV]")
    plt.ylabel("Predicted [eV]")
    if legend: plt.legend()
    plt.subplot(245)
    ediff = efs.get_array('energy').ravel()/N - train.get_array('energy').ravel()/N
    plt.hist(ediff, bins=50, log=True, label=f"{ediff.std():.05}");
    plt.xlabel(r"$E$ [eV]")
    if legend: plt.legend()

    train_forces = train.get_array("forces")

    plt.subplot(242, title="Force", aspect=1)
    plt.scatter(efs.get_array('forces').ravel(), train_forces.ravel(),
                marker='.', label=fit_job.name)
    plt.xlabel("True [eV/A]")
    plt.ylabel("Predicted [eV/A]")
    if legend: plt.legend()
    plt.subplot(246)
    fdiff = efs.get_array('forces').ravel() - train_forces.ravel()
    plt.hist(fdiff, bins=50, log=True, label=f"{fdiff.std():.05}");
    plt.xlabel(r"$F$ [eV/$\AA$]")
    if legend: plt.legend()

    train_stress = train.get_array("stress")

    train_hydrostatic = train_stress[:, :3].ravel()
    train_shear = train_stress[:, 3:].ravel()
    pred_hydrostatic = efs.get_array('stress')[:, :3].ravel()
    pred_shear = efs.get_array('stress')[:, 3:].ravel()

    plt.subplot(243, title="Hydrostatic Pressures", aspect=1)
    plt.scatter(train_hydrostatic, pred_hydrostatic,
                marker='.', label=fit_job.name)
    plt.xlabel("True [eV]")
    plt.ylabel("Predicted [eV]")
    if legend: plt.legend()
    plt.subplot(247)
    sdiff = pred_hydrostatic - train_hydrostatic
    plt.hist(sdiff, bins=50, log=True, label=f"{sdiff.std():.05}");
    plt.xlabel("$\Delta \sigma V$ [eV]")
    if legend: plt.legend()

    plt.subplot(244, title="Shear Pressures", aspect=1)
    plt.scatter(train_shear, pred_shear,
                marker='.', label=fit_job.name)
    plt.xlabel("True [eV]")
    plt.ylabel("Predicted [eV]")
    if legend: plt.legend()
    plt.subplot(248)
    sdiff = pred_shear - train_shear
    plt.hist(sdiff, bins=50, log=True, label=f"{sdiff.std():.05}");
    plt.xlabel("$\Delta \sigma V$ [eV]")
    if legend: plt.legend()

def energy_rmse(j):
    inpt = j['input/training_data'].to_object()
    N = inpt.get_array('length')
    train = np.squeeze(inpt.get_array('energy'))/N
    pred  = np.squeeze(j['output/training_efs'].to_object().get_array('energy'))/N
    return np.sqrt(np.mean((train-pred)**2))
def energy_mae(j):
    inpt = j['input/training_data'].to_object()
    N = inpt.get_array('length')
    train = np.squeeze(inpt.get_array('energy'))/N
    pred  = np.squeeze(j['output/training_efs'].to_object().get_array('energy'))/N
    # return np.abs(train-pred).mean()
    return np.abs(train-pred).max()
def force_rmse(j):
    inpt = j['input/training_data'].to_object()
    train = np.squeeze(inpt.get_array('forces'))
    pred  = np.squeeze(j['output/training_efs'].to_object().get_array('forces'))
    return np.sqrt(np.mean((train-pred)**2))
def force_mae(j):
    inpt = j['input/training_data'].to_object()
    N = inpt.get_array('length')
    train = np.squeeze(inpt.get_array('forces'))
    pred  = np.squeeze(j['output/training_efs'].to_object().get_array('forces'))
    return np.abs(train-pred).max()
def stress_hydro_rmse(j):
    train = np.squeeze(j['input/training_data'].to_object().get_array('stress'))
    pred  = np.squeeze(j['output/training_efs'].to_object().get_array('stress'))
    return np.sqrt(np.mean((train[:,:3]-pred[:,:3])**2))
def stress_hydro_mae(j):
    train = np.squeeze(j['input/training_data'].to_object().get_array('stress'))
    pred  = np.squeeze(j['output/training_efs'].to_object().get_array('stress'))
    return np.abs(train[:,:3]-pred[:,:3]).max()
def stress_shear_rmse(j):
    train = np.squeeze(j['input/training_data'].to_object().get_array('stress'))
    pred  = np.squeeze(j['output/training_efs'].to_object().get_array('stress'))
    return np.sqrt(np.mean((train[:,3:]-pred[:,3:])**2))
def stress_shear_mae(j):
    train = np.squeeze(j['input/training_data'].to_object().get_array('stress'))
    pred  = np.squeeze(j['output/training_efs'].to_object().get_array('stress'))
    return np.abs(train[:,3:]-pred[:,3:]).max()
def energy_spread(j):
    train = j['input/training_data'].to_object()
    return np.ptp(train.get_array('energy').ravel() / train.get_array('length'))

def analyze(fit_pr: Project, delete_existing_job=False):
    """
    Collect error metrics on a project of fits.

    Return MAE and RMSE of energies, forces and stresses for each fitted potential together with 

    Args:
        fit_pr (Project): project with Mlip jobs inside
        delete_existing_job (bool): if True recreate pyiron table from scratch

    Returns:
        pandas.DataFrame: return data frame with error metrics
    """
    def add(tab):
        tab.analysis_project = fit_pr
        tab.filter_function = tab.filter.job_type("Mlip")
        # GOTCHA: technically this can be anything; but we only ever fit() Alex' predefined potentials where the
        # name corresponds exactly to the level
        tab.add['level'] = lambda j: \
                int(read_generic_parameters(j['mlip_inp'], 'potential'))
        tab.add['rmin'] = lambda j: \
                read_generic_parameters(j['mlip_inp'], 'min_dist')
        tab.add['rmax'] = lambda j: \
                read_generic_parameters(j['mlip_inp'], 'max_dist')
        tab.add['restart'] = lambda j: j.name.endswith('restart')
        tab.add['energy_spread'] = energy_spread
        tab.add['energy_rmse'] = energy_rmse
        tab.add['energy_mae'] = energy_mae
        tab.add['force_rmse'] = force_rmse
        tab.add['force_mae'] = force_mae
        tab.add['stress_hydro_rmse'] = stress_hydro_rmse
        tab.add['stress_hydro_mae'] = stress_hydro_mae
        tab.add['stress_shear_rmse'] = stress_shear_rmse
        tab.add['stress_shear_mae'] = stress_shear_mae
        tab.add['training'] = lambda j: '|'.join(j.project.inspect(jid).name for jid in j.content.input['job_id_list'])
        tab.add['potential'] = lambda j: j['input/potential/Filename'][0]
        # tab.add['project'] = lambda j: j.project.name
        # tab.add['parent_project'] = lambda j: j.project['..'].name
        tab.add['potential'] = lambda j: os.path.join(j.working_directory, 'Trained.mtp_')
    tab = get_table(fit_pr, 'level_error_table', add,
                    delete_existing_job=delete_existing_job)
    df = tab.get_dataframe()

    # Melt table to have easy seaborn plotting later
    errors = ['energy_rmse', 'energy_mae', 'force_rmse', 'force_mae',
              'stress_hydro_rmse', 'stress_hydro_mae', 'stress_shear_rmse', 'stress_shear_mae']
    cols = [c for c in df.columns if c not in errors]
    df = df.melt(id_vars=cols, value_vars=errors, value_name='error')
    vv = df.variable.str.rsplit('_', expand=True, n=1)
    vv.columns = ['quantity', 'metric']
    df = pd.concat([df.drop('variable', axis='columns'), vv], axis='columns')
    df.metric = df.metric.str.upper()

    return df

def plot_error_vs_level(df, logy=True, **kwargs):
    return sns.relplot(
            data=df,
            kind='line', marker='o',
            x='level', y='error',
            col='quantity', row='metric',
            hue='max_dist', style='min_dist',
            facet_kws={'sharey': 'col'},
            **kwargs
    ).set(yscale='log' if logy else 'linear')

def plot_error_vs_max_dist(df, logy=True, **kwargs):
    return sns.relplot(
            data=df,
            kind='line', marker='o',
            x='max_dist', y='error',
            col='quantity', row='metric',
            hue='level', style='min_dist',
            facet_kws={'sharey': 'col'},
            **kwargs
    ).set(yscale='log' if logy else 'linear')

epilog="""
We lay out the project like this,

`root/training/containers`:     where we read the --containers from
`root/fits`:                    work exlusively in this project
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
            description='Fit MTPs to training data.',
            epilog=epilog,
            formatter_class=RawDescriptionHelpFormatter
    )

    parser.add_argument(
            '-l', '--level', type=int, nargs='+',
            help='The level of the potentials'
    )
    parser.add_argument(
            '-a', '--rmax', type=float, nargs='+',
            help='Outer cutoff for the potentials'
    )
    parser.add_argument(
            '-i', '--rmin', type=float, nargs='+',
            help='Inner cutoff for the data to consider'
    )

    parser.add_argument(
            '-p', '--project', default='fits',
            help='project to work in'
    )
    parser.add_argument(
            '--training-project', default='training',
            help='project that contains the training containers'
    )
    parser.add_argument(
            '-c', '--containers', type=str, nargs='+',
            default=('Everything',),
            help='names of training containers to use'
    )
    parser.add_argument(
            '--refit', action="store_true",
            help='Fit again to same data, if potential with given parameters already exists'
    )
    parser.add_argument(
            '--iterations', type=int,
            help='Maximum number of minimizer iterations during fitting'
    )
    parser.add_argument(
            '--energy-weight', type=float,
    )
    parser.add_argument(
            '--force-weight', type=float,
    )
    parser.add_argument(
            '--stress-weight', type=float,
    )


    args = parser.parse_args()

    fit_pr = Project(args.project)
    train_pr = fit_pr['..'].create_group(args.calculations_project)
    for contname in args.containers:
        cont = train_pr.load(contname)
        fit(
                fit_pr, cont,
                min_dist=args.rmin, max_dist=args.rmax,
                level=args.level,
                iterations=args.iterations,
                refit=args.refit,
                energy_weight=args.energy_weight,
                force_weight=args.force_weight,
                stress_weight=args.stress_weight,
        )

if __name__ == '__main__':
    main()
