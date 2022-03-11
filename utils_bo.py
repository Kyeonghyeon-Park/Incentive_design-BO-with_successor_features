import numpy as np
import os
import time

from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction


"""
More about bayes_opt: https://github.com/fmfn/BayesianOptimization. 
Because of unstable package, please rerun the code or change random_state if you have weird graph.
"""


def black_box_function(alpha):
    """
    Define the black box function which is the f(alpha).
    In this version, we manually put observations into the optimizer.
    Therefore, we don't iteratively run optimizer.maximize.

    Parameters
    ----------
    alpha : float
    """
    # Get results for alpha
    return round(alpha, 2)


def sqrt_beta(t=6, d=1, delta=0.5):
    """
    Coefficient of UCB in Shou and Di (2020).
    Check https://www.sciencedirect.com/science/article/pii/S0968090X20306525.

    Parameters
    ----------
    t : int
        t should be number of priors + 1.
    d : int
    delta : float

    Returns
    -------
    value : float
    """
    value = np.sqrt(2 * np.log(t**(d / 2 + 2) * np.pi**2 / (3 * delta)))
    return value


class BayesianOptimizationAlpha(BayesianOptimization):
    """
    Update self._gp to easily change alpha of GaussianProcessRegressor.
    """
    def __init__(self, f, pbounds, random_state=None, verbose=2, alpha=1e-6):
        super().__init__(f, pbounds, random_state=random_state, verbose=verbose,)
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )


def get_bo_optimizer(observations, *, pbounds=None, verbose=2, random_state=7, alpha=5e-3):
    """
    Get Bayesian optimization optimizer.

    Parameters
    ----------
    observations : dict
        {params: target}
        ex. observations = {0: 0.8287, 0.3: 0.8570, 0.5: 0.8961, 0.56: 0.9025, 0.63:0.8952, 0.8:0.8842, 1:0.8631}
    pbounds
    verbose
    random_state
    alpha

    Returns
    -------
    optimizer
    """
    if pbounds is None:
        pbounds = {'alpha': (0, 1)}
    optimizer = BayesianOptimizationAlpha(
        f=black_box_function,
        pbounds=pbounds,
        verbose=verbose,
        random_state=random_state,
        alpha=alpha,
    )
    for params, target in observations.items():
        optimizer.register(params=params, target=target)

    return optimizer


def get_acquisition_function(optimizer):
    """
    Get acquisition(or utility) function which is UCB in our version.

    Parameters
    ----------
    optimizer

    Returns
    -------
    acquisition_function
    """
    num_points = optimizer.space.__len__()
    kappa = sqrt_beta(t=num_points + 1)
    acquisition_function = UtilityFunction(kind="ucb", kappa=kappa, xi=0)

    return acquisition_function


def get_opt_and_acq(observations, **kwargs):
    """
    Get optimizer and acquisition function.

    Parameters
    ----------
    observations : dict

    Returns
    -------
    optimizer
    acquisition_function
    """
    optimizer = get_bo_optimizer(observations, **kwargs)
    acquisition_function = get_acquisition_function(optimizer)
    return optimizer, acquisition_function


def get_posterior(optimizer, x_obs, y_obs, grid):
    """
    Get mu and sigma of posterior distribution with observed points.

    Parameters
    ----------
    optimizer : BayesianOptimizationAlpha
    x_obs
    y_obs
    grid

    Returns
    -------
    mu
    sigma
    """
    optimizer._gp.fit(x_obs, y_obs)
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def get_obs(optimizer):
    """
    Get registered(or observed) points.

    Parameters
    ----------
    optimizer

    Returns
    -------
    x_obs : numpy.ndarray
        shape : (N, 1) where N is the number of observations.
    y_obs : numpy.ndarray
        shape : (N, )
    """
    x_obs = np.array([[res["params"]["alpha"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    return x_obs, y_obs


def plot_gp(optimizer, acquisition_function, x, is_only_acq=False, mode=''):
    """
    Plot posterior and acquisition function.
    Because of unknown issue, it (sometimes) cannot draw posterior and acquisition function simultaneously.
    Please try is_only_acq=False first and try is_only_acq=True if it shows weird graph.
    If figure shows weird posterior, please change random_state in the function get_bo_optimizer.
    ex. random_state=10 for ssd samples. random_state=7 for taxi samples.

    Parameters
    ----------
    optimizer
    acquisition_function
    x : numpy.ndarray
        x-axis of graph.
        We need the shape (-1, 1) for x because of acquisition_function.utility.
        ex. x = np.linspace(0, 1, 10000).reshape(-1, 1)
    is_only_acq : bool
    mode : str
        Only for set axis lim.
    """
    # Find a value of alpha to be evaluated.
    plot_time = time.time()
    next_point_suggestion = optimizer.suggest(acquisition_function) if not is_only_acq else None
    print("Next point suggestion: ", next_point_suggestion)
    utility = acquisition_function.utility(x, optimizer._gp, 0)

    # Plot settings.
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    # Observations.
    x_obs, y_obs = get_obs(optimizer)

    # Posterior.
    mu, sigma = get_posterior(optimizer, x_obs, y_obs, x)

    # Plot.
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label='Observations', color='k')
    axis.plot(x, mu, linestyle=(0, (5, 5)), color='k', label='Prediction')
    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='lightgray', ec='None', label='95% C. I.')

    if mode == 'taxi':
        axis.set_xlim((0, 1))
        axis.set_ylim((0.81, 0.92))
        acq.set_xlim((0, 1))
        acq.set_ylim((0.83, 0.935))
    elif mode == 'ssd':
        axis.set_xlim((0, 1))
        axis.set_ylim((None, None))
        acq.set_xlim((0, 1))
        acq.set_ylim((np.min(utility) - 10, np.max(utility) + 10))
    else:
        pass
    axis.set_xlabel(r'$\alpha$', fontdict={'size': 24})
    axis.set_ylabel(r'$f$', fontdict={'size': 24})

    acq.plot(x, utility, label='Acquisition function', color='k')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlabel(r'$\alpha$', fontdict={'size': 24})
    acq.set_ylabel('UCB', fontdict={'size': 24})

    axis.legend(loc='upper right', fontsize=20)
    axis.tick_params(axis='both', labelsize=20)
    acq.tick_params(axis='both', labelsize=20)
    acq.legend(loc='upper right', fontsize=20)

    time_str = time.strftime('%y%m%d_%H%M', time.localtime(plot_time))
    PATH = './BO/' + time_str + '/'
    os.makedirs(PATH, exist_ok=True)
    file_name = str(len(optimizer.space)) + 'steps_' + time_str + '.png'
    plt.savefig(PATH + file_name)

    plt.show()
