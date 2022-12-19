import numpy as np
import os
import time

from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.util import acq_max

"""
More about bayes_opt: https://github.com/fmfn/BayesianOptimization. 
"""


def black_box_function(alpha):
    """
    Define the black box function which is the f(alpha).
    In this version, we manually put observations into the optimizer.
    Therefore, we don't iteratively run optimizer.maximize.

    Parameters
    ----------
    alpha: float
    """
    # Get results for alpha
    return round(alpha, 2)


def sqrt_beta(t=6, d=1, delta=0.5):
    """
    Coefficient of UCB in Shou and Di (2020).
    Check https://www.sciencedirect.com/science/article/pii/S0968090X20306525.

    Parameters
    ----------
    t: int
        t should be the number of priors + 1.
    d: int
    delta: float

    Returns
    -------
    value: float
    """
    value = np.sqrt(2 * np.log(t**(d / 2 + 2) * np.pi**2 / (3 * delta)))
    return value


def get_proportion_main(num_samples, sensitivity=0.25, is_increasing=False):
    """
    Get the proportion of main(original or true) UCB.

    Parameters
    ----------
    num_samples: int
        Number of policies(or samples).
    sensitivity: float
        The high sensitivity gives a high proportion of main UCB for same number of samples.
    is_increasing: bool
        False if the proportion is constant.

    Returns
    -------
    proportion
    """
    if is_increasing:
        proportion = 1 / (1 + np.exp(-sensitivity * num_samples))
    else:
        proportion = 0.5
    return proportion


class BayesianOptimizationModified(BayesianOptimization):
    """
    Update self._gp to easily change alpha and length_scale_bounds of GaussianProcessRegressor.
    Build useful functions to easily access the GaussianProcessRegressor and other things.
    """
    def __init__(self, f, pbounds, random_state=None, verbose=2, length_scale_bounds=(1e-5, 1e5), alpha=1e-6):
        super().__init__(f, pbounds, random_state=random_state, verbose=verbose,)
        self._gp = GaussianProcessRegressor(
            kernel=Matern(length_scale_bounds=length_scale_bounds, nu=2.5,),
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

    def register(self, observations, **kwargs):
        """
        Register observed points.
        **kwargs is for removing PyCharm error.

        Parameters
        ----------
        observations: dict
            {params: target}
            ex. observations = {0: 0.8287, 0.3: 0.8570, 0.5: 0.8961, 0.56: 0.9025, 0.63:0.8952, 0.8:0.8842, 1:0.8631}
        """
        for params, target in observations.items():
            super().register(params=params, target=target)

    def fit(self):
        """
        Fit Gaussian process regression model.
        GaussianProcessRegressor requires
        1) x_obs: numpy.ndarray
            Feature vectors or other representations of training data.
            shape: (N, 1) where N is the number of observations.
        2) y_obs: numpy.ndarray
            Target values.
            shape: (N, )
        """
        x_obs, y_obs = self.get_obs()
        self._gp.fit(x_obs, y_obs)

    def get_posterior(self, grid):
        """
        Get mu and sigma of posterior distribution for given grid.

        Parameters
        ----------
        grid: numpy.ndarray
            Query points where the GP is evaluated.
            ex. x = np.linspace(0, 1, 10000).reshape(-1, 1)

        Returns
        -------
        mu: numpy.ndarray
            Mean of predictive distribution a query points.
            ndarray of shape (n_samples, [n_output_dims]).
        sigma: numpy.ndarray
            Standard deviation of predictive distribution at query points.
            ndarray of shape (n_samples,), optional.
            Only returned when `return_std` is True.
        """
        mu, sigma = self._gp.predict(grid, return_std=True)
        return mu, sigma

    def get_obs(self):
        """
        Get registered(or observed) points.

        Returns
        -------
        x_obs: numpy.ndarray
            Feature vectors or other representations of training data.
            shape: (N, 1) where N is the number of observations.
        y_obs: numpy.ndarray
            Target values.
            shape: (N, )
        """
        x_obs = np.array([[res["params"]["alpha"]] for res in self.res])
        y_obs = np.array([res["target"] for res in self.res])
        return x_obs, y_obs

    def get_acq(self):
        """
        Get acquisition(or utility) function which is UCB in our version.

        Returns
        -------
        acquisition_function: UtilityFunction
        """
        num_points = self._space.__len__()
        kappa = sqrt_beta(t=num_points + 1)
        acquisition_function = UtilityFunction(kind="ucb", kappa=kappa, xi=0)

        return acquisition_function

    def suggest(self, acquisition_function):
        """
        Suggest most promising point to probe next.

        Parameters
        ----------
        acquisition_function: UtilityFunction

        Returns
        -------
        suggested_point: dict
            ex. {'alpha': 0.5820942986281569}
        """
        if len(self._space) == 0:
            suggested_point = self._space.array_to_params(self._space.random_sample())
            return suggested_point

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=acquisition_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )
        suggested_point = self._space.array_to_params(suggestion)
        return suggested_point


def get_opt_and_acq(observations, pbounds=None, **kwargs):
    """
    Get the optimizer and the acquisition function.
    Optimizer is fitted to observed points(observations).

    Parameters
    ----------
    observations: dict
    pbounds: None or dict

    Returns
    -------
    optimizer: BayesianOptimizationModified
    acquisition_function: UtilityFunction
    """
    if pbounds is None:
        pbounds = {'alpha': (0, 1)}
    optimizer = BayesianOptimizationModified(
        f=black_box_function,
        pbounds=pbounds,
        **kwargs,
    )
    optimizer.register(observations)
    optimizer.fit()
    acquisition_function = optimizer.get_acq()
    return optimizer, acquisition_function


def plot_gp_utility(optimizer, utility, x, gp_lim=None, acq_lim=None, next_point_suggestion=None):
    """
    Plot posterior and values of acquisition function.
    220805 axis update.

    Parameters
    ----------
    optimizer: BayesianOptimizationModified
    utility: numpy.ndarray
    x: numpy.ndarray
        x-axis of graph. We need the shape (-1, 1) for x.
        ex. x = np.linspace(0, 1, 10000).reshape(-1, 1)
    gp_lim: None or List
        List of xlim and ylim for the GP graph.
        ex. gp_lim = [(0, 1), (0.81, 0.92)]
    acq_lim: None or List
        List of xlim and ylim for the acquisition graph.
        ex. acq_lim = [(0, 1), (0.83, 0.935)]
    next_point_suggestion: None or dict
        If next_point_suggestion is coming from acquisition function, it should be dict.
        Otherwise, it will be None.
        ex. next_point_suggestion = {'alpha': 0.5313201435572625}
    """
    plot_time = time.time()

    # Observations and posterior distributions.
    x_obs, y_obs = optimizer.get_obs()
    mu, sigma = optimizer.get_posterior(x)

    # Find a value of alpha to be evaluated.
    if next_point_suggestion is None:
        next_point_suggestion = "{'alpha': " + str(x[np.argmax(utility)].item()) + "}"
    print("Next point suggestion: ", next_point_suggestion)

    # Plot settings.
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    # Plot.
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label='Observations', color='k')
    axis.plot(x, mu, linestyle=(0, (5, 5)), color='k', label='Prediction')
    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='lightgray', ec='None', label='95% C. I.')

    if gp_lim is not None:
        axis.set_xlim(gp_lim[0])
        axis.set_ylim(gp_lim[1])
    if acq_lim is not None:
        acq.set_xlim(acq_lim[0])
        acq.set_ylim(acq_lim[1])

    # axis.set_xlabel(r'$\alpha$', fontdict={'size': 24})  # 220805
    # axis.set_ylabel(r'$f$', fontdict={'size': 24})  # 220805
    axis.set_xlabel(r'$\alpha$', fontdict={'size': 28})  # 220805
    axis.set_ylabel(r'$\mathcal{F}$', fontdict={'size': 28})  # 220805

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


def plot_gp_acq(optimizer, acquisition_function, x, gp_lim=None, acq_lim=None):
    """
    Plot posterior and acquisition function.
    If figure shows weird posterior, please change random_state or length_scale_bounds when you build the optimizer.

    Parameters
    ----------
    optimizer: BayesianOptimizationModified
    acquisition_function: UtilityFunction
    x: numpy.ndarray
        x-axis of graph.
        We need the shape (-1, 1) for x because of acquisition_function.utility.
        ex. x = np.linspace(0, 1, 10000).reshape(-1, 1)
    gp_lim: None or List
        List of xlim and ylim for the GP graph.
        ex. gp_lim=[(0, 1), (0.81, 0.92)]
    acq_lim: None or List
        List of xlim and ylim for the acquisition graph.
        ex. acq_lim=[(0, 1), (0.83, 0.935)]
    """
    # TODO: check float issues.
    # next_point_suggestion = optimizer.suggest(acquisition_function)
    next_point_suggestion = None
    utility = acquisition_function.utility(x, optimizer._gp, 0)
    plot_gp_utility(optimizer, utility, x, gp_lim=gp_lim, acq_lim=acq_lim, next_point_suggestion=next_point_suggestion)
