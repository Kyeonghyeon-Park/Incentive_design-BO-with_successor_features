#%%
import argparse
import numpy as np
import time
import os

from matplotlib import gridspec
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction


#%%
"""
Note
Current package is not stable. 
Therefore, if you have weird graph, please rerun the code.
You should set "add priors" part and t in sqrt_beta. 
"""
def black_box_function(alpha):
    """
    Define the black box function which is the f(alpha).
    We have to get the designer's objective after the algorithm converges.
    """
    # Get results for alpha
    return round(alpha, 2)


def sqrt_beta(t=6, d=1, delta=0.5):
    value = np.sqrt(2 * np.log(t**(d / 2 + 2) * np.pi**2 / (3 * delta)))
    return value


#%%
BO_start = time.time()
x = np.linspace(0, 1, 10000).reshape(-1, 1)
pbounds = {'alpha': (0, 1)}

#%% Build optimizer.
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2,
    random_state=14,
)

#%% Add priors.
optimizer.register(params=0, target=173.94)
optimizer.register(params=0.1, target=193.40)
# optimizer.register(params=0.33, target=193.04)
optimizer.register(params=0.33, target=194.50)
optimizer.register(params=0.43, target=215.00)
optimizer.register(params=0.5, target=278.06)
optimizer.register(params=0.6, target=168.14)
optimizer.register(params=0.8, target=40.62)
optimizer.register(params=1, target=64.74)

# Set sqrt_beta for UCB of our algorithm (we are not using constant term).
sqrt_beta = sqrt_beta(t=9)  # t should be number of priors + 1

# Set utility function.
utility_function = UtilityFunction(kind="ucb", kappa=sqrt_beta, xi=0)

# Set acquisition function.
next_point_suggestion = None
# I don't know but it doesn't work when we draw _gp and run next_point_suggestion simultaneously.
# If we want to draw the posterior figure, we have to deactivate this line.
# If we want to draw the acquisition function, we have to activate this line.
next_point_suggestion = optimizer.suggest(utility_function)

print("Next point suggestion : ", next_point_suggestion)
utility = utility_function.utility(x, optimizer._gp, 0)


#%% Plotting and visualizing the algorithm.
def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, x, utility):
    fig = plt.figure(figsize=(18, 12))
    steps = len(optimizer.space)

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["alpha"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)

    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label='Observations', color='k')
    axis.plot(x, mu, linestyle=(0, (5, 5)), color='k', label='Prediction')
    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='lightgray', ec='None', label='95% C. I.')
    axis.set_xlim((0, 1))
    axis.set_ylim((None, None))
    axis.set_ylabel(r'$f$', fontdict={'size': 24})
    axis.set_xlabel(r'$\alpha$', fontdict={'size': 24})

    acq.plot(x, utility, label='Utility Function', color='k')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((0, 1))
    acq.set_ylim((np.min(utility) - 10, np.max(utility) + 10))
    acq.set_ylabel('UCB', fontdict={'size': 24})
    acq.set_xlabel(r'$\alpha$', fontdict={'size': 24})

    axis.legend(loc='upper right', fontsize=20)
    axis.tick_params(axis='both', labelsize=20)

    acq.tick_params(axis='both', labelsize=20)
    acq.legend(loc='upper right', fontsize=20)


    PATH = './BO/' + time.strftime('%y%m%d_%H%M', time.localtime(BO_start)) + '/'
    os.makedirs(PATH, exist_ok=True)
    c_time = time.time()
    file_name = str(len(optimizer.space)) + 'steps_' + time.strftime('%y%m%d_%H%M', time.localtime(c_time)) + '.png'
    plt.savefig(PATH + file_name)

    plt.show()


#%% Run. (It will show the BO graph and print next suggested alpha.)
plot_gp(optimizer, x, utility)
