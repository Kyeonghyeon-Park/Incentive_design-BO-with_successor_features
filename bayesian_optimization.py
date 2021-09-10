#%%
import argparse
import numpy as np
import time
import os

from matplotlib import gridspec
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from actor_critic import ActorCritic


#%%
"""
To-do or check list
--------
Current package is not stable
Current BO setting is not same as the Shou and Di's paper (I almost use the initial setting of the package) 
"""

def get_args(alpha):
    """
    Build args which is the parameter setting of the actor-critic network

    Parameters
    ----------
    alpha : float
        Designer's decision (penalty for overcrowded grid)
        BO will decide the proper alpha

    Returns
    -------
    args : argparse.Namespace
        Return the args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--designer_alpha', default=alpha)
    parser.add_argument('--sample_size', default=4)
    parser.add_argument('--buffer_max_size', default=50)
    parser.add_argument('--max_episode_number', default=3500)
    parser.add_argument('--discount_factor', default=1)
    parser.add_argument('--epsilon', default=0.5)
    parser.add_argument('--mean_action_sample_number', default=5)
    parser.add_argument('--obj_weight', default=0.6)
    parser.add_argument('--lr_actor', default=0.0001)
    parser.add_argument('--lr_critic', default=0.001)
    parser.add_argument('--update_period', default=10)
    parser.add_argument('--trained', default=False)
    parser.add_argument('--PATH', default='')
    parser.add_argument('--filename', default='')

    args = parser.parse_args()

    return args


def black_box_function(alpha):
    """
    Define the black box function which is the f(alpha)
    We have to get the designer's objective after the algorithm converges

    Parameters
    ----------
    alpha : float
        Designer's decision (penalty for overcrowded grid)
        BO will decide the proper alpha

    Returns
    obj_for_BO : float
        Return the designer's objective for BO which is the average of the obj_ftn
    -------

    """
    args = get_args(alpha)
    model = ActorCritic(args)
    model.run()
    obj_for_BO = np.average(model.outcome['test']['obj_ftn'][-100:])
    return obj_for_BO


#%%
BO_start = time.time()
x = np.linspace(0, 1, 10000).reshape(-1, 1)
pbounds = {'alpha': (0, 1)}

#%%
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
)

#%% register가 기존 value 넣는 것인듯? 추가 절차 필요 없는듯
optimizer.register(params=0, target=0.8286)
optimizer.register(params=0.5, target=0.8964)
optimizer.register(params=0.7, target=0.8960)
optimizer.register(params=0.8, target=0.8876)
optimizer.register(params=1, target=0.8679)


#%% Plotting and visualizing the algorithm
def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, x):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size': 30}
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["alpha"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)

    # axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label='Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((0, 1))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size': 20})
    axis.set_xlabel('x', fontdict={'size': 20})

    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    next_point_suggestion = optimizer.suggest(utility_function)
    print("Next point suggestion : ", next_point_suggestion)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((0, 1))
    acq.set_ylim((np.min(utility) - 0.1, np.max(utility) + 0.1))
    acq.set_ylabel('Utility', fontdict={'size': 20})
    acq.set_xlabel('x', fontdict={'size': 20})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    PATH = './BO/' + time.strftime('%y%m%d_%H%M', time.localtime(BO_start)) + '/'
    os.makedirs(PATH, exist_ok=True)
    c_time = time.time()
    file_name = str(len(optimizer.space)) + 'steps_' + time.strftime('%y%m%d_%H%M', time.localtime(c_time)) + '.png'
    plt.savefig(PATH + file_name)

    plt.show()


#%%
plot_gp(optimizer, x)
# for BO_iter in range(5):
#     optimizer.maximize(init_points=0, n_iter=1)
#     plot_gp(optimizer, x)


#%%
# print("################## Final outcome ##################")
# print(optimizer.max)
# for i, res in enumerate(optimizer.res):
#     print("Iteration {}: \n\t{}".format(i, res))
# print("################## Final outcome ##################")
#
# logger = JSONLogger(path="./logs.json")
# optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

#%% etc
# probe 시 다음 maximize 호출 되었을 때 어느 point 진행할 지 예약
# optimizer.probe(
#     params={'alpha': 0.53},
#     lazy=True
# )
# optimizer.maximize(init_points=0, n_iter=0)

# optimizer.maximize(
#     init_points=0,  # exploration 전 탐색할 숫자?
#     n_iter=5
# )
