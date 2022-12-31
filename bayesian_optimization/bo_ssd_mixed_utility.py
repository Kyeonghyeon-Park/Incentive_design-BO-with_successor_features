import numpy as np
from scipy.spatial import distance
import torch

from utils import utils_bo
"""
This file is for getting BO results given observations. 
This file uses evaluation results, from evaluate_xxx_multi.py, to calculate misUCB. 
If you want to use original UCB, you can set:
proportion_main = 1 in line 47.
You should set line 30 that loads evaluation results.

The shape of data will be 
data = torch.load('evaluation_results_ssd.tar')
->
data = {alpha_env: {alpha_pol: {'rew': ndarray(num_tests=100, num_networks=1),
                                'pen': ndarray(num_tests=100, num_networks=1),
                                'inc': ndarray(num_tests=100, num_networks=1),
                                'obj': ndarray(num_tests=100, num_networks=1),
                                'x_axis': ndarray(num_networks=1,),
                                }
                    }
        }
"""
# x_axis points.
num_x_pts = 10000
x = np.linspace(0, 1, num_x_pts).reshape(-1, 1)

# Load data.
# data = torch.load("../evaluation_results_ssd.tar")
data = torch.load("../14_0.00,0.05,0.10,0.13,0.16,0.18,0.21,0.24,0.28,0.29,0.33,0.40,0.56,0.68,1.00_num_tests=50.tar")

alphas_env = list(data.keys())
alphas_pol = list(data[alphas_env[0]].keys())
num_env = len(alphas_env)
num_pol = len(alphas_pol)
objs = np.zeros([num_env, num_pol])
for i in range(num_env):
    for j in range(num_pol):
        objs[i][j] = data[alphas_env[i]][alphas_pol[j]]['obj'].mean()

# Set some variables.
num_pol = len(alphas_pol)
utilities = np.zeros([num_pol + 1, num_x_pts])
weights = np.ones([num_pol + 1, num_x_pts])
alpha_range = 1
proportion_main = utils_bo.get_proportion_main(num_pol, is_increasing=True)  # misUCB
# proportion_main = 1  # original UCB

# Original GP.
observations = {}
for i in range(num_pol):
    alpha_pol = alphas_pol[i]
    alpha_env = alphas_env[i]
    observations[alpha_env] = objs[i, i]

optimizer_main, acquisition_function_main = utils_bo.get_opt_and_acq(observations,
                                                                     random_state=20,
                                                                     length_scale_bounds=(5e-2, 1e5),
                                                                     alpha=5e-3,
                                                                     )
x_obs, y_obs = optimizer_main.get_obs()
utility = acquisition_function_main.utility(x, optimizer_main._gp, 0)
utilities[0] = utility

# GP for each policy.
for i in range(num_pol):
    alpha_pol = alphas_pol[i]
    weights[i + 1] = np.exp(-1.5 * distance.cdist([[alpha_pol]], x, 'euclidean'))
    observations = {}
    for j in range(num_pol):
        observations[alphas_env[j]] = objs[j, i]
    optimizer, acquisition_function = utils_bo.get_opt_and_acq(observations,
                                                               random_state=20,
                                                               length_scale_bounds=(5e-2, 1e5),
                                                               alpha=5e-3,
                                                               )

    x_obs, y_obs = optimizer.get_obs()
    utility = acquisition_function.utility(x, optimizer._gp, 0)
    utilities[i + 1] = utility

# Mixed utility.
weights[0, :] = weights[0, :] * proportion_main
weights[1:, :] = np.divide(weights[1:, :], np.sum(weights[1:, :], axis=0)) * (1 - proportion_main)
mixed_utility = np.sum(np.multiply(utilities, weights), axis=0)
utils_bo.plot_gp_utility(optimizer_main, mixed_utility, x)
