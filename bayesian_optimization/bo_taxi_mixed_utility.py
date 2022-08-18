import numpy as np
from scipy.spatial import distance
import torch

from utils import utils_bo
"""
This file is for getting BO results given observations. 
This file uses evaluation results, from evaluate_xxx_multi.py, to calculate misUCB. 
If you want to use original UCB, you can set:
proportion_main = 1 in line 27.
You should set line 18 that loads evaluation results.
"""
# x_axis points.
num_x_pts = 10000
x = np.linspace(0, 1, num_x_pts).reshape(-1, 1)

# Load data.
data = torch.load("../evaluation_results_taxi.tar")
alphas_env, alphas_pol, objs = data.values()

# Set some variables.
num_pol = len(alphas_pol)
utilities = np.zeros([num_pol + 1, num_x_pts])
weights = np.ones([num_pol + 1, num_x_pts])
alpha_range = 1
proportion_main = utils_bo.get_proportion_main(num_pol, is_increasing=True)
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
utils_bo.plot_gp_utility(optimizer_main,
                         mixed_utility,
                         x,
                         gp_lim=[(0, 1), (0.81, 0.91)],
                         )
