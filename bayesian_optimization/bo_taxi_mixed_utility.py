import numpy as np
from scipy.spatial import distance
import torch

from utils import utils_bo
"""
220426 Test for mixed utility. 
It uses saved results which comes from evaluate_taxi_multi.py. 

220518 Test for mixed utility with increasing proportion_main.
"""


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


# x_axis points.
num_x_pts = 10000
x = np.linspace(0, 1, num_x_pts).reshape(-1, 1)

# Load data.
# data = torch.load("../test.tar")
data = torch.load("../results/220518 objs/8_alpha=0,0.3,0.43,0.50,0.54,0.62,0.73,0.85,1.tar")
alphas_env, alphas_pol, objs = data.values()

# Set some variables.
num_pol = len(alphas_pol)
utilities = np.zeros([num_pol + 1, num_x_pts])
weights = np.ones([num_pol + 1, num_x_pts])
alpha_range = 1
# proportion_main = get_proportion_main(num_pol, is_increasing=True)
# proportion_main = get_proportion_main(num_pol, is_increasing=False)
proportion_main = 1

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
# utils_bo.plot_gp_simple(optimizer_main, utility, x)

# GP for each policy.
for i in range(num_pol):
    alpha_pol = alphas_pol[i]
    # weights[i+1] = 1 - distance.cdist([[alpha_pol]], x, 'euclidean') / alpha_range
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
    # utils_bo.plot_gp_simple(optimizer, utility, x)

# Mixed utility.
weights[0, :] = weights[0, :] * proportion_main
weights[1:, :] = np.divide(weights[1:, :], np.sum(weights[1:, :], axis=0)) * (1 - proportion_main)
mixed_utility = np.sum(np.multiply(utilities, weights), axis=0)
utils_bo.plot_gp_utility(optimizer_main, mixed_utility, x)
