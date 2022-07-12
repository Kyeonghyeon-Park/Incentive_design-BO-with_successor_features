import numpy as np
from scipy.spatial import distance
import torch

from utils import utils_bo
"""
220614 Test for ssd.
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
"""
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
data = torch.load("../evaluation_results_ssd.tar")
# data = torch.load("../results/220518 objs/8_alpha=0,0.3,0.43,0.50,0.54,0.62,0.73,0.85,1.tar")

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
proportion_main = get_proportion_main(num_pol, is_increasing=True)  # misUCB
# proportion_main = get_proportion_main(num_pol, is_increasing=False)  # misUCB_old
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
