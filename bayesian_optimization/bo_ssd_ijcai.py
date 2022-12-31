import numpy as np
from utils import utils_bo

'''
230101 Drawing GP of full BO results for IJCAI.
Because I built a full set of representative objective values for each alpha like a library, 
you should write alpha_dicts which the key is the value of alpha and the value is the order of alpha.
Orders of alphas are used to write the orders to the graph.
'''

START_ALPHA = 0.05
ACQUISITION_FUNCTION = 'misUCB'

if START_ALPHA not in [0.05, 0.33, 0.40, 0.56] or ACQUISITION_FUNCTION not in ['UCB', 'misUCB']:
    raise ValueError

alpha_dicts = {}
if START_ALPHA == 0.05:
    if ACQUISITION_FUNCTION == 'UCB':
        alpha_dicts = {0.00: 5,
                       0.05: 0,
                       0.08: 16,
                       0.13: 2,
                       0.14: 7,
                       0.16: 6,
                       0.21: 8,
                       0.24: 4,
                       0.28: 15,
                       0.29: 17,
                       0.33: 10,
                       0.40: 13,
                       0.47: 3,
                       0.56: 11,
                       0.65: 14,
                       0.75: 9,
                       0.86: 12,
                       1.00: 1,
                       }
    else:
        alpha_dicts = {0.00: 4,
                       0.05: 0,
                       0.10: 8,
                       0.13: 2,
                       0.16: 9,
                       0.18: 5,
                       0.21: 6,
                       0.24: 7,
                       0.28: 10,
                       0.29: 14,
                       0.33: 12,
                       0.40: 3,
                       0.56: 13,
                       0.68: 11,
                       1.00: 1,
                       }
elif START_ALPHA == 0.33:
    if ACQUISITION_FUNCTION == 'UCB':
        alpha_dicts = {0.00: 4,
                       0.05: 8,
                       0.10: 6,
                       0.15: 11,
                       0.21: 3,
                       0.26: 13,
                       0.28: 7,
                       0.30: 12,
                       0.33: 0,
                       0.40: 2,
                       0.56: 10,
                       0.68: 5,
                       0.86: 9,
                       1.00: 1,
                       }
    else:
        alpha_dicts = {0.00: 3,
                       0.05: 10,
                       0.08: 5,
                       0.14: 4,
                       0.20: 11,
                       0.24: 7,
                       0.26: 2,
                       0.28: 9,
                       0.33: 0,
                       0.56: 6,
                       0.79: 8,
                       1.00: 1,
                       }
elif START_ALPHA == 0.40:
    if ACQUISITION_FUNCTION == 'UCB':
        alpha_dicts = {0.00: 3,
                       0.05: 17,
                       0.08: 8,
                       0.10: 6,
                       0.13: 4,
                       0.14: 7,
                       0.16: 9,
                       0.20: 5,
                       0.24: 18,
                       0.26: 15,
                       0.28: 19,
                       0.30: 12,
                       0.33: 16,
                       0.40: 0,
                       0.47: 2,
                       0.56: 11,
                       0.68: 14,
                       0.75: 10,
                       0.86: 13,
                       1.00: 1,
                       }
    else:
        alpha_dicts = {0.00: 4,
                       0.05: 8,
                       0.08: 7,
                       0.10: 13,
                       0.13: 12,
                       0.16: 3,
                       0.20: 15,
                       0.25: 6,
                       0.28: 9,
                       0.30: 14,
                       0.33: 2,
                       0.40: 0,
                       0.56: 10,
                       0.68: 5,
                       0.86: 11,
                       1.00: 1,
                       }
elif START_ALPHA == 0.56:
    if ACQUISITION_FUNCTION == 'UCB':
        alpha_dicts = {0.00: 1,
                       0.07: 2,
                       0.14: 9,
                       0.21: 4,
                       0.24: 14,
                       0.25: 10,
                       0.26: 12,
                       0.27: 13,
                       0.28: 5,
                       0.30: 8,
                       0.40: 6,
                       0.56: 0,
                       0.75: 7,
                       0.86: 11,
                       1.00: 3,
                       }
    else:
        alpha_dicts = {0.00: 1,
                       0.07: 2,
                       0.14: 10,
                       0.20: 5,
                       0.25: 9,
                       0.28: 4,
                       0.30: 8,
                       0.40: 6,
                       0.56: 0,
                       0.75: 7,
                       0.86: 11,
                       1.00: 3,
                       }

# Observations.
observations_full = {
    0.00: 190.88,
    0.05: 195.78,
    0.07: 191.98,
    0.08: 198.23,
    0.10: 197.37,
    0.13: 201.87,
    0.14: 190.67,
    0.15: 202.64,
    0.16: 198.67,
    0.18: 195.99,
    0.20: 197.54,
    0.21: 196.15,
    0.24: 195.94,
    0.25: 191.96,
    0.26: 203.28,
    0.27: 197.80,
    0.28: 211.09,
    0.29: 205.77,
    0.30: 192.71,
    0.33: 197.12,
    0.40: 154.86,
    0.47: 145.41,
    0.56: 134.17,
    0.65: 102.74,
    0.68: 64.06,
    0.75: 81.63,
    0.79: 59.64,
    0.86: 28.07,
    1.00: 64.18,
}

# Selected observations.
alpha_lists = list(alpha_dicts.keys())
alpha_labels = list(alpha_dicts.values())
observations = {key: observations_full[key] for key in alpha_lists}

# Get optimizer and acquisition function.
optimizer, acquisition_function = utils_bo.get_opt_and_acq(observations,
                                                           random_state=20,
                                                           length_scale_bounds=(5e-2, 1e5),
                                                           alpha=5e-3,
                                                           # alpha=1e-5,
                                                           )

# Plot.
x = np.linspace(0, 1, 10000).reshape(-1, 1)
utils_bo.plot_gp_acq(optimizer,
                     acquisition_function,
                     x,
                     gp_lim=[(0, 1), (0, 220)],
                     acq_lim=[(0, 1), (None, None)],
                     point_labels=alpha_labels,
                     gp_only=True,
                     )

# acq.set_ylim((np.min(utility) - 10, np.max(utility) + 10))
