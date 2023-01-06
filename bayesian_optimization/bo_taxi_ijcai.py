import numpy as np
from utils import utils_bo

'''
230101 Drawing GP of full BO results for IJCAI.
Because I built a full set of representative objective values for each alpha like a library, 
you should write alpha_dicts which the key is the value of alpha and the value is the order of alpha.
Orders of alphas are used to write the orders to the graph.
'''

START_ALPHA = 0.93
ACQUISITION_FUNCTION = 'misUCB'

if START_ALPHA not in [0.30, 0.43, 0.63, 0.93] or ACQUISITION_FUNCTION not in ['UCB', 'misUCB']:
    raise ValueError

alpha_dicts = {}
if START_ALPHA == 0.30:
    if ACQUISITION_FUNCTION == 'UCB':
        alpha_dicts = {0.30: 0,
                       1.00: 1,
                       0.93: 2,
                       0.87: 3,
                       0.70: 4,
                       0.58: 5,
                       0.00: 6,
                       0.48: 7,
                       0.54: 8,
                       0.56: 9,
                       0.59: 10,
                       0.53: 11,
                       }
    else:
        alpha_dicts = {0.30: 0,
                       1.00: 1,
                       0.85: 2,
                       0.00: 3,
                       0.62: 4,
                       0.50: 5,
                       0.43: 6,
                       0.54: 7,
                       0.73: 8,
                       }
elif START_ALPHA == 0.43:
    if ACQUISITION_FUNCTION == 'UCB':
        alpha_dicts = {0.43: 0,
                       1.00: 1,
                       0.50: 2,
                       0.00: 3,
                       0.73: 4,
                       0.60: 5,
                       0.53: 6,
                       0.52: 7,
                       }
    else:
        alpha_dicts = {0.43: 0,
                       1.00: 1,
                       0.50: 2,
                       0.00: 3,
                       0.70: 4,
                       0.26: 5,
                       0.57: 6,
                       0.52: 7,
                       }
elif START_ALPHA == 0.63:
    if ACQUISITION_FUNCTION == 'UCB':
        alpha_dicts = {0.63: 0,
                       0.00: 1,
                       0.56: 2,
                       1.00: 3,
                       0.35: 4,
                       0.47: 5,
                       0.81: 6,
                       0.49: 7,
                       0.41: 8,
                       0.21: 9,
                       0.73: 10,
                       0.90: 11,
                       0.11: 12,
                       0.52: 13,
                       0.51: 14,
                       }
    else:
        alpha_dicts = {0.63: 0,
                       0.00: 1,
                       0.81: 2,
                       0.35: 3,
                       1.00: 4,
                       0.49: 5,
                       0.45: 6,
                       0.54: 7,
                       0.21: 8,
                       0.57: 9,
                       0.55: 10,
                       0.56: 11,
                       }
elif START_ALPHA == 0.93:
    if ACQUISITION_FUNCTION == 'UCB':
        alpha_dicts = {0.93: 0,
                       0.00: 1,
                       0.85: 2,
                       0.55: 3,
                       0.36: 4,
                       0.68: 5,
                       0.46: 6,
                       0.43: 7,
                       0.49: 8,
                       0.73: 9,
                       0.48: 10,
                       0.47: 11,
                       0.21: 12,
                       0.50: 13,
                       }
    else:
        alpha_dicts = {0.93: 0,
                       0.00: 1,
                       0.79: 2,
                       0.41: 3,
                       0.58: 4,
                       0.51: 5,
                       0.62: 6,
                       0.26: 7,
                       0.54: 8,
                       0.16: 9,
                       0.49: 10,
                       }

# Observations.
observations_full = {
    0.00: 0.8287,
    0.11: 0.8346,
    0.16: 0.8491,
    0.21: 0.8571,
    0.26: 0.8727,
    0.28: 0.8808,
    0.30: 0.8570,
    0.35: 0.8873,
    0.36: 0.8877,
    0.41: 0.8921,
    0.43: 0.8953,
    0.45: 0.8948,
    0.46: 0.8996,
    0.47: 0.9035,
    0.48: 0.8985,
    0.49: 0.8986,
    0.50: 0.9038,
    0.51: 0.9054,
    0.52: 0.9025,
    0.53: 0.9041,
    0.54: 0.9027,
    0.55: 0.8980,
    0.56: 0.9042,
    0.57: 0.9040,
    0.58: 0.9026,
    0.59: 0.8986,
    0.60: 0.8984,
    0.62: 0.8918,
    0.63: 0.8928,
    0.68: 0.8927,
    0.70: 0.8854,
    0.73: 0.8882,
    0.79: 0.8863,
    0.81: 0.8813,
    0.85: 0.8800,
    0.87: 0.8730,
    0.90: 0.8741,
    0.93: 0.8710,
    1.00: 0.8631,
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
                     gp_lim=[(0, 1), (0.81, 0.91)],
                     acq_lim=[(0, 1), (None, None)],
                     point_labels=alpha_labels,
                     gp_only=True,
                     )

# acq.set_ylim((np.min(utility) - 10, np.max(utility) + 10))
