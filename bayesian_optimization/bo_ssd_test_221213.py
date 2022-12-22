import numpy as np
from utils import utils_bo


START_ALPHA = 0.40
ACQUISITION_FUNCTION = 'UCB'

if START_ALPHA not in [0.05, 0.33, 0.40, 0.56] or ACQUISITION_FUNCTION not in ['UCB', 'misUCB']:
    raise ValueError

alpha_lists = []
if START_ALPHA == 0.05:
    if ACQUISITION_FUNCTION == 'UCB':
        alpha_lists = [0.05, 0.12, 1.00]
    else:
        alpha_lists = [0.05, 0.12, 1.00]
elif START_ALPHA == 0.33:
    if ACQUISITION_FUNCTION == 'UCB':
        alpha_lists = []
    else:
        alpha_lists = []
elif START_ALPHA == 0.40:
    if ACQUISITION_FUNCTION == 'UCB':
        alpha_lists = [0.00, 0.08, 0.10, 0.13, 0.14, 0.20, 0.40, 0.47, 1.00]
    else:
        alpha_lists = [0.17, 0.33, 0.40, 1.00]
elif START_ALPHA == 0.56:
    if ACQUISITION_FUNCTION == 'UCB':
        alpha_lists = [0.00, 0.07, 0.21, 0.56, 1.00]
    else:
        alpha_lists = [0.00, 0.07, 0.56]

# Selected observations.
observations_full = {
    0.00: 190.88,
    0.05: 195.78,
    0.07: 191.98,
    0.08: 198.23,
    0.10: 197.37,
    0.13: 201.87,
    0.14: 190.67,
    0.20: 197.54,
    0.21: 196.15,
    0.28: 211.09,
    0.33: 197.12,
    0.40: 154.86,
    0.47: 145.41,
    0.56: 134.17,
    0.86: 28.07,
    1.00: 64.18,
}

# Selected observations.
observations = {key: observations_full[key] for key in alpha_lists}

# Prev.
# UCB
# observations = {
#     0.00: 190.88,
#     0.05: 183.65,
#     0.10: 195.14,
#     0.15: 192.99,
#     0.21: 191.16,
#     0.26: 203.28,
#     0.28: 211.09,
#     0.30: 190.36,
#     0.33: 197.12,
#     0.40: 150.04,
#     0.56: 45.22,
#     0.69: 25.24,
#     0.86: 28.07,
#     1.00: 64.18,
# }
# misUCB
# observations = {
#     0.00: 190.88,
#     0.05: 194.67,
#     0.08: 192.50,
#     0.14: 191.71,
#     0.20: 190.88,
#     0.23: 185.71,
#     0.26: 203.28,
#     0.28: 211.09,
#     0.33: 191.78,
#     0.56: 45.22,
#     0.79: 59.68,
#     1.00: 60.80,
# }


optimizer, acquisition_function = utils_bo.get_opt_and_acq(observations,
                                                           random_state=20,
                                                           length_scale_bounds=(5e-2, 1e5),
                                                           alpha=5e-3,
                                                           # alpha=1e-5,
                                                           )

x = np.linspace(0, 1, 10000).reshape(-1, 1)

utils_bo.plot_gp_acq(optimizer,
                     acquisition_function,
                     x,
                     gp_lim=[(0, 1), (0, 220)],
                     acq_lim=[(0, 1), (None, None)],
                     )

# acq.set_ylim((np.min(utility) - 10, np.max(utility) + 10))
