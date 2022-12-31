import numpy as np
from utils import utils_bo

'''
220715 Drawing GP of full BO results for AAAI. 
'''
# Observations.
# UCB
observations = {
    0.00: 190.88,
    0.05: 183.65,
    0.10: 195.14,
    0.15: 192.99,
    0.21: 191.16,
    0.26: 203.28,
    0.28: 211.09,
    0.30: 190.36,
    0.33: 197.12,
    0.40: 150.04,
    0.56: 45.22,
    0.69: 25.24,
    0.86: 28.07,
    1.00: 64.18,
}
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
                                                           )

x = np.linspace(0, 1, 10000).reshape(-1, 1)

utils_bo.plot_gp_acq(optimizer,
                     acquisition_function,
                     x,
                     gp_lim=[(0, 1), (0, 220)],
                     acq_lim=[(0, 1), (None, None)],
                     )

# acq.set_ylim((np.min(utility) - 10, np.max(utility) + 10))
