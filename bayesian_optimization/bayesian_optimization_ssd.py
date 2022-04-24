import numpy as np
from utils import utils_bo

# Observations.
observations = {
    0.00: 173.94,
    0.10: 193.40,
    0.33: 194.50,
    0.43: 215.00,
    0.50: 278.06,
    0.60: 168.14,
    0.80: 40.62,
    1.00: 64.74
}

optimizer, acquisition_function = utils_bo.get_opt_and_acq(observations,
                                                           random_state=20,
                                                           length_scale_bounds=(5e-2, 1e5),
                                                           )

x = np.linspace(0, 1, 10000).reshape(-1, 1)

utils_bo.plot_gp(optimizer,
                 acquisition_function,
                 x,
                 gp_lim=[(0, 1), (None, None)],
                 acq_lim=[(0, 1), (None, None)],
                 )

# acq.set_ylim((np.min(utility) - 10, np.max(utility) + 10))