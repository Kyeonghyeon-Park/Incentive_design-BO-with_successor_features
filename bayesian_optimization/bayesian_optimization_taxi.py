import numpy as np
import utils_bo

# Observations.
observations = {0.00: 0.8287,
                0.30: 0.8570,
                0.50: 0.8961,
                0.56: 0.9025,
                0.63: 0.8952,
                0.80: 0.8842,
                1.00: 0.8631}

optimizer, acquisition_function = utils_bo.get_opt_and_acq(observations)

x = np.linspace(0, 1, 10000).reshape(-1, 1)

utils_bo.plot_gp(optimizer,
                 acquisition_function,
                 x,
                 is_only_acq=True,
                 mode='taxi')
