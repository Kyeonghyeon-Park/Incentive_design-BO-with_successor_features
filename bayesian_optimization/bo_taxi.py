import numpy as np
from utils import utils_bo


# Observations.
observations = {
    0.00: 0.8287,
    0.30: 0.8570,
    0.48: 0.8985,
    0.53: 0.9041,
    0.54: 0.9004,
    0.56: 0.9042,
    0.58: 0.9026,
    0.59: 0.8986,
    0.70: 0.8854,
    0.87: 0.8730,
    0.93: 0.8710,
    1.00: 0.8631,
}

# Get optimizer(GP) and acquisition function.
optimizer, acquisition_function = utils_bo.get_opt_and_acq(observations,
                                                           random_state=20,
                                                           length_scale_bounds=(5e-2, 1e5),
                                                           alpha=5e-3,
                                                           )

# x-axis of graph.
x = np.linspace(0, 1, 10000).reshape(-1, 1)

# Plot GP and acquisition function.
utils_bo.plot_gp_acq(optimizer,
                     acquisition_function,
                     x,
                     # gp_lim=[(0, 1), (0.81, 0.92)],
                     # acq_lim=[(0, 1), (0.83, 0.935)],
                     )
