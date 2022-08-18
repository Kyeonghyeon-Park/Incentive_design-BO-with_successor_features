import numpy as np
from utils import utils_bo
"""
This file is for getting BO results given observations. 
The form of observations should be:
observations = {alpha: f(alpha)}.

ex.
observations = {
    0.00: 10,
    0.50: 15,
    1.00: 10
}
"""
# Observations.
observations = {
    0.00: 10,
    0.50: 15,
    1.00: 10
}

optimizer, acquisition_function = utils_bo.get_opt_and_acq(observations,
                                                           random_state=20,
                                                           length_scale_bounds=(5e-2, 1e5),
                                                           )

x = np.linspace(0, 1, 10000).reshape(-1, 1)

utils_bo.plot_gp_acq(optimizer,
                     acquisition_function,
                     x,
                     gp_lim=[(0, 1), (None, None)],
                     acq_lim=[(0, 1), (None, None)],
                     )

