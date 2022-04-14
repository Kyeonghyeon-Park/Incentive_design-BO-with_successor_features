import copy

import matplotlib.pyplot as plt
import numpy as np

from utils import utils_bo
from utils_plot import remove_axis_margins, fill_between_3d

# Remove axes margins in 3D plot.
remove_axis_margins()

# Observations (or observed points).
observations = {0.00: 0.8287,
                0.30: 0.8570,
                0.50: 0.8961,
                0.56: 0.9025,
                0.63: 0.8952,
                0.80: 0.8842,
                1.00: 0.8631}

# BO.
# To use functions in utils_bo, we need the shape (-1, 1) for x.
optimizer, acquisition_function = utils_bo.get_opt_and_acq(observations)
x = np.linspace(0, 1, 10000).reshape(-1, 1)
x_obs, f_obs = utils_bo.get_obs(optimizer)
mu, sigma = utils_bo.get_posterior(optimizer, x_obs, f_obs, x)

# Build a figure.
# ax 상세 내용 https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})

# ax settings.
ax.set_xlabel(r'x-axis: $\alpha_i$', fontdict={'size': 20})
ax.set_ylabel(r'y-axis: $\pi^{\alpha_j}$', fontdict={'size': 20})
ax.set_zlabel(r'z-axis: $f(\alpha, \pi)$', fontdict={'size': 20})
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([min(f_obs) - 0.01, max(f_obs) + 0.01])
ax.margins(0)
ax.view_init(azim=-45)
stem_bottom = min(f_obs) - 0.01
x = x.flatten()
y = copy.deepcopy(x)

# Plot observations.
# stem 상세 내용 https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stem.html
obs = [x_obs.flatten(), x_obs.flatten(), f_obs]
ax.stem(*obs, bottom=stem_bottom, markerfmt='ko', label='Observations')

# Plot predicted means.
predict = [x, y, mu]
ax.plot(*predict, linestyle=(0, (5, 5)), color='k', label='Prediction')

# Plot 95% confidence intervals.
ci_lower = [x, y, mu - 1.9600 * sigma]
ci_upper = [x, y, mu + 1.9600 * sigma]
ax.plot(*ci_upper, lw=0, c='lightgray')
fill_between_3d(ax, *ci_lower, *ci_upper, mode=1, c='lightgray', alpha=0.6)

# Plot approximated gradient at alpha=test_point.
test_point = 0.3
line_range = 0.1
grad = -0.1728
grad_color = 'm'
f_value = np.take(f_obs, np.where(x_obs.flatten() == test_point)).item()
x_grad = [test_point - line_range, test_point + line_range]
y_grad = test_point * np.ones(len(x_grad))
f_grad = [f_value - grad * line_range, f_value + grad * line_range]
grad_line = [x_grad, y_grad, f_grad]
ax.stem(*grad_line, bottom=stem_bottom, markerfmt='')
ax.plot(*grad_line, color=grad_color, label='Approx. grad.',)

# Display legends.
ax.legend(loc='upper right', fontsize=20)

plt.show()

print("Figure displayed.")
