import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import utils_bo


# Patch start (removing axes margins in 3D plot).
# https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new
# Patch end.


def fill_between_3d(ax, x1, y1, z1, x2, y2, z2, mode=1, c='steelblue', alpha=0.6):
    """
    From https://github.com/artmenlope/matplotlib-fill_between-in-3D

    Function similar to the matplotlib.pyplot.fill_between function but
    for 3D plots.

    input:

        ax -> The axis where the function will plot.

        x1 -> 1D array. x coordinates of the first line.
        y1 -> 1D array. y coordinates of the first line.
        z1 -> 1D array. z coordinates of the first line.

        x2 -> 1D array. x coordinates of the second line.
        y2 -> 1D array. y coordinates of the second line.
        z2 -> 1D array. z coordinates of the second line.

    modes:
        mode = 1 -> Fill between the lines using the shortest distance between
                    both. Makes a lot of single trapezoids in the diagonals
                    between lines and then adds them into a single collection.

        mode = 2 -> Uses the lines as the edges of one only 3d polygon.

    Other parameters (for matplotlib):

        c -> the color of the polygon collection.
        alpha -> transparency of the polygon collection.

    """
    if mode == 1:

        for i in range(len(x1) - 1):
            verts = [(x1[i], y1[i], z1[i]), (x1[i + 1], y1[i + 1], z1[i + 1])] + \
                    [(x2[i + 1], y2[i + 1], z2[i + 1]), (x2[i], y2[i], z2[i])]

            ax.add_collection3d(Poly3DCollection([verts],
                                                 alpha=alpha,
                                                 linewidths=0,
                                                 color=c))

    if mode == 2:
        verts = [(x1[i], y1[i], z1[i]) for i in range(len(x1))] + \
                [(x2[i], y2[i], z2[i]) for i in range(len(x2))]

        ax.add_collection3d(Poly3DCollection([verts], alpha=alpha, color=c, label="test"))


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
ax.set_xlabel(r'x-axis: $\alpha$', fontdict={'size': 20})
ax.set_ylabel(r'y-axis: $\pi^\alpha$', fontdict={'size': 20})
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
