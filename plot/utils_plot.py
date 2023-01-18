from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axis3d import Axis
import numpy as np
import torch

from utils.utils_all import tile_ravel_multi_index

def remove_axis_margins():
    """
    Patch start (removing axes margins in 3D plot).
    https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
    """
    if not hasattr(Axis, "_get_coord_info_old"):
        def _get_coord_info_new(self, renderer):
            mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
            mins += deltas / 4
            maxs -= deltas / 4
            return mins, maxs, centers, deltas, tc, highs

        Axis._get_coord_info_old = Axis._get_coord_info
        Axis._get_coord_info = _get_coord_info_new


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
