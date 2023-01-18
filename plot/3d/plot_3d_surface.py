import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

from plot.utils_plot import remove_axis_margins

"""
This file requires surface data(or evaluation result) which name is "evaluate_result.tar".
"evaluate_result.tar" is coming from "evaluate_taxi.py". 
You can set "is_all_x_value" as "True" if you want to draw a surface with all x values. 
In this case, it requires data name (e.g. evaluate_result.tar) and x_obs. 
Otherwise, it requires x, y, f, x_obs, f_obs data manually. 
"""

# Remove axes margins in 3D plot.
remove_axis_margins()

# Set data.
is_all_x_value = False  # alpha가 0.01 간격일 때 True, 7 * 7 data일 때 False
if is_all_x_value:
    data = torch.load('evaluate_result.tar')
    x, y, f = data.values()
    x, y = np.meshgrid(x, y, indexing='ij')
    f = np.array(f)
    x_obs = np.array([0, 0.3, 0.5, 0.56, 0.63, 0.8, 1])
    f_obs = np.zeros(x_obs.size)
    for i in range(x_obs.size):
        f_obs[i] = f[int(x_obs[i] * 100), i]
else:
    x = np.array([0, 0.3, 0.5, 0.56, 0.63, 0.8, 1])
    x, y = np.meshgrid(x, x, indexing='ij')
    f = np.array([0.8288, 0.9053, 0.9611, 0.9774, 0.9770, 0.9876, 0.9894,
                  0.7689, 0.8555, 0.9185, 0.9369, 0.9364, 0.9484, 0.9506,
                  0.7289, 0.8224, 0.8901, 0.9099, 0.9094, 0.9223, 0.9247,
                  0.7169, 0.8136, 0.8831, 0.9007, 0.9011, 0.9138, 0.9180,
                  0.7029, 0.8020, 0.8732, 0.8912, 0.8916, 0.9047, 0.9090,
                  0.6689, 0.7738, 0.8491, 0.8682, 0.8686, 0.8824, 0.8870,
                  0.6289, 0.7408, 0.8209, 0.8411, 0.8415, 0.8563, 0.8612])
    f = f.reshape(-1, 7)
    x_obs, f_obs = [np.zeros(7) for _ in range(2)]
    for k in range(7):
        x_obs[k] = x[k][k]
        f_obs[k] = f[k][k]

# Build a figure.
# ax 상세 내용 https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})

# ax settings.
ax.set_xlabel(r'x-axis: $\alpha_i$', fontdict={'size': 20})
ax.set_ylabel(r'y-axis: $\pi^{\alpha_j}$', fontdict={'size': 20})
ax.set_zlabel(r'z-axis: $f(\alpha, \pi)$', fontdict={'size': 20})
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([np.min(f) - 0.01, np.max(f) + 0.01])
ax.margins(0)
ax.view_init(azim=-45)

surf = ax.plot_surface(x, y, f, cmap='coolwarm', linewidth=0, alpha=0.5)

obs = [x_obs, x_obs, f_obs]
ax.stem(*obs, bottom=np.min(f) - 0.01, markerfmt='ko', label='Observations')
ax.legend(loc='upper right', fontsize=20)

plt.show()
print("Figure displayed.")
