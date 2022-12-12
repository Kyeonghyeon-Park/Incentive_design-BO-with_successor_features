import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import utils_bo
from utils_plot import remove_axis_margins, fill_between_3d

"""
220725 BO code update에 따른 update. 
220407 이 파일의 목적은 multi-source information에 대한 graph를 그리기 위함.
Version a와 version b간 code의 upgrade는 없고 multi-information sources의 대상이 다름. 
Graph는 
1) Previous evaluated points에 대한 BO,
2) New point (여기서는 alpha=0.4) and previous policies에 대한 evaluations와 BO,
를 담고 있음. 
궁극적인 목적은, global optimum alpha^*를 찾기 위해, 
(1로부터 나온 exact) posterior mean and variance 뿐만 아니라 
부정확하지만 유용한 정보를 담고 있을 것이라 추측되는 cheaper approximation을 
optimization에 추가적으로 이용하고자 하는 것임. 
"""

# Remove axes margins in 3D plot.
remove_axis_margins()

# Observations (or observed points).
data = torch.load('evaluate_result.tar')
_, y_data, f_data = data.values()
x = np.array([0.00, 0.30, 0.50, 0.56, 0.63, 0.80, 1.00])
n = x.size  # num_policies
k = 2  # num_bos

obs_collections = [{} for _ in range(n+1)]
for i in range(n):
    obs_collections[0][x[i]] = f_data[40, i]
    obs_collections[1][x[i]] = f_data[int(x[i] * 100), i]

# BO.
# To use functions in utils_bo, we need the shape (-1, 1) for x_axis.
x_obs_collections = []
y_obs_collections = []
f_obs_collections = []
mu_collections = []
sigma_collections = []
x_axis = np.linspace(0, 1, 10000).reshape(-1, 1)
for i in range(k):
    opt, _ = utils_bo.get_opt_and_acq(obs_collections[i])
    x_obs, f_obs = opt.get_obs()  # x_obs: (n, 1), f_obs: (n,)
    mu, sigma = opt.get_posterior(x_axis)  # mu: (10000,), sigma: (10000,)
    y_obs_collections.append(x_obs.flatten())  # Be careful!
    f_obs_collections.append(f_obs)
    mu_collections.append(mu)
    sigma_collections.append(sigma)

x_obs_collections.append(0.40*np.ones(n))
x_obs_collections.append(x)

# Build a figure.
# ax 상세 내용 https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})

# ax settings.
f = np.array(f_obs_collections)
# f = np.array(f_obs_collections[1])
axis_min = np.min(f) - 0.01
axis_max = np.max(f) + 0.01
ax.set_xlabel(r'x-axis: $\alpha_i$', fontdict={'size': 20})
ax.set_ylabel(r'y-axis: $\pi^{\alpha_j}$', fontdict={'size': 20})
ax.set_zlabel(r'z-axis: $f(\alpha, \pi)$', fontdict={'size': 20})
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([axis_min, axis_max])
ax.margins(0)
ax.view_init(azim=-45)
stem_bottom = axis_min

x_axis_collections = []
x_axis_collections.append(np.linspace(0.40, 0.40, 10000))
x_axis_collections.append(x_axis.flatten())
y_axis = copy.deepcopy(x_axis.flatten())

# Plot observations.
# stem 상세 내용 https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stem.html
for i in range(k):
    obs = [x_obs_collections[i], y_obs_collections[i], f_obs_collections[i]]
    ax.stem(*obs, bottom=stem_bottom, markerfmt='ko', label='Observations')

# Plot predicted means.
for i in range(k):
    predict = [x_axis_collections[i], y_axis, mu_collections[i]]
    ax.plot(*predict, linestyle=(0, (5, 5)), color='k', label='Prediction')

# Plot 95% confidence intervals.
for i in range(k):
    ci_lower = [x_axis_collections[i], y_axis, mu_collections[i] - 1.9600 * sigma_collections[i]]
    ci_upper = [x_axis_collections[i], y_axis, mu_collections[i] + 1.9600 * sigma_collections[i]]
    ax.plot(*ci_upper, lw=0, c='lightgray')
    c = 'lightgray' if i == 0 else 'gray'
    fill_between_3d(ax, *ci_lower, *ci_upper, mode=1, c=c, alpha=0.6)

plt.show()

print("Figure displayed.")
