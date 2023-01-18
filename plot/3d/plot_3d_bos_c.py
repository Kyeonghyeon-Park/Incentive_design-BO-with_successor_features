import copy

import matplotlib.pyplot as plt
import numpy as np

from utils import utils_bo
from plot.utils_plot import remove_axis_margins, fill_between_3d

import matplotlib
matplotlib.rcParams["mathtext.fontset"] = 'cm'
# import matplotlib.font_manager as fm
# fm._rebuild()
# from matplotlib import rc
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
"""
220816 axis update. 
220812 color change. 
220804 axis update. 
220725 BO code update에 따른 update and axis update. 
220412 이 파일의 목적은 multi-source information에 대한 graph를 그리기 위함.
Version c는 version a의 변형으로, code의 upgrade보다는 alpha_i=alpha_j에 대한 BO만 그리고
나머지 점들은 수선의 발만 그리기 위함. 
Graph는 
1) Previous evaluated points에 대한 BO,
2) Previously evaluated points and policies에 대한 evaluations (점과 수선의 발만),
를 담고 있음. 
궁극적인 목적은, global optimum alpha^*를 찾기 위해, 
(1로부터 나온 exact) posterior mean and variance 뿐만 아니라 
부정확하지만 유용한 정보를 담고 있을 것이라 추측되는 cheaper approximation을 
optimization에 추가적으로 이용하고자 하는 것임. 
"""

# Remove axes margins in 3D plot.
remove_axis_margins()

# Observations (or observed points).
x = np.array([0.00, 0.30, 0.50, 0.56, 0.63, 0.80, 1.00])  # 1D array with n values(or samples)
f = np.array([0.8288, 0.9053, 0.9611, 0.9774, 0.9770, 0.9876, 0.9894,
              0.7689, 0.8555, 0.9185, 0.9369, 0.9364, 0.9484, 0.9506,
              0.7289, 0.8224, 0.8901, 0.9099, 0.9094, 0.9223, 0.9247,
              0.7169, 0.8136, 0.8831, 0.9007, 0.9011, 0.9138, 0.9180,
              0.7029, 0.8020, 0.8732, 0.8912, 0.8916, 0.9047, 0.9090,
              0.6689, 0.7738, 0.8491, 0.8682, 0.8686, 0.8824, 0.8870,
              0.6289, 0.7408, 0.8209, 0.8411, 0.8415, 0.8563, 0.8612])  # 1D array with n*n values(or samples)
n = x.size  # the number of policies
f = f.reshape(-1, n)

obs_collections = [{} for _ in range(n+1)]
for i in range(n):
    obs_collections[n][x[i]] = f[i][i]
    for j in range(n):
        obs_collections[i][x[j]] = f[i][j]

# BO.
# To use functions in utils_bo, we need the shape (-1, 1) for x_axis.
x_obs_collections = []
y_obs_collections = []
f_obs_collections = []
mu_collections = []
sigma_collections = []
x_axis = np.linspace(0, 1, 10000).reshape(-1, 1)
for i in range(n+1):
    opt, _ = utils_bo.get_opt_and_acq(obs_collections[i])
    x_obs, f_obs = opt.get_obs()  # x_obs: (n, 1), f_obs: (n,)
    mu, sigma = opt.get_posterior(x_axis)  # mu: (10000,), sigma: (10000,)
    y_obs_collections.append(x_obs.flatten())  # Be careful!
    f_obs_collections.append(f_obs)
    mu_collections.append(mu)
    sigma_collections.append(sigma)
for i in range(n):
    x_obs_collections.append(x[i]*np.ones(n))
x_obs_collections.append(x)

# Build a figure.
# ax 상세 내용 https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html
fig, ax = plt.subplots(dpi=600,figsize=(10, 10), subplot_kw={"projection": "3d"})

# ax settings.
# axis_min = np.min(f_obs_collections[n]) - 0.01
# axis_max = np.max(f_obs_collections[n]) + 0.01
axis_min = np.min(f) - 0.01
axis_max = np.max(f) + 0.01
# ax.set_xlabel(r'x-axis: $\mathbf{w}$', fontdict={'size': 20})  # 220816
# ax.set_ylabel(r'y-axis: $\pi_{\mathbf{w}}$', fontdict={'size': 20})  # 220816
# ax.set_zlabel(r'z-axis: $\tilde{\mathcal{F}}(\mathbf{w}, \pi)$', fontdict={'size': 20})  # 220816
# plt.rcParams['font.family'] = 'Helvetica'
# ax.set_xlabel(r'$x$-axis: $\mathbf{w}_e$ (for env.)', fontdict={'size': 20}, fontname='Helvetica')  # 220816
# ax.set_ylabel(r'$y$-axis: $\mathbf{w}_p$ (for policy $\pi_{\mathbf{w}_p}$)', fontdict={'size': 20}, fontname='Helvetica')  # 220816
# ax.set_zlabel(r'$z$-axis: $\tilde{\mathcal{F}}(\mathbf{w}_e, \pi_{\mathbf{w}_p})$', fontdict={'size': 20}, fontname='Helvetica')  # 220816
ax.set_xlabel(r'$x$-axis: $\mathbf{w}_e$ (for env.)', fontdict={'size': 20}, fontname='Times')  # 221230
ax.set_ylabel(r'$y$-axis: $\mathbf{w}_p$ (for policy $\pi_{\mathbf{w}_p}$)', fontdict={'size': 20}, fontname='Times')  # 221230
ax.set_zlabel(r'$z$-axis: $\tilde{\mathcal{F}}(\mathbf{w}_e, \pi_{\mathbf{w}_p})$', fontdict={'size': 20}, fontname='Times')  # 221230
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([axis_min, axis_max])
ax.margins(0)
ax.view_init(azim=-45)
stem_bottom = axis_min

x_axis_collections = []
for i in range(n):
    x_axis_collections.append(np.linspace(x[i], x[i], 10000))
x_axis_collections.append(x_axis.flatten())
y_axis = copy.deepcopy(x_axis.flatten())

# Plot observations.
# stem 상세 내용 https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stem.html
for i in range(n+1):
    obs = [x_obs_collections[i], y_obs_collections[i], f_obs_collections[i]]
    if i != n:
        # continue
        ax.stem(*obs, bottom=stem_bottom, markerfmt='ko', label='Observations', linefmt='None', basefmt='None')
    else:
        # ax.stem(*obs, bottom=stem_bottom, markerfmt='ko', label='Observations', )  # 220812
        ax.stem(*obs, bottom=stem_bottom, markerfmt='ko', label='Observations', linefmt='k:', basefmt='k-')  # 220812
# Plot predicted means.
for i in [n]:
    predict = [x_axis_collections[i], y_axis, mu_collections[i]]
    ax.plot(*predict, linestyle=(0, (5, 5)), color='k', label='Prediction')

# Plot 95% confidence intervals.
for i in [n]:
    ci_lower = [x_axis_collections[i], y_axis, mu_collections[i] - 1.9600 * sigma_collections[i]]
    ci_upper = [x_axis_collections[i], y_axis, mu_collections[i] + 1.9600 * sigma_collections[i]]
    ax.plot(*ci_upper, lw=0, c='lightgray')
    c = 'lightgray' if i != n else 'gray'
    fill_between_3d(ax, *ci_lower, *ci_upper, mode=1, c=c, alpha=0.6)

plt.show()

print("Figure displayed.")
