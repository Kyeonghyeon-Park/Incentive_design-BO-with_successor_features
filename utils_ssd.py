import matplotlib.pyplot as plt
import numpy as np
import torch


def get_plt_final(outcomes_l, outcomes_r, is_3000=False):
    """
    Get the figure of two final outcomes.
    This function uses the evaluation results.
    If you want to draw the outcome per 3000 episodes, you have to set is_3000=True.

    Examples
    ----------
    # Convergence of the lower-level
    dict_l = torch.load("./results_ssd_final/alpha=0.00/outcomes.tar")
    dict_r = torch.load("./results_ssd_final/alpha=1.00/outcomes.tar")
    outcomes_l = dict_l["obj_full"]
    outcomes_r = dict_r["obj_full"]
    utils_ssd.get_plt_final(outcomes_l, outcomes_r, is_3000=True)

    # Efficiency of the transfer: visual comparison
    dict_l = torch.load("./results_ssd_final/alpha=0.33 using alpha=0.50/outcomes.tar")
    dict_r = torch.load("./results_ssd_final/alpha=0.33/outcomes.tar")
    outcomes_l = dict_l["obj_full"]
    outcomes_r = dict_r["obj_full"]
    utils_ssd.get_plt_final(outcomes_l, outcomes_r, is_3000=True)

    Parameters
    ----------
    outcomes_l
        Outcomes which will be shown in the left figure
    outcomes_r
        Outcomes which will be shown in the right figure
    is_3000 : boolean
        True if we want to draw the outcome per 3000 episodes
    """
    def get_status(inputs):
        means = np.mean(inputs, axis=1)
        stds = np.std(inputs, axis=1)
        return means, stds

    if is_3000:
        outcomes_l = outcomes_l[2::3, :]
        outcomes_r = outcomes_r[2::3, :]
        x = 3000 * np.arange(1, 11)
        x_lim = [3000, 30000]
    else:
        x = 1000 * np.arange(1, 31)
        x_lim = [None, None]

    y_lim = [None, None]
    # y_lim = [None, 265]  # Convergence of the lower-level
    # y_lim = [0, 375]  # Efficiency of the transfer: visual comparison

    plt.figure(figsize=(30, 8))

    plt.subplot(1, 2, 1)
    means, stds = get_status(outcomes_l)
    plt.plot(x, means, label="Mean objective value", color=(0, 0, 1))
    plt.fill_between(x, means - stds, means + stds, color=(0.75, 0.75, 1))
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Value", fontsize=24)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.subplot(1, 2, 2)
    means, stds = get_status(outcomes_r)
    plt.plot(x, means, label="Mean objective value", color=(0, 0, 1))
    plt.fill_between(x, means - stds, means + stds, color=(0.75, 0.75, 1))
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Value", fontsize=24)
    plt.ylim(y_lim)
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.show()


def get_plt_final_aggregate(outcomes_l, outcomes_r, is_3000=False):
    """
    Get the figure of two final outcomes.
    This function uses the evaluation results.
    If you want to draw the outcome per 3000 episodes, you have to set is_3000=True.
    Unlike the previous function(get_plt_final), this figure put two outcomes into one figure.

    Examples
    ----------
    # Efficiency of the transfer: visual comparison
    dict_l = torch.load("./results_ssd_final/alpha=0.33 using alpha=0.50/outcomes.tar")
    dict_r = torch.load("./results_ssd_final/alpha=0.33/outcomes.tar")
    outcomes_l = dict_l["obj_full"]
    outcomes_r = dict_r["obj_full"]
    utils_ssd.get_plt_final_aggregate(outcomes_l, outcomes_r, is_3000=True)

    Parameters
    ----------
    outcomes_l
        Outcomes which will be shown in the left figure
    outcomes_r
        Outcomes which will be shown in the right figure
    is_3000 : boolean
        True if we want to draw the outcome per 3000 episodes
    """
    def get_status(inputs):
        means = np.mean(inputs, axis=1)
        stds = np.std(inputs, axis=1)
        return means, stds

    if is_3000:
        outcomes_l = outcomes_l[2::3, :]
        outcomes_r = outcomes_r[2::3, :]
        x = 3000 * np.arange(1, 11)
        x_lim = [3000, 30000]
    else:
        x = 1000 * np.arange(1, 31)
        x_lim = [None, None]

    # y_lim = [None, None]
    # y_lim = [None, 265]  # Convergence of the lower-level
    y_lim = [0, 375]  # Efficiency of the transfer: visual comparison

    plt.figure(figsize=(15, 8))

    means_l, stds_l = get_status(outcomes_l)
    plt.plot(x, means_l, label="Mean objective value (transfer)", alpha=0.5, color=(0, 0, 1))
    plt.fill_between(x, means_l - stds_l, means_l + stds_l, color=(0.75, 0.75, 1))
    means_r, stds_r = get_status(outcomes_r)
    plt.plot(x, means_r, label="Mean objective value (non-transfer)", alpha=0.5, color=(1, 0, 0))
    plt.fill_between(x, means_r - stds_r, means_r + stds_r, alpha=0.5, color=(1, 0.75, 0.75))

    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Value", fontsize=24)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.show()


def get_plt_cumulative_SKLD(skl_l, skl_r, is_3000=False):
    """
    Get the figure of two cumulative SKLDs (sum of KL divergences).
    This function uses the evaluation results.
    SKLD of the transfer scenario will be shown in the left.
    SKLD is divided by the maximum value of SKLDs.

    Examples
    ----------
    dict_l = torch.load("./results_ssd_final/alpha=0.33 using alpha=0.50/outcomes.tar")
    dict_r = torch.load("./results_ssd_final/alpha=0.33/outcomes.tar")
    outcomes_l = dict_l["skl_mean"]
    outcomes_r = dict_r["skl_mean"]
    utils_ssd.get_plt_cumulative_SKLD(outcomes_l, outcomes_r, is_3000=True)

    Parameters
    ----------
    skl_l : numpy.ndarray
        Array of SKLDs (size : 30)
        Each SKLD is the mean value for 50 tests
    skl_r : numpy.ndarray
    """
    def get_CSKLD(skl):
        cskld = np.zeros(len(skl))
        for i in range(len(skl)):
            cskld[i] = skl[i] if i == 0 else skl[i] + cskld[i - 1]
        return cskld

    if is_3000:
        skl_l = skl_l[2::3]
        skl_r = skl_r[2::3]
        x = 3000 * np.arange(1, 11)
        x_lim = [3000, 30000]
    else:
        x = 1000 * np.arange(1, 31)
        x_lim = [None, None]

    skl_l = skl_l / max(skl_l)
    skl_r = skl_r / max(skl_r)

    y_lim = [None, None]

    cskld_l = get_CSKLD(skl_l)
    cskld_r = get_CSKLD(skl_r)

    plt.figure(figsize=(16, 8))
    plt.plot(x, cskld_l, label="Cumulative SKLD (transfer)", color=(0, 0, 1))
    plt.plot(x, cskld_r, label="Cumulative SKLD (non-transfer)", color=(1, 0, 0))
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Cumulative SKLD", fontsize=24)
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.show()


def get_plt_cumulative_SKLD_multiseeds(skld_l_list, skld_r_list, is_3000=False, is_normalized=False):
    """
    Get the figure of two cumulative SKLDs(sum of KL divergences) for multiple random seeds.
    This function uses the evaluation results.
    SKLD of the transfer scenario will be shown in the left.
    If is_normalized is True, SKLD will be divided by the maximum value of SKLDs.

    Examples
    ----------
    skld_l_list = []
    skld_r_list = []
    skld_l_list_path = ["./results_ssd_final/alpha=0.33 using alpha=0.50 (5 seeds)/seed 1278 (original)/outcomes.tar",
                        "./results_ssd_final/alpha=0.33 using alpha=0.50 (5 seeds)/seed 1279/outcomes.tar",
                        ]
    skld_r_list_path = ["./results_ssd_final/alpha=0.33 (5 seeds)/seed 1267 (original)/outcomes.tar",
                        # "./results_ssd_final/alpha=0.33 (5 seeds)/seed 1268 (not converged to 200)/outcomes.tar",
                        "./results_ssd_final/alpha=0.33 (5 seeds)/seed 1269/outcomes.tar",
                        "./results_ssd_final/alpha=0.33 (5 seeds)/seed 1270/outcomes.tar",
                        # "./results_ssd_final/alpha=0.33 (5 seeds)/seed 1271 (not converged to 200)/outcomes.tar",
                        ]

    for i in skld_l_list_path:
        dict_l = torch.load(i)
        outcomes_l = dict_l["skl_mean"]
        skld_l_list.append(outcomes_l)
    for i in skld_r_list_path:
        dict_r = torch.load(i)
        outcomes_r = dict_r["skl_mean"]
        skld_r_list.append(outcomes_r)
    utils_ssd.get_plt_cumulative_SKLD_multiseeds(skld_l_list, skld_r_list, is_3000=True, is_normalized=True)

    Parameters
    ----------
    skld_l_list : List
        Each element of list is the numpy array of SKLDs (size : 30).
        Each SKLD is the mean value for 50 tests.
    skld_r_list : List
    is_3000 : bool
    is_normalized : bool
    """
    # num_seeds_l = len(skld_l_list)
    # num_seeds_r = len(skld_r_list)
    # num_episodes = len(skld_l_list[0][2::3]) if is_3000 else len(skld_l_list[0])

    # cskld_l = np.zeros([num_seeds_l, num_episodes])
    # cskld_r = np.zeros([num_seeds_r, num_episodes])

    def get_cskld(skld_list, is_3000, is_normalized):
        num_seeds = len(skld_list)
        num_episodes = len(skld_list[0][2::3]) if is_3000 else len(skld_list[0])
        cskld = np.zeros([num_seeds, num_episodes])

        for i in range(num_seeds):
            skld = skld_list[i][2::3] if is_3000 else skld_list[i]
            if is_normalized:
                skld = skld / max(skld)
            for j in range(num_episodes):
                cskld[i, j] = skld[j] + cskld[i, j - 1] if j != 0 else skld[j]

        return cskld

    x = 3000 * np.arange(1, 11) if is_3000 else 1000 * np.arange(1, 31)
    x_lim = [3000, 30000] if is_3000 else [None, None]
    y_lim = [None, None]

    cskld_l = get_cskld(skld_l_list, is_3000=is_3000, is_normalized=is_normalized)
    cskld_r = get_cskld(skld_r_list, is_3000=is_3000, is_normalized=is_normalized)

    means_l = np.mean(cskld_l, axis=0)
    stds_l = np.std(cskld_l, axis=0)
    means_r = np.mean(cskld_r, axis=0)
    stds_r = np.std(cskld_r, axis=0)

    plt.figure(figsize=(16, 8))
    plt.plot(x, means_l, label="Cumulative SKLD (transfer)", color=(0, 0, 1))
    plt.fill_between(x, means_l - stds_l, means_l + stds_l, color=(0.75, 0.75, 1))
    plt.plot(x, means_r, label="Cumulative SKLD (non-transfer)", alpha=0.5, color=(1, 0, 0))
    plt.fill_between(x, means_r - stds_r, means_r + stds_r, alpha=0.5, color=(1, 0.75, 0.75))

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Cumulative SKLD", fontsize=24)
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.show()

    # def get_CSKLD(skl):
    #     cskld = np.zeros(len(skl))
    #     for i in range(len(skl)):
    #         cskld[i] = skl[i] if i == 0 else skl[i] + cskld[i - 1]
    #     return cskld
    #
    # if is_3000:
    #     skl_l = skl_l[2::3]
    #     skl_r = skl_r[2::3]
    #     x = 3000 * np.arange(1, 11)
    #     x_lim = [3000, 30000]
    # else:
    #     x = 1000 * np.arange(1, 31)
    #     x_lim = [None, None]
    #
    # skl_l = skl_l / max(skl_l)
    # skl_r = skl_r / max(skl_r)
    #
    # y_lim = [None, None]
    #
    # cskld_l = get_CSKLD(skl_l)
    # cskld_r = get_CSKLD(skl_r)

    # plt.figure(figsize=(16, 8))
    # plt.plot(x, cskld_l, label="Cumulative SKLD (transfer)", color=(0, 0, 1))
    # plt.plot(x, cskld_r, label="Cumulative SKLD (non-transfer)", color=(1, 0, 0))
    # plt.xlim(x_lim)
    # plt.ylim(y_lim)
    # plt.xlabel("Episodes", fontsize=24)
    # plt.ylabel("Cumulative SKLD", fontsize=24)
    # plt.legend(loc='lower right', fontsize=20)
    # plt.tick_params(axis='both', labelsize=20)
    # plt.grid()
    #
    # plt.show()
