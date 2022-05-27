import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import utils_all
from sequential_social_dilemma_games.utility_funcs import make_video_from_image_dir


def make_dirs(args):
    path = "results_ssd/" + args.setting_name
    if path is None:
        path = os.path.abspath(os.path.dirname(__file__)) + "/results_ssd" + args.setting_name
        if not os.path.exists(path):
            os.makedirs(path)
    image_path = os.path.join(path, "frames/")
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    video_path = os.path.join(path, "videos/")
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    saved_path = os.path.join(path, "saved/")
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    return path, image_path, video_path, saved_path


def make_video(is_train, epi_num, fps, video_path, image_path):
    if is_train:
        video_name = "trajectory_train_episode_" + str(epi_num)
    else:
        video_name = "trajectory_test_episode_" + str(epi_num)
    make_video_from_image_dir(video_path, image_path, fps=fps, video_name=video_name)
    # Clean up images.
    for single_image_name in os.listdir(image_path):
        single_image_path = os.path.join(image_path, single_image_name)
        try:
            if os.path.isfile(single_image_path) or os.path.islink(single_image_path):
                os.unlink(single_image_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (single_image_path, e))


def save_data(args, env, episode_trained, decayed_eps, time_trained, outcomes, networks, path, name):
    """
    Save several data.
    """
    params = utils_all.get_networks_params(args, networks)
    actor_params, actor_opt_params, critic_params, critic_opt_params, psi_params, psi_opt_params = params

    torch.save({
        'args': args,
        'env': env,
        'episode_trained': episode_trained,
        'time_trained': time_trained,
        'decayed_eps': decayed_eps,
        'outcomes': outcomes,
        'actor': actor_params,
        'actor_opt': actor_opt_params,
        'psi': psi_params,
        'psi_opt': psi_opt_params,
        'critic': critic_params,
        'critic_opt': critic_opt_params,
    }, path + name)


def draw_or_save_plt(col_rews, col_rews_test, objs, objs_test, i=0, mode='draw', filename=''):
    def get_figure_components(inputs, i):
        rew = inputs[:i + 1]
        moving_avg_len = 20
        means, stds = [np.zeros(rew.size) for _ in range(2)]
        for j in range(rew.size):
            if j + 1 < moving_avg_len:
                rew_part = rew[:j + 1]
            else:
                rew_part = rew[j - moving_avg_len + 1:j + 1]
            means[j] = np.mean(rew_part)
            stds[j] = np.std(rew_part)
        return rew, means, stds

    x_axis = np.arange(i+1)
    y_axis_lim_rew = np.max(col_rews[:i + 1]) + 100
    y_axis_lim_rew_test = np.max(col_rews_test[:i + 1]) + 100
    y_axis_lim_obj = np.max(objs[:i + 1]) + 1
    y_axis_lim_obj_test = np.max(objs_test[:i + 1]) + 1
    plt.figure(figsize=(16, 14))

    plt.subplot(2, 2, 1)
    outs, means, stds = get_figure_components(col_rews, i)
    plt.plot(x_axis, means, label='Moving avg. of collective rewards', color=(0, 1, 0))
    plt.fill_between(x_axis, means - stds, means + stds, color=(0.85, 1, 0.85))
    plt.scatter(x_axis, outs, label='Collective rewards')
    plt.ylim([0, y_axis_lim_rew])
    plt.xlabel('Episodes (1000 steps per episode)', fontsize=20)
    plt.ylabel('Collective rewards per episode', fontsize=20)
    plt.title('Collective rewards (train)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=14)
    plt.grid()

    plt.subplot(2, 2, 2)
    outs, means, stds = get_figure_components(col_rews_test, i)
    plt.plot(x_axis, means, label='Moving avg. of collective rewards', color=(0, 1, 0))
    plt.fill_between(x_axis, means - stds, means + stds, color=(0.85, 1, 0.85))
    plt.scatter(x_axis, outs, label='Collective rewards')
    plt.ylim([0, y_axis_lim_rew_test])
    plt.xlabel('Episodes (1000 steps per episode)', fontsize=20)
    plt.ylabel('Collective rewards per episode', fontsize=20)
    plt.title('Collective rewards (test)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=14)
    plt.grid()

    plt.subplot(2, 2, 3)
    outs, means, stds = get_figure_components(objs, i)
    plt.plot(x_axis, means, label='Moving avg. of designer objectives', color=(0, 1, 0))
    plt.fill_between(x_axis, means - stds, means + stds, color=(0.85, 1, 0.85))
    plt.scatter(x_axis, outs, label='Designer objectives')
    plt.ylim([0, y_axis_lim_obj])
    plt.xlabel('Episodes (1000 steps per episode)', fontsize=20)
    plt.ylabel('Designer objectives per episode', fontsize=20)
    plt.title('Designer objectives (train)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=14)
    plt.grid()

    plt.subplot(2, 2, 4)
    outs, means, stds = get_figure_components(objs_test, i)
    plt.plot(x_axis, means, label='Moving avg. of designer objectives', color=(0, 1, 0))
    plt.fill_between(x_axis, means - stds, means + stds, color=(0.85, 1, 0.85))
    plt.scatter(x_axis, outs, label='Designer objectives')
    plt.ylim([0, y_axis_lim_obj_test])
    plt.xlabel('Episodes (1000 steps per episode)', fontsize=20)
    plt.ylabel('Designer objectives per episode', fontsize=20)
    plt.title('Designer objectives (test)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=14)
    plt.grid()

    if mode == 'draw':
        plt.show()
    elif mode == 'save':
        plt.savefig(filename)
    else:
        raise ValueError


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


def get_plt_test(outcomes):
    """
    Get plt of test results.

    Examples
    ----------
    get_plt_test(outcomes)

    Parameters
    ----------
    outcomes: dict
        ex. {'rew': ndarray: (num_tests, num_net),
             'pen': ndarray: (num_tests, num_net),
             'inc': ndarray: (num_tests, num_net),
             'obj': ndarray: (num_tests, num_net),
             'x_axis':  ndarray: (num_net,),
             }
    """
    def get_mean_and_std(inputs):
        mean = np.mean(inputs, axis=0)
        std = np.std(inputs, axis=0)
        return mean, std

    x_axis = outcomes['x_axis']
    plt.figure(figsize=(8, 14))

    plt.subplot(2, 1, 1)
    mean, std = get_mean_and_std(outcomes['rew'])
    plt.plot(x_axis, mean, label='Mean of collective rewards', color=(0, 1, 0))
    plt.fill_between(x_axis, mean - std, mean + std, color=(0.85, 1, 0.85))
    # plt.scatter(x_axis, outs, label='Collective rewards')
    # plt.ylim([0, y_axis_lim_rew])
    plt.xlabel('Episodes (100 steps per episode)', fontsize=20)
    plt.ylabel('Collective rewards per episode', fontsize=20)
    plt.title('Collective rewards (test)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=14)
    plt.grid()

    plt.subplot(2, 1, 2)
    mean, std = get_mean_and_std(outcomes['obj'])
    plt.plot(x_axis, mean, label='Mean of the designer objective', color=(0, 1, 0))
    plt.fill_between(x_axis, mean - std, mean + std, color=(0.85, 1, 0.85))
    # plt.scatter(x_axis, outs, label='Collective rewards')
    # plt.ylim([0, y_axis_lim_rew])
    plt.xlabel('Episodes (100 steps per episode)', fontsize=20)
    plt.ylabel("The designer's objective per episode", fontsize=20)
    plt.title("The designer's objective (test)", fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=14)
    plt.grid()

    plt.show()
