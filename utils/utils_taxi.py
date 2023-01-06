import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import torch
import torch.distributions as distributions

from taxi import TaxiEnv
from . import utils_all


def make_dirs(args):
    """
    Make directories of the current setting.

    Parameters
    ----------
    args: argparse.Namespace

    Returns
    -------
    path: str
    saved_path: str
    """
    path = "results_taxi/" + args.setting_name
    saved_path = os.path.join(path, "saved/")
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    return path, saved_path


def save_data(args, env, episode_trained, decayed_eps, time_start, outcomes, outcomes_t, skld, networks, path, name):
    """
    Save several data including training-and-test results.

    Parameters
    ----------
    args: argparse.Namespace
    env: TaxiEnv
    episode_trained: int
    decayed_eps: float
    time_start: float
    outcomes: list
        outcomes = [orr, osc, avg_rew, obj]
    outcomes_t: list
        outcomes_t = [orr_t, osc_t, avg_rew_t, obj_t]
    skld: numpy.ndarray
        Size: (args.num_episodes + 1, )
        ex. skld[i + 1] = utils_taxi.calculate_kl_divergence(networks, networks_final) if args.mode_kl_divergence else 0
    networks: Networks
    path: str
        File path.
    name: str
        File name.
    """
    params = utils_all.get_networks_params(args, networks)
    actor_params, actor_opt_params, critic_params, critic_opt_params, psi_params, psi_opt_params = params

    torch.save({
        'args': args,
        'env': env,
        'episode_trained': episode_trained,
        'time_trained': time.time() - time_start,
        'decayed_eps': decayed_eps,
        'outcomes': outcomes,
        'outcomes_t': outcomes_t,
        'skld': skld,
        'actor': actor_params,
        'actor_opt': actor_opt_params,
        'psi': psi_params,
        'psi_opt': psi_opt_params,
        'critic': critic_params,
        'critic_opt': critic_opt_params,
    }, path + name)


def get_one_hot_obs(obs, env):
    """
    Get one-hot encoded version of obs.
    obs is a list of ind_obs.

    Parameters
    ----------
    obs: List
        ex. [np.array([1, 0], ...)]
    env: taxi.TaxiEnv

    Returns
    -------
    one_hot_obs: numpy.ndarray
    """
    len_obs = len(obs)
    num_grids = env.num_grids
    epi_length = env.episode_length
    one_hot_obs = np.zeros([len(obs), num_grids, epi_length+1])

    for i in range(len_obs):
        ind_obs = obs[i]
        loc = ind_obs[0]
        time = ind_obs[1] if ind_obs[1] <= epi_length else epi_length
        one_hot_obs[i, loc, time] = 1

    return one_hot_obs


def make_vars(n, mode):
    """
    Return n list or dict.
    Not used.

    Parameters
    ----------
    n: int
    mode: str
        It should be 'list' or 'dict'.
    """
    for _ in range(n):
        if mode == 'list':
            yield []
        elif mode == 'dict':
            yield {}
        else:
            raise NotImplementedError("Possible options of mode are list and dict.")

def get_masked_categorical_dists(action_probs, masks):
    probs = torch.mul(action_probs, masks)
    action_dists = distributions.Categorical(probs)
    return action_dists





def print_status(args, i, orr, osc, avg_rew, obj, time_start, is_train=True):
    """
    This function prints major status for each episode.

    Parameters
    ----------
    args: argparse.Namespace
    i: int
        The number which represents the current episode.
    orr: numpy.ndarray
        Array of order response rate.
        Size : (1, num_episodes) or (num_tests, num_episodes)
    osc: numpy.ndarray
        Array of overall service charge.
        Size : (1, num_episodes) or (num_tests, num_episodes)
    avg_rew: numpy.ndarray
        Array of average rewards of all agents.
        Size : (1, num_episodes) or (num_tests, num_episodes)
    obj: numpy.ndarray
        Array of objective values.
        Size : (1, num_episodes) or (num_tests, num_episodes)
    time_start: float
        The time when the training starts.
    is_train: boolean
        True if train
    """
    update = "O" if (i + 1) % args.update_freq == 0 and is_train else "X"
    mode = "  Train  " if is_train else "Test(avg)"
    print(f"Process : {i + 1}/{args.num_episodes}, "
          f"Time : {time.time() - time_start:.2f}, "
          f"Update : {update}"
          ) if is_train else None
    print(f"    [{mode}]  "
          f"ORR : {np.mean(orr[:, i]):5.4f}, "
          f"OSC : {np.mean(osc[:, i]):5.4f}, "
          f"Avg. reward : {np.mean(avg_rew[:, i]):5.4f}, "
          f"Obj : {np.mean(obj[:, i]):5.4f}"
          )


def print_updated_q(networks):
    """
    Print current trained Q values for each location, action, and mean_action(0, 0.1, ..., 1).
    """
    torch.set_printoptions(linewidth=200, sci_mode=False)
    for loc in [1, 2]:
        for t in [0]:
            print("Q at (#", loc, ", ", t, ")")
            q_all = torch.zeros([11, networks.action_size])
            for i in range(11):
                m_act = i / 10
                with torch.no_grad():
                    ind_obs = np.array([loc, t])
                    obs_tensor = torch.tensor(get_one_hot_obs([ind_obs], networks.env), dtype=torch.float)
                    obs_tensor = obs_tensor.view(-1, networks.observation_size)
                    # obs_tensor = get_one_hot_obs_tensor(ind_obs, networks.observation_size)

                    m_act_tensor = torch.tensor(m_act, dtype=torch.float)
                    m_act_tensor = m_act_tensor.view(-1, networks.mean_action_size)

                    psi = networks.psi(obs_tensor, m_act_tensor)
                    q = torch.tensordot(psi, networks.w, dims=([2], [0]))
                    q_all[i] = q[0]
            print(q_all.transpose(1, 0))


def print_action_dist(networks):
    for loc in [1, 2]:
        for t in [0]:
            ind_obs = np.array([loc, t])
            obs_tensor = torch.tensor(get_one_hot_obs([ind_obs], networks.env), dtype=torch.float)
            obs_tensor = obs_tensor.view(-1, networks.observation_size)
            # obs_tensor = get_one_hot_obs_tensor(ind_obs, networks.observation_size)
            obs_mask = networks.get_masks([ind_obs])

            act_probs = networks.actor(obs_tensor)
            dists = get_masked_categorical_dists(act_probs, obs_mask)
            act_probs = dists.probs

            act_probs = act_probs.detach()
            print("Action distribution at (#", loc, ", ", t, ") : ", act_probs)


def calculate_kl_divergence(networks, networks_final):
    """
    Get kl divergence for the locations [1, 0] and [2, 0].

    Parameters
    ----------
    networks: Networks
    networks_final: Networks

    Returns
    -------
    kld: float
    """
    kld = 0
    for ind_obs in [np.array([1, 0]), np.array([2, 0])]:

        obs_tensor = torch.tensor(get_one_hot_obs([ind_obs], networks.env), dtype=torch.float)
        obs_tensor = obs_tensor.view(-1, networks.observation_size)
        # obs_tensor = get_one_hot_obs_tensor(ind_obs, networks.observation_size)
        obs_mask = networks.get_masks([ind_obs])  # obs_mask: tensor([[1., 1., 0., 1.]]) for ind_obs: [1 0]

        act_probs = networks.actor_target(obs_tensor)
        dists = get_masked_categorical_dists(act_probs, obs_mask)
        act_probs = dists.probs  # ex. act_probs: tensor([[0.3911, 0.4301, 0.0000, 0.1788]])

        act_probs_final = networks_final.actor_target(obs_tensor)
        dists_final = get_masked_categorical_dists(act_probs_final, obs_mask)
        act_probs_final = dists_final.probs  # ex. act_probs_final: tensor([[0.1724, 0.0000, 0.0000, 0.8276]])

        obs_mask = obs_mask.squeeze()
        act_probs = act_probs.squeeze()
        act_probs_final = act_probs_final.squeeze()

        for i in range(4):
            if obs_mask[i] != 0:
                p = act_probs_final[i].item()
                p = max(p, 1e-12)
                q = act_probs[i].item()
                q = max(q, 1e-12)
                kld -= p * np.log(q / p)

    return kld


def get_plt(outcomes, outcomes_t, i, mode="draw", filename=""):
    """
    Get the figure during the training.

    Parameters
    ----------
    outcomes
    outcomes_t
    i
    mode
    filename
    """
    def get_status(inputs, i):
        inputs = inputs[:, :i+1]
        means = np.mean(inputs, axis=0)
        stds = np.std(inputs, axis=0)
        return means, stds

    x = np.arange(i+1)
    y_lim_rew = 6
    y_lim_others = 1.1
    plt.figure(figsize=(16, 14))

    plt.subplot(2, 2, 1)
    means, stds = get_status(outcomes[2], i)
    plt.plot(x, means, label="Avg reward", color=(0, 0, 1))
    plt.fill_between(x, means - stds, means + stds, color=(0.75, 0.75, 1))
    plt.ylim([0, y_lim_rew])
    plt.xlabel("Episodes", fontsize=20)
    plt.ylabel("Average reward of agents", fontsize=20)
    plt.title("Average rewards (train)", fontdict={"fontsize": 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.grid()

    plt.subplot(2, 2, 2)
    means, stds = get_status(outcomes_t[2], i)
    plt.plot(x, means, label="Avg reward", color=(0, 0, 1))
    plt.fill_between(x, means - stds, means + stds, color=(0.75, 0.75, 1))
    plt.ylim([0, y_lim_rew])
    plt.xlabel("Episodes", fontsize=20)
    plt.xlabel("Episodes", fontsize=20)
    plt.title("Average rewards (test)", fontdict={"fontsize": 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.grid()

    idxs = [0, 1, 3]
    labels = ["ORR", "OSC", "Obj"]
    colors = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
    colors_fill = [(0.75, 0.75, 1), (1, 0.75, 0.75), (0.75, 1, 0.75)]

    plt.subplot(2, 2, 3)
    for j in range(3):
        means, stds = get_status(outcomes[idxs[j]], i)
        plt.plot(x, means, label=labels[j], color=colors[j])
        plt.fill_between(x, means - stds, means + stds, color=colors_fill[j])
    plt.ylim([0, y_lim_others])
    plt.xlabel("Episodes", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.title("Objectives (train)", fontdict={"fontsize": 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.grid()

    plt.subplot(2, 2, 4)
    for j in range(3):
        means, stds = get_status(outcomes_t[idxs[j]], i)
        plt.plot(x, means, label=labels[j], color=colors[j])
        plt.fill_between(x, means - stds, means + stds, color=colors_fill[j])
    plt.ylim([0, y_lim_others])
    plt.xlabel("Episodes", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.title("Objectives (test)", fontdict={"fontsize": 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.grid()

    if mode == 'draw':
        plt.show()
    elif mode == 'save':
        plt.savefig(filename)
    else:
        raise ValueError


def get_plt_skld(skld, i, filename=""):
    """
    Get the figure of the sum of KL divergences during the training.

    Parameters
    ----------
    skld: numpy.ndarray
        Array of sum of KL divergences.
    i: int
    filename: str
    """
    # skld : sum of kl divergences
    skld = skld[:i + 2]
    x = np.arange(i + 2)
    y_lim_all = np.max(skld) + 0.01
    y_lim_partial = 0.1

    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    plt.plot(x, skld, label="KL divergence", color=(0, 0, 1))
    plt.ylim([0, y_lim_all])
    plt.xlabel("Episodes", fontsize=20)
    plt.ylabel("Sum of KL divergences", fontsize=20)
    plt.title("Sum of KL divergences", fontdict={"fontsize": 24})
    plt.legend(loc='upper right', fontsize=20)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(x, skld, label="KL divergence", color=(0, 0, 1))
    plt.ylim([0, y_lim_partial])
    plt.xlabel("Episodes", fontsize=20)
    plt.ylabel("Sum of KL divergences", fontsize=20)
    plt.title("Sum of KL divergences", fontdict={"fontsize": 24})
    plt.legend(loc='upper right', fontsize=20)
    plt.grid()

    plt.savefig(filename)


def get_plt_final(outcomes_l, outcomes_r):
    """
    Get the figure of two outcomes.
    Unlike the previous function, this is for the final outcomes, i.e., after the training.

    Examples
    ----------
    data_l = torch.load("./results_taxi_final/alpha=0.63 using alpha=0.50/7499.tar")
    outcomes_l = data_l["outcomes_t"]
    data_r = torch.load("./results_taxi_final/alpha=0.63/7499.tar")
    outcomes_r = data_r["outcomes_t"]
    utils_taxi.get_plt_final(outcomes_l, outcomes_r)

    Parameters
    ----------
    outcomes_l
        Outcomes which will be shown in the left figure
    outcomes_r
        Outcomes which will be shown in the right figure
    """
    def get_status(inputs):
        means = np.mean(inputs, axis=0)
        stds = np.std(inputs, axis=0)
        return means, stds

    x = np.arange(7500)
    y_lim_rew = [2, 6]
    y_lim_others = [0, 1.1]

    plt.figure(figsize=(20, 16))

    plt.subplot(2, 2, 1)
    means, stds = get_status(outcomes_l[2])
    plt.plot(x, means, label="Avg reward", color=(0, 0, 1))
    plt.fill_between(x, means - stds, means + stds, color=(0.75, 0.75, 1))
    plt.ylim(y_lim_rew)
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Reward", fontsize=24)
    plt.title("Average rewards", fontdict={"fontsize": 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.subplot(2, 2, 2)
    means, stds = get_status(outcomes_r[2])
    plt.plot(x, means, label="Avg reward", color=(0, 0, 1))
    plt.fill_between(x, means - stds, means + stds, color=(0.75, 0.75, 1))
    plt.ylim(y_lim_rew)
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Reward", fontsize=24)
    plt.title("Average rewards", fontdict={"fontsize": 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    idxs = [0, 1, 3]
    labels = ["ORR", "OSC", "Obj"]
    colors = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
    colors_fill = [(0.75, 0.75, 1), (1, 0.75, 0.75), (0.75, 1, 0.75)]

    plt.subplot(2, 2, 3)
    for j in range(3):
        means, stds = get_status(outcomes_l[idxs[j]])
        plt.plot(x, means, label=labels[j], color=colors[j])
        plt.fill_between(x, means - stds, means + stds, color=colors_fill[j])
    plt.ylim(y_lim_others)
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Value", fontsize=24)
    plt.title("ORR, OSC, and Objective values", fontdict={"fontsize": 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.subplot(2, 2, 4)
    for j in range(3):
        means, stds = get_status(outcomes_r[idxs[j]])
        plt.plot(x, means, label=labels[j], color=colors[j])
        plt.fill_between(x, means - stds, means + stds, color=colors_fill[j])
    plt.ylim(y_lim_others)
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Value", fontsize=24)
    plt.title("ORR, OSC, and Objective values", fontdict={"fontsize": 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.show()


def get_plt_final_grayscale(outcomes_l, outcomes_r):
    """
    Get the figure of two outcomes.
    Unlike the previous function, this is for the final outcomes, i.e., after the training.

    Examples
    ----------
    data_l = torch.load("./results_taxi_final/alpha=0.63 using alpha=0.50/7499.tar")
    outcomes_l = data_l["outcomes_t"]
    data_r = torch.load("./results_taxi_final/alpha=0.63/7499.tar")
    outcomes_r = data_r["outcomes_t"]
    utils_taxi.get_plt_final(outcomes_l, outcomes_r)

    Parameters
    ----------
    outcomes_l
        Outcomes which will be shown in the left figure
    outcomes_r
        Outcomes which will be shown in the right figure
    """
    # def get_figure_components(inputs, i):
    #     rew = inputs[:i + 1]
    #     moving_avg_len = 20
    #     means, stds = [np.zeros(rew.size) for _ in range(2)]
    #     for j in range(rew.size):
    #         if j + 1 < moving_avg_len:
    #             rew_part = rew[:j + 1]
    #         else:
    #             rew_part = rew[j - moving_avg_len + 1:j + 1]
    #         means[j] = np.mean(rew_part)
    #         stds[j] = np.std(rew_part)
    #     return rew, means, stds
    def get_status(inputs):
        means = np.mean(inputs, axis=0)
        stds = np.std(inputs, axis=0)
        return means, stds

    x = np.arange(7500)
    y_lim_rew = [2, 6]
    y_lim_others = [0, 1.1]

    plt.figure(figsize=(20, 16))

    plt.subplot(2, 2, 1)
    means, stds = get_status(outcomes_l[2])
    plt.plot(x, means, label="Avg reward", color=(0, 0, 0))
    plt.fill_between(x, means - stds, means + stds, color=(0.75, 0.75, 0.75))
    plt.ylim(y_lim_rew)
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Reward", fontsize=24)
    plt.title("Average rewards", fontdict={"fontsize": 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.subplot(2, 2, 2)
    means, stds = get_status(outcomes_r[2])
    plt.plot(x, means, label="Avg reward", color=(0, 0, 0))
    plt.fill_between(x, means - stds, means + stds, color=(0.75, 0.75, 0.75))
    plt.ylim(y_lim_rew)
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Reward", fontsize=24)
    plt.title("Average rewards", fontdict={"fontsize": 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    idxs = [0, 1, 3]
    labels = ["ORR", "OSC", "Obj"]
    colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    colors_fill = [(0.75, 0.75, 0.75), (0.75, 0.75, 0.75), (0.75, 0.75, 0.75)]
    # colors_fill = [(1, 1, 1), (1, 1, 1), (0.75, 0.75, 0.75)]
    # linestyles = ['', '', '-']
    linestyles = ['-', '--', ':']

    plt.subplot(2, 2, 3)
    for j in range(3):
        means, stds = get_status(outcomes_l[idxs[j]])
        plt.plot(x, means, label=labels[j], color=colors[j], linestyle=linestyles[j])
        plt.fill_between(x, means - stds, means + stds, color=colors_fill[j])
    plt.ylim(y_lim_others)
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Value", fontsize=24)
    plt.title("ORR, OSC, and Objective values", fontdict={"fontsize": 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.subplot(2, 2, 4)
    for j in range(3):
        means, stds = get_status(outcomes_r[idxs[j]])
        plt.plot(x, means, label=labels[j], color=colors[j], linestyle=linestyles[j])
        plt.fill_between(x, means - stds, means + stds, color=colors_fill[j])
    plt.ylim(y_lim_others)
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Value", fontsize=24)
    plt.title("ORR, OSC, and Objective values", fontdict={"fontsize": 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.show()


def get_plt_final_grayscale_only_obj(outcomes_l, outcomes_r, font_settings=None):
    """
    Get the figure of two final outcomes.
    This function uses the evaluation results.
    Unlike the previous function(get_plt_final), this figure put two outcomes into one figure.
    220805: axis update.
    230105: update font settings.

    Examples
    ----------
    # Efficiency of the transfer: visual comparison
    import torch
    from utils import utils_ssd

    dict_l = torch.load("./results/211008 submitted version/results_ssd_final/alpha=0.33 using alpha=0.50 (2 seeds)/seed 1278 (original)/outcomes.tar")
    dict_r = torch.load("./results/211008 submitted version/results_ssd_final/alpha=0.33 (5 seeds)/seed 1267 (original)/outcomes.tar")
    outcomes_l = dict_l["obj_full"]
    outcomes_r = dict_r["obj_full"]
    utils_ssd.get_plt_final_aggregate(outcomes_l, outcomes_r, is_3000=False)

    Parameters
    ----------
    outcomes_l
        Outcomes which will be shown in the left figure
    outcomes_r
        Outcomes which will be shown in the right figure
    font_settings: None or dict

    """
    # mpl.rcParams['hatch.linewidth'] = 2
    period = 1
    def get_status(inputs):
        means = np.mean(inputs, axis=0)
        stds = np.std(inputs, axis=0)
        mov_avg_len = 20
        means_mov_avg, stds_mov_avg = [np.zeros(means.size) for _ in range(2)]
        for j in range(means.size):
            if j + 1 < mov_avg_len:
                means_part = means[:j + 1]
                stds_part = stds[:j + 1]
            else:
                means_part = means[j - mov_avg_len + 1:j + 1]
                stds_part = stds[j - mov_avg_len + 1:j + 1]
            means_mov_avg[j] = np.mean(means_part)
            stds_mov_avg[j] = np.mean(stds_part)
            # means_mov_avg[j] = means[j - mov_avg_len + 1:j + 1] if j + 1 >= mov_avg_len else means[:j + 1]
            # stds_mov_avg[j] = stds[j - mov_avg_len + 1:j + 1] if j + 1 >= mov_avg_len else stds[:j + 1]
        # means = means_mov_avg
        # stds = stds_mov_avg
        means = means[::period]
        stds = stds[::period]
        return means, stds

    # x = np.arange(7500)
    x = np.arange(0, 7500, period)
    x_lim = [0, 7500]
    y_lim = [0.65, 1.05]

    plt.figure(dpi=600, figsize=(15, 8))

    outcomes_l = outcomes_l[3]  # Only select obj among [orr, osc, avg_rew, obj].
    outcomes_r = outcomes_r[3]

    means_l, stds_l = get_status(outcomes_l)
    means_l = savgol_filter(means_l, 101, 3)
    stds_l = savgol_filter(stds_l, 101, 3)
    plt.plot(x, means_l, label="Mean objective value (SF-MFAC)", color=(0, 0, 0))
    plt.fill_between(x, means_l - stds_l, means_l + stds_l, color=(0.5, 0.5, 0.5))
    means_r, stds_r = get_status(outcomes_r)
    means_r = savgol_filter(means_r, 101, 3)
    stds_r = savgol_filter(stds_r, 101, 3)
    plt.plot(x, means_r, label="Mean objective value (MFAC)", alpha=0.5, color=(0, 0, 0), linestyle='--')
    plt.fill_between(x, means_r - stds_r, means_r + stds_r, alpha=0.5, color=(0.75, 0.75, 0.75), hatch='/')

    axis_size = 24
    legend_size = 20
    tick_size = 20
    if font_settings is not None:
        if 'axis_size' in font_settings.keys():
            axis_size = font_settings['axis_size']
        if 'legend_size' in font_settings.keys():
            legend_size = font_settings['legend_size']
        if 'tick_size' in font_settings.keys():
            tick_size = font_settings['tick_size']
        if 'font_name' in font_settings.keys():
            plt.rcParams['font.family'] = font_settings['font_name']
            plt.xlabel("Episodes", fontsize=axis_size, fontname=font_settings['font_name'])
            plt.ylabel(r"$\mathcal{F}$", fontsize=axis_size, fontname=font_settings['font_name'])

    else:
        plt.xlabel("Episodes", fontsize=axis_size)
        plt.ylabel(r"$\mathcal{F}$", fontsize=axis_size)

    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.legend(loc='lower right', fontsize=legend_size)
    plt.tick_params(axis='both', labelsize=tick_size)

    plt.grid()
    plt.savefig('../Driver_lower_level.png', bbox_inches='tight')
    plt.show()


def get_plt_cumulative_skld_multiseeds(skld_trans_list, skld_ntrans_list, is_normalized=False):
    """
    Get the figure of two cumulative SKLDs(sum of KL divergences) for multiple random seeds.
    SKLD of the transfer scenario will be shown in the left.
    If is_normalized is True, SKLD will be divided by the maximum value of SKLDs.

    Examples
    ----------
    skld_trans_list = []
    skld_ntrans_list = []
    for i in range(1234, 1239):
        trans_path = "./results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed "+str(i)+"/kl/7499.tar"
        ntrans_path = "./results_taxi_final/alpha=0.63 (5 seeds)/seed "+str(i)+"/kl/7499.tar"
        data_trans = torch.load(trans_path)
        data_ntrans = torch.load(ntrans_path)
        skld_trans_list.append(data_trans["skld"])
        skld_ntrans_list.append(data_ntrans["skld"])
    utils_taxi.get_plt_cumulative_skld_multiseeds(skld_trans_list, skld_ntrans_list)

    Parameters
    ----------
    skld_trans_list: List
        list of skld from dict_trained for multiple random seeds.
    skld_ntrans_list: List
        list of skld from dict_trained for multiple random seeds.
    is_normalized: bool
    """
    num_seeds = len(skld_trans_list)
    num_episodes = len(skld_trans_list[0])

    cskld_trans, cskld_ntrans = [np.zeros([num_seeds, num_episodes]) for _ in range(2)]
    for i in range(num_seeds):
        if is_normalized:
            skld_trans = skld_trans_list[i] / max(skld_trans_list[i])
            skld_ntrans = skld_ntrans_list[i] / max(skld_ntrans_list[i])
        else:
            skld_trans = skld_trans_list[i]
            skld_ntrans = skld_ntrans_list[i]

        for j in range(num_episodes):
            if j == 0:
                cskld_trans[i, j] = skld_trans[j]
                cskld_ntrans[i, j] = skld_ntrans[j]
            else:
                cskld_trans[i, j] = skld_trans[j] + cskld_trans[i, j - 1]
                cskld_ntrans[i, j] = skld_ntrans[j] + cskld_ntrans[i, j - 1]

    x = np.arange(num_episodes)
    y_lim = None

    means_trans = np.mean(cskld_trans, axis=0)
    stds_trans = np.std(cskld_trans, axis=0)
    means_ntrans = np.mean(cskld_ntrans, axis=0)
    stds_ntrans = np.std(cskld_ntrans, axis=0)

    plt.figure(figsize=(16, 8))

    plt.plot(x, means_trans, label="Cumulative SKLD (transfer)", color=(0, 0, 1))
    plt.fill_between(x, means_trans - stds_trans, means_trans + stds_trans, color=(0.75, 0.75, 1))
    plt.plot(x, means_ntrans, label="Cumulative SKLD (non-transfer)", alpha=0.5, color=(1, 0, 0))
    plt.fill_between(x, means_ntrans - stds_ntrans, means_ntrans + stds_ntrans, alpha=0.5, color=(1, 0.75, 0.75))

    plt.ylim([0, y_lim])
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Cumulative SKLD", fontsize=24)
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.show()
