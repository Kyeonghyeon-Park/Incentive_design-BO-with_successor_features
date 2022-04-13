import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as distributions

from main_taxi import *
from networks_taxi import Networks
from parsed_args_taxi import args
from taxi import TaxiEnv


def make_setting_txt(args, path):
    """
    Save current setting(args) to txt for easy check.

    Parameters
    ----------
    args
        args which contains current setting.
    path : str
        Path where txt file is stored.
    """
    txt_path = os.path.join(path, 'args.txt')
    f = open(txt_path, 'w')
    for arg in vars(args):
        content = arg + ': ' + str(getattr(args, arg)) + '\n'
        f.write(content)
    f.close()


def make_dirs(args):
    path = "results_taxi/" + args.setting_name
    saved_path = os.path.join(path, "saved/")
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    return path, saved_path


def save_data(args, env, episode_trained, decayed_eps, time_start, outcomes, outcomes_t, skl, networks, path, name):
    actor_params, actor_opt_params, critic_params, critic_opt_params, psi_params, psi_opt_params = [None] * 6

    if args.mode_ac:
        actor_params = networks.actor.state_dict()
        actor_opt_params = networks.actor_opt.state_dict()
    if args.mode_psi:
        psi_params = networks.psi.state_dict()
        psi_opt_params = networks.psi_opt.state_dict()
    else:
        critic_params = networks.critic.state_dict()
        critic_opt_params = networks.critic_opt.state_dict()

    torch.save({
        'args': args,
        'env': env,
        'episode_trained': episode_trained,
        'time_trained': time.time() - time_start,
        'decayed_eps': decayed_eps,
        'outcomes': outcomes,
        'outcomes_t': outcomes_t,
        'skl': skl,
        'actor': actor_params,
        'actor_opt': actor_opt_params,
        'psi': psi_params,
        'psi_opt': psi_opt_params,
        'critic': critic_params,
        'critic_opt': critic_opt_params,
    }, path + name)


def print_status(args, i, orr, osc, avg_rew, obj, time_start, is_train=True):
    """
    This function prints serveral status for each episode.

    Parameters
    ----------
    args
    i : int
        The number which represents the current episode.
    orr : numpy.ndarray
        Array of order response rate.
        Size : (1, num_episodes) or (num_tests, num_episodes)
    osc : numpy.ndarray
        Array of overall service charge.
        Size : (1, num_episodes) or (num_tests, num_episodes)
    avg_rew : numpy.ndarray
        Array of average rewards of all agents.
        Size : (1, num_episodes) or (num_tests, num_episodes)
    obj : numpy.ndarray
        Array of objective values.
        Size : (1, num_episodes) or (num_tests, num_episodes)
    time_start : float
        The time when the training starts.
    is_train : boolean
        True if train
    """
    update = "O" if (i + 1) % args.update_freq == 0 and is_train else "X"
    mode = "  Train  " if is_train else "Test(avg)"
    print(f"Process : {i}/{args.num_episodes}, "
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
    def get_one_hot_obs(ind_obs):
        """
        Text.

        Parameters
        ----------
        ind_obs : numpy.ndarray
            shape : (2, )

        Returns
        -------
        ind_obs : numpy.ndarray
            shape : (num_grids, episode_length + 1)
        """
        num_grids = 4
        episode_length = 2
        ind_obs_one_hot = np.zeros([num_grids, episode_length + 1])
        loc = ind_obs[0]
        time = ind_obs[1] if ind_obs[1] <= episode_length else episode_length
        ind_obs_one_hot[loc, time] = 1
        return ind_obs_one_hot

    torch.set_printoptions(linewidth=200, sci_mode=False)

    for loc in [1, 2]:
        for t in [0]:
            print("Q at (#", loc, ", ", t, ")")
            q_all = torch.zeros([11, networks.action_size])
            for i in range(11):
                m_act = i / 10
            # for m_act in np.arange(0.0, 1.1, 0.1):
                with torch.no_grad():
                    ind_obs = np.array([loc, t])
                    ind_obs_1 = get_one_hot_obs(ind_obs)
                    obs_tensor = torch.tensor([ind_obs_1], dtype=torch.float)
                    obs_tensor = obs_tensor.view(-1, networks.observation_size)

                    m_act_tensor = torch.tensor(m_act, dtype=torch.float)
                    m_act_tensor = m_act_tensor.view(-1, networks.mean_action_size)

                    psi = networks.psi(obs_tensor, m_act_tensor)
                    q = torch.tensordot(psi, networks.w, dims=([2], [0]))
                    q_all[i] = q[0]
            print(q_all.transpose(1, 0))


def print_action_dist(networks):
    def get_one_hot_obs(ind_obs):
        """
        Text.

        Parameters
        ----------
        ind_obs : numpy.ndarray
            shape : (2, )

        Returns
        -------
        ind_obs : numpy.ndarray
            shape : (num_grids, episode_length + 1)
        """
        num_grids = 4
        episode_length = 2
        ind_obs_one_hot = np.zeros([num_grids, episode_length + 1])
        loc = ind_obs[0]
        time = ind_obs[1] if ind_obs[1] <= episode_length else episode_length
        ind_obs_one_hot[loc, time] = 1
        return ind_obs_one_hot

    def get_masked_categorical(action_probs, masks):
        probs = torch.mul(action_probs, masks)
        action_dists = distributions.Categorical(probs)
        return action_dists

    for loc in [1, 2]:
        for t in [0]:
            ind_obs = np.array([loc, t])
            ind_obs_1 = get_one_hot_obs(ind_obs)
            obs_tensor = torch.tensor([ind_obs_1], dtype=torch.float)
            obs_tensor = obs_tensor.view(-1, networks.observation_size)
            obs_mask = networks.get_masks([ind_obs])

            act_probs = networks.actor(obs_tensor)
            dists = get_masked_categorical(act_probs, obs_mask)
            act_probs = dists.probs
            act_probs = act_probs.detach()
            print("Action distribution at (#", loc, ", ", t, ") : ", act_probs)


def calculate_kl_divergence(networks, networks_final):
    def get_one_hot_obs(ind_obs):
        """
        Text.

        Parameters
        ----------
        ind_obs : numpy.ndarray
            shape : (2, )

        Returns
        -------
        ind_obs : numpy.ndarray
            shape : (num_grids, episode_length + 1)
        """
        num_grids = 4
        episode_length = 2
        ind_obs_one_hot = np.zeros([num_grids, episode_length + 1])
        loc = ind_obs[0]
        time = ind_obs[1] if ind_obs[1] <= episode_length else episode_length
        ind_obs_one_hot[loc, time] = 1
        return ind_obs_one_hot

    def get_masked_categorical(action_probs, masks):
        probs = torch.mul(action_probs, masks)
        action_dists = distributions.Categorical(probs)
        return action_dists

    kl = 0

    for ind_obs in [np.array([1, 0]), np.array([2, 0])]:
        ind_obs_1 = get_one_hot_obs(ind_obs)
        obs_tensor = torch.tensor([ind_obs_1], dtype=torch.float)
        obs_tensor = obs_tensor.view(-1, networks.observation_size)
        obs_mask = networks.get_masks([ind_obs])  # obs_mask: tensor([[1., 1., 0., 1.]]) for ind_obs: [1 0]

        act_probs = networks.actor_target(obs_tensor)
        dists = get_masked_categorical(act_probs, obs_mask)
        act_probs = dists.probs  # ex. act_probs: tensor([[0.3911, 0.4301, 0.0000, 0.1788]])

        act_probs_final = networks_final.actor_target(obs_tensor)
        dists_final = get_masked_categorical(act_probs_final, obs_mask)
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
                kl -= p * np.log(q / p)

    return kl


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

    Returns
    -------

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


def get_plt_skl(skl, i, filename=""):
    """
    Get the figure of the sum of KL divergences during the training.

    Parameters
    ----------
    skl : numpy.ndarray
        Array of sum of KL divergences.
    i
    filename
    """
    # skl : sum of kl divergences
    skl = skl[:i + 2]
    x = np.arange(i + 2)
    y_lim_all = np.max(skl) + 0.01
    y_lim_partial = 0.1

    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    plt.plot(x, skl, label="KL divergence", color=(0, 0, 1))
    plt.ylim([0, y_lim_all])
    plt.xlabel("Episodes", fontsize=20)
    plt.ylabel("Sum of KL divergences", fontsize=20)
    plt.title("Sum of KL divergences", fontdict={"fontsize": 24})
    plt.legend(loc='upper right', fontsize=20)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(x, skl, label="KL divergence", color=(0, 0, 1))
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


def get_plt_cumulative_SKLD(skld_trans, skld_ntrans):
    """
    Get the figure of two cumulative SKLDs (sum of KL divergences).
    SKLD of the transfer scenario will be shown in the left.
    SKLD is divided by the maximum value of SKLDs.

    Examples
    ----------
    data_trans = torch.load("./results_taxi_final/alpha=0.63 using alpha=0.50/kl/7499.tar")
    skl_trans = data_trans["skl"]
    data_ntrans = torch.load("./results_taxi_final/alpha=0.63/kl/7499.tar")
    skl_ntrans = data_ntrans["skl"]
    utils_taxi.get_plt_cumulative_SKLD(skl_trans, skl_ntrans)

    Parameters
    ----------
    skld_trans
    skld_ntrans
    """
    # skld : sum of kl divergences
    skld_trans = skld_trans / max(skld_trans)
    skld_ntrans = skld_ntrans / max(skld_ntrans)

    x = np.arange(len(skld_trans))
    # y_lim = np.max([np.sum(skld_trans), np.sum(skld_ntrans)]) + 0.01
    # y_lim = 1000
    y_lim = None
    cskld_trans = np.zeros(len(skld_trans))
    cskld_ntrans = np.zeros(len(skld_ntrans))
    for i in range(len(skld_trans)):
        if i == 0:
            cskld_trans[i] = skld_trans[i]
            cskld_ntrans[i] = skld_ntrans[i]
        else:
            cskld_trans[i] = skld_trans[i] + cskld_trans[i - 1]
            cskld_ntrans[i] = skld_ntrans[i] + cskld_ntrans[i - 1]

    plt.figure(figsize=(16, 8))
    plt.plot(x, cskld_trans, label="Cumulative SKLD (transfer)", color=(0, 0, 1))
    plt.plot(x, cskld_ntrans, label="Cumulative SKLD (non-transfer)", color=(1, 0, 0))
    plt.ylim([0, y_lim])
    plt.xlabel("Episodes", fontsize=24)
    plt.ylabel("Cumulative SKLD", fontsize=24)
    plt.legend(loc='best', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()

    plt.show()


def get_plt_cumulative_SKLD_multiseeds(skld_trans_list, skld_ntrans_list, is_normalized=False):
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
        skld_trans_list.append(data_trans["skl"])
        skld_ntrans_list.append(data_ntrans["skl"])
    utils_taxi.get_plt_cumulative_SKLD_multiseeds(skld_trans_list, skld_ntrans_list)

    Parameters
    ----------
    skld_trans_list : list
    skld_ntrans_list : list
    is_normalized : bool
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


def get_approximated_gradient(network_path, w, w_bound, h=0.01, num_tests=100):
    """
    This function is for getting approximated gradient by calculating f(policy, w + h*e_i) - f(policy, w) / h.
    It will require several evaluations to get the gradient.

    Examples
    ----------
    file_path = "./results_taxi_final/alpha=0.80/7499.tar"
    w = np.array([0.80])
    w_bound = np.array([[0, 1]])
    w_grad = utils_taxi.get_approximated_gradient(file_path, w, w_bound, h=0.01, num_tests=2000)

    Parameters
    ----------
    network_path : str
        File path for the trained network
    w : np.array
        Weight (or reward parameter)
        ex. w = np.array([0.33, 0.5])
    w_bound : np.array
        List which contains bounds of w.
        Each row (call w_bound[i,:]) represents the bound of w[i].
        w[i] cannot over the bound.
        It will be used for getting gradients.
        For example, let w_bound = np.array([[0, 1]]) and w = np.array([1]).
        In this case, we can't calculate f(policy, w + h*e_i).
        It will get approximated gradient by calculating f(policy, w) - f(policy, w - h*e_i) / h.
    h : float
        Parameter for calculating approximated gradient (small value).
    num_tests : int
        The number of tests to calculate approximated gradients.
        This function evaluate "num_tests" times to get gradients and average them.

    Returns
    -------
    w_grad : np.array
    """
    def get_env_and_networks(args, prev_dict):
        env = TaxiEnv(args)
        networks = Networks(env, args)
        networks = load_networks(networks, args, prev_dict)
        # networks = load_networks(networks, prev_dict)
        return env, networks

    # def load_networks(networks, prev_dict):
    #     # TODO : make complete files for critics
    #     networks.actor.load_state_dict(prev_dict['actor'])
    #     networks.actor_target.load_state_dict(prev_dict['actor'])
    #     networks.psi.load_state_dict(prev_dict['psi'])
    #     networks.psi_target.load_state_dict(prev_dict['psi'])
    #     return networks

    set_random_seed(1234)
    w_grad = np.zeros(w.size)

    # Load previous data.
    prev_dict = torch.load(network_path)
    args = prev_dict['args']
    if args.lv_penalty != w:
        raise ValueError("args.lv_penalty and w are not the same. Please check your inputs.")

    # Calculate w_grad for each dimension.
    for i in range(w.size):
        # Try to build f(policy, w_f) - f(policy, w_b) / h (w_f: w_front, w_b: w_back)
        args_f, args_b = [copy.deepcopy(args) for _ in range(2)]
        if w[i] + h > w_bound[i, 1]:
            w_f, w_b = w, w - h * np.eye(w.size)[i]
        else:
            w_f, w_b = w + h * np.eye(w.size)[i], w
        args_f.lv_penalty = w_f
        args_b.lv_penalty = w_b

        # Build the environment and networks.
        env_f, networks_f = get_env_and_networks(args_f, prev_dict)
        env_b, networks_b = get_env_and_networks(args_b, prev_dict)

        # Build array for collecting objective values.
        obj_f, obj_b = [np.zeros(num_tests) for _ in range(2)]

        for j in range(num_tests):
            _, outcome_f = roll_out(networks=networks_f,
                                    env=env_f,
                                    args=args_f,
                                    decayed_eps=0,
                                    is_train=False)
            _, outcome_b = roll_out(networks=networks_b,
                                    env=env_b,
                                    args=args_b,
                                    decayed_eps=0,
                                    is_train=False)

            _, _, _, obj_f[j] = outcome_f
            _, _, _, obj_b[j] = outcome_b
            print(f"Dim: {i+1}/{w.size}, Tests: {j+1}/{num_tests}") if (((j+1) * 10) % num_tests == 0) else None

        w_grad[i] = (np.mean(obj_f) - np.mean(obj_b)) / h
        print(f"w_grad: {w_grad}")

    return w_grad


def set_random_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)


def load_networks(networks, args, dict_trained):
    if args.mode_ac:
        networks.actor.load_state_dict(dict_trained['actor'])
        networks.actor_target.load_state_dict(dict_trained['actor'])
    if args.mode_psi:
        networks.psi.load_state_dict(dict_trained['psi'])
        networks.psi_target.load_state_dict(dict_trained['psi'])
    else:
        networks.critic.load_state_dict(dict_trained['critic'])
        networks.critic_target.load_state_dict(dict_trained['critic'])
    return networks
