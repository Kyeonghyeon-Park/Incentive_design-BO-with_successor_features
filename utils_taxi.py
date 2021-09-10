import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as distributions

def make_setting_txt(args, path):
    """
    Save current setting(args) to txt for easy check

    Parameters
    ----------
    args
        args which contains current setting
    path : str
        Path where txt file is stored
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


def print_status(args, i, orr, osc, avg_rew, obj, time_start, is_train=True):
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


def get_plt(outcomes, outcomes_t, i, mode="draw", filename=""):
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


def save_data(args, env, episode_trained, decayed_eps, time_start, outcomes, outcomes_t, networks, path, name):
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
        'actor': actor_params,
        'actor_opt': actor_opt_params,
        'psi': psi_params,
        'psi_opt': psi_opt_params,
        'critic': critic_params,
        'critic_opt': critic_opt_params,
    }, path + name)


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
            print("Action distribution at (#", loc, ", ", t, ") : ", act_probs)
