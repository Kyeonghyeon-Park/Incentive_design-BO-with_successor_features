import os
import random
import shutil
import sys
import time

import numpy as np
import torch

from networks_ssd import Networks
from parsed_args_ssd import args
from sequential_social_dilemma_games.social_dilemmas.envs.env_creator import get_env_creator
import sequential_social_dilemma_games.utility_funcs as utility_funcs

"""
This code is for getting test results using trained networks.
It not only calculate objective values of trained networks, 
but calculate KL divergences using final(or last) trained networks.
To calculate sum of KL divergences along the trajectory, it run the episode with "episode length".
In addition, it run this process "num_tests" times to get all(and average) results of sum of KL divergences.
Lastly, it saves results in the "folder_name" folder. 

To run this code, you should set check line 115.
"""


def roll_out(networks, env, init_obs, epi_length):
    agent_ids = list(env.agents.keys())
    collective_reward = 0
    collective_feature = np.zeros(np.prod(env.feature_space.shape))

    init_m_act = {agent_id: np.zeros(env.action_space.n) for agent_id in agent_ids}

    obs = init_obs  # Initial observations.
    prev_m_act = init_m_act  # Initial previous mean actions which is only used for Boltzmann policy.

    # Run the simulation (or episode).
    for i in range(epi_length):
        # Select actions.
        act, act_probs = networks.get_actions(obs, prev_m_act, is_target=False)

        # Step.
        obs, act, rew, m_act, n_obs, fea = env.step(act)

        # Add one-transition sample to samples if is_train=True and update the collective_reward.
        collective_reward += sum(rew[agent_id] for agent_id in agent_ids)
        collective_feature += sum(fea[agent_id] for agent_id in agent_ids)

        sys.stdout.flush()

        # Update obs and prev_m_act for the next step.
        obs = n_obs
        prev_m_act = m_act

    # Reset the environment after roll_out.
    init_obs = env.reset()

    return None, init_obs, collective_reward, collective_feature


def roll_out_kl(networks_f, networks_i, env, init_obs, epi_length):
    """

    Parameters
    ----------
    networks_f : final networks
    networks_i : intermediate networks
    env
    init_obs
    epi_length

    Returns
    -------
    skl
    """
    agent_ids = list(env.agents.keys())
    actions = [i for i in range(env.action_space.n)]

    init_m_act = {agent_id: np.zeros(env.action_space.n) for agent_id in agent_ids}
    obs = init_obs  # Initial observations.
    prev_m_act = init_m_act  # Initial previous mean actions which is only used for Boltzmann policy.

    skl = 0

    # Run the simulation (or episode).
    for i in range(epi_length):
        # Select actions.
        act, act_probs_f = networks_f.get_actions(obs, prev_m_act, is_target=False)
        _, act_probs_i = networks_i.get_actions(obs, prev_m_act, is_target=False)

        for agent_id in agent_ids:
            for action in actions:
                p = max(act_probs_f[agent_id][action].item(), 1e-12)
                q = max(act_probs_i[agent_id][action].item(), 1e-12)
                skl -= p * np.log(q / p)

        # Step.
        obs, act, rew, m_act, n_obs, fea = env.step(act)

        # Update obs and prev_m_act for the next step.
        obs = n_obs
        prev_m_act = m_act

    # Reset the environment after roll_out.
    init_obs = env.reset()

    return init_obs, skl


args.setting_name = "setting_testing"
args.env = "harvest_modified"
args.h_dims_a = [256, 128, 64, 32]
args.h_dims_p = [256, 128, 64, 32]

##### You should set this part ######
# Unlike evaluate_ssd.py file, it returns the test result (episode 999 ~ episode 29999)
args.num_agents = 4
args.lv_penalty = 0.33
args.lv_incentive = 0.33

# Set folder path which contains network files to test.
folder_name = "./results_ssd_final/alpha=0.33 using alpha=0.50 (5 seeds)/seed 1279/"

# Set the number of tests
num_tests = 50
#####################################

# Network lists.
prev_list = []
for i in range(1, 31):
    file_name = str(i*1000-1).zfill(9)+".tar"
    full_name = folder_name + file_name
    prev_list.append(full_name)

# Seed setting.
rand_seed = 1234
random.seed(rand_seed)
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

# Build the environment.
env_creator = get_env_creator(args.env, args.num_agents, args)
env = env_creator(args.num_agents)
init_obs = env.reset()

# Build networks.
networks_f = Networks(env, args)
networks_i = Networks(env, args)

final_dict = torch.load(prev_list[-1])
networks_f.actor.load_state_dict(final_dict["actor"])
networks_f.psi.load_state_dict(final_dict["psi"])

# Outcomes.
rew_full, pen_full, inc_full, obj_full, skl_full = [np.zeros([len(prev_list), num_tests]) for _ in range(5)]
rew_mean, pen_mean, inc_mean, obj_mean, skl_mean = [np.zeros(len(prev_list)) for _ in range(5)]

time_start = time.time()

for j in range(len(prev_list)):
    prev_dict = torch.load(prev_list[j])
    networks_i.actor.load_state_dict(prev_dict["actor"])
    networks_i.psi.load_state_dict(prev_dict["psi"])

    print(f"-----------------------------")
    print(prev_list[j])

    for i in range(num_tests):
        print(f"Test num : {i} / {num_tests - 1}")
        _, init_obs, collective_reward, collective_feature = roll_out(networks=networks_i,
                                                                      env=env,
                                                                      init_obs=init_obs,
                                                                      epi_length=args.episode_length,)

        rew_full[j, i] = collective_reward
        pen_full[j, i] = -collective_feature[1] * args.lv_penalty
        inc_full[j, i] = collective_feature[2] * args.lv_incentive
        obj_full[j, i] = rew_full[j, i] + pen_full[j, i] - inc_full[j, i]

        init_obs, skl = roll_out_kl(networks_f=networks_f,
                                    networks_i=networks_i,
                                    env=env,
                                    init_obs=init_obs,
                                    epi_length=args.episode_length, )

        skl_full[j, i] = skl

    rew_mean[j] = np.mean(rew_full[j])
    pen_mean[j] = np.mean(pen_full[j])
    inc_mean[j] = np.mean(inc_full[j])
    obj_mean[j] = np.mean(obj_full[j])
    skl_mean[j] = np.mean(skl_full[j])

    print(f"Obj mean : {obj_mean[j]:.2f}")

outcomes = {"rew_full": rew_full,
            "pen_full": pen_full,
            "inc_full": inc_full,
            "obj_full": obj_full,
            "skl_full": skl_full,
            "rew_mean": rew_mean,
            "pen_mean": pen_mean,
            "inc_mean": inc_mean,
            "obj_mean": obj_mean,
            "skl_mean": skl_mean,
            }

torch.save(outcomes, folder_name+"outcomes.tar")

print(f"Time : {time.time() - time_start:.2f}")
