import copy
import glob
from pathlib import Path
import sys
import time

import numpy as np
import torch

from networks_ssd import Networks
from parsed_args_ssd import args
from sequential_social_dilemma_games.social_dilemmas.envs.env_creator import get_env_creator
from utils import utils_all
"""
220526 코딩 목적
해당 폴더 내에 있는 tar 파일들을 불러와 evaluation results를 얻기 위함. 
figure까지 그리기?
It will run roll_out function "num_tests" times to get all(and average) results. 

This code is for getting test results using trained networks.
It not only calculate objective values of trained networks, 
but calculate KL divergences using final(or last) trained networks.
To calculate sum of KL divergences along the trajectory, it run the episode with "episode length".
In addition, it run this process "num_tests" times to get all(and average) results of sum of KL divergences.
Lastly, it saves results in the "folder_name" folder. 

To run this code, you should set check line 115.
"""
# HERE #####
alphas_env = [1.00]
paths_pol_dir_dict = {
    # 0.00: "./results_ssd/setting_0_220525_1520/saved/*.tar",
    # 0.20: "./results_ssd/setting_5_220529_0009/saved/*.tar",
    # 0.20: "./results_ssd/setting_6_220605_0138/saved/*.tar",  # reuse
    # 0.33: "./results_ssd/setting_1_220528_0210/saved/*.tar",
    # 0.50: "./results_ssd/setting_3_220528_1314/saved/*.tar",
    # 0.50: "./results_ssd/setting_7_220605_1320/saved/*.tar",  # reuse
    # 0.80: "./results_ssd/setting_2_220528_0213/saved/*.tar",
    # 1.00: "./results_ssd/setting_4_220528_1710/saved/*.tar",
    1.00: "./results_ssd/setting_8_220605_2233/saved/*.tar",  # reuse
}
############


def roll_out_simple(networks, env, init_obs, epi_length):
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


paths_pol_dir = list(paths_pol_dir_dict.values())
alphas_pol = list(paths_pol_dir_dict.keys())
num_env = len(alphas_env)
num_pol = len(alphas_pol)
num_tests = 100

utils_all.set_random_seed(1236)

paths_pol = [glob.glob(paths_pol_dir[i]) for i in range(num_pol)]
x_axis_pol = [[int(Path(path).stem) for path in paths] for paths in paths_pol]

time_start = time.time()

# Outcomes
outcomes_all = {}

for i in range(num_env):
    outcomes_env = {}
    for j in range(num_pol):
        path_pol = paths_pol[j]
        num_net = len(path_pol)  # number of networks

        # Outcomes
        rew, pen, inc, obj = [np.zeros([num_tests, num_net]) for _ in range(4)]
        for k in range(num_net):
            # Build args.
            dict_pol = torch.load(path_pol[k])
            args_pol = dict_pol['args']
            args = copy.deepcopy(dict_pol['args'])
            args.setting_name = "setting_evaluation"
            args.lv_penalty = alphas_env[i]
            args.lv_incentive = alphas_env[i]

            # Build the environment.
            env_creator = get_env_creator(args.env, args.num_agents, args)
            env = env_creator(args.num_agents)
            init_obs = env.reset()

            # Build networks.
            networks = utils_all.load_networks(Networks(env, args_pol), args_pol, dict_pol)

            for l in range(num_tests):
                print(f"Env.: {alphas_env[i]} ({i + 1}/{num_env}), "
                      f"Pol.: {alphas_pol[j]} ({j + 1}/{num_pol}), "
                      f"Net.: {x_axis_pol[j][k]} ({k + 1}/{num_net}), "
                      f"Test: ({l + 1}/{num_tests})")
                _, init_obs, collective_reward, collective_feature = roll_out_simple(networks=networks,
                                                                                     env=env,
                                                                                     init_obs=init_obs,
                                                                                     epi_length=args.episode_length)

                rew[l, k] = collective_reward
                pen[l, k] = -collective_feature[1] * args.lv_penalty
                inc[l, k] = collective_feature[2] * args.lv_incentive
                obj[l, k] = rew[l, k] + pen[l, k] - inc[l, k]
        outcomes_env_pol = {'rew': rew,
                            'pen': pen,
                            'inc': inc,
                            'obj': obj,
                            'x_axis': np.array(x_axis_pol[j])}

        outcomes_env[alphas_pol[j]] = outcomes_env_pol
    outcomes_all[alphas_env[i]] = outcomes_env

print(f"Time : {time.time() - time_start:.2f}")

torch.save(outcomes_all, "evaluation_results_ssd.tar")

