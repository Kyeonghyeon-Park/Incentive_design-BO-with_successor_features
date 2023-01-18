import copy
import glob
from pathlib import Path
import sys
import time

import numpy as np
import torch

from networks_ssd import Networks
from sequential_social_dilemma_games.social_dilemmas.envs.env_creator import get_env_creator
from utils import utils_all
"""
This file is for getting evaluation results given alphas and policies. 
It requires "alphas_env", list that contains alphas of environments, 
and "paths_pol_dir_dict", dict that contains the directory of the file or the directory of files. 
It run "num_tests" times to get results for each alpha of environment and alpha of policy.
Set alphas_env and paths_pol_dir_dict in line 72.
221217) .tar file name은 그때까지 진행한 num_episodes가 되어야 함. 

ex. 
alphas_env = [0.00, 0.50, 1.00]

paths_pol_dir_dict = {
    0.00: "./folder_name/00002999.tar",
    0.50: "./folder_name/00002999.tar",
    1.00: "./folder_name/00002999.tar",
}

If you want to do evaluations using all networks in the folder,
paths_pol_dir_dict = {
    0.00: "./folder_name/*.tar",
    0.50: "./folder_name/*.tar",
    1.00: "./folder_name/*.tar",
}
"""


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


'''
# Examples
# alphas_env = [0.00, 0.50, 1.00]
#
# paths_pol_dir_dict = {
#     0.00: "./folder_name/00002999.tar",
#     0.50: "./folder_name/00002999.tar",
#     1.00: "./folder_name/00002999.tar",
# }

# alphas_env = [0.00]
#
# paths_pol_dir_dict = {
#     0.00: "./results_misucb_ucb_temp/3_misUCB_alpha=0.00_using_alpha=0.26/000029999.tar"
# }
'''
########################################################
'''
misUCB evaluation results for alpha=0.05.
'''
# misUCB
# alphas_env = [0.00, 0.05, 0.10, 0.13, 0.16, 0.18, 0.21, 0.24, 0.28, 0.29, 0.33, 0.40, 0.56, 0.68, 1.00]
########################################################
'''
misUCB evaluation results for alpha=0.40.
'''
# misUCB
# alphas_env = [0.00, 0.16, 0.33, 0.40, 0.68, 1.00]
########################################################
'''
misUCB evaluation results for alpha=0.56.
'''
# misUCB
# alphas_env = [0.00, 0.07, 0.14, 0.20, 0.25, 0.28, 0.30, 0.40, 0.56, 0.75, 0.86, 1.00]
########################################################
'''
Policies (이건 full dict 만들고 필요 없는 건 주석 처리)
'''
# paths_pol_dir_dict = {
#     0.00: "./results_ssd_IJCAI/lists of policies/0.00/000029999.tar",
#     0.05: "./results_ssd_IJCAI/lists of policies/0.05/000029999.tar",
#     0.10: "./results_ssd_IJCAI/lists of policies/0.10/000029999.tar",
#     0.13: "./results_ssd_IJCAI/lists of policies/0.13/000029999.tar",
#     0.16: "./results_ssd_IJCAI/lists of policies/0.16/000029999.tar",
#     0.18: "./results_ssd_IJCAI/lists of policies/0.18/000029999.tar",
#     0.21: "./results_ssd_IJCAI/lists of policies/0.21/000029999.tar",
#     0.24: "./results_ssd_IJCAI/lists of policies/0.24/000029999.tar",
#     0.28: "./results_ssd_IJCAI/lists of policies/0.28/000029999.tar",
#     0.29: "./results_ssd_IJCAI/lists of policies/0.29/000029999.tar",
#     0.33: "./results_ssd_IJCAI/lists of policies/0.33/000029999.tar",
#     0.40: "./results_ssd_IJCAI/lists of policies/0.40/000029999.tar",
#     # 0.47: "./results_ssd_IJCAI/lists of policies/0.47/000029999.tar",
#     0.56: "./results_ssd_IJCAI/lists of policies/0.56/000029999.tar",
#     0.68: "./results_ssd_IJCAI/lists of policies/0.68/000029999.tar",
#     1.00: "./results_ssd_IJCAI/lists of policies/1.00/000029999.tar",
# }
########################################################
'''
Code for 31 tar files.
'''
alphas_env = [0.05]
paths_pol_dir_dict = {
    0.05: "./results_ssd_IJCAI/alpha=0.05 using alpha=1.00 (3 seeds)/seed 1234/*.tar",
    # 0.05: "./results_ssd_IJCAI/alpha=0.05 using alpha=1.00 (3 seeds)/seed 1235/000029999.tar",
}
########################################################
paths_pol_dir = list(paths_pol_dir_dict.values())
alphas_pol = list(paths_pol_dir_dict.keys())
num_env = len(alphas_env)
num_pol = len(alphas_pol)
num_tests = 50

utils_all.set_random_seed(1236)

paths_pol = [sorted(glob.glob(paths_pol_dir[i])) for i in range(num_pol)]
x_axis_pol = [[int(Path(path).stem) for path in paths] for paths in paths_pol]

time_start = time.time()
print(f"Evaluation start time: {time.strftime('%y%m%d_%H%M', time.localtime(time_start))}")
# Outcomes
outcomes_all = {}

for i in range(num_env):
    outcomes_env = {}
    for j in range(num_pol):
        path_pol = paths_pol[j]
        num_net = len(path_pol)  # number of networks
        assert num_net != 0, f'There is no matched file for \'{paths_pol_dir[j]}\'.'

        # Outcomes
        rew, pen, inc, obj = [np.zeros([num_tests, num_net]) for _ in range(4)]
        for k in range(num_net):
            # Build args.
            dict_pol = torch.load(path_pol[k])
            args_pol = dict_pol['args']
            args_pol.mode_reuse_networks = False  # 2212127 추가
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

print(f"Total time : {time.time() - time_start:.2f}")
print(f"Finished time : "+time.strftime('%y%m%d_%H%M', time.localtime(time.time())))

start_time_tag = "_"+time.strftime('%y%m%d_%H%M', time.localtime(time_start))
torch.save(outcomes_all, "evaluation_results_ssd"+start_time_tag+".tar")
