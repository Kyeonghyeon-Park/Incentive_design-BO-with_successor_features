import copy
import os
import random
import shutil
import sys
import time

import numpy as np
import torch

from networks_ssd_lstm import NetworksLSTM
# import pathmagic
from parsed_args_lstm import args
from social_dilemmas.envs.env_creator import get_env_creator
import utility_funcs

'''
Notes

01) You should set sequential_social_dilemma_games to source root to avoid errors
    If you can't, try "import pathmagic"    
02) You will run this 'test_for_ssd.py' file but you should change settings in 'parsed_args.py'
'''


def make_vars(n, mode):
    for _ in range(n):
        if mode=='list':
            yield []
        elif mode=='dict':
            yield {}
        else:
            raise NotImplementedError("Possible options of mode are list and dict.")


def dict_append(d0, d1):
    """
    Add d1's values to d1's list
    d0 and d1 should have same keys

    Parameters
    ----------
    d0 : dict
    d1 : dict
    """
    for key in d0.keys():
        d0[key].append(d1[key])


def roll_out(networks, env, args, init_obs, epi_num, epi_length, is_draw=False):
    """
    Run the simulation over the rollout_fragment_length.

    Parameters
    ----------
    networks
    env
    args
    init_obs
    epi_num
    epi_length
    is_draw

    Returns
    -------
    obs_time
    act_time
    rew_time
    m_act_time
    fea_time
    init_obs
    """
    prev_steps = epi_num * epi_length
    if is_draw:
        print(f"Saving figures...")
        filename = image_path + "frame" + str(prev_steps).zfill(9) + ".png"
        env.render(filename=filename, i=prev_steps)

    agent_id_list = list(env.agents.keys())

    obs_time, act_time, rew_time, m_act_time, fea_time, hiddens = make_vars(6, 'dict')

    for agent_id in agent_id_list:
        obs_time[agent_id] = []
        act_time[agent_id] = []
        rew_time[agent_id] = []
        m_act_time[agent_id] = []
        fea_time[agent_id] = []
        hiddens[agent_id] = {x: torch.zeros(args.hidden_cell_size).unsqueeze(0).unsqueeze(0)
                             for x in ["a_hx", "a_cx", "c_hx", "c_cx"]}  # Initialize hidden state

    obs = init_obs  # Initial state.

    for i in range(epi_length):
        agents_actions_dict = dict()
        for agent_id in agent_id_list:
            hidden = (hiddens[agent_id]["a_hx"], hiddens[agent_id]["a_cx"])
            action, hidden = networks.get_action(obs[agent_id], hidden)
            agents_actions_dict[agent_id] = action
            hiddens[agent_id]["a_hx"] = hidden[0]
            hiddens[agent_id]["a_cx"] = hidden[1]

        obs, act, rew, m_act, n_obs, fea = env.step(agents_actions_dict)

        sys.stdout.flush()

        dict_append(obs_time, obs)
        dict_append(act_time, act)
        dict_append(rew_time, rew)
        dict_append(m_act_time, m_act)
        dict_append(fea_time, fea)

        obs = n_obs

        if is_draw:
            filename = image_path + "frame" + str(prev_steps+i+1).zfill(9) + ".png"
            env.render(filename=filename, i=prev_steps+i+1)

    if is_draw:
        video_name = "trajectory_episode_" + str(epi_num)
        utility_funcs.make_video_from_image_dir(video_path, image_path, fps=args.fps, video_name=video_name)
        # Clean up images
        for single_image_name in os.listdir(image_path):
            single_image_path = os.path.join(image_path, single_image_name)
            try:
                if os.path.isfile(single_image_path) or os.path.islink(single_image_path):
                    os.unlink(single_image_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (single_image_path, e))

    init_obs = env.reset()

    return obs_time, act_time, rew_time, m_act_time, fea_time, init_obs


# Seed setting
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

# Build environment
env_creator = get_env_creator(args.env, args.num_agents, args)
env = env_creator(args.num_agents)
init_obs = env.reset()

# Build networks
networks = NetworksLSTM(env, args)

# Build path for saving results
# TODO : move to the function
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

# Metrics
collective_rewards = np.zeros(args.episode_num)
time_start = time.time()

# Save current setting(args) to txt for easy check
utility_funcs.make_setting_txt(args, path)

# Batches for training
obs_batch, act_batch, rew_batch, m_act_batch, fea_batch = make_vars(5, 'list')

# Run
# TODO : 4 -> args.episode_num
for i in range(4):
    # Visualization
     # TODO : change False -> True, decide record frequency
    is_draw = False if (i == 0 or (i + 1) % 10) else False
    obs_time, act_time, rew_time, m_act_time, fea_time, init_obs = roll_out(networks,
                                                                            env,
                                                                            args,
                                                                            init_obs,
                                                                            epi_num=i,
                                                                            epi_length=args.episode_length,
                                                                            is_draw=is_draw,
                                                                            )

    for agent_id in obs_time.keys():
        obs_batch.append(obs_time[agent_id])
        act_batch.append(act_time[agent_id])
        rew_batch.append(rew_time[agent_id])
        m_act_batch.append(m_act_time[agent_id])
        fea_batch.append(fea_time[agent_id])

    collective_rewards[i] = sum(sum(rew_time[agent_id]) for agent_id in rew_time.keys())

    # Update networks
    if (i + 1) % args.update_freq == 0:
        print(f"Updating networks...")
        # TODO : Update networks and reset batches
        raise NotImplementedError

    # Print status
    print(f"Process : {i}/{args.episode_num}, "
          f"Time : {time.time() - time_start:.2f}, "
          f"Collective reward : {collective_rewards[i]}")

    # Draw collective rewards
    if (i + 1) % 20 == 0:
        utility_funcs.draw_or_save_plt_new(collective_rewards, mode='draw')

    # Save several things
    if (i + 1) % 500 == 0:
        time_trained = time.time() - time_start
        filename = str(i).zfill(9) + '.tar'
        filename_plt = saved_path + 'collective_rewards_' + str(i).zfill(9) + '.png'
        utility_funcs.draw_or_save_plt_new(collective_rewards, mode='save', filename=filename_plt)
        utility_funcs.save_data_new(args=args,
                                    env=env,
                                    time_trained=time_trained,
                                    collective_rewards=collective_rewards,
                                    networks=networks,
                                    path=saved_path,
                                    name=filename)

