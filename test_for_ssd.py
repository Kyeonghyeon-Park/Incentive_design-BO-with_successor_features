import os
import sys
import random
import shutil
import time

import torch
import numpy as np

# import pathmagic
from parsed_args import args
from networks_ssd import Networks
from social_dilemmas.envs.env_creator import get_env_creator
import utility_funcs

'''
Notes

01) You should set sequential_social_dilemma_games to source root to avoid errors
    If you can't, try "import pathmagic"    
02) You will run this 'test_for_ssd.py' file but you should change settings in 'parsed_args.py'
'''

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
horizon = args.horizon

# Build environment
env_creator = get_env_creator(args.env, args.num_agents, args)
env = env_creator(args.num_agents)
_ = env.reset()

# Build networks
networks = Networks(env, args)

# exploration prob.
epsilon = args.epsilon

# Build path for saving images
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
agents = list(env.agents.values())
rewards = dict()
for agent in agents:
    rewards[agent.agent_id] = []
    rewards['collective_reward'] = []

buffers = []
time_start = time.time()

# Save current setting(args) to txt for easy check
utility_funcs.make_setting_txt(args, path)

# Run
for i in range(horizon):
    agents = list(env.agents.values())
    agents_actions_dict = dict()
    for agent in agents:
        # Action selection for agents
        exploration = np.random.rand(1)[0]
        decayed_epsilon = epsilon - epsilon * 0.98 * i / horizon * args.mode_epsilon_decay
        if exploration < decayed_epsilon:
            action = np.random.randint(networks.action_dim)
        else:
            observation = env.single_map_to_idx(env.symbol_view_ind(agent))
            prev_mean_action = env.prev_mean_action[agent.agent_id]
            action = networks.get_action(observation, prev_mean_action, is_target=True)
        agents_actions_dict[agent.agent_id] = action

    obs, rew, dons, info, fea, exp_color, exp_idx = env.step(agents_actions_dict)

    # Update metrics
    collective_reward = 0
    for agent_id in exp_idx.keys():
        prev_cum_reward = 0 if i == 0 else rewards[agent_id][-1]
        curr_reward = exp_idx[agent_id]['reward']
        rewards[agent_id].append(prev_cum_reward + curr_reward)
        collective_reward = collective_reward + curr_reward

    prev_cum_collective_reward = 0 if i == 0 else rewards['collective_reward'][-1]
    rewards['collective_reward'].append(prev_cum_collective_reward + collective_reward)

    sys.stdout.flush()

    # For visualization
    if image_path is not None:
        filename = image_path + "frame" + str(i).zfill(9) + ".png"
        if (i < 100) or (9000 <= i % 10000 < 9100) or (9900 <= i % 10000 < 10000):
            env.render(filename=filename, i=i)

    # Add samples to buffer
    buffers.append(exp_idx)
    buffers = buffers[-args.buffer_size:]

    # Update networks
    if (i + 1) % 5 == 0:
        samples = random.choices(buffers, k=args.K)
        networks.update_networks(samples)

    # Update target networks
    if (i + 1) % 50 == 0:
        networks.update_target_networks()

    # Print status
    if (i + 1) % 10 == 0:
        print('Process : {i}/{horizon}, '
              'Time : {time:.2f}, '
              'Collective reward: {reward}'.format(i=str(i),
                                                   horizon=str(horizon),
                                                   time=time.time() - time_start,
                                                   reward=str(rewards['collective_reward'][-1]),
                                                   ))

    # Draw collective reward
    if (i + 1) % 100 == 0:
        utility_funcs.draw_or_save_plt(rewards['collective_reward'], mode='draw')

    # Get videos
    if (i + 1) == 100 or (i + 1) % 10000 == 9100 or (i + 1) % 10000 == 0:
        video_name = 'trajectory_' + str(i).zfill(9)
        utility_funcs.make_video_from_image_dir(video_path, image_path, fps=args.fps, video_name=video_name)
        # Clean up images
        for single_image_name in os.listdir(image_path):
            single_image_path = os.path.join(image_path, single_image_name)
            try:
                if os.path.isfile(single_image_path) or os.path.islink(single_image_path):
                    os.unlink(single_image_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (single_image_path, e))
        # shutil.rmtree(image_path)

    # Save several things
    if (i + 1) % 10000 == 0:
        time_trained = time.time() - time_start
        filename = str(i).zfill(9) + '.tar'
        filename_plt = saved_path + 'collective_reward_' + str(i).zfill(9) + '.png'
        utility_funcs.draw_or_save_plt(rewards['collective_reward'], mode='save', filename=filename_plt)
        utility_funcs.save_data(args=args,
                                env=env,
                                buffers=buffers,
                                time_trained=time_trained,
                                rewards=rewards,
                                networks=networks,
                                path=saved_path,
                                name=filename)

    # Reset the environment
    if (i + 1) % args.episode_length == 0:
        _ = env.reset()
