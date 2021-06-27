import os
import sys
import argparse
import random
import shutil

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
'''

# TODO 1. Build {Actor network, critic network, psi network} or {critic network, psi network}
# TODO 2. Action selection based on {actor network} or {critic network, psi network}
# TODO 3. Update network

# TODO : trained 추가?

random.seed(1234)
np.random.seed(1234)
horizon = args.horizon

# Build environment
env_creator = get_env_creator(args.env, args.num_agents, args)
env = env_creator(args.num_agents)
_ = env.reset()

# Build networks
networks = Networks(env, args)

# exploration prob.
epsilon = 0

# Build path for saving images
path = "saved_video"
image_path = os.path.join(path, "frames/")
if not os.path.exists(image_path):
    os.makedirs(image_path)

# Not needed right now
# shape = env.world_map.shape
# full_obs = [np.zeros((shape[0], shape[1], 3), dtype=np.uint8) for i in range(horizon)]

buffers = []

for i in range(horizon):
    agents = list(env.agents.values())
    agents_actions_dict = dict()
    for agent in agents:
        # Action selection for agents
        exploration = np.random.rand(1)[0]
        if exploration < epsilon:
            action = np.random.randint(networks.action_dim)
        else:
            observation = env.single_map_to_idx(env.symbol_view_ind(agent))
            action = networks.get_action(observation, is_target=False)
        agents_actions_dict[agent.agent_id] = action

    obs, rew, dons, info, fea, exp_color, exp_idx = env.step(agents_actions_dict)
    # TODO. 이 reward를 더해서 cumulative reward로 만들기

    sys.stdout.flush()

    # For visualization
    if image_path is not None:
        env.render(filename=image_path + "frame" + str(i).zfill(6) + ".png")
        if i % 10 == 0:
            print("Saved frame " + str(i) + "/" + str(horizon))

    # rgb_arr = env.full_map_to_colors()
    # full_obs[i] = rgb_arr.astype(np.uint8)

    # Add samples to buffer
    buffers.append(exp_idx)

    # TODO 3. Update network

# Get videos
if path is None:
    path = os.path.abspath(os.path.dirname(__file__)) + "/saved_video"
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)

fps = 3
video_name = args.env + "_trajectory"
utility_funcs.make_video_from_image_dir(path, image_path, fps=fps, video_name=video_name)

# Clean up images
shutil.rmtree(image_path)
