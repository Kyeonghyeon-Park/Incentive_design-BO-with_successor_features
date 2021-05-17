import os
import sys
import argparse
import random
import shutil

import numpy as np

# import pathmagic

from config.default_args import add_default_args
from social_dilemmas.envs.env_creator import get_env_creator
import utility_funcs

'''
Notes

01) You should set sequential_social_dilemma_games to source root to avoid errors
    If you can't, try "import pathmagic"    
'''

# TODO 1. Build Actor network, critic network, psi network
# TODO 2. Action selection based on actor network
# TODO 3. Update network

# Initial setting
parser = argparse.ArgumentParser()
add_default_args(parser)

# Our initial setting
parser.add_argument("--lv_penalty", type=float, default=0, help="Penalty level for agents who eat apple")
parser.add_argument("--lv_incentive", type=float, default=0, help="Incentive level for agents who clean the river")
parsed_args = parser.parse_args()

# Our setting
parsed_args.env = 'cleanup_modified'
parsed_args.num_agents = 3  # Maximum 10 agents
parsed_args.lv_penalty = 0.5
parsed_args.lv_incentive = 0.3

random.seed(1234)
np.random.seed(1234)
horizon = 200  # Maximum period for the run

# Build environment
env_creator = get_env_creator(parsed_args.env, parsed_args.num_agents, parsed_args)
env = env_creator(parsed_args.num_agents)
observations = env.reset()

path = "saved_video"
image_path = os.path.join(path, "frames/")
if not os.path.exists(image_path):
    os.makedirs(image_path)
shape = env.world_map.shape
full_obs = [np.zeros((shape[0], shape[1], 3), dtype=np.uint8) for i in range(horizon)]

buffers = []

for i in range(horizon):
    agents = list(env.agents.values())
    action_dim = env.action_space.n
    agents_actions_dict = dict()
    for agent in agents:
        # Action selection for agents
        #############
        # TODO 2. Action selection based on actor network based on current observations
        rand_action = np.random.randint(action_dim)
        #############
        agents_actions_dict[agent.agent_id] = rand_action

    obs, rew, dons, info, fea, exp = env.step(agents_actions_dict)

    sys.stdout.flush()

    # For visualization
    if image_path is not None:
        env.render(filename=image_path + "frame" + str(i).zfill(6) + ".png")
        if i % 10 == 0:
            print("Saved frame " + str(i) + "/" + str(horizon))
    rgb_arr = env.full_map_to_colors()
    full_obs[i] = rgb_arr.astype(np.uint8)

    # Add samples to buffer
    buffers.append(exp)

    # TODO 3. Update network

# Get videos
if path is None:
    path = os.path.abspath(os.path.dirname(__file__)) + "/saved_video"
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)

fps = 5
video_name = parsed_args.env + "_trajectory"
utility_funcs.make_video_from_image_dir(path, image_path, fps=fps, video_name=video_name)

# Clean up images
shutil.rmtree(image_path)
