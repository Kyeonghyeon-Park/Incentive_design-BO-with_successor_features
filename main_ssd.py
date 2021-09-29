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
Notes

01) You will run this 'main_ssd.py' file but you should change settings in 'parsed_args_ssd.py'
"""


def roll_out(networks, env, args, init_obs, epi_num, epi_length, decayed_eps, paths, is_draw=False, is_train=True):
    """
    Run the simulation over epi_length and get samples from it.

    Parameters
    ----------
    networks
    env
    args
    init_obs
    epi_num
    epi_length
    decayed_eps
    paths
    is_draw
    is_train

    Returns
    ----------
    samples : list
        List of samples which are tuples.
        Length of samples will be the epi_length.
        ex. [(obs, act, rew, m_act, fea), (obs, act, rew, m_act, fea), ...]
    init_obs : list
        It is the list of numpy.ndarray.
        Initial observation of agents after reset().
    collective_reward : int
        Collective reward of this episode.
    collective_feature : np.ndarray
        Collective feature of this episode.
        This is used for calculating total_incentives and total_penalties.
        ex. np.array([x, y])
    """
    image_path, video_path, saved_path = paths

    agent_ids = list(env.agents.keys())
    prev_steps = epi_num * epi_length
    samples = [None] * epi_length
    collective_reward = 0
    collective_feature = np.zeros(np.prod(env.feature_space.shape))

    # TODO : we can move init_m_act into env.reset()
    init_m_act = {agent_id: np.zeros(env.action_space.n) for agent_id in agent_ids}

    obs = init_obs  # Initial observations.
    prev_m_act = init_m_act  # Initial previous mean actions which is only used for Boltzmann policy.

    # Run the simulation (or episode).
    for i in range(epi_length):
        # Select actions.
        if is_train:
            rand_prob = np.random.rand(1)[0]
            if rand_prob < decayed_eps:
                act = {agent_ids[j]: np.random.randint(networks.action_size) for j in range(len(agent_ids))}
                act_probs = {agent_ids[j]: "Random" for j in range(len(agent_ids))}
            else:
                act, act_probs = networks.get_actions(obs, prev_m_act)
        else:
            act, act_probs = networks.get_actions(obs, prev_m_act)

        # Save the image.
        if is_draw:
            print(f"Run the episode with saving figures...") if i == 0 else None
            filename = image_path + "frame" + str(prev_steps + i).zfill(9) + ".png"
            env.render(filename=filename, i=prev_steps + i, act_probs=act_probs)

        # Step.
        obs, act, rew, m_act, n_obs, fea = env.step(act)

        # Add one-transition sample to samples if is_train=True and update the collective_reward.
        if is_train:
            samples[i] = (obs, act, rew, m_act, n_obs, fea)
        collective_reward += sum(rew[agent_id] for agent_id in agent_ids)
        collective_feature += sum(fea[agent_id] for agent_id in agent_ids)

        sys.stdout.flush()

        # Update obs and prev_m_act for the next step.
        obs = n_obs
        prev_m_act = m_act

        # Save the last image.
        if is_draw and i == epi_length - 1:
            act_probs = {agent_ids[j]: "End" for j in range(len(agent_ids))}
            filename = image_path + "frame" + str(prev_steps + i + 1).zfill(9) + ".png"
            env.render(filename=filename, i=prev_steps + i + 1, act_probs=act_probs)

    # Save the video.
    utility_funcs.make_video(is_train, epi_num, args, video_path, image_path) if is_draw else None

    # Reset the environment after roll_out.
    init_obs = env.reset()

    return samples, init_obs, collective_reward, collective_feature


def get_decayed_eps(prev_decayed_eps, i, args):
    if args.mode_epsilon_decay:
        if args.epsilon_decay_ver == 'linear':
            decayed_eps = args.epsilon * (1 - 0.98 * i / args.episode_num)
        elif args.epsilon_decay_ver == 'exponential':
            decayed_eps = max(prev_decayed_eps * 0.9999, 0.01)
        else:
            raise ValueError("The version of epsilon decay is not matched with current implementation.")
    else:
        decayed_eps = prev_decayed_eps
    return decayed_eps


# Seed setting.
rand_seed = args.random_seed
random.seed(rand_seed)
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

# Build the environment.
env_creator = get_env_creator(args.env, args.num_agents, args)
env = env_creator(args.num_agents)
init_obs = env.reset()

# Build networks
networks = Networks(env, args)

# Initial exploration probability
decayed_eps = args.epsilon

# Build paths for saving images.
path, image_path, video_path, saved_path = utility_funcs.make_dirs(args)
paths = [image_path, video_path, saved_path]

# Metrics
collective_rewards, collective_rewards_test = [np.zeros(args.episode_num) for _ in range(2)]
total_penalties, total_penalties_test = [np.zeros(args.episode_num) for _ in range(2)]
total_incentives, total_incentives_test = [np.zeros(args.episode_num) for _ in range(2)]
objectives, objectives_test = [np.zeros(args.episode_num) for _ in range(2)]
time_start = time.time()

# Save current setting(args) to txt for easy check
utility_funcs.make_setting_txt(args, path)

# Buffer
buffer = []

# Run
for i in range(args.episode_num):
    # Option for visualization.
    is_draw = (True and args.mode_draw) if (i == 0 or (i + 1) % args.save_freq == 0) else False

    # Decayed exploration probability
    decayed_eps = get_decayed_eps(decayed_eps, i, args)

    # Run roll_out function. (We can get 1,000 samples and collective reward of this episode)
    samples, init_obs, collective_reward, collective_feature = roll_out(networks=networks,
                                                                        env=env,
                                                                        args=args,
                                                                        init_obs=init_obs,
                                                                        epi_num=i,
                                                                        epi_length=args.episode_length,
                                                                        decayed_eps=decayed_eps,
                                                                        paths=paths,
                                                                        is_draw=is_draw,
                                                                        is_train=True,
                                                                        )

    buffer += samples
    collective_rewards[i] = collective_reward
    total_penalties[i] = collective_feature[0] * args.lv_penalty
    total_incentives[i] = collective_feature[1] * args.lv_incentive
    objectives[i] = collective_rewards[i] + total_penalties[i] - total_incentives[i]

    buffer = buffer[-args.buffer_size:]

    # Update networks
    if (i + 1) % args.update_freq == 0:
        k_samples = random.choices(buffer, k=args.K)
        networks.update_networks(k_samples)

    # Update target networks
    if (i + 1) % args.update_freq_target == 0:
        networks.update_target_networks()

    # Print status
    update = "O" if (i + 1) % args.update_freq == 0 else "X"
    print(f"Process : {i}/{args.episode_num}, "
          f"Time : {time.time() - time_start:.2f}, "
          f"Collective reward : {collective_rewards[i]:.2f}, "
          f"Objective : {objectives[i]:.2f}, "
          f"Update : {update}, "
          f"Train")

    # Test
    if args.mode_test:
        samples, init_obs, collective_reward, collective_feature = roll_out(networks=networks,
                                                                            env=env,
                                                                            args=args,
                                                                            init_obs=init_obs,
                                                                            epi_num=i,
                                                                            epi_length=args.episode_length,
                                                                            decayed_eps=decayed_eps,
                                                                            paths=paths,
                                                                            is_draw=is_draw,
                                                                            is_train=False,
                                                                            )

        collective_rewards_test[i] = collective_reward
        total_penalties_test[i] = collective_feature[0] * args.lv_penalty
        total_incentives_test[i] = collective_feature[1] * args.lv_incentive
        objectives_test[i] = collective_rewards_test[i] + total_penalties_test[i] - total_incentives_test[i]

        # Print status
        print(f"Process : {i}/{args.episode_num}, "
              f"Time : {time.time() - time_start:.2f}, "
              f"Collective reward : {collective_rewards_test[i]:.2f}, "
              f"Objective : {objectives_test[i]:.2f}, "
              f"Test")

    # Draw collective rewards
    if (i + 1) % 20 == 0 and args.mode_draw:
        utility_funcs.draw_or_save_plt_v4(collective_rewards,
                                          collective_rewards_test,
                                          objectives,
                                          objectives_test,
                                          i=i,
                                          mode='draw',
                                          )

    # Save several things
    if (i + 1) % args.save_freq == 0:
        time_trained = time.time() - time_start
        filename = str(i).zfill(9) + '.tar'
        filename_plt = saved_path + 'outcomes_' + str(i).zfill(9) + '.png'
        utility_funcs.draw_or_save_plt_v4(collective_rewards,
                                          collective_rewards_test,
                                          objectives,
                                          objectives_test,
                                          i=i,
                                          mode='save',
                                          filename=filename_plt,
                                          )
        utility_funcs.save_data_v4(args=args,
                                   env=env,
                                   episode_trained=i,
                                   decayed_eps=decayed_eps,
                                   time_trained=time_trained,
                                   outcomes={'collective_rewards': collective_rewards,
                                             'collective_rewards_test': collective_rewards_test,
                                             'total_penalties': total_penalties,
                                             'total_penalties_test': total_penalties_test,
                                             'total_incentives': total_incentives,
                                             'total_incentives_test': total_incentives_test,
                                             'objectives': objectives,
                                             'objectives_test': objectives_test,
                                             },
                                   networks=networks,
                                   path=saved_path,
                                   name=filename,
                                   )
