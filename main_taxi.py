import random
import time

import numpy as np
import torch

from networks_taxi import Networks

from utils import utils, utils_taxi
from parsed_args_taxi import args
from taxi import TaxiEnv


def roll_out(networks, env, decayed_eps, is_train=True):
    epi_length = env.episode_length
    samples = [None] * epi_length
    fare_infos = np.array([0, 0], dtype=float)
    for global_time in range(epi_length):
        available_agents = env.get_available_agents(global_time)
        av_obs = {agent_id: env.obs[agent_id] for agent_id in available_agents}
        # Select actions.
        is_random = True if np.random.rand(1)[0] < decayed_eps and is_train else False
        act = networks.get_actions(av_obs, is_random=is_random)

        # Step.
        sample, fare_info = env.step(act, global_time)
        samples[global_time] = sample if is_train else None
        fare_infos += fare_info

    outcome = get_outcome(env, fare_infos)
    env.reset()

    return samples, outcome


def get_decayed_eps(i, init_eps):
    decayed_eps = max(init_eps - 0.01 * (i // 20), 0.01)
    return decayed_eps


def get_outcome(env, fare_infos):
    weight = 3 / 5
    num_demands = env.demand[:, 3].shape[0]
    num_fulfilled_demands = np.sum(env.demand[:, 3])
    orr = num_fulfilled_demands / num_demands
    osc = fare_infos[0] / fare_infos[1]
    avg_rew = (fare_infos[1] - fare_infos[0]) / env.num_agents
    obj = weight * orr + (1 - weight) * (1 - osc)
    outcome = orr, osc, avg_rew, obj  # outcome will be tuple
    return outcome


def get_final_networks(env, args):
    """
    If args.mode_kl_divergence is True (= want to calculate kl divergences),
    we have to get final(last trained) networks.

    Parameters
    ----------
    env: TaxiEnv
    args: Namespace

    Returns
    -------
    networks_final: Networks
    """
    if args.mode_kl_divergence:
        networks_final = Networks(env, args)
        dict_trained = torch.load(args.file_path_final)
        networks_final = utils.load_networks(networks_final, args, dict_trained)
    else:
        networks_final = None
    return networks_final


if __name__ == "__main__":
    # Set the random seed.
    utils.set_random_seed(args.random_seed)

    # Build the environment.
    env = TaxiEnv(args)

    # Build networks.
    networks = Networks(env, args)
    networks_final = get_final_networks(env, args)

    # Initial exploration probability.
    init_eps = args.epsilon

    # Build paths for saving files.
    path, saved_path = utils_taxi.make_dirs(args)

    # Buffer.
    buffer = []

    # Metrics.
    orr, osc, avg_rew, obj = [np.zeros([1, args.num_episodes]) for _ in range(4)]
    orr_t, osc_t, avg_rew_t, obj_t = [np.zeros([args.num_tests, args.num_episodes]) for _ in range(4)]
    time_start = time.time()

    # Save current setting(args) to txt for easy check.
    utils.make_setting_txt(args, path)

    # KL divergence.
    skl = np.zeros(args.num_episodes + 1)
    skl[0] = utils_taxi.calculate_kl_divergence(networks, networks_final) if args.mode_kl_divergence else 0

    # Run.
    for i in range(args.num_episodes):
        # Decayed exploration probability.
        decayed_eps = get_decayed_eps(i, init_eps)

        # Run roll_out function for the training (We can get samples and outcomes of this episode).
        samples, outcome = roll_out(networks, env,decayed_eps, is_train=True)
        orr[0, i], osc[0, i], avg_rew[0, i], obj[0, i] = outcome
        buffer += samples
        buffer = buffer[-args.buffer_size:]

        # Update networks.
        if (i + 1) % args.update_freq == 0:
            k_samples = random.choices(buffer, k=args.K)
            networks.update_networks(k_samples)

        # Update target networks.
        if (i + 1) % args.update_freq_target == 0:
            networks.update_target_networks()

        # Print status.
        utils_taxi.print_status(args, i, orr, osc, avg_rew, obj, time_start, is_train=True)

        # Run roll_out function for the testing.
        for j in range(args.num_tests):
            _, outcome = roll_out(networks, env, decayed_eps, is_train=False)
            orr_t[j, i], osc_t[j, i], avg_rew_t[j, i], obj_t[j, i] = outcome

        # Print status.
        utils_taxi.print_status(args, i, orr_t, osc_t, avg_rew_t, obj_t, time_start, is_train=False)

        # KL divergence.
        skl[i + 1] = utils_taxi.calculate_kl_divergence(networks, networks_final) if args.mode_kl_divergence else 0

        # Draw outcomes.
        if (i + 1) % args.draw_freq == 0 and args.mode_draw:
            outcomes = [orr, osc, avg_rew, obj]
            outcomes_t = [orr_t, osc_t, avg_rew_t, obj_t]
            utils_taxi.get_plt(outcomes, outcomes_t, i, mode="draw")

        # Print values and policies.
        if (i + 1) % 1 == 0:
            utils_taxi.print_action_dist(networks)

        if (i + 1) % 10 == 0:
            utils_taxi.print_updated_q(networks)

        # Save the figure and data.
        if (i + 1) % args.save_freq == 0:
            outcomes = [orr, osc, avg_rew, obj]
            outcomes_t = [orr_t, osc_t, avg_rew_t, obj_t]
            filename = str(i).zfill(4) + ".tar"
            filename_plt = saved_path + "outcomes_" + str(i).zfill(4) + ".png"
            filename_plt_skl = saved_path + "skl_" + str(i).zfill(4) + ".png"
            utils_taxi.get_plt(outcomes, outcomes_t, i, mode="save", filename=filename_plt)
            utils_taxi.get_plt_skl(skl, i, filename=filename_plt_skl)
            utils_taxi.save_data(args=args,
                                 env=env,
                                 episode_trained=i,
                                 decayed_eps=decayed_eps,
                                 time_start=time_start,
                                 outcomes=outcomes,
                                 outcomes_t=outcomes_t,
                                 skl=skl,
                                 networks=networks,
                                 path=saved_path,
                                 name=filename,
                                 )
