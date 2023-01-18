import argparse
import torch

from utils import utils_all


def add_default_args(parser):
    """
    Build default ArgumentParser.

    Parameters
    ----------
    parser: argparse.ArgumentParser
    """
    # Setting for the description.
    parser.add_argument("--description", type=str, default='Experiment',
                        help="General description for this experiment (or setting). It is only used for the reminder.")
    parser.add_argument("--setting_name", type=str, default='setting_0',
                        help="Setting name for the current setup. This name will be used for the folder name.")

    # Setting for the environment.
    parser.add_argument("--grid_size", type=int, default=2, help="Grid size (each axis).")
    parser.add_argument("--num_agents", type=int, default=100, help="Number of agents.")

    # Setting for the reward designer's problem.
    parser.add_argument("--lv_penalty", type=float, default=0, help="Penalty level for the oversupplied grid.")

    # Setting for the networks.
    parser.add_argument("--mode_ac", type=bool, default=True, help="Mode selection (Actor-critic/psi or critic/psi).")
    parser.add_argument("--mode_psi", type=bool, default=False, help="Mode selection (critic or psi).")
    parser.add_argument("--h_dims_a", type=list, default=[], help="Default layer size for actor hidden layers.")
    parser.add_argument("--h_dims_c", type=list, default=[], help="Default layer size for critic hidden layers.")
    parser.add_argument("--h_dims_p", type=list, default=[], help="Default layer size for psi hidden layers.")
    parser.add_argument("--lr_a", type=float, default=0, help="Default learning rate for the actor network.")
    parser.add_argument("--lr_c", type=float, default=0, help="Default learning rate for the critic network.")
    parser.add_argument("--lr_p", type=float, default=0, help="Default learning rate for the psi network.")
    parser.add_argument("--gamma", type=float, default=1, help="Discount factor.")

    # Setting for the experiment.
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes.")
    parser.add_argument("--episode_length", type=int, default=2, help="Episode length for the experiment.")
    parser.add_argument("--epsilon", type=float, default=0.9, help="Epsilon for exploration.")
    parser.add_argument("--num_tests", type=int, default=30, help="Number of tests for each episode.")
    parser.add_argument("--random_seed", type=int, default=1234, help="Random seed.")

    # Setting for the learning.
    parser.add_argument("--K", type=int, default=4, help="Number of samples from the buffer.")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Maximum buffer size.")
    parser.add_argument("--num_mean_actions", type=int, default=5, help="Number of samples for the mean action.")
    parser.add_argument("--mode_lr_decay", type=bool, default=True, help="True if we do learning rate decay.")
    parser.add_argument("--update_freq", type=int, default=5,
                        help="Update frequency of networks (unit : episode).")
    parser.add_argument("--update_freq_target", type=int, default=50,
                        help="Update frequency of target networks (unit : episode).")
    parser.add_argument("--tau", type=float, default=1,
                        help="Learning rate of target networks. If tau is 1, it is hard update.")
    parser.add_argument("--mode_reuse_networks", type=bool, default=False,
                        help="True if we reuse other networks for the initialization.")
    parser.add_argument("--file_path", type=str, default="",
                        help="File path of the results of other networks. "
                             "ex. args.file_path='./results_ssd/setting_14/saved/000011999.tar'")

    # Setting for the draw and the save.
    parser.add_argument("--draw_freq", type=int, default=100,
                        help="Draw frequency of results (unit: episode).")
    parser.add_argument("--save_freq", type=int, default=100,
                        help="Save frequency of results and networks (unit: episode).")
    parser.add_argument("--mode_draw", type=bool, default=True,
                        help="True if we draw a figure(plot) during the training.")
    parser.add_argument("--mode_print_policy_and_q", type=bool, default=True,
                        help="True if we print a action distribution and a Q value during the training.")

    # Setting for the KL divergence.
    parser.add_argument("--mode_kl_divergence", type=bool, default=False,
                        help="True if we reuse previous networks for calculating a KL divergence.")
    parser.add_argument("--file_path_final", type=str, default="",
                        help="File path of the final results of other networks. "
                             "We use same args of these results and train again to calculate the KL divergence. "
                             "You should notice that we have to overlap the description, or other things. "
                             "ex. args.file_path='./results_ssd/setting_14/saved/000011999.tar'")


def overlap_setting(args):
    """
    Overlap some arguments of args of last-trained networks with new ones if we try to calculate KL-divergence.
    To calculate KL-divergence, we compare the last-trained network and the current (training) network.
    Thus, we load the last-trained network's setting(=args).
    Then, we update some arguments with the setting of the current args.

    Parameters
    ----------
    args: argparse.Namespace

    Returns
    -------
    args_trained: argparse.Namespace
    """
    if args.mode_kl_divergence:
        # Caution.
        print("##############################################################")
        print("You should notice that arg is overlapped (KL divergence mode).")
        print("##############################################################")

        # Information from trained files.
        dict_trained = torch.load(args.file_path_final)
        args_trained = dict_trained['args']

        # Overlap information.
        args_trained.description = args.description
        args_trained.setting_name = args.setting_name
        args_trained.random_seed = args.random_seed
        args_trained.mode_kl_divergence = args.mode_kl_divergence
        args_trained.file_path_final = args.file_path_final
    else:
        args_trained = args
    return args_trained


parser = argparse.ArgumentParser()
add_default_args(parser)
args = parser.parse_args()

""" Setting for the description. """
args.description = "Taxi. alpha=0.93 using alpha=0.00. random seed=1243."
args.setting_name = "alpha=0.93_using_alpha=0.00_random_seed=1243"+utils_all.get_current_time_tag()

""" Setting for the environment. """
# args.grid_size = 2
# args.num_agents = 100

""" Setting for the reward designer's problem. """
args.lv_penalty = 0.93

""" Setting for the networks. """
# args.mode_ac = True
args.mode_psi = True
# args.h_dims_a = [32, 16, 8]
args.h_dims_a = [32, 16, 8]
# args.h_dims_c = []
args.h_dims_p = [64, 32, 16]
args.lr_a = 0.0001
# args.lr_c = 0.001
args.lr_p = 0.0005
# args.gamma = 1

""" Setting for the experiment. """
args.num_episodes = 7500
# args.episode_length = 2
args.epsilon = 0.5
args.num_tests = 20
args.random_seed = 1243

""" Setting for the learning. """
args.K = 8
args.buffer_size = 100
# args.num_mean_actions = 5
args.mode_lr_decay = False
args.update_freq = 1
args.update_freq_target = 10
# args.tau = 1
args.mode_reuse_networks = True
# args.file_path = "./results/results_taxi_final/alpha=0.30/7499.tar"
args.file_path = "./results_taxi_IJCAI/lists of policies/0.00/7499.tar"

""" Setting for the draw and the save. """
args.draw_freq = 50
args.save_freq = 500
args.mode_draw = False
args.mode_print_policy_and_q = False

""" Setting for the KL divergence."""
args.mode_kl_divergence = False
# args.file_path_final = "./results/results_taxi_final/alpha=0.63 using alpha=0.50/7499.tar"

""" Validate the setting. """
utils_all.validate_setting(args)

""" Overlap the setting if we calculate the KL divergence. """
args = overlap_setting(args)

""" Execute main_taxi.py file. """
exec(open('main_taxi.py').read())
