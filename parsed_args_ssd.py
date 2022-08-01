import argparse

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
    parser.add_argument("--env", type=str, default="cleanup_modified",
                        help="Name of the environment to use. Can be cleanup_modified or harvest_modified.")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents. Maximum number of agents is 10.")

    # Setting for the reward designer's problem.
    parser.add_argument("--lv_penalty", type=float, default=0, help="Penalty level for agents who eat apple.")
    parser.add_argument("--lv_incentive", type=float, default=0,
                        help="Incentive level for agents 1) who clean the river if the environment is cleanup, "
                             "2) who don't eat apple if the environment is harvest.")

    # Setting for the networks.
    parser.add_argument("--mode_ac", type=bool, default=True, help="Mode selection (Actor-critic/psi or critic/psi).")
    parser.add_argument("--mode_psi", type=bool, default=False, help="Mode selection (critic or psi).")
    parser.add_argument("--h_dims_a", type=list, default=[], help="Default layer size for actor hidden layers.")
    parser.add_argument("--h_dims_c", type=list, default=[], help="Default layer size for critic hidden layers.")
    parser.add_argument("--h_dims_p", type=list, default=[], help="Default layer size for psi hidden layers.")
    parser.add_argument("--lr_a", type=float, default=0, help="Default learning rate for the actor network.")
    parser.add_argument("--lr_c", type=float, default=0, help="Default learning rate for the critic network.")
    parser.add_argument("--lr_p", type=float, default=0, help="Default learning rate for the psi network.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")

    # Setting for the experiment.
    parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes.")
    parser.add_argument("--episode_length", type=int, default=1000, help="Episode length for the experiment.")
    parser.add_argument("--epsilon", type=float, default=0.9, help="Epsilon for exploration.")
    parser.add_argument("--mode_epsilon_decay", type=bool, default=True, help="True if we do epsilon decay.")
    parser.add_argument("--epsilon_decay_ver", type=str, default="linear",
                        help="If mode_epsilon_decay is True, we have to choose the version of epsilon decay."
                             "'linear', 'exponential' can be used.")
    parser.add_argument("--boltz_beta", type=float, default=1,
                        help="Parameter for Boltzmann policy.")
    parser.add_argument("--mode_test", type=bool, default=False,
                        help="True if we do test during the learning. It require double time.")
    parser.add_argument("--random_seed", type=int, default=1234, help="Random seed.")

    # Setting for the learning.
    parser.add_argument("--K", type=int, default=200, help="Number of samples from the buffer.")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Maximum buffer size.")
    parser.add_argument("--mode_lr_decay", type=bool, default=True, help="True if we do learning rate decay.")
    parser.add_argument("--update_freq", type=int, default=5,
                        help="Update frequency of networks (unit : episode).")
    parser.add_argument("--update_freq_target", type=int, default=50,
                        help="Update frequency of target networks (unit : episode).")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="Learning rate of target networks. If tau is 1, it is hard update.")
    parser.add_argument("--mode_one_hot_obs", type=bool, default=True,
                        help="True if we use one-hot encoded observations.")
    parser.add_argument("--mode_reuse_networks", type=bool, default=False,
                        help="True if we reuse other networks for the initialization.")
    parser.add_argument("--file_path", type=str, default='',
                        help="File path of the results of other networks. "
                             "ex. args.file_path='./results_ssd/setting_14/saved/000011999.tar'")

    # Setting for the draw and the save.
    parser.add_argument("--fps", type=int, default=3, help="Frame per second for videos")
    parser.add_argument("--draw_freq", type=int, default=100,
                        help="Frequency of drawing results (unit : episode).")
    parser.add_argument("--save_freq", type=int, default=1000,
                        help="Frequency of saving results and networks (unit : episode).")
    parser.add_argument("--mode_draw", type=bool, default=True,
                        help="True if we draw plt during the training.")

    # TODO : remove this (This task have low priorities)
    # Deprecated arguments (only works for building the environment.).
    parser.add_argument("--use_collective_reward", action="store_true", default=False,
                        help="Train using collective reward instead of individual reward.",)


parser = argparse.ArgumentParser()
add_default_args(parser)
args = parser.parse_args()

""" Setting for the description. """
args.description = 'SSD environment.'
args.setting_name = 'setting_0'+utils_all.get_current_time_tag()

""" Setting for the environment. """
args.env = 'harvest_modified_v2'
args.num_agents = 4

""" Setting for the reward designer's problem. """
args.lv_penalty = 0.00
args.lv_incentive = 0.00

""" Setting for the networks. """
# args.mode_ac = True
args.mode_psi = True
args.h_dims_a = [256, 128, 64, 32]
# args.h_dims_c = [256, 128, 64, 32]
args.h_dims_p = [256, 128, 64, 32]
args.lr_a = 0.0001
# args.lr_c = 0.001
args.lr_p = 0.001
# args.gamma = 0.99

""" Setting for the experiment. """
args.num_episodes = 30000
args.episode_length = 1000
args.epsilon = 0.95
# args.mode_epsilon_decay = True
args.epsilon_decay_ver = 'linear'
# args.boltz_beta = 1.0
# args.mode_test = False
args.random_seed = 1234

""" Setting for the learning. """
args.K = 400
args.buffer_size = 1000000
args.mode_lr_decay = True
args.update_freq = 1
args.update_freq_target = 1
# args.tau = 0.01
# args.mode_one_hot_obs = True
args.mode_reuse_networks = False
args.file_path = './some_paths/000029999.tar'

""" Setting for the draw and the save. """
# args.fps = 3
args.draw_freq = 100000000  # not drawing figure during the training
args.save_freq = 1000
args.mode_draw = True

""" Validate the setting. """
utils_all.validate_setting(args)

""" Execute main_ssd.py file. """
exec(open('main_ssd.py').read())
