import argparse


def add_default_args(parser):
    # Initial setting
    parser.add_argument("--description", type=str, default='Experiment',
                        help="General description for this experiment (or setting). It is only used for the reminder.",)
    parser.add_argument("--setting_name", type=str, default='setting_0',
                        help="Setting name for the current setup. This name will be used for the folder name.",)
    parser.add_argument("--env", type=str, default="cleanup",
                        help="Name of the environment to use. Can be cleanup_modified or harvest_modified.",)
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents.")

    # Setting for the incentive designer's problem
    parser.add_argument("--lv_penalty", type=float, default=0, help="Penalty level for agents who eat apple.")
    parser.add_argument("--lv_incentive", type=float, default=0, help="Incentive level for agents who clean the river.")

    # Setting for the Networks
    parser.add_argument("--mode_ac", type=bool, default=True, help="Mode selection (Actor-critic/psi or critic/psi).")
    parser.add_argument("--mode_psi", type=bool, default=False, help="Mode selection (critic or psi).")
    parser.add_argument("--h_dims_a", type=list, default=[], help="Default layer size for actor hidden layers.")
    parser.add_argument("--h_dims_c", type=list, default=[], help="Default layer size for critic hidden layers.")
    parser.add_argument("--h_dims_p", type=list, default=[], help="Default layer size for psi hidden layers.")
    parser.add_argument("--lr_a", type=float, default=0, help="Default learning rate for the actor network.")
    parser.add_argument("--lr_c", type=float, default=0, help="Default learning rate for the critic network.")
    parser.add_argument("--lr_p", type=float, default=0, help="Default learning rate for the psi network.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")

    # Setting for the experiment
    parser.add_argument("--episode_num", type=int, default=200, help="Number of episodes.")
    parser.add_argument("--episode_length", type=int, default=1000, help="Episode length for the experiment.")
    parser.add_argument("--epsilon", type=float, default=0.9,
                        help="Epsilon for exploration. Not used yet. (Do we need this?)")
    parser.add_argument("--mode_epsilon_decay", type=bool, default=True,
                        help="True if we do epsilon decay. Not used yet. (Do we need this?)")
    parser.add_argument("--boltz_beta", type=float, default=1,
                        help="Parameter for Boltzmann policy. Not used yet.")

    # Setting for the learning
    parser.add_argument("--K", type=int, default=200, help="Number of samples from the buffer.")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Maximum buffer size.")
    parser.add_argument("--update_freq", type=int, default=5,
                        help="Update frequency of networks (unit : episode).")
    # TODO : If we do soft update, we don't need this anymore
    parser.add_argument("--update_freq_target", type=int, default=50,
                        help="Update frequency of target networks (unit : episode).")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="Learning rate of target networks. If tau is 1, it is hard update.")

    # Setting for the save
    parser.add_argument("--fps", type=int, default=3, help="Frame per second for videos")
    parser.add_argument("--save_freq", type=int, default=1000,
                        help="Save frequency of results and networks (unit : episode).")

    # TODO : remove this (This task have low priorities)
    # Deprecated arguments (only works for building the environment.)
    parser.add_argument("--use_collective_reward", action="store_true",default=False,
                        help="Train using collective reward instead of individual reward.",)


def validate_setting(args):
    if args.mode_ac:
        assert len(args.h_dims_a) != 0 and args.lr_a != 0, "Actor network setting error."
    if args.mode_psi:
        assert len(args.h_dims_p) != 0 and args.lr_p != 0, "Psi network setting error."
    else:
        assert len(args.h_dims_c) != 0 and args.lr_c != 0, "Critic network setting error."


parser = argparse.ArgumentParser()
add_default_args(parser)
args = parser.parse_args()

# Our setting
args.description = 'Experiment for testing the new code. ' \
                   'Soft update is built.'
args.setting_name = 'setting_12'
args.env = 'cleanup_modified'
args.num_agents = 3  # Maximum 10 agents
# args.lv_penalty = 0.5
# args.lv_incentive = 0.3
args.h_dims_a = [64, 32, 16]
args.h_dims_c = [128, 64, 32]
args.lr_a = 0.0001
args.lr_c = 0.001
args.episode_num = 20000
args.epsilon = 0.95
args.buffer_size = 5000000
args.update_freq = 1
args.update_target_freq = 1


# Validate setting
validate_setting(args)
