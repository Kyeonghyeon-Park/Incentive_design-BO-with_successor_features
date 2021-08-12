import argparse
from config.default_args import add_default_args

# Initial setting
parser = argparse.ArgumentParser()
add_default_args(parser)

# Our initial setting
# Setting for the incentive designer's problem
parser.add_argument("--lv_penalty", type=float, default=0, help="Penalty level for agents who eat apple")
parser.add_argument("--lv_incentive", type=float, default=0, help="Incentive level for agents who clean the river")
# Setting for the Networks
parser.add_argument("--mode_ac", type=bool, default=True, help="Mode selection (Actor-critic/psi or critic/psi)")
parser.add_argument("--mode_psi", type=bool, default=False, help="Mode selection (critic or psi)")
parser.add_argument("--h_dims_a", type=list, default=[], help="Default layer size for actor hidden layers")
parser.add_argument("--h_dims_c", type=list, default=[], help="Default layer size for critic hidden layers")
parser.add_argument("--h_dims_p", type=list, default=[], help="Default layer size for psi hidden layers")
parser.add_argument("--lr_a", type=float, default=0, help="Default learning rate for the actor network")
parser.add_argument("--lr_c", type=float, default=0, help="Default learning rate for the critic network")
parser.add_argument("--lr_p", type=float, default=0, help="Default learning rate for the psi network")
# Setting for the experiment
parser.add_argument("--horizon", type=int, default=20000, help="Maximum step number for the run")
parser.add_argument("--episode_length", type=int, default=1000, help="Episode length for the experiment")
parser.add_argument("--epsilon", type=float, default=0.9, help="Epsilon for exploration")
parser.add_argument("--mode_epsilon_decay", type=bool, default=True, help="True if we do epsilon decay")
parser.add_argument("--boltz_beta", type=float, default=1, help="Parameter for Boltzmann policy")
# Setting for the learning
parser.add_argument("--K", type=int, default=16, help="Number of samples from buffer")
parser.add_argument("--buffer_size", type=int, default=1024, help="Buffer size")
# Setting for the save
parser.add_argument("--fps", type=int, default=3, help="Frame per second for videos")
parser.add_argument("--setting_name", type=str, default='setting_0', help="Setting name for the current setup")
parser.add_argument("--description", type=str, default='Experiment', help="Description for this experiment (setting)")

args = parser.parse_args()

# Our setting
args.description = 'Experiment for testing the new beam (3x3 grid beam)'
args.setting_name = 'setting_6'
args.env = 'cleanup_modified'
args.num_agents = 3  # Maximum 10 agents
# args.lv_penalty = 0.5
# args.lv_incentive = 0.3
args.horizon = 400000
args.h_dims_a = [64, 32, 16]
args.h_dims_c = [128, 64, 32]
args.lr_a = 0.0001
args.lr_c = 0.001

# Validate setting
if args.mode_ac:
    assert len(args.h_dims_a) != 0 and args.lr_a != 0, "Actor network setting error"
if args.mode_psi:
    assert len(args.h_dims_p) != 0 and args.lr_p != 0, "Psi network setting error"
else:
    assert len(args.h_dims_c) != 0 and args.lr_c != 0, "Critic network setting error"
assert args.horizon % args.episode_length == 0, "Maximum step number cannot be divided by episode length"
