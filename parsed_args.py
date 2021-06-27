import argparse
from config.default_args import add_default_args

# Initial setting
parser = argparse.ArgumentParser()
add_default_args(parser)

# Our initial setting
parser.add_argument("--lv_penalty", type=float, default=0, help="Penalty level for agents who eat apple")
parser.add_argument("--lv_incentive", type=float, default=0, help="Incentive level for agents who clean the river")
parser.add_argument("--horizon", type=int, default=200, help="Maximum period for the run")
parser.add_argument("--mode_ac", type=bool, default=True, help="Mode selection (Actor-critic/psi or critic/psi)")
parser.add_argument("--mode_psi", type=bool, default=False, help="Mode selection (critic or psi)")
parser.add_argument("--h_dims_a", type=list, default=[], help="Default layer size for actor hidden layers")
parser.add_argument("--h_dims_c", type=list, default=[], help="Default layer size for critic hidden layers")
parser.add_argument("--h_dims_p", type=list, default=[], help="Default layer size for psi hidden layers")
args = parser.parse_args()

# Our setting
args.env = 'cleanup_modified'
args.num_agents = 3  # Maximum 10 agents
args.lv_penalty = 0.5
args.lv_incentive = 0.3
args.h_dims_a = [128, 64, 32, 16]
args.h_dims_c = [256, 128, 64, 32]

# Validate setting
if args.mode_ac:
    assert len(args.h_dims_a) != 0, "No layers for the actor network"
if args.mode_psi:
    assert len(args.h_dims_p) != 0, "No layers for the psi network"
else:
    assert len(args.h_dims_c) != 0, "No layers for the critic network"

