import argparse
import numpy as np

from actor_psi import *


designer_alpha = 0.8  # Given by Bayesian optimization

"""
Initial settings for learning
Set several parameters using parser

Parameters
----------
lr_actor : float
    Learning rate for the actor network
lr_psi : float
    Learning rate for the psi network
actor_loss_type : str
    It is used for calculating v_observation using q value
    Actually, we should "avg" option
    Other things are just for testing
    Option : "avg", "max", "mix"
update_period : int
    Period for updating the actor network and the psi network
discount_factor : float
    Discount factor for learning
designer_alpha : float
    Designer's decision (penalty for overcrowded grid)
    This value is given by Bayesian optimization
    Range : 0~1
epsilon : float
    Probability of random action for exploration
    Range : 0~1
epsilon_decay : boolean
    True if epsilon is decayed during the learning
buffer_max_size : int
    Maximum buffer size for training
    If the number of elements in the buffer is bigger than buffer_max_size, buffer remain recent elements
sample_size : int
    Sample size for learning networks
mean_action_sample_number : int
    Sample size for expectation over mean action
    To calculate the mean action at the next observation, we have to get the value of the mean action
    Therefore, we sample (calculate) the value of the mean action
max_episode_number : int
    Maximum episode number for the training
obj_weight : float
    Weight of the ORR
    Objective is weighted average of the ORR and (1-OSC)
reuse_actor_and_psi : boolean
    True if we want to reuse the actor network and the psi network for initialization
    If true and there is no previous network, network is random initialized
reuse_type_and_alpha : dict
    If we want to reuse the actor network and the psi network, 
    It is composed of type and alpha value
    alpha value is only needed when type is "specific"
    If type is "recent", actor-psi network will be initialized with the most recent one
    If type is "nearest", (compare with current alpha) nearest alpha's network will be used for initialization 
    If type is "specific", it will use specific alpha's network to initialize the current actor-psi network
        Not decided yet : multiple networks for same alpha
    type option : "recent", "nearest", "specific"
learn_more : boolean
    True if we want to learn the network more
    You should write PATH and filename to get learned network (which should be learned more)
    You should set max_episode_number bigger than those of learned network
PATH : str
    ex. args.PATH = './results/a_lr=0.0001_alpha=0.7/'
filename : str
    ex. args.filename = 'all.tar'
"""

parser = argparse.ArgumentParser()
# Parameters for networks
parser.add_argument('--lr_actor', default=0.0005)
parser.add_argument('--lr_psi', default=0.01)
parser.add_argument('--actor_loss_type', default="avg")

# Parameters for the learning
parser.add_argument('--update_period', default=10)
parser.add_argument('--discount_factor', default=1)
parser.add_argument('--designer_alpha', default=designer_alpha)
parser.add_argument('--epsilon', default=0.5)
parser.add_argument('--epsilon_decay', default=True)
parser.add_argument('--buffer_max_size', default=50)
parser.add_argument('--sample_size', default=8)
parser.add_argument('--mean_action_sample_number', default=5)
parser.add_argument('--max_episode_number', default=5500)

# Parameters for the outcome and objective
parser.add_argument('--obj_weight', default=0.6)

# Parameters for reusing previous networks
parser.add_argument('--reuse_actor_and_psi', default=True)
parser.add_argument('--reuse_type_and_alpha', default={'type': "recent", 'alpha': 0})

# Parameters for learn the network more
parser.add_argument('--learn_more', default=False)
parser.add_argument('--PATH', default='')
parser.add_argument('--filename', default='')

args = parser.parse_args()

"""
If you want to learn trained network more, try to do this code

Codes
-------
args.learn_more = True
args.PATH = './results/a_lr=0.0001_alpha=0.7/'
args.filename = 'all.tar'
args.max_episode_number = 3500
"""

"""
Try other settings

Samples
-------
# args.learn_more = True
# args.PATH = './results/a_lr=0.0001_alpha=0.5/201031_1823/'
# args.filename = 'all_3499episode.tar'
args.max_episode_number = 5500
args.designer_alpha = 0.5609
args.lr_actor = 0.0005
args.lr_critic = 0.01
# args.actor_loss_type = "mix"
"""
args.designer_alpha = 0.65

"""
Run the model
"""

torch.manual_seed(1238)
model = ActorPsi(args)
model.run()
f_alpha = np.average(model.outcome['test']['obj_ftn'][-100:])
print(f_alpha)
