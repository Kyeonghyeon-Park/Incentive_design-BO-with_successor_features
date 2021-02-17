import argparse
import numpy as np

from actor_critic import *


designer_alpha = 0.8  # Bayesian optimization으로

parser = argparse.ArgumentParser()
parser.add_argument('--designer_alpha', default=designer_alpha)
parser.add_argument('--sample_size', default=8)
parser.add_argument('--buffer_max_size', default=50)
parser.add_argument('--max_episode_number', default=3500)
parser.add_argument('--discount_factor', default=1)
parser.add_argument('--epsilon', default=0.5)
parser.add_argument('--epsilon_decay', default=False)
parser.add_argument('--mean_action_sample_number', default=5)
parser.add_argument('--obj_weight', default=0.6)
parser.add_argument('--lr_actor', default=0.0001)
parser.add_argument('--lr_critic', default=0.001)
parser.add_argument('--actor_loss_type', default="avg")
parser.add_argument('--update_period', default=10)
parser.add_argument('--trained', default=False)
parser.add_argument('--PATH', default='')
parser.add_argument('--filename', default='')

args = parser.parse_args()

#############
# if want to learn trained network more, try to do this code
# args.trained = True
# args.PATH = './weights/a_lr=0.0001_alpha=0.7/'
# args.filename = 'all.tar'
# args.max_episode_number = 3500
#############
# try other settings
# args.trained = True
# args.PATH = './weights/a_lr=0.0001_alpha=0.5/201031_1823/'
# args.filename = 'all_3499episode.tar'
args.max_episode_number = 5500
args.designer_alpha = 0.5609
args.lr_actor = 0.0005
args.lr_critic = 0.01
# args.actor_loss_type = "mix"
#############
torch.manual_seed(1238)
model = ActorCritic(args)
model.run()
f_alpha = np.average(model.outcome['test']['obj_ftn'][-100:])
print(f_alpha)
