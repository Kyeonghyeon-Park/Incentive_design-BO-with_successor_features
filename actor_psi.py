# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import os

from grid_world import ShouAndDiTaxiGridGame
from utils import *

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
To-do or check list
1)  I build the reuse function to initialize the network
    Not decided yet : multiple networks for same w (if args.reuse_type_and_w is {'type': "specific", 'w': ###}
2)  alpha should be 0.XXXX
+)  We could decide the stopping point (In my algorithm, I just train the network a fixed number of times)
    If we try to find the benefit of the successor feature network, 
    we have to say that SF network helps the convergence speed or the performance
"""

# %% Define the actor network and the psi network
class Actor(nn.Module):
    """
    Actor network
    """
    def __init__(self, net_size):
        """
        Create a new actor network
        Build the MLP using the net_size

        Attributes
        ----------
        layers : torch.nn.modules.container.ModuleList
            Container for layers
        num_layers : int
            Number of layers except the input layer

        Parameters
        ----------
        net_size : list
            Layer size (list of dimensions)
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(net_size) - 1

        for i in range(self.num_layers):
            fc_i = nn.Linear(net_size[i], net_size[i + 1])
            self.layers.append(fc_i)

    def forward(self, x):
        """
        Forward

        Parameters
        ----------
        x : torch.Tensor
            Input for the actor network (actor_input)

        Returns
        -------
        x : torch.Tensor
            Return the action probability using softmax
        """
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        x = F.softmax(x, dim=-1)
        return x


class Psi(nn.Module):
    """
    Psi network (deep successor feature network)
    """
    def __init__(self, net_size):
        """
        Create a new psi network
        Build the MLP using the net_size

        Attributes
        ----------
        layers : torch.nn.modules.container.ModuleList
            Container for layers
        num_layers : int
            Number of layers except the input layer

        Parameters
        ----------
        net_size : list
            Layer size (list of dimensions)
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(net_size) - 1

        for i in range(self.num_layers):
            fc_i = nn.Linear(net_size[i], net_size[i + 1])
            self.layers.append(fc_i)

    def forward(self, x):
        """
        Forward

        Parameters
        ----------
        x : torch.Tensor
            Input for the psi network (psi_input)

        Returns
        -------
        x : torch.Tensor
            Return the successor feature
        """
        for i in range(self.num_layers):
            x = self.layers[i](x)  # [12, 64, 32, 16, 2]
            if i < self.num_layers - 1:  # if 0 < 4
                x = F.relu(x)

        return x


def init_weights(m):
    """
    Define the initialization function for the layers
    Use the kaiming_normal because xavier_normal is not good for ReLU

    Parameters
    ----------
    m
        Type of the layer
    """
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


# %% Define the main body
class ActorPsi(object):
    """
    Define the actor-psi network
    """
    def __init__(self, args):
        """
        Create a new actor-psi network
        Load the successor feature network and actor network of previous w (alpha)
        to initialize the network if args.##### is True

        Attributes
        ----------
        world : grid_world.ShouAndDiTaxiGridGame
            Define the taxi grid game
        actor_layer : list
            Dimension of layers of the actor network
        psi_layer : list
            Dimension of layers of the psi network
        actor : Actor
            Actor network
        psi : Psi
            Psi network (deep successor feature network)
        lr_actor : float
            Learning rate for the actor network
        lr_psi : float
            Learning rate for the psi network
        actor_loss_type : str
            "avg" or "max" or "mix"
            How to calculate the v_observation (only use "avg")
        optimizerA : torch.optim.adam.Adam
            Optimizer for the actor network
        optimizerP : torch.optim.adam.Adam
            Optimizer for the psi network
        update_period : int
            Period for updating the actor network and the psi network
        discount_factor : float
            Discount factor
        designer_alpha : float
            Designer's decision (penalty for overcrowded grid)
        epsilon : float
            Probability of random action for exploration
        epsilon_decay : boolean
            True if epsilon is decayed during the learning
        buffer : list
            Buffer
        buffer_max_size : int
            Maximum size of the buffer
            If the size of the buffer becomes bigger than buffer_max_size, erase the oldest data
        K : int
            Sample size
        mean_action_sample_number : int
            To calculate the mean action at the next observation, we have to sample the value of the mean action
            Sample size
        trained_episode_number : int
            Trained episode number of the pre-trained network (for learn the trained network more)
            This is not the episode number of previous alpha (w)
        ####################################
        trained_time : float
            Trained time of the pre-trained network (for learn the trained network more)
            This is not the trained time of previous alpha (w)
        ####################################
        overall_time : dict
            Dict of cumulative time: train, test, total
        max_episode_number : int
            Maximum episode number for the training
        obj_weight : float
            Weight of the ORR
            Objective is weighted average of the ORR and (1-OSC)
        test_size : int
            Number of evaluations for testing
            We should take mean for getting performance
        outcome : dict
            Outcome for previous episodes
        previous_networks : list
            List of w and previous learned networks
            If args.reuse_psi_and_actor is True and , network will initialize the network with previous one
            Row : [alpha, actor, psi]
            It is saved in './weights_and_networks'
        previous_information : str
            Information of previous networks in this learning (just text)
        actor_target : Actor
            Target actor network (update per "update_period" episodes)
        psi_target : Psi
            Target psi network (update per "update_period" episodes)

        Parameters
        ----------
        args : argparse.Namespace
        """
        # Generate the game
        self.world = ShouAndDiTaxiGridGame()

        # Generate the actor network and psi network
        self.actor_layer = [7, 32, 16, 8, 4]
        self.psi_layer = [12, 64, 32, 16, 2]
        self.actor = Actor(self.actor_layer)
        self.psi = Psi(self.psi_layer)

        # Parameters and instances for networks
        self.actor.apply(init_weights)
        self.psi.apply(init_weights)
        self.lr_actor = args.lr_actor
        self.lr_psi = args.lr_psi
        self.actor_loss_type = args.actor_loss_type
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizerP = optim.Adam(self.psi.parameters(), lr=self.lr_psi)

        # Parameters and instances for the learning
        self.update_period = args.update_period
        self.discount_factor = args.discount_factor
        self.designer_alpha = args.designer_alpha
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.buffer = []
        self.buffer_max_size = args.buffer_max_size
        self.K = args.sample_size
        self.mean_action_sample_number = args.mean_action_sample_number
        self.trained_episode_number = 0
        self.overall_time = {'train': [],
                             'test': [],
                             'total': [],
                             }
        # self.trained_time = 0
        self.max_episode_number = args.max_episode_number

        # Parameters and instances for the outcome and objective
        self.obj_weight = args.obj_weight
        self.test_size = args.test_size
        self.outcome = {'train': {'ORR': [[] for i in range(1)],
                                  'OSC': [[] for i in range(1)],
                                  'avg_reward': [[] for i in range(1)],
                                  'obj_ftn': [[] for i in range(1)],
                                  },
                        'test': {'ORR': [[] for i in range(self.test_size)],
                                 'OSC': [[] for i in range(self.test_size)],
                                 'avg_reward': [[] for i in range(self.test_size)],
                                 'obj_ftn': [[] for i in range(self.test_size)],
                                 },
                        }

        # Parameters and instances for reusing previous networks
        if not os.path.isdir('./weights_and_networks/'):
            os.makedirs('./weights_and_networks/')
        try:
            previous_networks = torch.load('./weights_and_networks/previous_networks.tar')
            self.previous_networks = previous_networks['previous_networks']
        except FileNotFoundError:
            self.previous_networks = []
        self.previous_information = ''

        # Reuse previous networks to initialize
        if args.reuse_actor_and_psi:
            assert len(self.previous_networks) != 0, "There is no previous network"
            if args.reuse_type_and_alpha['type'] == "recent":
                self.actor.load_state_dict(self.previous_networks[-1][1])
                self.psi.load_state_dict(self.previous_networks[-1][2])
                self.previous_information = 'Reuse recent network, alpha='+str(self.previous_networks[-1][0])
            elif args.reuse_type_and_alpha['type'] == "nearest":
                alphas = [item[0] for item in self.previous_networks]
                idx = min(range(len(alphas)), key=lambda i: abs(alphas[i]-self.designer_alpha))
                self.actor.load_state_dict(self.previous_networks[idx][1])
                self.psi.load_state_dict(self.previous_networks[idx][2])
                self.previous_information = 'Reuse nearest network, alpha='+str(self.previous_networks[idx][0])
            else:  # args.reuse_type_and_w['type'] == "specific"
                alphas = [item[0] for item in self.previous_networks]
                assert args.reuse_type_and_alpha['alpha'] in alphas, "Chosen alpha is not in the previous network"
                idx = alphas.index(args.reuse_type_and_alpha['alpha'])
                self.actor.load_state_dict(self.previous_networks[idx][1])
                self.psi.load_state_dict(self.previous_networks[idx][2])
                self.previous_information = 'Reuse specific network, alpha='+str(self.previous_networks[idx][0])
            print(self.previous_information)

        # Learn the network more
        if args.learn_more:
            data = torch.load(args.PATH + args.filename)
            self.actor.load_state_dict(data['actor'])
            self.psi.load_state_dict(data['psi'])
            self.lr_actor = data['parameters']['lr_actor']
            self.lr_psi = data['parameters']['lr_psi']
            self.actor_loss_type = data['parameters']['actor_loss_type']
            self.optimizerA = optim.Adam(self.actor.parameters())
            self.optimizerA.load_state_dict(data['optimizerA'])
            self.optimizerP = optim.Adam(self.psi.parameters())
            self.optimizerP.load_state_dict(data['optimizerP'])
            self.update_period = data['update_period']
            self.discount_factor = data['parameters']['discount_factor']
            self.designer_alpha = data['parameters']['designer_alpha']
            self.epsilon = data['parameters']['epsilon']
            self.epsilon_decay = data['parameters']['epsilon_decay']
            self.buffer = data['buffer']
            self.buffer_max_size = data['parameters']['buffer_max_size']
            self.K = data['parameters']['sample_size']
            self.mean_action_sample_number = data['parameters']['mean_action_sample_number']
            self.trained_episode_number = data['parameters']['max_episode_number']
            self.overall_time = data['overall_time']
            # self.trained_time = data['total_time']
            self.max_episode_number = args.max_episode_number
            self.obj_weight = data['parameters']['obj_weight']
            self.test_size = data['parameters']['test_size']
            self.outcome = data['outcome']

        # Build target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.psi_target = copy.deepcopy(self.psi)

    def get_outcome(self, overall_fare, mode, idx=0):
        """
        Define the outcome function (add the current result)
        Outcome will be saved in the list which is for the specific test
        idx is always 0 for the training because there is no multiple outcomes for the training

        Parameters
        ----------
        overall_fare : np.array
            (fee, fare)
        mode : str
            'train' or 'test'
        idx : int
            Test number (outcome will be saved in the list which is for the test number)
        """
        # Order response rate / do not consider no demand case in the game
        total_request = self.world.demand[:, 3].shape[0]
        fulfilled_request = np.sum(self.world.demand[:, 3])
        self.outcome[mode]['ORR'][idx].append(fulfilled_request / total_request)

        # Overall service charge ratio
        if overall_fare[1] != 0:
            self.outcome[mode]['OSC'][idx].append(overall_fare[0] / overall_fare[1])
        else:
            self.outcome[mode]['OSC'][idx].append(0)

        # Average reward of all agents
        self.outcome[mode]['avg_reward'][idx].append((overall_fare[1] - overall_fare[0]) / self.world.number_of_agents)
        self.outcome[mode]['obj_ftn'][idx].append(self.obj_weight * self.outcome[mode]['ORR'][idx][-1]
                                                  + (1 - self.obj_weight) * (1 - self.outcome[mode]['OSC'][idx][-1]))

    def get_location_agent_number_and_prob(self, joint_observation, current_time):
        """
        Define the function that returns agents' number and action distribution for each location
        It uses the target actor network

        Parameters
        ----------
        joint_observation : np.array
            Joint observation of agents
        current_time : int
            Agent's current time

        Returns
        -------
        agent_num : list
            List of the number of agents in the grid
        action_dist_set : list
            List of the action distribution for each location
        """
        agent_num = []
        action_dist_set = []
        for loc in range(4):
            agent_num.append(np.sum((joint_observation[:, 0] == loc) & (joint_observation[:, 1] == current_time)))
            action_dist = get_action_dist(self.actor_target, [loc, current_time])
            action_dist_set.append(action_dist)

        return agent_num, action_dist_set

    def sample_mean_action(self, observation, action, agent_num, action_dist_set):
        """
        Sample mean action using observation, action, agent_num, and action_dist_set

        Parameters
        ----------
        observation : np.array
            Observation of the agent
        action : int
            Action of the agent
        agent_num : list
            List of the number of agents in the grid
        action_dist_set : list
            List of the action distribution for each location

        Returns
        -------
        mean_action : float
            Return the value of mean action
        """
        temp_observation = self.world.move_agent(observation, action)
        temp_location = temp_observation[0]
        temp_time = temp_observation[1]
        local_demand = np.argwhere((self.world.demand[:, 0] == temp_time) & (self.world.demand[:, 1] == temp_location))[
                       :, 0]
        local_demand_num = local_demand.shape[0]
        loc_agent_num = 0

        # Sampling the number which represents the number of agents who want to go to the location of available action
        for loc in range(4):
            num = agent_num[loc]
            prob = action_dist_set[loc].probs[0][action].detach().numpy()
            loc_agent_num = loc_agent_num + np.random.binomial(num, prob)
        try:
            mean_action = local_demand_num / loc_agent_num
        except ZeroDivisionError:
            if local_demand_num > 0:
                mean_action = 1
            else:
                mean_action = 0

        return mean_action

    def get_psi_expectation_over_mean_action(self, observation, action, agent_num, action_dist_set):
        """
        Define the function for expectation over mean action using sampling
        We need psi expectation over mean action for updating psi network
        We can get q expectation over mean action using this
        Therefore, we don't have to return the q expectation

        Parameters
        ----------
        observation : np.array
            Observation of the agent
        action : int
            Action of the agent
        agent_num : list
            List of the number of agents in the grid
        action_dist_set : list
            List of the action distribution for each location

        Returns
        -------
        psi_observation_action : torch.Tensor
            Return the psi expectation over mean action
        """
        psi_observation_action = torch.tensor([0, 0])

        sample_number = self.mean_action_sample_number

        for expectation_over_mean_action in range(sample_number):
            mean_action = self.sample_mean_action(observation, action, agent_num, action_dist_set)
            psi_input = get_psi_input(observation, action, mean_action)
            psi_observation_action = psi_observation_action + self.psi_target(psi_input) / sample_number

        return psi_observation_action

    def calculate_actor_loss(self, sample, agent_id):
        """
        Define the actor loss function for one sample and agent id
        It uses psi_expectation_over_mean_action and w to calculate the q value
        Parameters
        ----------
        sample : list
            Sample from the buffer
        agent_id : int
            Agent's id

        Returns
        -------
        actor_loss : torch.tensor
            Return the actor loss for the specific agent
        """
        w = np.array([1, self.designer_alpha])
        observation = sample[0][agent_id]
        action = sample[1][agent_id]
        with torch.no_grad():
            agent_num, action_dist_set = self.get_location_agent_number_and_prob(sample[0], observation[1])
            available_action_set = self.world.get_available_action_from_location(observation[0])
            q_observation_set = torch.zeros(4)
            v_observation_avg = 0
            target_action_dist = get_action_dist(self.actor_target, observation)
            for available_action in available_action_set:
                psi_expectation_over_mean_action = self.get_psi_expectation_over_mean_action(observation,
                                                                                            available_action,
                                                                                            agent_num,
                                                                                            action_dist_set)
                q_value = get_q(psi_expectation_over_mean_action, w)

                q_observation_set[available_action] = q_value
                v_observation_avg = v_observation_avg + target_action_dist.probs[0][available_action] \
                                    * q_observation_set[available_action]

            v_observation_max = max(q_observation_set)

        action = torch.tensor(action)
        action_dist = get_action_dist(self.actor, observation)
        q_observation = q_observation_set[action]

        if self.actor_loss_type == "avg":
            v_observation = v_observation_avg
        elif self.actor_loss_type == "max":
            v_observation = v_observation_max
        else:  # self.actor_loss_type == "mix"
            v_observation = 1/2 * (v_observation_avg + v_observation_max)
        actor_loss = - (q_observation - v_observation) * action_dist.log_prob(action)

        return actor_loss

    def calculate_psi_loss(self, sample, agent_id):
        """
        Define the psi loss function for one sample and agent id
        We get next_action using q (q for next state is calculated by psi expectation and w)

        Parameters
        ----------
        sample : list
            Sample from the buffer
        agent_id : int
            Agent's id

        Returns
        -------
        psi_loss : torch.Tensor
            Return the psi loss for the specific agent
        """
        w = np.array([1, self.designer_alpha])
        observation = sample[0][agent_id]
        action = sample[1][agent_id]
        # reward = sample[2][agent_id]
        mean_action = sample[3][agent_id]
        next_observation = sample[4][agent_id]
        phi = sample[5][agent_id]
        with torch.no_grad():
            if next_observation[1] != self.world.max_episode_time:
                available_action_set = self.world.get_available_action_from_location(next_observation[0])

                q_next_observation_set = []
                psi_next_observation_set = []
                # get each location's agent numbers and action distributions from next_joint_observation
                agent_num, action_dist_set = self.get_location_agent_number_and_prob(sample[4], next_observation[1])

                for available_action in available_action_set:
                    psi_next_observation_action = self.get_psi_expectation_over_mean_action(next_observation,
                                                                                            available_action,
                                                                                            agent_num,
                                                                                            action_dist_set)
                    q_next_observation_action = get_q(psi_next_observation_action, w)

                    q_next_observation_set.append(q_next_observation_action)
                    psi_next_observation_set.append(psi_next_observation_action)
                # Get next action which has maximum q value
                max_idx = np.argmax(q_next_observation_set)
                # Get next psi
                max_psi_next_observation = psi_next_observation_set[max_idx]

            else:
                max_psi_next_observation = torch.tensor([[0, 0]])
        # test
        max_psi_next_observation = torch.tensor([[0, 0]])
        psi_input = get_psi_input(observation, action, mean_action)
        phi = torch.tensor([phi])
        # print("phi")
        # print(phi)
        # print("psi results")
        # print(self.psi(psi_input))
        # print("max_psi_next_observation")
        # print(max_psi_next_observation)
        psi_loss = phi + self.discount_factor * max_psi_next_observation - self.psi(psi_input)
        # print("psi_loss")
        # print(psi_loss)

        return psi_loss

    def train(self):
        """
        Define the train function to train the actor network and the psi network
        """
        self.world.initialize_game(random_grid=False)
        global_time = 0
        overall_fare = np.array([0, 0], 'float')

        while global_time is not self.world.max_episode_time:

            available_agent = self.world.get_available_agent(global_time)

            joint_action = []  # available agents' joint action
            for agent_id in available_agent:
                available_action_set = self.world.get_available_action(agent_id)
                exploration = np.random.rand(1)[0]
                if exploration < self.epsilon:
                    random_action = np.random.choice(available_action_set)
                    joint_action.append(random_action.item())
                else:
                    action_dist = get_action_dist(self.actor_target, self.world.joint_observation[agent_id])
                    # runtime error if specific probability is too small
                    action = action_dist.sample()
                    joint_action.append(action.item())

            # After the step, add (o, a, r, a_bar, o_prime, f) to the replay buffer B (only train)
            if len(available_agent) != 0:
                buffer, overall_fare = self.world.step(available_agent, joint_action, self.designer_alpha, self.buffer,
                                                       overall_fare, train=True)
                self.buffer = buffer[-self.buffer_max_size:]
            global_time += 1

        # Get outcome of train episode
        self.get_outcome(overall_fare, mode='train')

        # Update the network
        sample_id_list = np.random.choice(len(self.buffer), self.K, replace=True)
        actor_loss = torch.tensor([[0]])
        psi_loss = torch.tensor([[0, 0]])

        update_count = 0

        for sample_id in sample_id_list:
            sample = self.buffer[sample_id]
            if sample[0][0][1] != 0:
                continue
            for agent_id in range(self.world.number_of_agents):
                if sample[1][agent_id] is not None:
                    actor_loss = actor_loss + self.calculate_actor_loss(sample, agent_id)
                    psi_loss = psi_loss + self.calculate_psi_loss(sample, agent_id) ** 2
                    update_count += 1
                else:
                    continue

        # not divided by K (Because all agents are homogeneous, all experiences (# of available agents) used for update)
        if update_count != 0:
            actor_loss = actor_loss / update_count
            psi_loss = psi_loss / update_count

            self.optimizerA.zero_grad()
            self.optimizerP.zero_grad()
            actor_loss.backward()
            psi_loss.backward(torch.Tensor([[1, 1]]))
            self.optimizerA.step()
            self.optimizerP.step()

    def evaluate(self, idx):
        """
        Define the evaluate function to evaluate the trained actor network
        Network will evaluate multiple times to get the mean

        Parameters
        ----------
        idx : int
            Test number
        """
        self.world.initialize_game(random_grid=False)
        global_time = 0
        overall_fare = np.array([0, 0], 'float')

        while global_time is not self.world.max_episode_time:
            available_agent = self.world.get_available_agent(global_time)
            joint_action = []
            for agent_id in available_agent:
                action_dist = get_action_dist(self.actor, self.world.joint_observation[agent_id])
                if global_time == 0 and agent_id in [0, len(available_agent) - 1] and idx == 0:
                    loc = self.world.joint_observation[agent_id][0]
                    print(f"Agent in #{loc}'s action prob. : {action_dist.probs}")
                    # """Will be modified to show the agent's location (#1 and #2) and action probability"""
                    # print(agent_id, action_dist.probs)
                action = action_dist.sample()  # runtime error if specific probability is too small
                joint_action.append(action.item())
            if len(available_agent) != 0:
                buffer, overall_fare = self.world.step(available_agent, joint_action, self.designer_alpha, self.buffer,
                                                       overall_fare, train=False)
            global_time += 1

        self.get_outcome(overall_fare, mode='test', idx=idx)

    def save_network(self):
        """
        Save previous networks to the folder
        """
        self.previous_networks.append([self.designer_alpha, self.actor.state_dict(), self.psi.state_dict()])
        save_previous_networks(self.previous_networks)

    def save_model(self, PATH, episode):
        """
        Define the save function to save the model and the results

        Parameters
        ----------
        PATH : str
            PATH name
        episode : int
            Number of trained episodes
        """
        if not os.path.isdir(PATH):
            os.makedirs(PATH)
        filename = 'all_' + str(episode) + 'episode.tar'
        torch.save({
            'actor_layer': self.actor_layer,
            'psi_layer': self.psi_layer,
            'actor': self.actor.state_dict(),
            'psi': self.psi.state_dict(),
            'optimizerA': self.optimizerA.state_dict(),
            'optimizerP': self.optimizerP.state_dict(),
            'parameters': {'lr_actor': self.lr_actor,
                           'lr_psi': self.lr_psi,
                           'actor_loss_type': self.actor_loss_type,
                           'update_period': self.update_period,
                           'discount_factor': self.discount_factor,
                           'designer_alpha': self.designer_alpha,
                           'epsilon': self.epsilon,
                           'epsilon_decay': self.epsilon_decay,
                           'buffer_max_size': self.buffer_max_size,
                           'sample_size': self.K,
                           'mean_action_sample_number': self.mean_action_sample_number,
                           'max_episode_number': self.max_episode_number,
                           'obj_weight': self.obj_weight,
                           'test_size': self.test_size,
                           },
            'buffer': self.buffer,
            'outcome': self.outcome,
            'overall_time': self.overall_time,
            'previous_information': self.previous_information,
        }, PATH + filename)

    def run(self):
        """
        Run the network (train and test)
        """
        np.random.seed(seed=1234)
        run_start = time.time()
        for episode in np.arange(self.trained_episode_number, self.max_episode_number):
            train_start = time.time()
            self.train()
            train_end = time.time()
            test_start = train_end
            for idx in range(self.test_size):
                with torch.no_grad():
                    self.evaluate(idx)
            test_end = time.time()
            self.overall_time['train'].append(train_end-train_start)
            self.overall_time['test'].append(test_end-test_start)
            self.overall_time['total'].append(test_end-train_start)

            if (episode + 1) % self.update_period == 0:
                self.actor_target = copy.deepcopy(self.actor)
                self.psi_target = copy.deepcopy(self.psi)

                with torch.no_grad():
                    print_updated_q_using_psi(self.psi, self.designer_alpha)
                    print_action_distribution(self.actor)
                print_information_per_n_episodes(self.outcome, self.overall_time, episode)
                draw_plt_test(self.outcome, episode)

            if self.epsilon_decay:
                if (episode + 1) % 20 == 0:
                    self.epsilon = max(self.epsilon - 0.01, 0.01)

            if (episode + 1) % 1 == 0:
                if self.previous_information == '':
                    PATH = './results/alpha=' + str(round(self.designer_alpha, 4)) +'/' \
                           + time.strftime('%y%m%d_%H%M', time.localtime(run_start)) + '/'
                else:
                    PATH = './results/alpha=' + str(round(self.designer_alpha, 4)) + '(' + self.previous_information \
                           + ')/' + time.strftime('%y%m%d_%H%M', time.localtime(run_start)) + '/'

                self.save_model(PATH, episode)

        self.save_network()


