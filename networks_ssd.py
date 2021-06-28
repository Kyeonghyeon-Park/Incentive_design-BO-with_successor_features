import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import copy
import os


class Actor(nn.Module):
    """
    Actor network
    """
    def __init__(self, net_dims):
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
        net_dims : list
            Layer size (list of dimensions)
            ex. [[10, 64], [64, 64], [64, 4]]
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(net_dims)

        for i in range(self.num_layers):
            fc_i = nn.Linear(net_dims[i][0], net_dims[i][1])
            self.layers.append(fc_i)

    def forward(self, x):
        """
        Forward

        Parameters
        ----------
        x : torch.Tensor
            Input for the actor network (observation)

        Returns
        -------
        x : torch.Tensor
            Return the action probability using softmax (action)
        """
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        x = F.softmax(x, dim=-1)
        return x


class Critic(nn.Module):
    """
    Critic network
    """
    def __init__(self, net_dims):
        """
        Create a new critic network
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
        self.num_layers = len(net_dims)

        for i in range(self.num_layers):
            fc_i = nn.Linear(net_dims[i][0], net_dims[i][1])
            self.layers.append(fc_i)

    def forward(self, observation, action, mean_action):
        """
        Forward

        Parameters
        ----------
        observation : torch.Tensor
            Observation for the critic network
        action : torch.Tensor
            Action for the critic network
        mean_action : torch.Tensor
            Mean action for the critic network

        Returns
        -------
        x : torch.Tensor
            Return the q value
        """
        x = self.layers[0](observation)
        x = F.relu(x)
        x = torch.cat((x, action, mean_action), dim=1)

        for i in range(1, self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        return x


def init_weights(m):
    """
    Define the initialization function for the layers

    Parameters
    ----------
    m
        Type of the layer
    """
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def make_layer_dims(observation_dim, action_dim, feature_dim, hidden_dims, mode='actor'):
    layer_dims = []
    if mode == 'actor':
        for i in range(len(hidden_dims)):
            if i == 0:
                layer_dim = [observation_dim, hidden_dims[i]]
            elif i < len(hidden_dims) - 1:
                layer_dim = [hidden_dims[i - 1], hidden_dims[i]]
            else:  # last layer
                layer_dim = [hidden_dims[i], action_dim]
            layer_dims.append(layer_dim)
    elif mode == 'critic':
        for i in range(len(hidden_dims)):
            if i == 0:
                layer_dim = [observation_dim, hidden_dims[i]]
            elif i == 1:
                layer_dim = [hidden_dims[i - 1] + action_dim * 2, hidden_dims[i]]
            elif i < len(hidden_dims) - 1:
                layer_dim = [hidden_dims[i - 1], hidden_dims[i]]
            else:  # last layer
                layer_dim = [hidden_dims[i], 1]
            layer_dims.append(layer_dim)
    elif mode == 'psi':
        for i in range(len(hidden_dims)):
            if i == 0:
                layer_dim = [observation_dim, hidden_dims[i]]
            elif i == 1:
                layer_dim = [hidden_dims[i - 1] + action_dim * 2, hidden_dims[i]]
            elif i < len(hidden_dims) - 1:
                layer_dim = [hidden_dims[i - 1], hidden_dims[i]]
            else:  # last layer
                layer_dim = [hidden_dims[i], feature_dim]
            layer_dims.append(layer_dim)
    return layer_dims


class Networks(object):
    """
    Define networks (actor-critic / actor-psi / critic / psi)
    """
    def __init__(self, env, args):
        self.action_dim = env.action_space.n
        self.observation_dim = np.prod(env.observation_dim)
        self.feature_dim = env.feature_dim
        self.actor_layers = []
        self.critic_layers = []
        self.psi_layers = []
        self.args = args
        if args.mode_ac:
            self.actor_layers = make_layer_dims(self.observation_dim,
                                                self.action_dim,
                                                self.feature_dim,
                                                args.h_dims_a,
                                                mode='actor')
            self.actor = Actor(self.actor_layers)
            self.actor.apply(init_weights)
            self.actor_target = copy.deepcopy(self.actor)
        if args.mode_psi:
            self.psi_layers = make_layer_dims(self.observation_dim,
                                              self.action_dim,
                                              self.feature_dim,
                                              args.h_dims_p,
                                              mode='psi')
            self.psi = Critic(self.psi_layers)
            self.psi.apply(init_weights)
            self.psi_target = copy.deepcopy(self.psi)
        else:
            self.critic_layers = make_layer_dims(self.observation_dim,
                                                 self.action_dim,
                                                 self.feature_dim,
                                                 args.h_dims_c,
                                                 mode='critic')
            self.critic = Critic(self.critic_layers)
            self.critic.apply(init_weights)
            self.critic_target = copy.deepcopy(self.critic)

    def get_action(self, observation, prev_mean_action, is_target=False):
        if self.args.mode_ac:
            if is_target:
                action_prob = self.actor_target(observation)
            else:
                action_prob = self.actor(observation)
            action_dist = distributions.Categorical(action_prob)
            action = action_dist.sample()
        else:  # TODO : We have to get action from Boltzmann policy (using prev_mean_action)
            pass
            if self.args.mode_psi:
                pass
                # Make value using psi and weight
            else:
                pass
            action = np.random.randint(self.action_dim)
        return action
