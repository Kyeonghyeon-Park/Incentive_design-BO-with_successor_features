import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def make_vars(n, mode):
    for _ in range(n):
        if mode == 'list':
            yield []
        elif mode == 'dict':
            yield {}
        else:
            raise NotImplementedError("Possible options of mode are list and dict.")


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


def make_layer_dims(observation_size, action_size, feature_size, hidden_dims, mode='actor'):
    """
    Make the list of layer dimensions.
    Each element is the dimension of layers ([input_dim, output_dim]).
    Unlike previous implementation (taxi example), action is added into second layer.
    In addition, the dimension of mean_action is assumed to be same as the dimension of action.

    Parameters
    ----------
    observation_size : int
    action_size : int
    feature_size : int
    hidden_dims : list
        List of hidden layers' size
    mode : str
        'actor' or 'critic' or 'psi'

    Returns
    -------
    layer_dims : list
        List of list
        Each element is the dimension of layers ([input_dim, output_dim])
    """
    layer_dims = []
    if mode == 'actor':
        for i in range(len(hidden_dims)):
            if i == 0:
                layer_dim = [observation_size, hidden_dims[i]]
            else:
                layer_dim = [hidden_dims[i - 1], hidden_dims[i]]
            layer_dims.append(layer_dim)
        layer_dim = [hidden_dims[-1], action_size]
        layer_dims.append(layer_dim)
    elif mode == 'critic':
        for i in range(len(hidden_dims)):
            if i == 0:
                layer_dim = [observation_size, hidden_dims[i]]
            elif i == 1:
                layer_dim = [hidden_dims[i - 1] + action_size, hidden_dims[i]]
            else:
                layer_dim = [hidden_dims[i - 1], hidden_dims[i]]
            layer_dims.append(layer_dim)
        layer_dim = [hidden_dims[-1], action_size]
        layer_dims.append(layer_dim)
    elif mode == 'psi':
        for i in range(len(hidden_dims)):
            if i == 0:
                layer_dim = [observation_size, hidden_dims[i]]
            elif i == 1:
                layer_dim = [hidden_dims[i - 1] + action_size, hidden_dims[i]]
            else:
                layer_dim = [hidden_dims[i - 1], hidden_dims[i]]
            layer_dims.append(layer_dim)
        layer_dim = [hidden_dims[-1], action_size * feature_size]
        layer_dims.append(layer_dim)
    else:
        raise ValueError
    return layer_dims


class Actor(nn.Module):
    """
    Actor network based on MLP structure.
    """

    def __init__(self, obs_size, act_size, fea_size, hidden_dims):
        """
        Create a new actor network.
        The network is composed of linear (or fully connected) layers.
        After the linear layer, except the last case, we use ReLU for the activation function.
        Lastly, we use softmax to return the action probabilities.

        Parameters
        ----------
        obs_size
        act_size
        fea_size
        hidden_dims : list
            Dimensions of hidden layers.
            ex. if hidden_dims = [128, 64, 32],
                layer_dims = [[observation_size, 128], [128, 64], [64, 32], [32, action_size]].
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.observation_size = obs_size
        self.action_size = act_size
        self.feature_size = fea_size
        layer_dims = make_layer_dims(self.observation_size,
                                     self.action_size,
                                     self.feature_size,
                                     hidden_dims,
                                     mode="actor")

        for layer_dim in layer_dims:
            fc_i = nn.Linear(layer_dim[0], layer_dim[1])
            self.layers.append(fc_i)

        self.num_layers = len(self.layers)

    def forward(self, x):
        """
        Forward propagation.
        Input of the actor network will be the batches of individual observations.

        Parameters
        ----------
        x : torch.Tensor
            Input for the actor network (observation)
            The shape should be (N, input_size)
            input_size is observation_size which is np.prod(observation_space.shape).
            ex. observation_size = 15 * 15.

        Returns
        -------
        x : torch.Tensor
            Return the action probability using softmax (action)
            The shape will be (N, output_size: action_size)
        """
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        x = F.softmax(x, dim=-1)

        return x


class Critic(nn.Module):
    """
    Critic network based on MLP structure.
    """

    def __init__(self, obs_size, act_size, fea_size, hidden_dims):
        """
        Create a new critic network.
        The network is composed of LSTM layer and fully connected layer.
        The network is composed of linear (or fully connected) layers.
        After the linear layer, except the last case, we use ReLU for the activation function.

        Parameters
        ----------
        obs_size
        act_size
        fea_size
        hidden_dims : list
            Dimensions of hidden layers.
            ex. if hidden_dims = [128, 64, 32],
                layer_dims = [[observation_size, 128], [128 + action_size, 64], [64, 32], [32, action_size]].
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.observation_size = obs_size
        self.action_size = act_size
        self.feature_size = fea_size
        layer_dims = make_layer_dims(self.observation_size,
                                     self.action_size,
                                     self.feature_size,
                                     hidden_dims,
                                     mode="critic")

        for layer_dim in layer_dims:
            fc_i = nn.Linear(layer_dim[0], layer_dim[1])
            self.layers.append(fc_i)

        self.num_layers = len(self.layers)

    def forward(self, observation, mean_action):
        """
        Forward propagation.
        Input of the critic network will be the concatenated batches of individual observations and mean actions.
        Unlike the previous implementation, Critic will return the q values for all actions.

        Parameters
        ----------
        observation : torch.Tensor
            Batches of individual observations
            The shape should be (N, observation_size)
            ex. observation_size = 15 * 15.
        mean_action : torch.Tensor
            Batches of individual mean actions
            The shape should be (N, action_size)
            ex. action_size = 6.

        Returns
        -------
        x : torch.Tensor
            Return the q value for all actions
            The shape will be (N, action_size)
            ex. action_size = 6.
        """
        x = self.layers[0](observation)
        x = F.relu(x)
        x = torch.cat((x, mean_action), dim=-1)
        for i in range(1, self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        x.view(-1, self.action_size)

        return x


class Psi(nn.Module):
    """
    Psi network based on MLP structure.
    """

    def __init__(self, obs_size, act_size, fea_size, hidden_dims):
        """
        Create a new psi (successor feature) network.
        The network is composed of LSTM layer and fully connected layer.
        The network is composed of linear (or fully connected) layers.
        After the linear layer, except the last case, we use ReLU for the activation function.
        We will reshape the last outcome to show the features for all actions.

        Parameters
        ----------
        obs_size
        act_size
        fea_size
        hidden_dims : list
            Dimensions of hidden layers.
            ex. if hidden_dims = [128, 64, 32],
                layer_dims = [[observation_size, 128], [128 + action_size, 64], [64, 32], [32, action_size * feature_size]].
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.observation_size = obs_size
        self.action_size = act_size
        self.feature_size = fea_size
        layer_dims = make_layer_dims(self.observation_size,
                                     self.action_size,
                                     self.feature_size,
                                     hidden_dims,
                                     mode="psi")

        for layer_dim in layer_dims:
            fc_i = nn.Linear(layer_dim[0], layer_dim[1])
            self.layers.append(fc_i)

        self.num_layers = len(self.layers)

    def forward(self, observation, mean_action):
        """
        Forward propagation.
        Input of the psi network will be the concatenated batches of individual observations and mean actions.
        Unlike the previous implementation, Psi will return the successor features (psi) for all actions.
        Therefore, the shape of the outcome of Psi will be (N, action_size, feature_size).

        Parameters
        ----------
        observation : torch.Tensor
            Batches of individual observations
            The shape should be (N, observation_size)
            ex. observation_size = 15 * 15.
        mean_action : torch.Tensor
            Batches of individual mean actions
            The shape should be (N, action_size)
            ex. action_size = 6.

        Returns
        -------
        x : torch.Tensor
            Return the psi value for all actions
            The shape will be (N, action_size, feature_size)
            ex. action_size = 6.
        """
        x = self.layers[0](observation)
        x = F.relu(x)
        x = torch.cat((x, mean_action), dim=-1)
        for i in range(1, self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        x = x.view(-1, self.action_size, self.feature_size)

        return x


class Networks(object):
    """
    Define networks (actor-critic / actor-psi / critic / psi)
    """

    def __init__(self, env, args):
        self.observation_size = np.prod(env.observation_space.shape)
        self.action_size = env.action_space.n
        self.feature_size = np.prod(env.feature_space.shape)
        self.args = args
        if self.args.mode_ac:
            self.actor = Actor(self.observation_size, self.action_size, self.feature_size, self.args.h_dims_a)
            self.actor.apply(init_weights)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.args.lr_a)
        if self.args.mode_psi:
            self.psi = Psi(self.observation_size, self.action_size, self.feature_size, self.args.h_dims_p)
            self.psi.apply(init_weights)
            self.psi_target = copy.deepcopy(self.psi)
            self.psi_opt = optim.Adam(self.psi.parameters(), lr=self.args.lr_p)
        else:  # use critic
            self.critic = Critic(self.observation_size, self.action_size, self.feature_size, self.args.h_dims_c)
            self.critic.apply(init_weights)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.args.lr_c)

    # TODO : build a Boltzmann policy if we don't use actor-critic or actor-psi
    def get_boltzmann_policy(self, q_values):
        """
        Get Boltzmann policy using batches of q values.

        Parameters
        ----------
        q_values : torch.Tensor
            Q values for possible actions (size : [N, self.action_size])

        Returns
        -------
        policy : torch.Tensor
            Probabilities for possible actions (size : [N, self.action_size])
        """
        policy = None
        raise NotImplementedError

    def get_actions(self, obs, prev_m_act, is_target=False):
        """
        Get actions.

        Parameters
        ----------
        obs : dict
            dict of observations of agents.
            ex. obs['agent-0'] = np.array(15,15)
        prev_m_act : dict
            dict of previous mean actions of agents.
            ex. prev_m_act['agent-0'] = np.zeros(env.action_space.n)
        is_target : bool
            True if it uses the target network.
            But I think we only use the target network when we use this function.

        Returns
        -------
        actions : dict
            dict of actions of agents.
            ex. actions['agent-0'] = 3 (int)
        """
        agent_ids = list(obs.keys())
        observation = list(obs.values())
        prev_mean_action = list(prev_m_act.values())
        with torch.no_grad():
            tensors = self.to_tensors(obs=observation, m_act=prev_mean_action)
            if self.args.mode_ac:
                if is_target:
                    action_probs = self.actor_target(tensors['obs'])
                else:
                    action_probs = self.actor(tensors['obs'])
                action_dists = distributions.Categorical(action_probs)
                actions = action_dists.sample()
                actions = {agent_ids[i]: actions[i].item() for i in range(len(agent_ids))}
            else:  # TODO : We have to get action from Boltzmann policy (using tensors['m_act'])
                if self.args.mode_psi:
                    pass
                    # Make value using psi and weight
                else:
                    pass
                raise NotImplementedError

        return actions

    def preprocess(self, samples):
        """
        First, from samples, this function gather same categories and make lists for each category.
        Second, for each elements (observation, rewards, ... ), this function change the dict of something
        into the list.
        All agents are treated as homogeneous agents.
        In other words, if the joint observation comes in, this function will make the list of individual observations.

        Parameters
        ----------
        samples : list
            ex. [(obs, act, rew, m_act, n_obs, fea), (obs, act, rew, m_act, n_obs, fea), ...]
            ex. obs = {'agent-0': np.array(15,15), 'agent-1': np.array(15,15), ...}

        Returns
        -------
        obs : list
            list of individual observations.
            ex. obs = [np.array(15,15), np.array(15,15), ...]
        act : list
            ex. act = [3,2,1,1,1,0,5,4,...]
        rew : list
            ex. rew = [0,0,0,1,0,1,0,1,1,...]
        m_act : list
            ex. [np.array(action_size), np.array(action_size), ...]
        n_obs : list
            ex. [np.array(15,15), np.array(15,15), ...]
        fea : list
            ex. [np.array(feature_size), np.array(feature_size), ...]
        """
        obs, act, rew, m_act, n_obs, fea = make_vars(6, mode="list")
        for sample in samples:
            obs += list(sample[0].values())
            act += list(sample[1].values())
            rew += list(sample[2].values())
            m_act += list(sample[3].values())
            n_obs += list(sample[4].values())
            fea += list(sample[5].values())

        return obs, act, rew, m_act, n_obs, fea

    def to_tensors(self, obs=None, act=None, rew=None, m_act=None, n_obs=None, fea=None):
        """
        Make list of inputs to tensors.

        Parameters
        ----------
        obs : list
            List of individual observations.
            ex. obs = [np.array(15,15), np.array(15,15), ...]
        act : list
            List of individual actions.
            ex. act = [3,2,1,1,1,0,5,4,...]
        rew : list
            List of individual rewards.
            ex. rew = [0,0,0,1,0,1,0,1,1,...]
        m_act : list
            List of individual mean actions.
            ex. m_act = [np.array(action_size), np.array(action_size), ...]
        n_obs
            List of individual mean actions.
            ex. next_obs = [np.array(15,15), np.array(15,15), ...]
        fea
            List of individual features.
            ex. [np.array(feature_size), np.array(feature_size), ...]

        Returns
        -------
        tensors : dict
            Dict of tensors which are tensor versions of obs, act, rew, m_act, n_obs, fea
        """
        tensors = {i: None for i in ['obs', 'act', 'rew', 'm_act', 'n_obs', 'fea']}

        if obs is not None:
            obs_tensor = torch.tensor(obs, dtype=torch.float)
            obs_tensor = obs_tensor.view(-1, self.observation_size)  # Shape should be (N, observation_size)
            tensors['obs'] = obs_tensor
        if act is not None:
            act_tensor = torch.tensor(act)  # Shape should be (N, )
            tensors['act'] = act_tensor
            # TODO : remove
            # act_tensor = torch.tensor(act)
            # act_tensor = F.one_hot(act_tensor, num_classes=self.action_size)
            # act_tensor = act_tensor.type(torch.float)
            # act_tensor - act_tensor.view(-1, self.action_size)  # Shape should be (N, action_size)
            # tensors['act'] = act_tensor
        if rew is not None:
            rew_tensor = torch.tensor(rew, dtype=torch.float)
            rew_tensor = rew_tensor.view(-1, 1)  # Shape should be (N, 1)
            tensors['rew'] = rew_tensor
        if m_act is not None:
            mean_act_tensor = torch.tensor(m_act, dtype=torch.float)
            mean_act_tensor = mean_act_tensor.view(-1, self.action_size)  # Shape should be (N, action_size)
            tensors['m_act'] = mean_act_tensor
        if n_obs is not None:
            n_obs_tensor = torch.tensor(n_obs, dtype=torch.float)
            n_obs_tensor = n_obs_tensor.view(-1, self.observation_size)  # Shape should be (N, observation_size)
            tensors['n_obs'] = n_obs_tensor
        if fea is not None:
            fea_tensor = torch.tensor(fea, dtype=torch.float)
            fea_tensor = fea_tensor.view(-1, self.feature_size)  # Shape should be (N, feature_size)
            tensors['fea'] = fea_tensor

        return tensors

    # TODO : add descriptions
    def calculate_losses(self, tensors):
        actor_loss, psi_loss, critic_loss = [None] * 3
        if self.args.mode_ac:
            actor_loss = self.calculate_actor_loss(tensors)
        if self.args.mode_psi:
            psi_loss = self.calculate_psi_loss(tensors)
        else:
            critic_loss = self.calculate_critic_loss(tensors)

        return actor_loss, psi_loss, critic_loss

    def calculate_actor_loss(self, tensors):
        obs = tensors['obs']  # Shape : (N, observation_size)
        act = tensors['act']  # Shape : (N, )
        m_act = tensors['m_act']  # Shape : (N, action_size)

        with torch.no_grad():
            # Get q values from the psi/critic target network
            if self.args.mode_psi:
                # TODO
                # psi_target = self.psi_target(obs, m_act)
                # multiply w
                # get q_target
                raise NotImplementedError
            else:
                q_target = self.critic_target(obs, m_act)  # Shape : (N, action_size)
            # Get action probabilities from the actor target network
            act_probs_target = self.actor_target(obs)

            # Get v values using q values and action probabilities
            v_target = torch.sum(q_target * act_probs_target, dim=-1).view(-1, 1)

        # Get action probabilities from the actor network
        act_probs = self.actor(obs)
        act_dist = distributions.Categorical(act_probs)

        # Get actor loss using values and probabilities
        q_target = q_target[torch.arange(q_target.size(0)), act].view(-1, 1)
        actor_loss = - (q_target - v_target) * act_dist.log_prob(act).view(-1, 1)
        actor_loss = torch.mean(actor_loss)

        return actor_loss

    def calculate_psi_loss(self, tensors):
        # TODO
        with torch.no_grad():
            pass
        raise NotImplementedError

    def calculate_critic_loss(self, tensors):
        obs = tensors['obs']
        act = tensors['act']
        rew = tensors['rew']
        m_act = tensors['m_act']
        n_obs = tensors['n_obs']
        with torch.no_grad():
            # Get q values from the psi/critic target network
            q_target_n = self.critic_target(n_obs, m_act)

            # Get action probabilities from the actor target network or the Boltzmann policy
            if self.args.mode_ac:
                act_probs_target_n = self.actor_target(n_obs)
            else:
                act_probs_target_n = self.get_boltzmann_policy(q_target_n)

            # Get v values using q values and action probabilities
            v_target_n = torch.sum(q_target_n * act_probs_target_n, dim=-1).view(-1, 1)

        # Get actor loss using values
        q = self.critic(obs, m_act)
        q = q[torch.arange(q.size(0)), act].view(-1, 1)
        critic_loss = (rew + v_target_n - q) ** 2
        critic_loss = torch.mean(critic_loss)

        return critic_loss

    def update_networks(self, samples):
        obs, act, rew, m_act, n_obs, fea = self.preprocess(samples)
        tensors = self.to_tensors(obs=obs, act=act, rew=rew, m_act=m_act, n_obs=n_obs, fea=fea)
        actor_loss, psi_loss, critic_loss = self.calculate_losses(tensors)

        if self.args.mode_ac:
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        if self.args.mode_psi:
            self.psi_opt.zero_grad()
            psi_loss.backward()
            self.psi_opt.step()
        else:
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

    def update_target_networks(self):
        if self.args.mode_ac:
            self.actor_target = copy.deepcopy(self.actor)
        if self.args.mode_psi:
            self.psi_target = copy.deepcopy(self.psi)
        else:
            self.critic_target = copy.deepcopy(self.critic)
