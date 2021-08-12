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
        if mode=='list':
            yield []
        elif mode=='dict':
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


def make_layer_dims(observation_dim, action_dim, feature_dim, hidden_dims, mode='actor'):
    """
    Make the list of layer dimensions
    Each element is the dimension of layers ([input_dim, output_dim])
    Unlike previous implementation (taxi example), action is added into second layer
    In addition, the dimension of mean_action is assumed to be same as the dimension of action

    Parameters
    ----------
    observation_dim : int
    action_dim : int
    feature_dim : int
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
                layer_dim = [observation_dim, hidden_dims[i]]
            else:
                layer_dim = [hidden_dims[i - 1], hidden_dims[i]]
            layer_dims.append(layer_dim)
        layer_dim = [hidden_dims[-1], action_dim]
        layer_dims.append(layer_dim)
    elif mode == 'critic':
        for i in range(len(hidden_dims)):
            if i == 0:
                layer_dim = [observation_dim, hidden_dims[i]]
            elif i == 1:
                layer_dim = [hidden_dims[i - 1] + action_dim * 2, hidden_dims[i]]
            else:
                layer_dim = [hidden_dims[i - 1], hidden_dims[i]]
            layer_dims.append(layer_dim)
        layer_dim = [hidden_dims[-1], 1]
        layer_dims.append(layer_dim)
    elif mode == 'psi':
        for i in range(len(hidden_dims)):
            if i == 0:
                layer_dim = [observation_dim, hidden_dims[i]]
            elif i == 1:
                layer_dim = [hidden_dims[i - 1] + action_dim * 2, hidden_dims[i]]
            else:
                layer_dim = [hidden_dims[i - 1], hidden_dims[i]]
            layer_dims.append(layer_dim)
        layer_dim = [hidden_dims[-1], feature_dim]
        layer_dims.append(layer_dim)
    else:
        raise ValueError
    return layer_dims


class Actor(nn.Module):
    """
    Actor network based on MLP structure.
    """
    def __init__(self, net_dims):
        """
        Create a new actor network.
        The network is composed of linear (or fully connected) layers.
        After the linear layer, except the last case, we use ReLU for the activation function.
        Lastly, we use softmax to return the action probabilities.

        Attributes
        ----------
        layers : torch.nn.modules.container.ModuleList
            Container for layers
        num_layers : int
            Number of layers

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
            The shape will be (N, output_size)
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
    def __init__(self, net_dims):
        """
        Create a new critic network.
        The network is composed of LSTM layer and fully connected layer.
        The network is composed of linear (or fully connected) layers.
        After the linear layer, except the last case, we use ReLU for the activation function.

        Attributes
        ----------
        layers : torch.nn.modules.container.ModuleList
            Container for layers
        num_layers : int
            Number of layers

        Parameters
        ----------
        net_size : list
            Layer size (list of dimensions)
            ex. [[10, 64], [64, 64], [64, 1]]
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(net_dims)

        for i in range(self.num_layers):
            fc_i = nn.Linear(net_dims[i][0], net_dims[i][1])
            self.layers.append(fc_i)

    def forward(self, observation, action, mean_action):
        """
        Forward propagation.
        Input of the critic (or psi) network will be the concatenated batches of individual observations, actions,
        and mean actions.

        Parameters
        ----------
        observation : torch.Tensor
            Batches of individual observations
            The shape should be (N, observation_size)
            ex. observation_size = 15 * 15.
        action : torch.Tensor
            Batches of individual actions
            Individual action is one-hot encoded action.
            The shape should be (N, action_size)
            ex. action_size = 6.
        mean_action : torch.Tensor
            Batches of individual mean actions
            The shape should be (N, action_size)
            ex. action_size = 6.

        Returns
        -------
        x : torch.Tensor
            Return the q value (or psi value)
        """

        x = self.layers[0](observation)
        x = F.relu(x)
        x = torch.cat((x, action, mean_action), dim=1)
        for i in range(1, self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        return x

class Networks(object):
    """
    Define networks (actor-critic / actor-psi / critic / psi)
    """
    def __init__(self, env, args):
        self.observation_size = np.prod(env.observation_space.shape)
        self.action_size = env.action_space.n
        # TODO : get feature_space from the environment.
        self.feature_size = 2
        self.actor_layers = []
        self.critic_layers = []
        self.psi_layers = []
        self.args = args
        if self.args.mode_ac:
            self.actor_layers = make_layer_dims(self.observation_size,
                                                self.action_size,
                                                self.feature_size,
                                                self.args.h_dims_a,
                                                mode='actor')
            self.actor.apply(init_weights)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.args.lr_a)

        if self.args.mode_psi:
            self.psi_layers = make_layer_dims(self.observation_size,
                                              self.action_size,
                                              self.feature_size,
                                              self.args.h_dims_p,
                                              mode='psi')
            self.psi = Critic(self.psi_layers)
            self.psi.apply(init_weights)
            self.psi_target = copy.deepcopy(self.psi)
            self.psi_opt = optim.Adam(self.psi.parameters(), lr=self.args.lr_p)
        else:
            self.critic_layers = make_layer_dims(self.observation_size,
                                                 self.action_size,
                                                 self.feature_size,
                                                 self.args.h_dims_c,
                                                 mode='critic')
            self.critic = Critic(self.critic_layers)
            self.critic.apply(init_weights)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.args.lr_c)

    # TODO : build a Boltzmann policy if we don't use actor-critic or actor-psi
    def get_boltzmann_policy(self, q_values):
        """
        Get Boltzmann policy using q values.

        Parameters
        ----------
        q_values : torch.tensor
            Q values for possible actions (size : [1, self.action_dim])

        Returns
        -------
        policy : torch.tensor
            Probabilities for possible actions (size : [1, self.action_dim])
        """
        policy = None
        raise NotImplementedError

    def get_actions(self, obs, prev_m_act, is_target=False):
        """
        Get actions.
        update할 때는 상관 없는데 이건 Agent id 별로 돌려줘야 할듯

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
                    action_probs = self.actor_target(tensors['obs'], tensors['m_act'])
                else:
                    action_probs = self.actor(tensors['obs'], tensors['m_act'])
                action_dists = distributions.Categorical(action_probs)
                actions = action_dists.sample()
                actions = {agent_ids[i]: actions[i].item() for i in range(len(agent_ids))}
            else:  # TODO : We have to get action from Boltzmann policy (using prev_mean_action)
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
            Dict of tensors which is composed of obs, act, rew, m_act, n_obs, fea
        """
        tensors = {i: None for i in ['obs', 'act', 'rew', 'm_act', 'n_obs', 'fea']}

        if obs is not None:
            obs_tensor = torch.tensor(obs, dtype=torch.float)
            obs_tensor = obs_tensor.view(-1, self.observation_size)  # Shape should be (N, observation_size)
            tensors['obs'] = obs_tensor
        if act is not None:
            act_tensor = torch.tensor(act)
            act_tensor = F.one_hot(act_tensor, num_classes=self.action_size)
            act_tensor = act_tensor.type(torch.float)
            act_tensor - act_tensor.view(-1, self.action_size)  # Shape should be (N, action_size)
            tensors['act'] = act_tensor
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

    def update_networks(self, samples):
        obs, act, rew, m_act, n_obs, fea = self.preprocess(samples)
        # tensors = self.to_tensors(obs=obs, act=act, rew=rew, m_act=m_act, n_obs=n_obs, fea=fea)
        # 위에서 얻은 것들로 loss 계산해서 update

    def update_target_networks(self):
        if self.args.mode_ac:
            self.actor_target = copy.deepcopy(self.actor)
        if self.args.mode_psi:
            self.psi_target = copy.deepcopy(self.psi)
        else:
            self.critic_target = copy.deepcopy(self.critic)


##############################################
    #
    # def to_tensor(self, obs=None, act=None, rew=None, mean_act=None, next_obs=None, fea=None):
    #     obs_tensor, act_tensor, rew_tensor, mean_act_tensor, next_obs_tensor, fea_tensor = None, None, None, None, None, None
    #     if obs is not None:
    #         obs_tensor = torch.zeros(self.env_observation_dim.tolist(), dtype=torch.float)
    #         for i in range(obs.shape[0]):
    #             for j in range(obs.shape[1]):
    #                 obs_tensor[i, j, obs[i, j]] = 1
    #         obs_tensor = obs_tensor.view(1, -1)
    #     if act is not None:
    #         act_tensor = torch.zeros([1, self.action_dim], dtype=torch.float)
    #         act_tensor[0, act] = 1
    #     if rew is not None:
    #         rew_tensor = torch.tensor([[rew]], dtype=torch.float)
    #     if mean_act is not None:
    #         mean_act_tensor = torch.tensor(mean_act, dtype=torch.float)
    #         mean_act_tensor = mean_act_tensor.view(1, -1)
    #     if next_obs is not None:
    #         next_obs_tensor = torch.zeros(self.env_observation_dim.tolist(), dtype=torch.float)
    #         for i in range(next_obs.shape[0]):
    #             for j in range(next_obs.shape[1]):
    #                 next_obs_tensor[i, j, next_obs[i, j]] = 1
    #         next_obs_tensor = next_obs_tensor.view(1, -1)
    #     if fea is not None:
    #         fea_tensor = torch.tensor([fea], dtype=torch.float)
    #     return obs_tensor, act_tensor, rew_tensor, mean_act_tensor, next_obs_tensor, fea_tensor
    #
    # def get_tensors(self, individual_dict):
    #     observation = individual_dict['observation']
    #     action = individual_dict['action']
    #     reward = individual_dict['reward']
    #     mean_action = individual_dict['mean_action']
    #     next_observation = individual_dict['next_observation']
    #     feature = individual_dict['feature']
    #     return self.to_tensor(obs=observation,
    #                           act=action,
    #                           rew=reward,
    #                           mean_act=mean_action,
    #                           next_obs=next_observation,
    #                           fea=feature)

    # def get_boltzmann_policy(self, q_values):
    #     """
    #
    #     Parameters
    #     ----------
    #     q_values : torch.tensor
    #         Q values for possible actions (size : [1, self.action_dim])
    #
    #     Returns
    #     -------
    #     policy : torch.tensor
    #         Probabilities for possible actions (size : [1, self.action_dim])
    #     """
    #     policy = None
    #     raise NotImplementedError

    def calculate_actor_loss(self, sample):
        """
        This function returns the sample actor loss using the sample
        It gets q target values, v target values, and action target probabilities
        It also gets action probabilities
        It calculates sample actor loss using the above things

        Parameters
        ----------
        sample : dict
            Sample which contains observation, action, reward, mean_action, next_observation, feature

        Returns
        -------
        sample_actor_loss : torch.tensor
            (size : [1, 1])
        """
        sample_actor_loss = torch.zeros([1, 1], dtype=torch.float)
        for agent in sample.values():
            # Get (formatted) tensors from the sample
            action_int = agent['action']
            observation, action, reward, mean_action, next_observation, feature = self.get_tensors(agent)

            with torch.no_grad():
                # Get q values from the psi/critic target network
                q_target_obs = torch.zeros([1, self.action_dim], dtype=torch.float)
                for act_int in range(self.action_dim):
                    _, possible_action, _, _, _, _ = self.to_tensor(act=act_int)
                    if self.args.mode_psi:
                        # TODO
                        raise NotImplementedError
                    else:
                        q_target_obs[0, act_int] = self.critic_target(observation, possible_action, mean_action)

                # Get action probabilities from the actor target network
                act_target_probs = self.actor_target(observation)
                act_target_dist = distributions.Categorical(act_target_probs)
                target_probs = act_target_dist.probs

                # Get v values using q values and action probabilities
                v_target_obs = torch.tensordot(q_target_obs, target_probs, dims=2).view(1, -1)

            # Get action probabilities from the actor network
            act_probs = self.actor(observation)
            act_dist = distributions.Categorical(act_probs)

            # Get actor loss and sample actor loss using values and probabilities
            q_target_obs_act = q_target_obs[0, action_int]
            act = torch.tensor(action_int, dtype=torch.float)
            actor_loss = - (q_target_obs_act - v_target_obs) * act_dist.log_prob(act)
            sample_actor_loss = sample_actor_loss + actor_loss

        return sample_actor_loss

    def calculate_critic_loss(self, sample):
        """
        This function returns the sample critic loss using the sample
        It gets q target values, v target values, and action target probabilities of next observation
        You should notice that q target values of next observation uses current mean action
        Current mean action is the prev mean action of next observation (ref. MFQ paper and POMFQ paper)
        It also gets q values of observation
        It calculates sample critic loss using the above things

        Parameters
        ----------
        sample : dict
            Sample which contains observation, action, reward, mean_action, next_observation, feature

        Returns
        -------
        sample_critic_loss : torch.tensor
            (size : [1, 1])
        """
        sample_critic_loss = torch.zeros([1, 1], dtype=torch.float)
        for agent in sample.values():
            # Get (formatted) tensors from the sample
            observation, action, reward, mean_action, next_observation, feature = self.get_tensors(agent)

            with torch.no_grad():
                # Get q values from the critic target network
                q_target_n_obs = torch.zeros([1, self.action_dim], dtype=torch.float)
                for n_act_int in range(self.action_dim):
                    _, next_action, _, _, _, _ = self.to_tensor(act=n_act_int)
                    q_target_n_obs[0, n_act_int] = self.critic_target(next_observation, next_action, mean_action)

                # Get action probabilities from the actor target network or the Boltzmann policy
                if self.args.mode_ac:
                    n_act_probs = self.actor_target(next_observation)
                    n_act_dist = distributions.Categorical(n_act_probs)
                    probs = n_act_dist.probs
                else:
                    probs = self.get_boltzmann_policy(q_target_n_obs)

                # Get v values using q values and action probabilities
                v_target_n_obs = torch.tensordot(q_target_n_obs, probs, dims=2).view(1, -1)

            # Get critic loss and sample critic loss using values and probabilities
            critic_loss = (reward + v_target_n_obs - self.critic(observation, action, mean_action)) ** 2
            sample_critic_loss = sample_critic_loss + critic_loss

        return sample_critic_loss

    def calculate_psi_loss(self, sample):
        """
        This function returns the sample psi loss using the sample
        It gets psi target values, expected psi target values, and action target probabilities of next observation
        You should notice that psi target values of next observation uses current mean action
        Current mean action is the prev mean action of next observation (ref. MFQ paper and POMFQ paper)
        It also gets q target values using psi target values and weight
        It also gets psi values of observation
        It calculates sample psi loss using the above things

        Parameters
        ----------
        sample : dict
            Sample which contains observation, action, reward, mean_action, next_observation, feature

        Returns
        -------
        sample_psi_loss : torch.tensor
            (size : [1, self.feature_dim])
        """
        sample_psi_loss = torch.zeros([1, self.feature_dim], dtype=torch.float)
        w = torch.tensor([[self.args.lv_penalty, self.args.lv_incentive]], dtype=torch.float)
        for agent in sample.values():
            # Get (formatted) tensors from the sample
            observation, action, reward, mean_action, next_observation, feature = self.get_tensors(agent)

            with torch.no_grad():
                # Get psi values and q values from the psi target network and the weight
                psi_target_n_obs = torch.zeros([1, self.action_dim, self.feature_dim], dtype=torch.float)
                q_target_n_obs = torch.zeros([1, self.action_dim], dtype=torch.float)
                for n_act_int in range(self.action_dim):
                    _, next_action, _, _, _, _ = self.to_tensor(act=n_act_int)
                    x = self.psi_target(next_observation, next_action, mean_action)
                    psi_target_n_obs[0, n_act_int] = x
                    q_target_n_obs[0, n_act_int] = torch.tensordot(x, w, dims=2)

                # Get action probabilities from the actor target network or the Boltzmann policy
                if self.args.mode_ac:
                    n_act_probs = self.actor_target(next_observation)
                    n_act_dist = distributions.Categorical(n_act_probs)
                    probs = n_act_dist.probs
                else:
                    probs = self.get_boltzmann_policy(q_target_n_obs)

                # Get expected psi values of next observation using psi values and action probabilities
                expected_psi_target_n_obs = torch.tensordot(psi_target_n_obs, probs, dims=([0, 1], [0, 1])).view(1, -1)
                expected_psi_target_n_obs = expected_psi_target_n_obs.type(torch.float)

            # Get psi loss and sample psi loss using expected psi values and probabilities
            psi_loss = (feature + expected_psi_target_n_obs - self.psi(observation, action, mean_action)) ** 2
            sample_psi_loss = sample_psi_loss + psi_loss

        return sample_psi_loss

    def calculate_losses(self, samples):
        """
        This function returns several losses using samples

        Parameters
        ----------
        samples : list
            List of dict which is the sample from the buffer

        Returns
        -------
        loss_actor : torch.tensor
        loss_psi : torch.tensor
        loss_critic : torch.tensor
        """
        loss_actor = torch.zeros([1, 1], dtype=torch.float)
        loss_critic = torch.zeros([1, 1], dtype=torch.float)
        loss_psi = torch.zeros([1, self.feature_dim], dtype=torch.float)
        count = 0
        for sample in samples:
            if self.args.mode_ac:
                sample_actor_loss = self.calculate_actor_loss(sample)
                loss_actor = loss_actor + sample_actor_loss
            if self.args.mode_psi:
                sample_psi_loss = self.calculate_psi_loss(sample)
                loss_psi = loss_psi + sample_psi_loss
            else:
                sample_critic_loss = self.calculate_critic_loss(sample)
                loss_critic = loss_critic + sample_critic_loss
            count = count + len(sample)
        loss_actor = loss_actor / count
        loss_psi = loss_psi / count
        loss_critic = loss_critic / count

        return loss_actor, loss_psi, loss_critic

    def update_networks(self, samples):
        loss_actor, loss_psi, loss_critic = self.calculate_losses(samples)

        self.opt_actor.zero_grad() if self.args.mode_ac else None
        self.opt_psi.zero_grad() if self.args.mode_psi else self.opt_critic.zero_grad()
        loss_actor.backward() if self.args.mode_ac else None
        loss_psi.backward() if self.args.mode_psi else loss_critic.backward()
        self.opt_actor.step() if self.args.mode_ac else None
        self.opt_psi.step() if self.args.mode_psi else self.opt_critic.step()

    def update_target_networks(self):
        if self.args.mode_ac:
            self.actor_target = copy.deepcopy(self.actor)
        if self.args.mode_psi:
            self.psi_target = copy.deepcopy(self.psi)
        else:
            self.critic_target = copy.deepcopy(self.critic)
