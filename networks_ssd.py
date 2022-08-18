import copy

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.utils_all import init_weights
from utils.utils_ssd import make_vars


def make_layer_dims(observation_size, action_size, feature_size, hidden_dims, mode='actor'):
    """
    Make the list of layer dimensions.
    Each element is the dimension of layers ([input_dim, output_dim]).
    Unlike previous implementation (taxi example), action is added into second layer.
    In addition, the dimension of mean_action is assumed to be same as the dimension of action.

    Parameters
    ----------
    observation_size: int
    action_size: int
    feature_size: int
    hidden_dims: List
        List of hidden layers' size.
    mode: str
        'actor' or 'critic' or 'psi'.

    Returns
    -------
    layer_dims: list
        List of list
        Each element is the dimension of layers ([input_dim, output_dim]).
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
        obs_size: int
        act_size: int
        fea_size: int
        hidden_dims: list
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
        x: torch.Tensor
            Input for the actor network (observation).
            The shape should be (N, input_size).
            input_size is observation_size which is np.prod(observation_space.shape).
            ex. observation_size = 15 * 15.

        Returns
        -------
        x: torch.Tensor
            Return the action probability using softmax (action).
            The shape will be (N, output_size: action_size).
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
        The network is composed of linear (or fully connected) layers.
        After the linear layer, except the last case, we use ReLU for the activation function.

        Parameters
        ----------
        obs_size: int
        act_size: int
        fea_size: int
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
        self.args = args
        self.observation_num_classes = env.observation_space.high.max() + 1
        self.observation_size = self.get_observation_size(env)
        self.action_size = env.action_space.n
        self.feature_size = np.prod(env.feature_space.shape)
        self.w = self.get_w()
        self.actor, self.actor_target = self.get_network("actor")
        self.psi, self.psi_target = self.get_network("psi")
        self.critic, self.critic_target = self.get_network("critic")
        self.reuse_networks()
        self.actor_opt, self.actor_skd = self.get_opt_and_skd("actor")
        self.psi_opt, self.psi_skd = self.get_opt_and_skd("psi")
        self.critic_opt, self.critic_skd = self.get_opt_and_skd("critic")

    def get_observation_size(self, env):
        """
        Return observation_size which depends on the mode_one_hot_obs.

        Returns
        -------
        observation_size: numpy.ndarray
        """
        if self.args.mode_one_hot_obs:
            observation_size = np.prod(env.observation_space.shape) * self.observation_num_classes
        else:
            observation_size = np.prod(env.observation_space.shape)
        return observation_size

    def get_w(self):
        """
        Return w which depends on the environment.

        Returns
        -------
        w: torch.Tensor
        """
        if "cleanup" in self.args.env:
            w = torch.tensor([1 - self.args.lv_penalty, self.args.lv_incentive], dtype=torch.float)
        elif "harvest" in self.args.env:
            w = torch.tensor([1, self.args.lv_penalty, self.args.lv_incentive], dtype=torch.float)
        else:
            raise NotImplementedError
        return w

    def get_network(self, mode):
        """
        Build network and target network.

        Parameters
        ----------
        mode: str
            It should be 'actor', 'psi', or 'critic'.

        Returns
        -------
        network: None or Actor or Psi or Critic
        network_target: None or Actor or Psi or Critic
        """
        network, network_target = [None] * 2
        if self.args.mode_ac and mode == "actor":
            network = Actor(self.observation_size, self.action_size, self.feature_size, self.args.h_dims_a)
            network.apply(init_weights)
            network_target = copy.deepcopy(network)
        elif self.args.mode_psi and mode == "psi":
            network = Psi(self.observation_size, self.action_size, self.feature_size, self.args.h_dims_p)
            network.apply(init_weights)
            network_target = copy.deepcopy(network)
        elif (not self.args.mode_psi) and mode == "critic":
            network = Critic(self.observation_size, self.action_size, self.feature_size, self.args.h_dims_c)
            network.apply(init_weights)
            network_target = copy.deepcopy(network)
        return network, network_target

    def get_opt_and_skd(self, mode):
        """
        Get the optimizer and the learning rate scheduler.

        Parameters
        ----------
        mode: str
            It should be 'actor', 'psi', or 'critic'.

        Returns
        -------
        opt: None or optim.Adam
        skd: None or optim.lr_scheduler.StepLR
        """
        opt, skd = [None] * 2
        if self.args.mode_ac and mode == "actor":
            opt = optim.Adam(self.actor.parameters(), lr=self.args.lr_a)
        elif self.args.mode_psi and mode == "psi":
            opt = optim.Adam(self.psi.parameters(), lr=self.args.lr_p)
        elif (not self.args.mode_psi) and mode == "critic":
            opt = optim.Adam(self.critic.parameters(), lr=self.args.lr_c)
        if self.args.mode_lr_decay and opt is not None:
            skd = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.999)
        return opt, skd

    def reuse_networks(self):
        """
        Update network parameters if we reuse previous networks.
        """
        if self.args.mode_reuse_networks:
            prev_dict = torch.load(self.args.file_path)
            if self.args.mode_ac:
                self.actor.load_state_dict(prev_dict['actor'])
                self.actor_target.load_state_dict(prev_dict['actor'])
            if self.args.mode_psi:
                self.psi.load_state_dict(prev_dict['psi'])
                self.psi_target.load_state_dict(prev_dict['psi'])
            else:
                self.critic.load_state_dict(prev_dict['critic'])
                self.critic_target.load_state_dict(prev_dict['critic'])

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

    def get_actions(self, obs, prev_m_act, is_target=True):
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
        observation = np.array(list(obs.values()))
        prev_mean_action = np.array(list(prev_m_act.values()))
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
                actions_probs = {agent_ids[i]: action_probs[i] for i in range(len(agent_ids))}
            else:  # TODO : We have to get action from Boltzmann policy (using tensors['m_act'])
                if self.args.mode_psi:
                    pass
                    # Make value using psi and weight
                else:
                    pass
                raise NotImplementedError

        return actions, actions_probs

    def preprocess(self, samples):
        """
        First, from samples, this function gather same categories and make lists for each category.
        Second, for each elements (observation, rewards, ... ), this function change the dict of something
        into the list.
        All agents are treated as homogeneous agents.
        In other words, if the joint observation comes in, this function will make the list of individual observations.
        Lastly, lists will be changed into numpy.ndarrays.

        Parameters
        ----------
        samples : list
            ex. [(obs, act, rew, m_act, n_obs, fea), (obs, act, rew, m_act, n_obs, fea), ...]
            ex. obs = {'agent-0': np.array(15,15), 'agent-1': np.array(15,15), ...}

        Returns
        -------
        obs: numpy.ndarray
            (N, 15, 15) array of individual observations (15 can be changed).
            ex. obs = np.array([np.array(15,15), np.array(15,15), ...])
        act: numpy.ndarray
            (N,) array of individual actions.
            ex. act = np.array([3,2,1,1,1,0,5,4,...])
        rew: numpy.ndarray
            (N,) array of individual rewards.
            ex. rew = np.array([0,0,0,1,0,1,0,1,1,...])
        m_act: numpy.ndarray
            (N, action_size) array of individual mean actions.
            ex. m_act = np.array([np.array(action_size), np.array(action_size), ...])
        n_obs: numpy.ndarray
            (N, 15, 15) array of individual next observations (15 can be changed).
            ex. n_obs = np.array([np.array(15,15), np.array(15,15), ...])
        fea: numpy.ndarray
            (N, feature_size) array of individual features.
            ex. fea = np.array([np.array(feature_size), np.array(feature_size), ...])
        """
        obs, act, rew, m_act, n_obs, fea = make_vars(6, mode="list")
        for sample in samples:
            obs += list(sample[0].values())
            act += list(sample[1].values())
            rew += list(sample[2].values())
            m_act += list(sample[3].values())
            n_obs += list(sample[4].values())
            fea += list(sample[5].values())

        obs = np.array(obs)
        act = np.array(act)
        rew = np.array(rew)
        m_act = np.array(m_act)
        n_obs = np.array(n_obs)
        fea = np.array(fea)

        return obs, act, rew, m_act, n_obs, fea

    def to_tensors(self, obs=None, act=None, rew=None, m_act=None, n_obs=None, fea=None):
        """
        Make ndarray of inputs to tensors.
        If args.mode_one_hot_obs, observations will be changed into one-hot encoded version.

        Parameters
        ----------
        obs: None or numpy.ndarray
            ndarray of individual observations.
            ex. obs = np.array([np.array(15,15), np.array(15,15), ...])
        act: None or numpy.ndarray
            ndarray of individual actions.
            ex. act = np.array([3,2,1,1,1,0,5,4,...])
        rew: None or numpy.ndarray
            ndarray of individual rewards.
            ex. rew = np.array([0,0,0,1,0,1,0,1,1,...])
        m_act: None or numpy.ndarray
            ndarray of individual mean actions.
            ex. m_act = np.array([np.array(action_size), np.array(action_size), ...])
        n_obs: None or numpy.ndarray
            ndarray of individual next observations.
            ex. n_obs = np.array([np.array(15,15), np.array(15,15), ...])
        fea: None or numpy.ndarray
            ndarray of individual features.
            ex. fea = np.array([np.array(feature_size), np.array(feature_size), ...])

        Returns
        -------
        tensors : dict
            Dict of tensors which are tensor versions of obs, act, rew, m_act, n_obs, fea
            ex. shape of tensors['obs'] = (N, observation_size: 15 * 15 * 6)
        """
        tensors = {i: None for i in ['obs', 'act', 'rew', 'm_act', 'n_obs', 'fea']}
        with torch.no_grad():
            if obs is not None:
                if self.args.mode_one_hot_obs:
                    # F.one_hot takes tensor with index values of shape (*) and returns a tensor of shape (*, num_classes)
                    obs_tensor = torch.tensor(obs, dtype=torch.int64)
                    obs_tensor = F.one_hot(obs_tensor, num_classes=self.observation_num_classes)
                    obs_tensor = obs_tensor.type(torch.float)
                    obs_tensor = obs_tensor.view(-1, self.observation_size)  # Shape should be (N, observation_size)
                    tensors['obs'] = obs_tensor
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float)
                    obs_tensor = obs_tensor.view(-1, self.observation_size)  # Shape should be (N, observation_size)
                    tensors['obs'] = obs_tensor
            if act is not None:
                act_tensor = torch.tensor(act, dtype=torch.int64)  # Shape should be (N, )
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
                if self.args.mode_one_hot_obs:
                    # F.one_hot takes tensor with index values of shape (*) and returns a tensor of shape (*, num_classes)
                    n_obs_tensor = torch.tensor(n_obs, dtype=torch.int64)
                    n_obs_tensor = F.one_hot(n_obs_tensor, num_classes=self.observation_num_classes)
                    n_obs_tensor = n_obs_tensor.type(torch.float)
                    n_obs_tensor = n_obs_tensor.view(-1, self.observation_size)  # Shape should be (N, observation_size)
                    tensors['n_obs'] = n_obs_tensor
                else:
                    n_obs_tensor = torch.tensor(n_obs, dtype=torch.float)
                    n_obs_tensor = n_obs_tensor.view(-1, self.observation_size)  # Shape should be (N, observation_size)
                    tensors['n_obs'] = n_obs_tensor
            if fea is not None:
                fea = np.asarray(fea)
                fea_tensor = torch.tensor(fea, dtype=torch.float)
                fea_tensor = fea_tensor.view(-1, self.feature_size)  # Shape should be (N, feature_size)
                tensors['fea'] = fea_tensor

        return tensors

    def calculate_losses(self, tensors):
        """
        Calculate losses.

        Parameters
        ----------
        tensors: dict
            Dict which contains tensors of samples.

        Returns
        -------
        actor_loss: None or torch.Tensor
            ex. tensor(-0.2188, grad_fn=<MeanBackward0>)
        psi_loss: None or torch.Tensor
            ex. tensor([20.1826,  3.6522], grad_fn=<MeanBackward1>)
        critic_loss: None or torch.Tensor
        """
        actor_loss, psi_loss, critic_loss = [None] * 3
        if self.args.mode_ac:
            actor_loss = self.calculate_actor_loss(tensors)
        if self.args.mode_psi:
            psi_loss = self.calculate_psi_loss(tensors)
        else:
            critic_loss = self.calculate_critic_loss(tensors)

        return actor_loss, psi_loss, critic_loss

    def calculate_actor_loss(self, tensors):
        """
        Calculate an actor loss.

        Parameters
        ----------
        tensors: dict
            Dict which contains tensors of samples.
        Returns
        -------
        actor_loss: torch.Tensor
            ex. tensor(-0.2188, grad_fn=<MeanBackward0>)
        """
        obs = tensors['obs']  # Shape : (N, observation_size)
        act = tensors['act']  # Shape : (N, )
        m_act = tensors['m_act']  # Shape : (N, action_size)

        with torch.no_grad():
            # Get q values from the psi/critic target network
            if self.args.mode_psi:
                psi_target = self.psi_target(obs, m_act)  # Shape : (N, action_size, feature_size)
                q_target = torch.tensordot(psi_target, self.w, dims=([2], [0]))  # Shape : (N, action_size)
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
        """
        Calculate a psi loss.

        Parameters
        ----------
        tensors: dict
            Dict which contains tensors of samples.
        Returns
        -------
        psi_loss: torch.Tensor
            Size: (2, ).
            ex. tensor([20.1826,  3.6522], grad_fn=<MeanBackward1>)
        """
        obs = tensors['obs']
        act = tensors['act']  # Shape : (N, )
        m_act = tensors['m_act']
        n_obs = tensors['n_obs']
        fea = tensors['fea']

        with torch.no_grad():
            # Get psi values from the psi target network
            psi_target_n = self.psi_target(n_obs, m_act)  # (N, action_size, feature_size)

            # Get action probabilities from the actor target network or the Boltzmann policy (N, action_size)
            if self.args.mode_ac:
                act_probs_target_n = self.actor_target(n_obs)
            else:
                act_probs_target_n = self.get_boltzmann_policy(psi_target_n)

            # Get expected psi using psi and action probabilities
            expected_psi_target_n = torch.bmm(act_probs_target_n.unsqueeze(1), psi_target_n)  # (N, 1, feature_size)
            expected_psi_target_n = expected_psi_target_n.view(-1, self.feature_size)  # (N, feature_size)
        # Get psi loss
        psi = self.psi(obs, m_act)  # (N, action_size, feature_size)
        psi = psi[torch.arange(psi.size(0)), act]  # (N, feature_size)
        psi_loss = (fea + self.args.gamma * expected_psi_target_n - psi) ** 2  # (N, feature_size)
        psi_loss = torch.mean(psi_loss, dim=0)  # (feature_size, )

        return psi_loss

    def calculate_critic_loss(self, tensors):
        """
        Calculate a critic loss.

        Parameters
        ----------
        tensors: dict
            Dict which contains tensors of samples.
        Returns
        -------
        critic_loss: torch.Tensor
        """
        obs = tensors['obs']
        act = tensors['act']
        rew = tensors['rew']
        m_act = tensors['m_act']
        n_obs = tensors['n_obs']
        with torch.no_grad():
            # Get q values from the critic target network
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
        critic_loss = (rew + self.args.gamma * v_target_n - q) ** 2
        critic_loss = torch.mean(critic_loss)

        return critic_loss

    def update_networks(self, samples):
        """
        Update networks(actor, critic, psi) using samples.

        Parameters
        ----------
        samples: list
            List of N samples.
        """
        obs, act, rew, m_act, n_obs, fea = self.preprocess(samples)
        tensors = self.to_tensors(obs=obs, act=act, rew=rew, m_act=m_act, n_obs=n_obs, fea=fea)
        actor_loss, psi_loss, critic_loss = self.calculate_losses(tensors)

        if self.args.mode_ac:
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self.actor_skd.step() if self.args.mode_lr_decay else None
        if self.args.mode_psi:
            self.psi_opt.zero_grad()
            psi_loss.backward(torch.ones(self.feature_size.item()))
            self.psi_opt.step()
            self.psi_skd.step() if self.args.mode_lr_decay else None
        else:
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            self.critic_skd.step() if self.args.mode_lr_decay else None

    def update_target_network(self, network, target_network):
        """
        Update target network using network's parameters.

        Parameters
        ----------
        network: Actor or Psi or Critic
        target_network: Actor or Psi or Critic
        """
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(param.data * self.args.tau + target_param.data * (1.0 - self.args.tau))

    def update_target_networks(self):
        """
        Update target networks (Actor, Psi, Critic).
        """
        if self.args.mode_ac:
            self.update_target_network(self.actor, self.actor_target)
        if self.args.mode_psi:
            self.update_target_network(self.psi, self.psi_target)
        else:
            self.update_target_network(self.critic, self.critic_target)
