import copy

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


def get_one_hot_obs(ind_obs):
    """
    Text.

    Parameters
    ----------
    ind_obs : numpy.ndarray
        shape : (2, )

    Returns
    -------
    ind_obs : numpy.ndarray
        shape : (num_grids, episode_length + 1)
    """
    num_grids = 4
    episode_length = 2
    ind_obs_one_hot = np.zeros([num_grids, episode_length + 1])
    loc = ind_obs[0]
    time = ind_obs[1] if ind_obs[1] <= episode_length else episode_length
    ind_obs_one_hot[loc, time] = 1
    return ind_obs_one_hot


def get_masked_categorical(action_probs, masks):
    probs = torch.mul(action_probs, masks)
    action_dists = distributions.Categorical(probs)
    return action_dists


def make_layer_dims(observation_size, action_size, mean_action_size, feature_size, hidden_dims, mode='actor'):
    """
    Make the list of layer dimensions.
    Each element is the dimension of layers ([input_dim, output_dim]).
    Unlike previous implementation (taxi example), action is added into second layer.
    In addition, the dimension of mean_action is assumed to be same as the dimension of action.

    Parameters
    ----------
    observation_size : int
    action_size : int
    mean_action_size : int
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
                layer_dim = [hidden_dims[i - 1] + mean_action_size, hidden_dims[i]]
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
                layer_dim = [hidden_dims[i - 1] + mean_action_size, hidden_dims[i]]
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
    def __init__(self, obs_size, act_size, m_act_size, fea_size, hidden_dims):
        """
        Create a new actor network.
        The network is composed of linear (or fully connected) layers.
        After the linear layer, except the last case, we use ReLU for the activation function.
        Lastly, we use softmax to return the action probabilities.
        You should notice that we do not directly use these probabilities because we have masks.

        Parameters
        ----------
        obs_size
        act_size
        m_act_size
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
        self.mean_action_size = m_act_size
        self.feature_size = fea_size
        layer_dims = make_layer_dims(self.observation_size,
                                     self.action_size,
                                     self.mean_action_size,
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
    def __init__(self, obs_size, act_size, m_act_size, fea_size, hidden_dims):
        """
        Create a new critic network.
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
        self.mean_action_size = m_act_size
        self.feature_size = fea_size
        layer_dims = make_layer_dims(self.observation_size,
                                     self.action_size,
                                     self.mean_action_size,
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
        mean_action : torch.Tensor
            Batches of individual mean actions
            The shape should be (N, mean_action_size)

        Returns
        -------
        x : torch.Tensor
            Return the q value for all actions
            The shape will be (N, action_size)
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

    def __init__(self, obs_size, act_size, m_act_size, fea_size, hidden_dims):
        """
        Create a new psi (successor feature) network.
        The network is composed of linear (or fully connected) layers.
        After the linear layer, except the last case, we use ReLU for the activation function.
        We will reshape the last outcome to show the features for all actions.

        Parameters
        ----------
        obs_size
        act_size
        m_act_size
        fea_size
        hidden_dims : list
            Dimensions of hidden layers.
            ex. if hidden_dims = [128, 64, 32],
                layer_dims = [[observation_size, 128], [128 + mean_action_size, 64], [64, 32], [32, action_size * feature_size]].
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.observation_size = obs_size
        self.action_size = act_size
        self.mean_action_size = m_act_size
        self.feature_size = fea_size
        layer_dims = make_layer_dims(self.observation_size,
                                     self.action_size,
                                     self.mean_action_size,
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
        mean_action : torch.Tensor
            Batches of individual mean actions
            The shape should be (N, mean_action_size)

        Returns
        -------
        x : torch.Tensor
            Return the psi value for all actions
            The shape will be (N, action_size, feature_size)
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
        self.env = env
        self.args = args
        self.observation_size = env.num_grids * (env.episode_length + 1)
        self.action_size = env.num_grids
        self.mean_action_size = 1
        self.feature_size = 2
        self.w = torch.tensor([1, env.lv_penalty], dtype=torch.float)
        self.mask = torch.tensor(env.get_mask())
        self.actor, self.actor_target = self.get_network("actor")
        self.psi, self.psi_target = self.get_network("psi")
        self.critic, self.critic_target = self.get_network("critic")
        self.reuse_networks()
        self.actor_opt, self.actor_skd = self.get_opt_and_skd("actor")
        self.psi_opt, self.psi_skd = self.get_opt_and_skd("psi")
        self.critic_opt, self.critic_skd = self.get_opt_and_skd("critic")

    def get_network(self, mode):
        network, network_target = [None] * 2
        if self.args.mode_ac and mode == "actor":
            network = Actor(self.observation_size, self.action_size, self.mean_action_size, self.feature_size, self.args.h_dims_a)
            network.apply(init_weights)
            network_target = copy.deepcopy(network)
        elif self.args.mode_psi and mode == "psi":
            network = Psi(self.observation_size, self.action_size, self.mean_action_size, self.feature_size, self.args.h_dims_p)
            network.apply(init_weights)
            network_target = copy.deepcopy(network)
        elif (not self.args.mode_psi) and mode == "critic":
            network = Critic(self.observation_size, self.action_size, self.mean_action_size, self.feature_size, self.args.h_dims_c)
            network.apply(init_weights)
            network_target = copy.deepcopy(network)
            raise NotImplementedError("Critic is not tested yet in the current version.")
        return network, network_target

    def reuse_networks(self):
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

    def get_opt_and_skd(self, mode):
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

    def get_actions(self, av_obs, is_target=True, is_random=False):
        av_agent_ids = list(av_obs.keys())
        obs = list(av_obs.values())
        masks = self.get_masks(obs)
        if is_random:
            action_probs = torch.ones([len(av_agent_ids), self.action_size]) / self.action_size
        else:
            tensors = self.to_tensors(obs=obs)
            action_probs = self.actor_target(tensors["obs"]) if is_target else self.actor(tensors["obs"])
        action_dists = get_masked_categorical(action_probs, masks)
        actions = action_dists.sample()
        actions = {av_agent_ids[i]: actions[i].item() for i in range(len(av_agent_ids))}
        return actions

    def get_masks(self, obs):
        # obs : av_obs, list of ind_obs
        masks = torch.zeros([len(obs), self.action_size])
        for i in range(len(obs)):
            loc = obs[i][0]
            masks[i] = self.mask[loc]
        return masks

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

    def preprocess(self, samples):
        obs, act, rew, m_act, n_obs, fea, done, m_act_s = make_vars(8, mode="list")
        for sample in samples:
            av_agent_ids = [k for k, v in sample[1].items() if v is not None]
            m_act_samples = self.get_mean_action_samples(sample, av_agent_ids)
            obs += [sample[0][agent_id] for agent_id in av_agent_ids]
            act += [sample[1][agent_id] for agent_id in av_agent_ids]
            rew += [sample[2][agent_id] for agent_id in av_agent_ids]
            m_act += [sample[3][agent_id] for agent_id in av_agent_ids]
            n_obs += [sample[4][agent_id] for agent_id in av_agent_ids]
            fea += [sample[5][agent_id] for agent_id in av_agent_ids]
            done += [sample[6][agent_id] for agent_id in av_agent_ids]
            m_act_s += [m_act_samples for _ in av_agent_ids]

        return obs, act, rew, m_act, n_obs, fea, done, m_act_s

    def get_mean_action_samples(self, sample, av_agent_ids):
        av_obs = {agent_id: sample[0][agent_id] for agent_id in av_agent_ids}
        t = sample[0][av_agent_ids[0]][1]
        m_act_samples = np.zeros([self.args.num_mean_actions, self.action_size])
        for i in range(self.args.num_mean_actions):
            act = self.get_actions(av_obs, is_target=True, is_random=False)
            temp_obs = self.env.move_agents(av_obs, act)
            _, ds_ratios, _ = self.env.match_orders(temp_obs, t)
            m_act_samples[i] = np.minimum(ds_ratios, np.ones(self.action_size))
        return m_act_samples

    def to_tensors(self, obs=None, act=None, rew=None, m_act=None, n_obs=None, fea=None, done=None, m_act_s=None):
        keys = ['obs', 'act', 'rew', 'm_act', 'n_obs', 'fea', 'done', 'obs_mask', 'n_obs_mask', 'm_act_s']
        tensors = {i: None for i in keys}
        if obs is not None:
            masks = self.get_masks(obs)  # Shape : (N, action_size)
            obs_tensor = torch.tensor([get_one_hot_obs(ind_obs) for ind_obs in obs], dtype=torch.float)  # Shape : (N, num_grids, episode_length + 1)
            obs_tensor = obs_tensor.view(-1, self.observation_size)  # Shape : (N, observation_size)
            tensors['obs'] = obs_tensor
            tensors['obs_mask'] = masks
        if act is not None:
            act_tensor = torch.tensor(act)  # Shape : (N, )
            tensors['act'] = act_tensor
        if rew is not None:
            rew_tensor = torch.tensor(rew, dtype=torch.float)
            rew_tensor = rew_tensor.view(-1, 1)  # Shape : (N, 1)
            tensors['rew'] = rew_tensor
        if m_act is not None:
            m_act_tensor = torch.tensor(m_act, dtype=torch.float)
            m_act_tensor = m_act_tensor.view(-1, self.mean_action_size)  # Shape should be (N, mean_action_size)
            tensors['m_act'] = m_act_tensor
        if n_obs is not None:
            n_masks = self.get_masks(n_obs)  # Shape : (N, action_size)
            n_obs_tensor = torch.tensor([get_one_hot_obs(ind_obs) for ind_obs in n_obs], dtype=torch.float)  # Shape : (N, num_grids, episode_length + 1)
            n_obs_tensor = n_obs_tensor.view(-1, self.observation_size)  # Shape : (N, observation_size)
            tensors['n_obs'] = n_obs_tensor
            tensors['n_obs_mask'] = n_masks
        if fea is not None:
            fea_tensor = torch.tensor(fea, dtype=torch.float)
            fea_tensor = fea_tensor.view(-1, self.feature_size)  # Shape : (N, feature_size)
            tensors['fea'] = fea_tensor
        if done is not None:
            done_tensor = torch.tensor(done)
            done_tensor = done_tensor.view(-1, 1)  # Shape : (N, 1)
            tensors['done'] = done_tensor
        if m_act_s is not None:
            m_act_s_tensor = torch.tensor(m_act_s, dtype=torch.float)
            m_act_s_tensor = m_act_s_tensor.view(-1, self.args.num_mean_actions, self.action_size)  # Shape : (N, num_mean_actions, action_size)
            tensors['m_act_s'] = m_act_s_tensor
        return tensors

    def calculate_losses(self, tensors):
        actor_loss, psi_loss, critic_loss = [None] * 3
        if self.args.mode_ac:
            actor_loss = self.calculate_actor_loss(tensors)
        if self.args.mode_psi:
            psi_loss = self.calculate_psi_loss(tensors)
        else:
            critic_loss = self.calculate_critic_loss(tensors)
        return actor_loss, psi_loss, critic_loss

    def get_expected_psi_target(self, obs, m_act_s):
        """

        Parameters
        ----------
        obs : torch.Tensor
            Shape : (N, observation_size)
        m_act_s : torch.Tensor
            Shape : (N, num_mean_actions, action_size)
        """
        expected_psi_target = torch.zeros(obs.shape[0], self.action_size, self.feature_size)  # Shape : (N, action_size, feature_size)
        for loc in range(self.action_size):
            expected_psi_target_loc = torch.zeros(obs.shape[0], self.action_size, self.feature_size)
            for i in range(self.args.num_mean_actions):
                m_act = m_act_s[:, i, loc]
                m_act = m_act.view(-1, 1)  # Shape : (N, mean_action_size: 1)
                psi_target = self.psi_target(obs, m_act)
                expected_psi_target_loc += psi_target / self.args.num_mean_actions
            expected_psi_target[:, loc, :] = expected_psi_target_loc[:, loc, :]
        return expected_psi_target

    def calculate_actor_loss(self, tensors):
        obs = tensors['obs']  # Shape : (N, observation_size)
        obs_mask = tensors['obs_mask']  # Shape : (N, action_size)
        act = tensors['act']  # Shape : (N, )
        m_act = tensors['m_act']  # Shape : (N, mean_action_size: 1)
        m_act_s = tensors['m_act_s']  # Shape : (N, num_mean_actions, action_size)

        with torch.no_grad():
            # Get q values from the psi/critic target network
            if self.args.mode_psi:
                psi_target = self.get_expected_psi_target(obs, m_act_s)
                # psi_target = self.psi_target(obs, m_act)  # Shape : (N, action_size, feature_size)
                q_target = torch.tensordot(psi_target, self.w, dims=([2], [0]))  # Shape : (N, action_size)
            else:
                q_target = self.critic_target(obs, m_act)  # Shape : (N, action_size)
            # Get action probabilities from the actor target network
            act_probs_target = self.actor_target(obs)
            dists_target = get_masked_categorical(act_probs_target, obs_mask)
            act_probs_target = dists_target.probs

            # Get v values using q values and action probabilities
            v_target = torch.sum(q_target * act_probs_target, dim=-1).view(-1, 1)  # Shape : (N, 1)

        # Get action probabilities from the actor network
        act_probs = self.actor(obs)
        dists = get_masked_categorical(act_probs, obs_mask)

        # Get actor loss using values and probabilities
        q_target = q_target[torch.arange(q_target.size(0)), act].view(-1, 1)  # Shape : (N, 1)
        actor_loss = - (q_target - v_target) * dists.log_prob(act).view(-1, 1)
        actor_loss = torch.mean(actor_loss)

        return actor_loss

    def calculate_psi_loss(self, tensors):
        obs = tensors['obs']  # Shape : (N, observation_size)
        act = tensors['act']  # Shape : (N, )
        m_act = tensors['m_act']  # Shape : (N, 1)
        n_obs = tensors['n_obs']  # Shape : (N, observation_size)
        n_obs_mask = tensors['n_obs_mask']  # Shape : (N, action_size)
        fea = tensors['fea']  # Shape : (N, feature_size)
        done = tensors['done']  # Shape : (N, 1)

        with torch.no_grad():
            # Get psi values of next observation from the psi target network
            psi_target_n = self.psi_target(n_obs, m_act)  # (N, action_size, feature_size)

            # Get action probabilities from the actor target network or the Boltzmann policy (N, action_size)
            if self.args.mode_ac:
                act_probs_target_n = self.actor_target(n_obs)
            else:
                act_probs_target_n = self.get_boltzmann_policy(psi_target_n)
            dists_target_n = get_masked_categorical(act_probs_target_n, n_obs_mask)
            act_probs_target_n = dists_target_n.probs

            # Get expected psi using psi and action probabilities
            expected_psi_target_n = torch.bmm(act_probs_target_n.unsqueeze(1), psi_target_n)  # (N, 1, feature_size)
            expected_psi_target_n = expected_psi_target_n.view(-1, self.feature_size)  # (N, feature_size)
            expected_psi_target_n = expected_psi_target_n * ~done  # (N, feature_size)

        # Get psi loss
        psi = self.psi(obs, m_act)  # (N, action_size, feature_size)
        psi = psi[torch.arange(psi.size(0)), act]  # (N, feature_size)
        psi_loss = (fea + expected_psi_target_n - psi) ** 2  # (N, feature_size)
        psi_loss = torch.mean(psi_loss, dim=0)  # (feature_size, )

        return psi_loss

    def calculate_critic_loss(self, tensors):
        obs = tensors['obs']
        act = tensors['act']
        rew = tensors['rew']
        m_act = tensors['m_act']
        n_obs = tensors['n_obs']
        n_obs_mask = tensors['n_obs_mask']
        with torch.no_grad():
            # Get q values from the critic target network
            q_target_n = self.critic_target(n_obs, m_act)

            # Get action probabilities from the actor target network or the Boltzmann policy
            if self.args.mode_ac:
                act_probs_target_n = self.actor_target(n_obs)
            else:
                act_probs_target_n = self.get_boltzmann_policy(q_target_n)
            dists_target_n = get_masked_categorical(act_probs_target_n, n_obs_mask)
            act_probs_target_n = dists_target_n.probs

            # Get v values using q values and action probabilities
            v_target_n = torch.sum(q_target_n * act_probs_target_n, dim=-1).view(-1, 1)

        # Get actor loss using values
        q = self.critic(obs, m_act)
        q = q[torch.arange(q.size(0)), act].view(-1, 1)
        critic_loss = (rew + v_target_n - q) ** 2
        critic_loss = torch.mean(critic_loss)

        return critic_loss

    def update_networks(self, samples):
        obs, act, rew, m_act, n_obs, fea, done, m_act_s = self.preprocess(samples)
        tensors = self.to_tensors(obs=obs, act=act, rew=rew, m_act=m_act, n_obs=n_obs, fea=fea, done=done, m_act_s=m_act_s)
        actor_loss, psi_loss, critic_loss = self.calculate_losses(tensors)

        if self.args.mode_ac:
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self.actor_skd.step() if self.args.mode_lr_decay else None
        if self.args.mode_psi:
            self.psi_opt.zero_grad()
            psi_loss.backward(torch.ones(self.feature_size))
            self.psi_opt.step()
            self.psi_skd.step() if self.args.mode_lr_decay else None
        else:
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            self.critic_skd.step() if self.args.mode_lr_decay else None

    def update_target_network(self, network, target_network):
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(param.data * self.args.tau + target_param.data * (1.0 - self.args.tau))

    def update_target_networks(self):
        if self.args.mode_ac:
            self.update_target_network(self.actor, self.actor_target)
        if self.args.mode_psi:
            self.update_target_network(self.psi, self.psi_target)
        else:
            self.update_target_network(self.critic, self.critic_target)
