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
1) We have to build the successor feature network
2) We have to decide the stopping point (In my algorithm, I just train the network a fixed number of times)
   If we try to find the benefit of the successor feature network, 
   we have to say that SF network helps the convergence speed or the performance
"""

# %% Define the actor network and the critic network
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


class Critic(nn.Module):
    """
    Critic network
    """
    def __init__(self, net_size):
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
            Input for the critic network (critic_input)

        Returns
        -------
        x : torch.Tensor
            Return the q value
        """
        for i in range(self.num_layers):
            x = self.layers[i](x)     #[12, 64, 32, 16, 2] 
            if i < self.num_layers - 1: #if 0 < 4
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
class ActorCritic(object):
    """
    Define the actor-critic network
    """
    def __init__(self, args):
        """
        Create a new actor-critic network

        Attributes
        ----------
        world : grid_world.ShouAndDiTaxiGridGame
            Define the taxi grid game
        actor_layer : list
            Dimension of layers of the actor network
        critic_layer : list
            Dimension of layers of the critic network
        actor : Actor
            Actor network
        critic : Critic
            Critic network
        update_period : int
            Period for updating the actor network and the critic network
        lr_actor : float
            Learnig rate for the actor network
        lr_critic : float
            Learning rate for the critic network
        actor_loss_type : str
            "avg" or "max" or "mix"
            How to calculate the v_observation
        optimizerA : torch.optim.adam.Adam
            Optimizer for the actor network
        optimizerC : torch.optim.adam.Adam
            Optimizer for the critic network
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
        obj_weight : float
            Weight of the ORR
            Objective is weighted average of the ORR and (1-OSC)
        outcome : dict
            Outcome for previous episodes
        trained_episode_number : int
            Trained episode number of the pre-trained network (for learn the trained network more)
        trained_time : float
            Trained time of the pre-trained network (for learn the trained network more)
        max_episode_number : int
            Maximum episode number for the training
        actor_target : Actor
            Target actor network (update per "update_period" episodes)
        critic_target : Critic
            Target critic network (update per "update_period" episodes)

        Parameters
        ----------
        args : argparse.Namespace
        """
        # Generate the game
        self.world = ShouAndDiTaxiGridGame()

        # Generate the actor network and critic network
        self.actor_layer = [7, 32, 16, 8, 4]
        self.critic_layer = [12, 64, 32, 16, 2] #last changes for 2 to match SF DIM
        self.actor = Actor(self.actor_layer)
        self.critic = Critic(self.critic_layer)

        # Define the parameters
        self.update_period = args.update_period

        if not args.trained:  # if we don't use the trained model (to train more), initialize the parameters
            self.actor.apply(init_weights)
            self.critic.apply(init_weights)
            self.lr_actor = args.lr_actor
            self.lr_critic = args.lr_critic
            self.actor_loss_type = args.actor_loss_type
            self.optimizerA = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            self.optimizerC = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
            self.discount_factor = args.discount_factor
            self.designer_alpha = args.designer_alpha
            self.epsilon = args.epsilon
            self.epsilon_decay = args.epsilon_decay
            self.buffer = []
            self.buffer_max_size = args.buffer_max_size
            self.K = args.sample_size
            self.mean_action_sample_number = args.mean_action_sample_number
            self.obj_weight = args.obj_weight
            self.outcome = {'train': {'ORR': [], 'OSC': [], 'avg_reward': [], 'obj_ftn': []},
                            'test': {'ORR': [], 'OSC': [], 'avg_reward': [], 'obj_ftn': []}
                            }
            self.trained_episode_number = 0
            self.trained_time = 0
            self.max_episode_number = args.max_episode_number
        else:
            data = torch.load(args.PATH + args.filename)
            self.actor.load_state_dict(data['actor'])
            self.critic.load_state_dict(data['critic'])
            try:
                self.lr_actor = data['parameters']['lr_actor']
            except:
                self.lr_actor = 0.0001
            try:
                self.lr_critic = data['parameters']['lr_critic']
            except:
                self.lr_critic = 0.001
            try:
                self.actor_loss_type = data['parameters']['actor_loss_type']
            except:
                self.actor_loss_type = "avg"
            self.optimizerA = optim.Adam(self.actor.parameters())
            self.optimizerA.load_state_dict(data['optimizerA'])
            self.optimizerC = optim.Adam(self.critic.parameters())
            self.optimizerC.load_state_dict(data['optimizerC'])
            self.discount_factor = data['parameters']['discount_factor']
            self.designer_alpha = data['parameters']['alpha']
            try:
                self.epsilon = data['parameters']['epsilon']
            except:
                self.epsilon = 0.5
            try:
                self.epsilon_decay = data['parameters']['epsilon_decay']
            except:
                self.epsilon_decay = False
            try:
                self.buffer = data['buffer']
            except:
                self.buffer = []
            try:
                self.buffer_max_size = data['parameters']['buffer_max_size']
            except:
                self.buffer_max_size = 50
            try:
                self.K = data['parameters']['sample_size']
            except:
                self.K = data['parameters']['buffer_size']
            self.mean_action_sample_number = data['parameters']['mean_action_sample_number']
            self.obj_weight = data['parameters']['obj_weight']
            self.outcome = data['outcome']
            self.trained_episode_number = data['parameters']['max_episode_number']
            self.max_episode_number = args.max_episode_number
            self.trained_time = data['total_time']

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

    def get_outcome(self, overall_fare, mode):
        """
        Define the outcome function (add the current result)

        Parameters
        ----------
        overall_fare : np.array
            (fee, fare)
        mode : str
            'train' or 'test'
        """
        # Order response rate / do not consider no demand case in the game
        total_request = self.world.demand[:, 3].shape[0]
        fulfilled_request = np.sum(self.world.demand[:, 3])
        self.outcome[mode]['ORR'].append(fulfilled_request / total_request)

        # Overall service charge ratio 
        if overall_fare[1] != 0:
            self.outcome[mode]['OSC'].append(overall_fare[0] / overall_fare[1])
        else:
            self.outcome[mode]['OSC'].append(0)

        # Average reward of all agents
        self.outcome[mode]['avg_reward'].append((overall_fare[1] - overall_fare[0]) / self.world.number_of_agents)
        self.outcome[mode]['obj_ftn'].append(self.obj_weight * self.outcome[mode]['ORR'][-1]
                                             + (1 - self.obj_weight) * (1 - self.outcome[mode]['OSC'][-1]))

    def get_location_agent_number_and_prob(self, joint_observation, current_time):
        """
        Define the function that returns agents' number and action distribution for each location

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

    def get_sf_expectation_over_mean_action(self, observation, action, agent_num, action_dist_set):
        """
        Define the function for expectation over mean action using sampling

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
        sf_observation_action : float
            Return the Q value expectation over mean action calculated by SF
        """
        sf_observation_action = 0
        w = np.array([1,self.designer_alpha])
        temp_observation = self.world.move_agent(observation, action)
        temp_location = temp_observation[0]
        temp_time = temp_observation[1]
        local_demand = np.argwhere((self.world.demand[:, 0] == temp_time) & (self.world.demand[:, 1] == temp_location))[
                       :, 0]
        local_demand_num = local_demand.shape[0]

        sample_number = self.mean_action_sample_number

        for expectation_over_mean_action in range(sample_number):
            loc_agent_num = 0
            for loc in range(4):
                num = agent_num[loc]
                prob = action_dist_set[loc].probs[0][action].detach().numpy()
                loc_agent_num = loc_agent_num + np.random.binomial(num, prob)
            try:
                mean_action_sample = local_demand_num / loc_agent_num
            except ZeroDivisionError:
                if local_demand_num > 0:
                    mean_action_sample = 1
                else:
                    mean_action_sample = 0

            critic_input = get_critic_input(observation, action, mean_action_sample)
            # q = psi.T w 
            psi = np.array(self.critic_target(critic_input))
            psiT = psi.reshape(w.shape)  
            sf_observation_action = sf_observation_action + np.sum(psiT * w)/ sample_number

        return sf_observation_action

    def calculate_actor_loss(self, sample, agent_id): #modified to work with sf instead of Q
        """
        Define the actor loss function for one sample and agent id

        Parameters
        ----------
        sample : list
            Sample from the buffer
        agent_id : int
            Agent's id

        Returns
        -------
        actor_loss : float
            Return the actor loss for the specific agent
        """
        observation = sample[0][agent_id]
        w = np.array([1,self.designer_alpha])
        action = sample[1][agent_id]
        with torch.no_grad(): 
            agent_num, action_dist_set = self.get_location_agent_number_and_prob(sample[0], observation[1])
            available_action_set = self.world.get_available_action_from_location(observation[0])
            sf_observation_set = torch.zeros(4)
            v_observation_avg = 0
            target_action_dist = get_action_dist(self.actor_target, observation)
            for available_action in available_action_set:
                sf_observation_set[available_action] = self.get_sf_expectation_over_mean_action(observation,
                                                                                              available_action,
                                                                                              agent_num,
                                                                                              action_dist_set)
                v_observation_avg = v_observation_avg + target_action_dist.probs[0][available_action] \
                                    * sf_observation_set[available_action]

            v_observation_max = max(sf_observation_set) #single value

        action = torch.tensor(action)
        action_dist = get_action_dist(self.actor, observation)
        sf_observation = sf_observation_set[action]

        if self.actor_loss_type == "avg":
            v_observation = v_observation_avg
        elif self.actor_loss_type == "max":
            v_observation = v_observation_max
        else:  # self.actor_loss_type == "mix"
            v_observation = 1/2 * (v_observation_avg + v_observation_max)
        actor_loss = - (sf_observation  - v_observation) * action_dist.log_prob(action)

        return actor_loss

    def calculate_critic_loss(self, sample, agent_id):
        """
        Define the critic loss function for one sample and agent id

        Parameters
        ----------
        sample : list
            Sample from the buffer
        agent_id : int
            Agent's id

        Returns
        -------
        critic_loss : float
            Return the actor loss for the specific agent
        """
        observation = sample[0][agent_id]
        action = sample[1][agent_id]
        phi = sample[5][agent_id] #5 is joint feature
        mean_action = sample[3][agent_id]
        next_observation = sample[4][agent_id]
        with torch.no_grad():
            if next_observation[1] != self.world.max_episode_time:
                available_action_set = self.world.get_available_action_from_location(next_observation[0])

                sf_next_observation = []
                # get each location's agent numbers and action distributions from next_joint_observation
                agent_num, action_dist_set = self.get_location_agent_number_and_prob(sample[4], next_observation[1])

                # sampling the number which represents the number of agents who want to go to the location of available action
                for available_action in available_action_set:
                    sf_next_observation_action = self.get_sf_expectation_over_mean_action(next_observation,
                                                                                        available_action,
                                                                                        agent_num, action_dist_set)
                    sf_next_observation.append(sf_next_observation_action)
                # max_q_next_observation = (np.max(q_next_observation)).clone().detach()
                max_sf_next_observation = np.max(sf_next_observation) #CHECK THIS
            else:
                max_sf_next_observation = 0
            # temporal test
            #max_sf_next_observation = 0
        critic_input = get_critic_input(observation, action, mean_action)
        #print(phi)
        phi = torch.tensor(phi.flatten())
        #print(phi)
        critic_loss = phi + self.discount_factor * max_sf_next_observation - self.critic(critic_input) 

        return critic_loss

    def train(self): #modify to work with sf
        """
        Define the train function to train the actor network and the critic network
        """
        self.world.initialize_game(random_grid=False)
        global_time = 0
        overall_fare = np.array([0, 0], 'float')
        #overall_phi =  np.zeros(self.world.number_of_agents) #initialize to 0
        #overall_rfit = [] #save reward obtained by phi^T * w

        #w = np.array([1,self.designer_alpha]) #weight vector

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
        
        # Get outcome of train episode by phi^T * w
        #for i in range(len(overall_phi)):
        #    if  overall_phi[i] == None:
        #        rfit = overall_phi[i] #appends none to the array
        #    else:             
        #        phi =  np.array(overall_phi[i])
        #        phiT = phi.reshape(w.shape)  
        #        r_fit = np.sum(phiT * w) 
            
        #    overall_rfit.append(rfit) 

        #self.get_outcome(overall_rfit,  mode='train') #update using sf fare
        

        # Update the network
        sample_id_list = np.random.choice(len(self.buffer), self.K, replace=True)
        actor_loss = torch.tensor([[0]])
        critic_loss = torch.tensor([[0]])

        update_count = 0

        for sample_id in sample_id_list:
            sample = self.buffer[sample_id]
            if sample[0][0][1] != 0:
                continue
            for agent_id in range(self.world.number_of_agents):
                if sample[1][agent_id] is not None:
                    actor_loss = actor_loss + self.calculate_actor_loss(sample, agent_id)
                    critic_loss = critic_loss + self.calculate_critic_loss(sample, agent_id) ** 2
                    update_count += 1
                else:
                    continue

        # not divided by K (Because all agents are homogeneous, all experiences (# of available agents) used for update)
        if update_count != 0:
            actor_loss = actor_loss / update_count
            critic_loss = critic_loss / update_count

            self.optimizerA.zero_grad()
            self.optimizerC.zero_grad()
            actor_loss.backward()
            #critic_loss.backward() changed to return [1,2]
            critic_loss.backward(torch.Tensor([[1, 1]])) #[1,2]
            self.optimizerA.step()
            self.optimizerC.step()

    def evaluate(self): #modify this to evaluate SF network
        """
        Define the evaluate function to evaluate the trained actor network
        """
        self.world.initialize_game(random_grid=False)
        global_time = 0
        overall_fare = np.array([0, 0], 'float')
        #overall_phi =  np.zeros(self.world.number_of_agents) #initialize to 0
        #overall_rfit = [] #save reward obtained by phi^T * w
        #w = [1,self.designer_alpha] #weight vector

        while global_time is not self.world.max_episode_time:
            available_agent = self.world.get_available_agent(global_time)
            joint_action = []
            for agent_id in available_agent:
                action_dist = get_action_dist(self.actor, self.world.joint_observation[agent_id])
                if global_time == 0 and agent_id in [0, len(available_agent) - 1]:
                    print(agent_id, action_dist.probs)
                action = action_dist.sample()  # runtime error if specific probability is too small
                joint_action.append(action.item())
            if len(available_agent) != 0:
                buffer, overall_fare = self.world.step(available_agent, joint_action, self.designer_alpha, self.buffer,
                                                       overall_fare, train=False)
            global_time += 1

        self.get_outcome(overall_fare, mode='test')

    def save_model(self, total_time, PATH, episode):
        """
        Define the save function to save the model and the results

        Parameters
        ----------
        total_time : float
            Current learning time
        PATH : str
            PATH name
        episode : int
            Number of trained episodes
        """
        if not os.path.isdir(PATH):
            os.makedirs(PATH)
        filename = 'all_' + str(episode) + 'episode.tar'
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_layer': self.actor_layer,
            'optimizerA': self.optimizerA.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_layer': self.critic_layer,
            'optimizerC': self.optimizerC.state_dict(),
            'parameters': {'sample_size': self.K,
                           'buffer_max_size': self.buffer_max_size,
                           'alpha': self.designer_alpha,
                           'max_episode_number': self.max_episode_number,
                           'mean_action_sample_number': self.mean_action_sample_number,
                           'discount_factor': self.discount_factor,
                           'epsilon': self.epsilon,
                           'epsilon_decay': self.epsilon_decay,
                           'obj_weight': self.obj_weight,
                           'lr_actor': self.lr_actor,
                           'lr_critic': self.lr_critic,
                           'actor_loss_type': self.actor_loss_type
                           },
            'buffer': self.buffer,
            'outcome': self.outcome,
            'total_time': total_time
        }, PATH + filename)

    def run(self):
        """
        Run the network (train and test)
        """
        np.random.seed(seed=1234)
        start = time.time()
        for episode in np.arange(self.trained_episode_number, self.max_episode_number):
            self.train()
            with torch.no_grad():
                self.evaluate()
            if (episode + 1) % self.update_period == 0:
                self.actor_target = copy.deepcopy(self.actor)
                self.critic_target = copy.deepcopy(self.critic)

                with torch.no_grad():
                    print_updated_q(self.critic, self.designer_alpha)
                    print_action_distribution(self.actor)
                print_information_per_n_episodes(self.outcome, episode, start)
                draw_plt(self.outcome)

            if self.epsilon_decay:
                if (episode + 1) % 50 == 0:
                    self.epsilon = max(self.epsilon - 0.01, 0.01)

            if (episode + 1) % 100 == 0:
                total_time = self.trained_time + time.time() - start
                PATH = './weights/a_lr=' + str(self.lr_actor) + '_alpha=' + str(round(self.designer_alpha, 4)) + '/' \
                       + time.strftime('%y%m%d_%H%M', time.localtime(start)) + '/'
                self.save_model(total_time, PATH, episode)

    #def reset_model(self, total_time, PATH, episode): #function to reset learning but keep SF of previous iteration

        # reset successor features and replay buffer

        # reset task information

