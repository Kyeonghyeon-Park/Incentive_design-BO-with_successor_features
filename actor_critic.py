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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Class for SF network
class SF(): 
    def psi_learning(self, env, psi, epsilon, N, lrn_rate = 0.02):
        """
        Args:
                env (GridWorld): The grid where the algorithm will be applied
                psi ([[floa]]): Successor feature 
                epsilon (int) : exploration rate, the probability to take a random path
                N (int) : number of samples
                Tmax (int) : the limit of episodes

        Returns:
                psi ([[float]]) : updated successor features
                pol (Policy object) : optimal policy according to the psi-learning
                V ([[float]]) : Values computed during the algorithm ??
                w_stock ([[float]]) : list successive value of w
        """
        phi = self.phi
        gamma = env.gamma #check where gamma is defined.
        Tmax = env.max_episode_time 
        # initialize a policy
        size = self.size

        pol = Policy(env) #generates policy
        t = 1
        alpha = [] #LR?
        for i1,i2 in enumerate(env.state_actions) : # state_actions define available actions in each state, modify
            alpha.append([])
            for j1, j2 in enumerate(i2):
                alpha[i1].append(0.5)

        # Null or random initialiation?
        w = np.zeros(len(phi[0][0])) #dimension depends on phi lenght
        w_stock = []
        rewards = []

        rang = range(N)
        for n in rang:

            state = env.reset() # Have to create a reset function for SF. 
            t_lim = 0

            # show the last episode of a round

            while(t_lim < Tmax):

                greedy = np.random.rand(1) > epsilon

                action = pol.get_action(state) if greedy else np.random.choice(env.state_actions[state]) #check with get_action_dist function #check partial state observation

                # To update alpha
                idx_action = env.state_actions[state].index(action)

                q_tmp = []
                prev_state = state
                state, reward =  env.step( available_agent, joint_action, designer_alpha, buffer, overall_fare, train=True) #env.step(state, action) #check step function 
                #check if new version has this function and keeps its arguments

                # Compute the next expected Q-values
                for idx, new_action in enumerate(env.state_actions[state]):
                    q_tmp.append(np.dot(psi[state][new_action], w))

                q_tmp = np.array(q_tmp)
                # Select the best action (random among the maximum) for the next step
                idx_env = np.random.choice(np.flatnonzero(q_tmp == q_tmp.max()))
               

                # Update the policy
                pol.update_action(state, env.state_actions[state][idx_env]) #check policy update with partial observation.

                # Update Psi, the successor feature
                TD_phi = phi[prev_state][action] + gamma*psi[state][pol.get_action(state)] - psi[prev_state][action]
                psi[prev_state][action] = psi[prev_state][action] + alpha[prev_state][idx_action] * TD_phi

                # Update w by gradient descent
                err = np.dot(phi[prev_state][action], w) - reward
                w = w - lrn_rate * phi[prev_state][action] * err / np.log(n+2) # smoothing convergence

                alpha[prev_state][idx_action] = 1./((1/alpha[prev_state][idx_action]) + 1.)
                t_lim += 1
            rewards.append(reward * gamma**(t_lim-1))
            w_stock.append(w)

        return psi, pol, rewards, w_stock






# %% Define the actor network and the critic network
class Actor(nn.Module):
    def __init__(self, net_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(net_size) - 1

        for i in range(self.num_layers):
            fc_i = nn.Linear(net_size[i], net_size[i + 1])
            self.layers.append(fc_i)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        x = F.softmax(x, dim=-1)

        return x


class Critic(nn.Module):
    def __init__(self, net_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(net_size) - 1

        for i in range(self.num_layers):
            fc_i = nn.Linear(net_size[i], net_size[i + 1])
            self.layers.append(fc_i)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        return x


# %% Define the initialization function
def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


# %% Define the actor and critic input generation(conversion) function (categorical data)
def get_actor_input(observation):
    # [0, 1, 2, 3 : location / 4, 5, 6 : time]
    actor_input_numpy = np.zeros(7)
    location = observation[0]
    current_time = observation[1]
    actor_input_numpy[location] = 1
    if current_time > 2:
        actor_input_numpy[6] = 1
    else:
        actor_input_numpy[4 + current_time] = 1
    actor_input = torch.FloatTensor(actor_input_numpy).unsqueeze(0)

    return actor_input


def get_critic_input(observation, action, mean_action):
    # [0, 1, 2, 3 : location / 4, 5, 6 : time / 7, 8, 9, 10 : action / 11 : mean action]
    critic_input_numpy = np.zeros(12)
    location = observation[0]
    current_time = observation[1]
    critic_input_numpy[location] = 1
    if current_time > 2:
        critic_input_numpy[6] = 1
    else:
        critic_input_numpy[4 + current_time] = 1
    critic_input_numpy[4] = current_time
    critic_input_numpy[7 + action] = 1
    critic_input_numpy[11] = np.min([mean_action, 1])
    critic_input = torch.FloatTensor(critic_input_numpy).unsqueeze(0)

    return critic_input


# %% Define the action distribution generation function given actor network and observation (=pi_network(a_i|o_i))
def get_action_dist(actor_network, observation):
    actor_input = get_actor_input(observation)
    action_prob = actor_network(actor_input)
    if observation[0] == 0:
        available_action_torch = torch.tensor([1, 1, 1, 0])
    elif observation[0] == 1:
        available_action_torch = torch.tensor([1, 1, 0, 1])
    elif observation[0] == 2:
        available_action_torch = torch.tensor([1, 0, 1, 1])
    else:
        available_action_torch = torch.tensor([0, 1, 1, 1])
    action_dist = distributions.Categorical(torch.mul(action_prob, available_action_torch))

    return action_dist


# %% draw the graph of outcome (avg reward, ORR, OSC, obj. of train and test)
def draw_plt(outcome):
    plt.figure(figsize=(16, 14))

    plt.subplot(2, 2, 1)
    plt.plot(outcome['train']['avg_reward'], label='Avg reward train')
    plt.ylim([0, 6])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(outcome['test']['avg_reward'], label='Avg reward test')
    plt.ylim([0, 6])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(outcome['train']['ORR'], label='ORR train')
    plt.plot(outcome['train']['OSC'], label='OSC train')
    plt.plot(outcome['train']['obj_ftn'], label='Obj train')
    plt.ylim([0, 1.1])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(outcome['test']['ORR'], label='ORR test')
    plt.plot(outcome['test']['OSC'], label='OSC test')
    plt.plot(outcome['test']['obj_ftn'], label='Obj test')
    plt.ylim([0, 1.1])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()

    plt.show()


def draw_plt_avg(outcome, moving_avg_length):
    outcome_avg = {}
    for i in outcome:
        outcome_avg[i] = {}
        for j in outcome[i]:
            outcome_avg[i][j] = {}
            measure_avg = []
            for k in range(len(outcome[i][j])):
                if k < moving_avg_length - 1:
                    measure_avg.append(np.average(outcome[i][j][:k + 1]))
                else:
                    measure_avg.append(np.average(outcome[i][j][k - moving_avg_length + 1:k + 1]))
            outcome_avg[i][j] = measure_avg
    draw_plt(outcome_avg)


# %% Define the main body
class ActorCritic(object):
    def __init__(self, args):
        # Generate the game
        self.world = ShouAndDiTaxiGridGame()

        # Generate the actor network and critic network
        self.actor_layer = [7, 32, 16, 8, 4]
        self.critic_layer = [12, 64, 32, 16, 1]
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

    # Define the outcome function (add the current result)
    def get_outcome(self, overall_fare, mode):
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

    # Define the function that returns locations' agents' number and action distribution
    def get_location_agent_number_and_prob(self, joint_observation, current_time):
        agent_num = []
        action_dist_set = []
        for loc in range(4):
            agent_num.append(np.sum((joint_observation[:, 0] == loc) & (joint_observation[:, 1] == current_time)))
            action_dist = get_action_dist(self.actor_target, [loc, current_time])
            action_dist_set.append(action_dist)

        return agent_num, action_dist_set

    # Define the function for expectation over mean action using sampling
    def get_q_expectation_over_mean_action(self, observation, action, agent_num, action_dist_set):
        q_observation_action = 0

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

            q_observation_action = q_observation_action + self.critic_target(critic_input) / sample_number

        return q_observation_action

    # Define the actor loss function for one sample and agent id
    def calculate_actor_loss(self, sample, agent_id):
        observation = sample[0][agent_id]
        action = sample[1][agent_id]
        with torch.no_grad():
            agent_num, action_dist_set = self.get_location_agent_number_and_prob(sample[0], observation[1])
            available_action_set = self.world.get_available_action_from_location(observation[0])
            q_observation_set = torch.zeros(4)
            v_observation_avg = 0
            target_action_dist = get_action_dist(self.actor_target, observation)
            for available_action in available_action_set:
                q_observation_set[available_action] = self.get_q_expectation_over_mean_action(observation,
                                                                                              available_action,
                                                                                              agent_num,
                                                                                              action_dist_set)
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

    # Define the critic loss function for one sample and agent id
    def calculate_critic_loss(self, sample, agent_id):
        observation = sample[0][agent_id]
        action = sample[1][agent_id]
        reward = sample[2][agent_id]
        mean_action = sample[3][agent_id]
        next_observation = sample[4][agent_id]
        with torch.no_grad():
            if next_observation[1] != self.world.max_episode_time:
                available_action_set = self.world.get_available_action_from_location(next_observation[0])

                q_next_observation = []
                # get each location's agent numbers and action distributions from next_joint_observation
                agent_num, action_dist_set = self.get_location_agent_number_and_prob(sample[4], next_observation[1])

                # sampling the number which represents the number of agents who want to go to the location of available action
                for available_action in available_action_set:
                    q_next_observation_action = self.get_q_expectation_over_mean_action(next_observation,
                                                                                        available_action,
                                                                                        agent_num, action_dist_set)
                    q_next_observation.append(q_next_observation_action)
                # max_q_next_observation = (np.max(q_next_observation)).clone().detach()
                max_q_next_observation = np.max(q_next_observation)
            else:
                max_q_next_observation = 0
            # temporal test
            max_q_next_observation = 0
        critic_input = get_critic_input(observation, action, mean_action)
        reward = torch.tensor(reward)
        critic_loss = reward + self.discount_factor * max_q_next_observation - self.critic(critic_input)

        return critic_loss

    # Define the train function to train the actor network and the critic network
    def train(self):
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

            # After the step, add (o, a, r, a_bar, o_prime) to the replay buffer B (only train)
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
            critic_loss.backward()
            self.optimizerA.step()
            self.optimizerC.step()

    # Define the evaluate function to evaluate the trained actor network
    def evaluate(self):
        self.world.initialize_game(random_grid=False)
        global_time = 0
        overall_fare = np.array([0, 0], 'float')

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

    # Define the outcome visualization functions
    def print_updated_q(self):
        np.set_printoptions(precision=2, linewidth=np.inf)
        for location in range(4):
            for agent_time in range(3):
                print("Q at (#", location, ", ", agent_time, ")")
                for action in range(4):
                    q = []
                    for mean_action in np.arange(0.0, 1.1, 0.1):
                        critic_input = get_critic_input([location, agent_time], action, mean_action)
                        q_value = self.critic(critic_input)
                        q.append(q_value.item())
                    q = np.array(q)
                    print(q)

    def print_action_distribution(self):
        for location in range(4):
            for agent_time in range(3):
                action_dist = get_action_dist(self.actor, [location, agent_time])
                print("Action distribution at (#", location, ", ", agent_time, ") : ", action_dist.probs[0].numpy())

    def print_information_per_n_episodes(self, episode, start):
        print("########################################################################################")
        print(f"| Episode : {episode:4} | total time : {time.time() - start:5.2f} |")
        print(
            f"| train ORR : {self.outcome['train']['ORR'][episode]:5.2f} "
            f"| train OSC : {self.outcome['train']['OSC'][episode]:5.2f} "
            f"| train Obj : {self.outcome['train']['obj_ftn'][episode]:5.2f} "
            f"| train avg reward : {self.outcome['train']['avg_reward'][episode]:5.2f} |")
        print(
            f"|  test ORR : {self.outcome['test']['ORR'][episode]:5.2f} "
            f"|  test OSC : {self.outcome['test']['OSC'][episode]:5.2f} "
            f"|  test Obj : {self.outcome['test']['obj_ftn'][episode]:5.2f} "
            f"|  test avg reward : {self.outcome['test']['avg_reward'][episode]:5.2f} |")
        print("########################################################################################")

    def save_model(self, total_time, PATH, episode):
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
                    self.print_updated_q()
                    self.print_action_distribution()
                self.print_information_per_n_episodes(episode, start)
                draw_plt(self.outcome)

            if self.epsilon_decay:
                if (episode + 1) % 50 == 0:
                    self.epsilon = max(self.epsilon - 0.01, 0.01)

            if (episode + 1) % 100 == 0:
                total_time = self.trained_time + time.time() - start
                PATH = './weights/a_lr=' + str(self.lr_actor) + '_alpha=' + str(round(self.designer_alpha, 4)) + '/' \
                       + time.strftime('%y%m%d_%H%M', time.localtime(start)) + '/'
                self.save_model(total_time, PATH, episode)



