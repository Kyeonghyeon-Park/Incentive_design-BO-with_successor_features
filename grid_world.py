import numpy as np
import copy


class ShouAndDiTaxiGridGame:
    # Define the taxi grid world setting
    def __init__(self):
        self.grid_size = 2
        self.max_episode_time = 2  # maximum global time
        self.number_of_grids = self.grid_size * self.grid_size
        self.fare = np.zeros((self.number_of_grids, self.number_of_grids))  # fare of demand (O-D)
        self.travel_time = np.ones((self.number_of_grids, self.number_of_grids), int)  # travel time between grids
        self.number_of_agents = 100
        self.demand = np.empty((0, 4))  # (time, origin, destination, fulfilled)
        self.joint_observation = np.zeros((self.number_of_agents, 2), int)  # (location, current time)

    # Initialize(reset) the joint observation and the initial position (joint observation = state)
    # joint observation's row = each agent's observation : (location, agent's current time)
    def initialize_joint_observation(self, random_grid):
        self.joint_observation = np.zeros((self.number_of_agents, 2), int)
        if random_grid:
            for i in range(self.number_of_agents):
                self.joint_observation[i][0] = np.random.randint(4)
        else:
            for i in range(self.number_of_agents):
                if i < self.number_of_agents/2:
                    self.joint_observation[i][0] = 1
                else:
                    self.joint_observation[i][0] = 2

    # Initialize the demand / demand's row :  (time, origin, destination, fulfilled or not)
    def initialize_demand(self):
        total_demand = []
        for i in range(int(self.number_of_agents/2)):
            total_demand.append([1, 3, 1, 0])
        for j in range(int(self.number_of_agents/5)):
            total_demand.append([1, 0, 2, 0])
        self.demand = np.asarray(total_demand)

    # Initialize the fare structure (Origin-Destination)
    def initialize_fare(self):
        self.fare[3, 1] = 10
        self.fare[0, 2] = 4.9

    # Initialize the travel time matrix (Origin-Destination) / at least 1
    def initialize_travel_time(self):
        for i in range(self.number_of_grids):
            self.travel_time[i, 3-i] = 2

    # Initialize the game
    def initialize_game(self, random_grid):
        self.initialize_joint_observation(random_grid)
        self.initialize_demand()
        self.initialize_fare()
        self.initialize_travel_time()

    # Get available agents whose current time is same as the global time
    # [available_agent_id, available_agent_id, ...]
    def get_available_agent(self, global_time):
        available_agent = np.argwhere(self.joint_observation[:, 1] == global_time)[:, 0]

        return available_agent  # numpy.ndarray

    # Get available actions from location / action : (0 : go to grid #0, 1 : go to grid #1, ...)
    def get_available_action_from_location(self, agent_loc):
        if agent_loc == 0:
            available_action_set = [0, 1, 2]
        elif agent_loc == 1:
            available_action_set = [0, 1, 3]
        elif agent_loc == 2:
            available_action_set = [0, 2, 3]
        else:
            available_action_set = [1, 2, 3]

        return np.asarray(available_action_set, int)  # numpy.ndarray

    # Get available actions for each agent /
    def get_available_action(self, agent_id):
        agent_loc = self.joint_observation[agent_id][0]
        return self.get_available_action_from_location(agent_loc)  # numpy.ndarray

    # For individual agent, calculate the temporal move using current observation and action (before order matching)
    # If travel time is not 1, it should be change
    # noinspection PyMethodMayBeStatic
    def move_agent(self, observation, action):
        temp_observation = [action, observation[1] + 1]

        return np.asarray(temp_observation)

    # calculate the joint temporal move for available agents (before order matching)
    def move_available_agent(self, available_agent, joint_action):
        temp_joint_observation = []
        for agent in range(available_agent.shape[0]):
            agent_id = available_agent[agent]
            observation = self.joint_observation[agent_id]
            action = joint_action[agent]
            temp_joint_observation.append(self.move_agent(observation, action))

        return np.asarray(temp_joint_observation)

    # Order matching after the temporal move (demand of the location and agents)
    # return the array which row means (agent_id, matched_demand_id)
    def match_agent_and_demand(self, available_agent, temp_joint_observation):
        temp_time = temp_joint_observation[0, 1]
        order_matching = []
        for location in range(4):
            # local_demand is not full demand info.
            local_demand = np.argwhere((self.demand[:, 0] == temp_time) & (self.demand[:, 1] == location))[:, 0]
            np.random.shuffle(local_demand)
            local_supply = available_agent[temp_joint_observation[:, 0] == location]
            np.random.shuffle(local_supply)
            if local_demand.shape[0] == 0:
                continue
            for demand_id in local_demand:
                if local_supply.shape[0] != 0:
                    order_matching.append([local_supply[0], demand_id])
                    local_supply = local_supply[1:]
                    self.demand[demand_id, 3] = 1
                else:
                    break

        return np.asarray(order_matching)

    # Move function for matched agents
    def move_with_demand(self, observation, order):
        destination = order[2]
        arrival_time = observation[1] + self.travel_time[order[1], order[2]]
        observation = np.array([destination, arrival_time])

        return observation

    # After the temporal move of available agents (before matching) calculate the demand to supply ratio for each locations
    def get_demand_to_supply_ratio(self, temp_joint_observation):
        temp_time = temp_joint_observation[0, 1]
        DS_ratio = []
        for location in range(4):
            local_demand = np.argwhere((self.demand[:, 0] == temp_time) & (self.demand[:, 1] == location))[:, 0]
            num_agents = np.sum(temp_joint_observation[:, 0] == location)
            num_orders = local_demand.shape[0]
            if num_agents == 0:
                DS_ratio.append(float("inf"))
            else:
                DS_ratio.append(num_orders / num_agents)

        return np.asarray(DS_ratio)

    # After the temporal move of available agents (before matching) calculate the service charge ratio for each locations
    def get_service_charge_ratio(self, designer_alpha, DS_ratio):
        SC_ratio = []
        for location in range(4):
            if DS_ratio[location] <= 1:
                SC_ratio.append(designer_alpha * (1 - DS_ratio[location]))
            else:
                SC_ratio.append(0)

        return np.asarray(SC_ratio)

    # Do the state transition and save the sample to the buffer given available agents and joint actions
    # Assumption : travel time = 1
    def step(self, available_agent, joint_action, designer_alpha, buffer, overall_fare, train=True):
        # Buffer and overall fare are defined externally
        temp_joint_observation = self.move_available_agent(available_agent, joint_action)
        order_matching = self.match_agent_and_demand(available_agent, temp_joint_observation)
        DS_ratio = self.get_demand_to_supply_ratio(temp_joint_observation)
        SC_ratio = self.get_service_charge_ratio(designer_alpha, DS_ratio)

        current_joint_observation = copy.deepcopy(self.joint_observation)

        overall_joint_action = []
        overall_joint_reward = []
        overall_joint_mean_action = []

        idx = 0
        for agent_id in range(self.number_of_agents):
            if agent_id in available_agent:  # same as agent_id
                action = joint_action[idx]
                temp_observation = temp_joint_observation[idx]
                if len(order_matching) != 0 and agent_id in order_matching[:, 0]:
                    order_id = order_matching[order_matching[:, 0] == agent_id][0, 1]
                    order = self.demand[order_id]
                    next_observation = self.move_with_demand(temp_observation, order)
                    fare = self.fare[order[1], order[2]]
                    service_charge = SC_ratio[temp_observation[0]]
                    reward = fare * (1 - service_charge)
                    # print
                    # if not isinstance(service_charge, float):
                    #     print(service_charge)
                    overall_fare = overall_fare + np.array([fare * service_charge, fare])
                else:
                    next_observation = temp_observation
                    reward = 0

                mean_action = DS_ratio[temp_observation[0]]
                self.joint_observation[agent_id] = next_observation
                idx = idx + 1

            else:
                action = None
                reward = None
                mean_action = None

            overall_joint_action.append(action)
            overall_joint_reward.append(reward)
            overall_joint_mean_action.append(mean_action)

        next_joint_observation = copy.deepcopy(self.joint_observation)
        if train:
            buffer.append([current_joint_observation, overall_joint_action, overall_joint_reward,
                           overall_joint_mean_action, next_joint_observation])

        return buffer, overall_fare

