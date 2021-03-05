import numpy as np
import copy


class ShouAndDiTaxiGridGame:
    def __init__(self):
        """
        Define the taxi grid world setting

        Attributes
        ----------
        grid_size : int
            Size of grid world
        number_of_grids : int
            Total number of grids
        max_episode_time : int
            Maximum global time (= maximum episode length)
        fare : np.array
            Fare of demand (O-D)
            Size : (number_of_grids, number_of_grids)
        travel_time : np.array
            Travel time between grids
            Size : (number_of_grids, number_of_grids)
        number_of_agents : int
            Total number of agents
        demand : np.array
            Demands (or requests, orders) in the world
            Row : (time : int, origin : int, destination : int, fulfilled : boolean)
            Size : (total number of demands, 4)
        joint_observation : np.array
            Joint observation of all agents (each row : each agent's observation)
            Row : (agent's location : int, agent's current time : int)
            Size : (number_of_agents, 2)
        """
        self.grid_size = 2
        self.number_of_grids = self.grid_size * self.grid_size
        self.max_episode_time = 2
        self.fare = np.zeros((self.number_of_grids, self.number_of_grids))
        self.travel_time = np.ones((self.number_of_grids, self.number_of_grids), int)
        self.number_of_agents = 100
        self.demand = np.empty((0, 4))
        self.joint_observation = np.zeros((self.number_of_agents, 2), int)

    def initialize_joint_observation(self, random_grid):
        """
        (Randomly) Initialize the joint observation and the initial position (joint observation = state)

        Parameters
        ----------
        random_grid : boolean
            True if agents' positions are randomly initialized
        """
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

    def initialize_demand(self):
        """
        Initialize the demand
        """
        total_demand = []
        for i in range(int(self.number_of_agents/2)):
            total_demand.append([1, 3, 1, 0])
        for j in range(int(self.number_of_agents/5)):
            total_demand.append([1, 0, 2, 0])
        self.demand = np.asarray(total_demand)

    def initialize_fare(self):
        """
        Initialize the fare structure (Origin-Destination)
        """
        self.fare[3, 1] = 10
        self.fare[0, 2] = 4.9

    def initialize_travel_time(self):
        """
        Initialize the travel time matrix (Origin-Destination)
        Minimum travel time is 1
        """
        for i in range(self.number_of_grids):
            self.travel_time[i, 3-i] = 2

    def initialize_game(self, random_grid):
        """
        Initialize the game

        Parameters
        ----------
        random_grid : boolean
            True if agents' positions are randomly initialized
        """
        self.initialize_joint_observation(random_grid)
        self.initialize_demand()
        self.initialize_fare()
        self.initialize_travel_time()

    def get_available_agent(self, global_time):
        """
        Get available agents whose current time is same as the global time
        Number of available agents is not same as the number of agents
        Example : [available_agent_id, available_agent_id, ...]

        Parameters
        ----------
        global_time : int
            World (or episode)'s time

        Returns
        -------
        available_agent : np.array (numpy.ndarray)
            Return the array of available agents
            Size : number of available agents
        """
        available_agent = np.argwhere(self.joint_observation[:, 1] == global_time)[:, 0]

        return available_agent

    def get_available_action_from_location(self, agent_loc):
        """
        Get available actions from location
        action : (0 : go to grid #0, 1 : go to grid #1, ...)

        Parameters
        ----------
        agent_loc : int
            Agent's location

        Returns
        -------
        available_action_set : np.array (numpy.ndarray)
            Return the array of available action set
        """
        if agent_loc == 0:
            available_action_set = [0, 1, 2]
        elif agent_loc == 1:
            available_action_set = [0, 1, 3]
        elif agent_loc == 2:
            available_action_set = [0, 2, 3]
        else:
            available_action_set = [1, 2, 3]
        available_action_set = np.asarray(available_action_set, int)

        return available_action_set

    def get_available_action(self, agent_id):
        """
        Get available actions for each agent

        Parameters
        ----------
        agent_id : int
            Agent's id (0 to number_of_agents - 1)

        Returns
        -------
        np.array (numpy.ndarray)
            Return the array of available actions for the specific agent
        """
        agent_loc = self.joint_observation[agent_id][0]

        return self.get_available_action_from_location(agent_loc)

    # noinspection PyMethodMayBeStatic
    def move_agent(self, observation, action):
        """
        Calculate the temporal move using current observation and action for individual agent
        It is not the final move (move before the order matching)
        If travel time is not 1, this part should be change

        Parameters
        ----------
        observation : np.array
            Observation of the agent (location, time)
        action : int
            Action of the agent (index of grid)

        Returns
        -------
        temp_observation : np.array
            Return the temporal observation
        """
        temp_observation = np.asarray([action, observation[1] + 1])

        return temp_observation

    def move_available_agent(self, available_agent, joint_action):
        """
        Calculate the joint temporal move for available agents (before order matching)

        Parameters
        ----------
        available_agent : np.array
            The array of available agents
        joint_action : list
            Joint action of available agents

        Returns
        -------
        temp_joint_observation : np.array
            Return the temporal observations of available agents
            Size : (number of available agents, 2)
        """
        temp_joint_observation = []
        for agent in range(available_agent.shape[0]):
            agent_id = available_agent[agent]
            observation = self.joint_observation[agent_id]
            action = joint_action[agent]
            temp_joint_observation.append(self.move_agent(observation, action))
        temp_joint_observation = np.asarray(temp_joint_observation)

        return temp_joint_observation

    def match_agent_and_demand(self, available_agent, temp_joint_observation):
        """
        Order matching after the temporal move (demand of the location and agents)

        Parameters
        ----------
        available_agent : np.array
            The array of available agents
        temp_joint_observation : np.array
            The array of the temporal joint observation of available agents

        Returns
        -------
        order_matching : np.array
            Return the array of order matching
            Row : (agent_id, matched_demand_id)
        """
        temp_time = temp_joint_observation[0, 1]
        order_matching = []
        for location in range(self.number_of_grids):
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
        order_matching = np.asarray(order_matching)

        return order_matching

    def move_with_demand(self, observation, order):
        """
        Move function for matched agent using the order

        Parameters
        ----------
        observation : np.array
            Observation of the agent (location, time)
        order : np.array
            Matched order for the agent (time : int, origin : int, destination : int, fulfilled : boolean)

        Returns
        -------
        observation : np.array
            Final observation for the matched agent (location, time)
        """
        destination = order[2]
        arrival_time = observation[1] + self.travel_time[order[1], order[2]]
        observation = np.array([destination, arrival_time])

        return observation

    def get_demand_to_supply_ratio(self, temp_joint_observation):
        """
        After the temporal move of available agents (before matching),
        calculate the demand to supply ratio for each locations
        If there is no agent, inf will be added

        Parameters
        ----------
        temp_joint_observation : np.array
            Temporal joint observation of available agents

        Returns
        -------
        DS_ratio : np.array
            Return the demand to supply ratio of each grid using the temporal joint observation and demands
            Size : number_of_grids
        """
        temp_time = temp_joint_observation[0, 1]
        DS_ratio = []
        for location in range(self.number_of_grids):
            local_demand = np.argwhere((self.demand[:, 0] == temp_time) & (self.demand[:, 1] == location))[:, 0]
            num_agents = np.sum(temp_joint_observation[:, 0] == location)
            num_orders = local_demand.shape[0]
            if num_agents == 0:
                DS_ratio.append(float("inf"))
            else:
                DS_ratio.append(num_orders / num_agents)

        DS_ratio = np.asarray(DS_ratio)

        return DS_ratio

    def get_service_charge_ratio(self, designer_alpha, DS_ratio):
        """
        After the temporal move of available agents (before matching),
        calculate the service charge ratio for each locations
        Service charge ratio is affected by the alpha

        Parameters
        ----------
        designer_alpha : float
            Designer's decision (penalty for overcrowded grid)
        DS_ratio : np.array
            Demand to supply ratio of each grid
            Size : number_of_grids

        Returns
        -------
        SC_ratio : np.array
            Service charge ratio of each grid
            Size : number_of_grids
        """
        SC_ratio = []
        for location in range(self.number_of_grids):
            if DS_ratio[location] <= 1:
                SC_ratio.append(designer_alpha * (1 - DS_ratio[location]))
            else:
                SC_ratio.append(0)
        SC_ratio = np.asarray(SC_ratio)

        return SC_ratio

    def step(self, available_agent, joint_action, designer_alpha, buffer, overall_fare, train=True):
        """
        Do the state transition and save the sample to the buffer given available agents and joint actions
        Buffer and overall fare are defined externally
        Overall fare will be cumulated if there is a matched request in the episode
        New sample will be appended in the buffer at the training phase
        Assumption : travel time = 1

        Parameters
        ----------
        available_agent : np.array
            The array of available agents
        joint_action : list
            Joint action of available agents
        designer_alpha : float
            Designer's decision (penalty for overcrowded grid)
        buffer : list
            Buffer which contains the information of all agents
            Row : (joint observation, joint action, joint reward, joint mean action, next joint observation, joint feature)
        overall_fare : np.array
            Overall fare in the episode (Fee, Fare)
            Size : (1, 2)
        train : boolean
            True if train phase

        Returns
        -------
        buffer : list
            Buffer after the step
        overall_fare : np.array
            Overall fare after the step
        """
        temp_joint_observation = self.move_available_agent(available_agent, joint_action)
        order_matching = self.match_agent_and_demand(available_agent, temp_joint_observation)
        DS_ratio = self.get_demand_to_supply_ratio(temp_joint_observation)
        SC_ratio = self.get_service_charge_ratio(designer_alpha, DS_ratio)  # Return the SC ratio for each grid

        current_joint_observation = copy.deepcopy(self.joint_observation)

        overall_joint_action = []
        overall_joint_reward = []
        overall_joint_mean_action = []
        overall_joint_feature = []

        idx = 0
        for agent_id in range(self.number_of_agents):
            if agent_id in available_agent:
                action = joint_action[idx]
                temp_observation = temp_joint_observation[idx]
                # Whether agent is matched with any order or not
                if len(order_matching) != 0 and agent_id in order_matching[:, 0]:
                    order_id = order_matching[order_matching[:, 0] == agent_id][0, 1]
                    order = self.demand[order_id]
                    next_observation = self.move_with_demand(temp_observation, order)
                    fare = self.fare[order[1], order[2]]
                    service_charge = SC_ratio[temp_observation[0]]  # SC ratio for specific grid
                    reward = fare * (1 - service_charge)  # Reward calculation
                    demand_to_supply_ratio = DS_ratio[temp_observation[0]]
                    if demand_to_supply_ratio > 1:
                        feature = [fare, 0]
                    else:
                        feature = [fare, - fare * (1 - demand_to_supply_ratio)]
                    overall_fare = overall_fare + np.array([fare * service_charge, fare])
                else:
                    next_observation = temp_observation
                    reward = 0
                    feature = [0, 0]

                mean_action = DS_ratio[temp_observation[0]]
                self.joint_observation[agent_id] = next_observation
                idx = idx + 1

            else:
                action = None
                reward = None
                mean_action = None
                feature = None

            overall_joint_action.append(action)
            overall_joint_reward.append(reward)
            overall_joint_mean_action.append(mean_action)
            overall_joint_feature.append(feature)

        next_joint_observation = copy.deepcopy(self.joint_observation)
        if train:
            buffer.append([current_joint_observation, overall_joint_action, overall_joint_reward,
                           overall_joint_mean_action, next_joint_observation, overall_joint_feature])

        return buffer, overall_fare

