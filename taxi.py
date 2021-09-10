import numpy as np
import copy

class TaxiEnv:
    def __init__(self, args):
        self.grid_size = args.grid_size
        self.num_grids = self.grid_size * self.grid_size
        self.num_agents = args.num_agents
        self.episode_length = args.episode_length
        self.lv_penalty = args.lv_penalty
        self.fare = self.get_fare()
        self.travel_time = self.get_travel_time()
        self.demand = None
        self.obs = None
        self.reset()

    def get_fare(self):
        fare = np.zeros((self.num_grids, self.num_grids))
        fare[3, 1] = 10
        fare[0, 2] = 4.9
        return fare

    def get_travel_time(self):
        travel_time = np.ones((self.num_grids, self.num_grids), dtype=int)
        for i in range(self.num_grids):
            travel_time[i, 3-i] = 2
        return travel_time

    def reset_obs(self):
        """
        Initialize the observation.
        ex. obs = {0: np.array([1, 0], 1: np.array([2, 0])} (agent_id: np.array([location, time]))
        """
        half = self.num_agents / 2
        self.obs = {i: np.array([1, 0], dtype=int) if i < half else np.array([2, 0], dtype=int) for i in range(self.num_agents)}
        # obs = np.zeros((self.num_agents, 2), int)
        # half = self.num_agents / 2
        # for i in range(self.num_agents):
        #     obs[i, 0] = 1 if i < half else 2
        # self.obs = obs

    def reset_demand(self):
        """
        Initialize the demand.
        Each row is the individual demand (=(time, origin, destination, fulfilled or not)).
        """
        demand = []
        for i in range(int(self.num_agents / 2)):
            demand.append([1, 3, 1, 0])
        for j in range(int(self.num_agents / 5)):
            demand.append([1, 0, 2, 0])
        self.demand = np.asarray(demand, dtype=int)

    def reset(self):
        self.reset_obs()
        self.reset_demand()

    def get_available_actions(self, loc):
        available_actions = np.array([[0, 1, 2],
                                      [0, 1, 3],
                                      [0, 2, 3],
                                      [1, 2, 3]], dtype=int)
        return available_actions[loc]

    def get_mask(self):
        mask = np.array([[1, 1, 1, 0],
                         [1, 1, 0, 1],
                         [1, 0, 1, 1],
                         [0, 1, 1, 1]], dtype=int)
        return mask

    def get_random_actions(self, available_agents):
        actions = {}
        for agent_id in available_agents:
            loc = self.obs[agent_id][0]
            available_actions = self.get_available_actions(loc)
            random_action = np.random.choice(available_actions)
            actions[agent_id] = random_action
        return actions

    def get_available_agents(self, global_time):
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
        available_agent : list
            Return the list of ids of available agents
        """
        available_agents = []
        for agent_id, ind_obs in self.obs.items():  # ind_obs : individual observation
            if ind_obs[1] == global_time:
                available_agents.append(agent_id)
        return available_agents

    def move_agent(self, ind_obs, ind_act):
        loc = ind_act
        time = ind_obs[1] + self.travel_time[ind_obs[0], loc]
        temp_ind_obs = np.array([loc, time])
        return temp_ind_obs

    def move_agent_with_demand(self, temp_ind_obs, origin, destination):
        arrival_time = temp_ind_obs[1] + self.travel_time[origin, destination]
        ind_obs = np.array([destination, arrival_time])
        return ind_obs

    def move_agents(self, av_obs, actions):
        """
        Move (available) agents.

        Parameters
        ----------
        actions

        Returns
        -------

        """
        temp_obs = {}
        for agent_id in actions.keys():
            ind_obs = av_obs[agent_id]
            ind_act = actions[agent_id]
            temp_obs[agent_id] = self.move_agent(ind_obs, ind_act)
        return temp_obs

    def get_ratios(self, num_orders, num_agents):
        ds_ratio = float("inf") if num_agents == 0 else num_orders / num_agents
        sc_ratio = self.lv_penalty * (1 - ds_ratio) if ds_ratio <= 1 else 0
        return ds_ratio, sc_ratio

    def match_orders(self, temp_obs, global_time):
        matched_orders = {}
        ds_ratios = np.zeros(self.num_grids)
        sc_ratios = np.zeros(self.num_grids)
        temp_time = global_time + 1
        locs_supply = [[] for _ in range(self.num_grids)]
        for agent_id, ind_obs in temp_obs.items():
            locs_supply[ind_obs[0]].append(agent_id)  # no error even if there is no element in temp_obs
        for loc in range(self.num_grids):
            loc_demand = np.argwhere((self.demand[:, 0] == temp_time) & (self.demand[:, 1] == loc))[:, 0]
            # If there is no demand which satisfies conditions, loc_demand will be ndarray: (0,)
            np.random.shuffle(loc_demand)  # It works even if there is no element in the array
            num_orders = loc_demand.shape[0]
            # np.random.shuffle() changes the original array
            loc_supply = np.array(locs_supply[loc])
            np.random.shuffle(loc_supply)
            num_agents = loc_supply.shape[0]
            ds_ratio, sc_ratio = self.get_ratios(num_orders, num_agents)
            ds_ratios[loc] = ds_ratio
            sc_ratios[loc] = sc_ratio
            if loc_demand.shape[0] != 0:
                for demand_id in loc_demand:
                    if loc_supply.shape[0] != 0:
                        matched_orders[loc_supply[0]] = demand_id
                        loc_supply = loc_supply[1:]
                    else:
                        break
        return matched_orders, ds_ratios, sc_ratios

    def step(self, actions, global_time):
        """
        # available actions 받아서 all agents에 대한 actions return
        # actions = {available_agent_id: action}
        # 바깥에서 None 처리하는 가공하기 (networks 안에서)

        Parameters
        ----------
        actions
        global_time

        Returns
        -------

        """
        av_obs = {agent_id: self.obs[agent_id] for agent_id in actions.keys()}

        temp_obs = self.move_agents(av_obs, actions)
        matched_orders, ds_ratios, sc_ratios = self.match_orders(temp_obs, global_time)
        obs = copy.deepcopy(self.obs)
        act = {}
        rew = {}
        m_act = {}
        fea = {}
        done = {}
        fare_info = np.array([0, 0], dtype=float)
        for agent_id in range(self.num_agents):
            if agent_id in actions.keys():
                temp_ind_obs = temp_obs[agent_id]
                sc_ratio = sc_ratios[temp_ind_obs[0]]
                ds_ratio = ds_ratios[temp_ind_obs[0]]

                if agent_id in matched_orders.keys():
                    order_id = matched_orders[agent_id]
                    self.demand[order_id, 3] = 1  # check that the order is matched
                    order = self.demand[order_id]
                    origin = order[1]
                    destination = order[2]
                    n_ind_obs = self.move_agent_with_demand(temp_ind_obs, origin, destination)
                    fare = self.fare[origin, destination]
                    reward = fare * (1 - sc_ratio)
                    feature = np.array([fare, 0]) if ds_ratio > 1 else np.array([fare, - fare * (1 - ds_ratio)])
                    fare_info += np.array([fare * sc_ratio, fare])
                else:
                    n_ind_obs = temp_ind_obs
                    reward = 0
                    feature = np.array([0, 0])
                mean_action = min(ds_ratio, 1)
                self.obs[agent_id] = n_ind_obs
                action = actions[agent_id]
            else:
                action, reward, mean_action, feature = [None] * 4
            act[agent_id] = action
            rew[agent_id] = reward
            m_act[agent_id] = mean_action
            fea[agent_id] = feature
            # TODO : come back to self.episode_length
            done[agent_id] = True if self.obs[agent_id][1] >= self.episode_length - 1 else False
        n_obs = copy.deepcopy(self.obs)

        return (obs, act, rew, m_act, n_obs, fea, done), fare_info
