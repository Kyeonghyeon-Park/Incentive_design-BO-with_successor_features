from gym.spaces import Box
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand

from sequential_social_dilemma_games.social_dilemmas.envs.agent import HarvestAgent, HarvestAgentModified
from sequential_social_dilemma_games.social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from sequential_social_dilemma_games.social_dilemmas.envs.map_env import MapEnv, MapEnvModified
from sequential_social_dilemma_games.social_dilemmas.maps import HARVEST_MAP

APPLE_RADIUS = 2

# Add custom actions to the agent
# _HARVEST_ACTIONS = {"FIRE": 5}  # {Action name: length of firing range}
_HARVEST_ACTIONS = {}

# SPAWN_PROB = [0, 0.005, 0.02, 0.05]  # Original spawn rate
SPAWN_PROB = [0, 0.005, 0.01, 0.015]


HARVEST_VIEW_SIZE = 7


class HarvestEnv(MapEnv):
    def __init__(
        self,
        ascii_map=HARVEST_MAP,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,
    ):
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            HARVEST_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])

    @property
    def action_space(self):
        return DiscreteWithDType(8, dtype=np.uint8)

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")

    def custom_action(self, agent, action):
        agent.fire_beam(b"F")
        updates = self.update_map_fire(
            agent.pos.tolist(), agent.get_orientation(), self.all_actions["FIRE"], fire_char=b"F",
        )
        return updates

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if (
                                0 <= x + j < self.world_map.shape[0]
                                and self.world_map.shape[1] > y + k >= 0
                            ):
                                if self.world_map[x + j, y + k] == b"A":
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = random_numbers[r]
                r += 1
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, b"A"))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples


SYMBOL_TO_NUM_HARVEST = {
    b" ": 0,
    b"@": 1,
    b"A": 2,
    b"P": 3,
}

NUM_TO_SYMBOL_HARVEST = {
    0: b" ",
    1: b"@",
    2: b"A",
    3: b"P",
}


class HarvestEnvModified(MapEnvModified):
    def __init__(
        self,
        ascii_map=HARVEST_MAP,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,
        lv_penalty=0,
        lv_incentive=0,
    ):
        self.lv_penalty = lv_penalty
        self.lv_incentive = lv_incentive
        self.num_new_apples = 0
        self.avg_spawn_rate = 0
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            HARVEST_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])

    def reset(self):
        self.num_new_apples = 0
        obs = super().reset()
        obs = self.map_to_idx(obs)

        return obs

    def step(self, actions):
        obs, act, rew, m_act, n_obs, fea = super().step(actions)
        obs = self.map_to_idx(obs)
        n_obs = self.map_to_idx(n_obs)
        rew, fea = self.get_rew_and_fea(rew, fea)

        return obs, act, rew, m_act, n_obs, fea

    @property
    def action_space(self):
        """
        Return the Discrete class (class from gym), Discrete.sample() -> return randint.
        Unlike the original action_space, we will (or possibly) decrease the possible actions.
        ex. remove rotation, remove fire_beam, etc.
        Therefore, action_space should consider the varying number of actions rather than the fixed number.
        Previously, action_space returns DiscreteWithDType(8, dtype=np.uint8).

        Returns
        -------
        object : Discrete
            The Discrete class (class from gym).
        """
        return DiscreteWithDType(len(self.all_actions), dtype=np.uint8)

    @property
    def observation_space(self):
        """
        Return the Box class (class from gym), Box.shape -> return shape.
        Unlike the original observation_space, we don't use rgb representation.
        We will use SYMBOL_TO_NUM_CLEANUP to build the observation.
        Therefore, obs_space = Box(low=0, high=3, shape=(2 * self.view_len + 1, 2 * self.view_len + 1),dtype=np.unit8)

        Returns
        -------
        obs_space : Box
            The Box class (class from gym).
        """
        obs_space = Box(
            low=0,
            high=max(SYMBOL_TO_NUM_HARVEST.values()),
            shape=(2 * self.view_len + 1, 2 * self.view_len + 1),
            dtype=np.uint8,
        )
        return obs_space

    @property
    def feature_space(self):
        """
        Return the Box class (class from gym), Box.shape -> return shape.

        Returns
        -------
        feature_space : Box
            The Box class (class from gym).
        """
        fea_space = Box(
            low=0,
            high=1,
            shape=(3,),
            dtype=np.uint8,
        )
        return fea_space

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgentModified(agent_id,
                                         spawn_point,
                                         rotation,
                                         grid,
                                         view_len=HARVEST_VIEW_SIZE,
                                         lv_penalty=self.lv_penalty,
                                         lv_incentive=self.lv_incentive,)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")

    def custom_action(self, agent, action):
        agent.fire_beam(b"F")
        updates = self.update_map_fire(agent.pos.tolist(),
                                       agent.get_orientation(),
                                       self.all_actions["FIRE"],
                                       fire_char=b"F",
                                       )
        return updates

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        self.num_new_apples = len(new_apples)
        self.update_map(new_apples)

    def spawn_apples(self):
        """
        In addition to the original 'spawn_apples' function, we calculate avg_spawn_rate in this function.
        Construct the apples spawned in this step.
        (New) Get an average spawn rate of apple spawn points which has no apple and agent.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """
        new_apple_points = []
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points))
        r = 0
        sum_spawn_rate = 0
        num_spawn_rate = 0  # Number of apple spawn points which has no apple and agent.

        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                num_spawn_rate += 1
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if 0 <= x + j < self.world_map.shape[0] and self.world_map.shape[1] > y + k >= 0:
                                if self.world_map[x + j, y + k] == b"A":
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                sum_spawn_rate += spawn_prob
                rand_num = random_numbers[r]
                r += 1
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, b"A"))

        avg_spawn_rate = sum_spawn_rate / num_spawn_rate if num_spawn_rate != 0 else max(SPAWN_PROB)

        # avg_spawn_rate = sum_spawn_rate / len(self.apple_points)
        self.avg_spawn_rate = avg_spawn_rate

        return new_apple_points

    def get_rew_and_fea(self, rew, fea):
        avg_spawn_rate = self.avg_spawn_rate
        max_spawn_rate = max(SPAWN_PROB)
        status = 1 - avg_spawn_rate / max_spawn_rate
        for key in rew.keys():
            if rew[key]:  # rew[key] == 1
                rew[key] = 1 - self.lv_penalty * status
                fea[key] = np.array([1, -status, 0])
            else:
                rew[key] = self.lv_incentive * status
                fea[key] = np.array([0, 0, status])

        return rew, fea

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples

    def single_map_to_idx(self, grid):
        """
        Get a 2D array of numbers representing the map.
        Symbols(b' ', b'@', ...) will be changed into numbers using SYMBOL_TO_NUM_HARVEST.

        Parameters
        -------
        grid

        Returns
        -------
        grid_idx:
            2D array of numbers representing the map.
        """
        grid_idx = np.full(
            (grid.shape[0], grid.shape[1]),
            fill_value=0,
            dtype=np.uint8,
        )
        for row in range(grid_idx.shape[0]):
            for col in range(grid_idx.shape[1]):
                grid_idx[row, col] = SYMBOL_TO_NUM_HARVEST[grid[row, col]]

        return grid_idx

    def map_to_idx(self, grids):
        """
        Get a Dict of 2D arrays.
        Each key of grids is the agent_id.
        2D arrays are filled with numbers and represents the grids.
        Symbols(b' ', b'@', ...) will be changed into numbers using SYMBOL_TO_NUM_HARVEST

        Parameters
        ----------
        grids

        Returns
        -------
        grids_idx : Dict
            Dict of 2D arrays
        """
        grids_idx = dict()
        for agent_id in grids.keys():
            grids_idx[agent_id] = self.single_map_to_idx(grids[agent_id])

        return grids_idx

    def render(self, filename=None, i=0, act_probs=None):
        """
        Creates an image of the map to plot or save.
        In addition to the original render, it contains the several status.

        Args
        ----
        filename: str
            If a string is passed, will save the image to disk at this location.
        i: int
            Current number of step. It will be the title of this image
        act_probs: dict
            Dict of action probs for agents in the current observation.
        """
        rgb_arr = self.full_map_to_colors()
        step = 'step='+str(i).zfill(9)
        # Print new apples instead of apple prob.
        new_apple_spawn = 'new_apple_spawn=' + '{num}'.format(num=self.num_new_apples)
        texts = [step, new_apple_spawn]

        fig = plt.figure()
        spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1])
        fig_graph = fig.add_subplot(spec[0])
        fig_graph.imshow(rgb_arr, interpolation="nearest")
        fig_status = fig.add_subplot(spec[1])
        fig_status.axis('off')
        for i in range(len(texts)):
            fig_status.text(0, 0.95 - 0.1 * i, texts[i], horizontalalignment='left', verticalalignment='center',
                            transform=fig_status.transAxes)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close(fig)

