import random

import numpy as np
from numpy.random import rand

from social_dilemmas.envs.agent import CleanupAgent, CleanupAgentModified
from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv, MapEnvModified
from social_dilemmas.maps import CLEANUP_MAP

# Add custom actions to the agent
_CLEANUP_ACTIONS = {"FIRE": 5, "CLEAN": 5}  # length of firing beam, length of cleanup beam

# Custom colour dictionary
# PKH : Original one but colors were wrong
# CLEANUP_COLORS = {
#     b"C": np.array([100, 255, 255], dtype=np.uint8),  # Cyan cleaning beam
#     b"S": np.array([113, 75, 24], dtype=np.uint8),  # Light grey-blue stream cell
#     b"H": np.array([99, 156, 194], dtype=np.uint8),  # Brown waste cells
#     b"R": np.array([113, 75, 24], dtype=np.uint8),  # Light grey-blue river cell
# }
CLEANUP_COLORS = {
    b"C": np.array([100, 255, 255], dtype=np.uint8),  # Cyan cleaning beam
    b"S": np.array([99, 156, 194], dtype=np.uint8),  # Light grey-blue stream cell
    b"H": np.array([113, 75, 24], dtype=np.uint8),  # Brown waste cells
    b"R": np.array([99, 156, 194], dtype=np.uint8),  # Light grey-blue river cell
}

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

CLEANUP_VIEW_SIZE = 7

thresholdDepletion = 0.4
thresholdRestoration = 0.0
# wasteSpawnProbability = 0.5
# appleRespawnProbability = 0.05
# PKH : test
wasteSpawnProbability = 0.0
appleRespawnProbability = 0.5


class CleanupEnv(MapEnv):
    def __init__(
        self,
        ascii_map=CLEANUP_MAP,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,
    ):
        super().__init__(
            ascii_map,
            _CLEANUP_ACTIONS,
            CLEANUP_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
        )

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get(b"H", 0) + counts_dict.get(b"R", 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == b"B":
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == b"S":
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == b"H":
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] in [b"H", b"R"]:
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == b"R":
                    self.river_points.append([row, col])

        self.color_map.update(CLEANUP_COLORS)

    @property
    def action_space(self):
        return DiscreteWithDType(9, dtype=np.uint8)

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.single_update_map(waste_start_point[0], waste_start_point[1], b"H")
        for river_point in self.river_points:
            self.single_update_map(river_point[0], river_point[1], b"R")
        for stream_point in self.stream_points:
            self.single_update_map(stream_point[0], stream_point[1], b"S")
        self.compute_probabilities()

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == "FIRE":
            agent.fire_beam(b"F")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"F",
            )
        elif action == "CLEAN":
            agent.fire_beam(b"C")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"C",
                cell_types=[b"H"],
                update_char=[b"R"],
                blocking_cells=[b"H"],
            )
        return updates

    def custom_map_update(self):
        """"Update the probabilities and then spawn"""
        self.compute_probabilities()
        self.update_map(self.spawn_apples_and_waste())

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgent(
                agent_id, spawn_point, rotation, map_with_agents, view_len=CLEANUP_VIEW_SIZE,
            )
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points) + len(self.waste_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                rand_num = random_numbers[r]
                r += 1
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, b"A"))

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != b"H":
                    rand_num = random_numbers[r]
                    r += 1
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, b"H"))
                        break
        return spawn_points

    def compute_probabilities(self):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                spawn_prob = (
                    1
                    - (waste_density - thresholdRestoration)
                    / (thresholdDepletion - thresholdRestoration)
                ) * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get(b"H", 0)
        free_area = self.potential_waste_area - current_area
        return free_area


class CleanupEnvModified(MapEnvModified):
    def __init__(
        self,
        ascii_map=CLEANUP_MAP,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,
        lv_penalty=0,
        lv_incentive=0,
    ):
        self.lv_penalty = lv_penalty
        self.lv_incentive = lv_incentive
        super().__init__(
            ascii_map,
            _CLEANUP_ACTIONS,
            CLEANUP_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
        )
        # PKH : Agents are defined by self.setup_agents() (run in super().__init__(~))

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        # PKH : unique = [b' ', b'@', b'B', b'H', b'P', b'R', b'S'], components of base_map
        # PKH : b' ' : the symbol which means the empty space
        # PKH : b'@' : the symbol which means the wall
        # PKH : b'B' : the symbol which means the possible apple spawn point
        # PKH : b'H' : the symbol which means the real waste and the possible waste spawn point
        # PKH : b'P' : the symbol which means the possible player spawn point
        # PKH : b'R' : the symbol which means the river and the possible waste spawn point
        # PKH : b'S' : same as the river
        # PKH : counts = [124, 82, 103, 56, 10, 63, 12]
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get(b"H", 0) + counts_dict.get(b"R", 0)
        # PKH : dict.get(str_key, default_value) : return the key's value, default 0
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()

        # make a list of the potential apple and waste spawn points
        self.apple_points = []  # PKH : possible apple spawn points, each element : location of the point
        self.waste_start_points = []  # PKH : waste points when start
        self.waste_points = []  # PKH : possible waste spawn points
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.spawn_points.append([row, col])  # PKH : possible player spawn points
                elif self.base_map[row, col] == b"B":
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == b"S":
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == b"H":
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] in [b"H", b"R"]:
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == b"R":
                    self.river_points.append([row, col])

        self.color_map.update(CLEANUP_COLORS)

    @property
    def action_space(self):
        return DiscreteWithDType(9, dtype=np.uint8)
        # PKH : return the Discrete class (class from gym), Discrete.sample() -> return randint

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.single_update_map(waste_start_point[0], waste_start_point[1], b"H")
        for river_point in self.river_points:
            self.single_update_map(river_point[0], river_point[1], b"R")
        for stream_point in self.stream_points:
            self.single_update_map(stream_point[0], stream_point[1], b"S")
        self.compute_probabilities()

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == "FIRE":  # PKH : put self.all_actions["FIRE"] into fire_len, current value is 5
            agent.fire_beam(b"F")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"F",
            )

        elif action == "CLEAN":
            agent.fire_beam(b"C")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"C",
                cell_types=[b"H"],
                update_char=[b"R"],
                blocking_cells=[b"H"],
            )
            # PKH : fire_char : type of fire, cell_types : the cell which can be affected by this beam
            # PKH : update_char : if the above cell is affected by beam, state of the cell will be changed into this
            # PKH : (ex. waste->river)
            # PKH : blocking_cells : this cell can block the beam
            if len(updates) != 0:
                agent_id = agent.agent_id
                self.agents[agent_id].clean(b"C")
        return updates

    def custom_map_update(self):
        """"Update the probabilities and then spawn"""
        self.compute_probabilities()
        self.update_map(self.spawn_apples_and_waste())

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgentModified(
                agent_id, spawn_point, rotation, map_with_agents, view_len=CLEANUP_VIEW_SIZE,
                lv_penalty=self.lv_penalty, lv_incentive=self.lv_incentive,
            )
            self.agents[agent_id] = agent  # agent is added to dict

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points) + len(self.waste_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                rand_num = random_numbers[r]
                r += 1
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, b"A"))

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != b"H":
                    rand_num = random_numbers[r]
                    r += 1
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, b"H"))
                        break
        return spawn_points

    def compute_probabilities(self):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                spawn_prob = (
                    1
                    - (waste_density - thresholdRestoration)
                    / (thresholdDepletion - thresholdRestoration)
                ) * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get(b"H", 0)
        free_area = self.potential_waste_area - current_area
        return free_area
