import random

import gymnasium as gym
import numpy as np


class LineEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Discrete(3)  # Actions: 0, 1, 2
        self.state_space = gym.spaces.Discrete(10)  # States: 0, 1, ..., 9
        self.terminal_states = [0]
        self._agent_location = None

        self._action_to_direction = {0: -1, 1: 0, 2: 1}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._agent_location = 8
        return self._agent_location

    def get_next_state_and_reward(self, state, action):
        if state in self.terminal_states:
            return state, 0  # Assuming zero reward at the terminal state
        direction = self._action_to_direction[action]
        next_state = np.clip(state + direction, 0, self.state_space.n - 1)
        reward = -1
        return next_state, reward

    def step(self, action):
        self._agent_location, reward = self.get_next_state_and_reward(
            self._agent_location, action
        )
        done = self._agent_location in self.terminal_states
        return self._agent_location, reward, done, False, {}

    def render(self):
        pass


class GridEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 4x4 grid
        self.action_space = gym.spaces.Discrete(4)  # Actions: 0, 1, 2, 3
        self.state_space = gym.spaces.Discrete(16)  # States: 0, 1, ..., 15

        # (0, 0) is bottom left, (3, 0) is bottom right
        # (3, 0) is top left, (3, 3) is top right
        # i.e. x-y coordinates
        self._state_to_grid = {i: (i % 4, i // 4) for i in range(self.state_space.n)}
        self._grid_to_state = dict(
            zip(self._state_to_grid.values(), self._state_to_grid.keys())
        )

        # U D L R
        self._action_to_direction = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

        self.terminal_grids = [(0, 0), (3, 3)]
        self.terminal_states = [
            self._grid_to_state[grid] for grid in self.terminal_grids
        ]

        self._agent_location = None

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._agent_location = 9
        return self._agent_location

    def get_next_state_and_reward(self, state, action):
        if state in self.terminal_states:
            return state, 0
        direction = np.array(self._action_to_direction[action])
        location = np.array(self._state_to_grid[state])
        next_location = tuple(np.clip(location + direction, 0, 3))
        next_state = self._grid_to_state[next_location]
        reward = -1
        return next_state, reward

    def step(self, action):
        self._agent_location, reward = self.get_next_state_and_reward(
            self._agent_location, action
        )
        done = self._agent_location in self.terminal_states
        return self._agent_location, reward, done, False, {}

    def render(self):
        pass


class MazeEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.maze = [
            "010000G0",
            "01011011",
            "11110001",
            "00100111",
            "01111100",
            "01000110",
            "1S111011",
            "01001110",
        ][::-1]

        self.maze = [
            "S000000G",
            "10000001",
            "11111111",
            "10000001",
            "11111111",
            "10000001",
            "11111111",
            "00000000",
        ][::-1]

        # self.maze = [
        #     "00000000",
        #     "00000000",
        #     "00000000",
        #     "00000000",
        #     "00000000",
        #     "00000000",
        #     "00000000",
        #     "S111111G",
        # ][::-1]

        self.maze = [
            "11111111111G",
            "100000000001",
            "100000000001",
            "111111110001",
            "000010010001",
            "111110011111",
            "100010010000",
            "100011111111",
            "100000000001",
            "111001000001",
            "100001000001",
            "S11111111111",
        ][::-1]

        # self.maze = [
        #     "11G",
        #     "101",
        #     "S11",
        # ][::-1]

        self.action_space = gym.spaces.Discrete(4)  # Actions: 0, 1, 2, 3
        self._agent_location = None

        self._state_to_grid = {}
        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                if self.maze[y][x] != "0":
                    self._state_to_grid[len(self._state_to_grid)] = (x, y)

        self.state_space = gym.spaces.Discrete(len(self._state_to_grid))
        self._grid_to_state = dict(
            zip(self._state_to_grid.values(), self._state_to_grid.keys())
        )

        self.terminal_grids = []
        self.start_grid = None

        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                if self.maze[y][x] == "G":
                    self.terminal_grids.append((x, y))
                elif self.maze[y][x] == "S":
                    self.start_grid = (x, y)

        self.terminal_states = [
            self._grid_to_state[grid] for grid in self.terminal_grids
        ]

        self._action_to_direction = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._agent_location = self.start_grid
        self._agent_state = self._grid_to_state[self._agent_location]
        return self._agent_state

    def get_next_state_and_reward(self, state, action):
        if state in self.terminal_states:
            return state, 0
        direction = np.array(self._action_to_direction[action])
        location = np.array(self._state_to_grid[state])
        next_location = tuple(location + direction)
        # Check if next_location is within bounds and not a wall
        if next_location in self._state_to_grid.values():
            next_state = self._grid_to_state[next_location]
        else:
            next_state = state  # Bump into wall or out of bounds, stay in current state

        reward = -1 + 1e-6 * state  # arbitrary symmetry breaking to ensure morse fns
        return next_state, reward

    def step(self, action):
        self._agent_state, reward = self.get_next_state_and_reward(
            self._agent_state, action
        )
        done = self._agent_state in self.terminal_states
        return self._agent_state, reward, done, False, {}

    def render(self):
        pass


from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.AldousBroder import AldousBroder
from mazelib.generate.Kruskal import Kruskal


class ProcMaze(MazeEnv):
    def __init__(self, size=20, num_loops=None, seed=None):
        super().__init__()
        self.seed = seed

        m = Maze(self.seed)
        m.generator = AldousBroder(size, size)
        m.generate()
        self.maze = np.logical_not(m.grid).astype(int)
        self.maze[0, 1] = 2
        self.maze[0, -2] = 3
        self.maze[-1, 1] = 3
        self.maze[-1, -2] = 3

        if num_loops is not None:
            self.add_loop(num_loops)

        self.action_space = gym.spaces.Discrete(4)  # Actions: 0, 1, 2, 3
        self._agent_location = None

        self._state_to_grid = {}
        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                if self.maze[y][x] != 0:
                    self._state_to_grid[len(self._state_to_grid)] = (x, y)

        self.state_space = gym.spaces.Discrete(len(self._state_to_grid))
        self._grid_to_state = dict(
            zip(self._state_to_grid.values(), self._state_to_grid.keys())
        )

        self.terminal_grids = []
        self.start_grid = None

        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                if self.maze[y][x] == 3:
                    self.terminal_grids.append((x, y))
                elif self.maze[y][x] == 2:
                    self.start_grid = (x, y)

        self.terminal_states = [
            self._grid_to_state[grid] for grid in self.terminal_grids
        ]

        self._action_to_direction = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

    def add_loop(self, num_loops=1):
        # Find indices of 0s that are not on the edges
        zero_indices = np.argwhere(self.maze == 0)
        non_edge_zero_indices = [
            idx
            for idx in zero_indices
            if idx[0] != 0
            and idx[0] != self.maze.shape[0] - 1
            and idx[1] != 0
            and idx[1] != self.maze.shape[1] - 1
        ]

        # Filter indices that are adjacent to exactly two 1s
        filtered_indices = []
        for idx in non_edge_zero_indices:
            row, col = idx
            adjacent_ones = 0
            if self.maze[row - 1, col] == 1:
                adjacent_ones += 1
            if self.maze[row + 1, col] == 1:
                adjacent_ones += 1
            if self.maze[row, col - 1] == 1:
                adjacent_ones += 1
            if self.maze[row, col + 1] == 1:
                adjacent_ones += 1
            if adjacent_ones == 2:
                filtered_indices.append(idx)

        random.seed(self.seed)
        selected_indices = random.sample(
            filtered_indices, min(num_loops, len(filtered_indices))
        )

        for idx in selected_indices:
            self.maze[idx[0], idx[1]] = 1
