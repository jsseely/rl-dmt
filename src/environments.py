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
        """A simple function to print the state of the environment, given the agent's location."""
        print("-" * self.state_space.n)
        for i in range(self.state_space.n):
            if i == self._agent_location:
                print("A", end="")
            elif i in self.terminal_states:
                print("T", end="")
            else:
                # print unicode for light box
                print("\u25A1", end="")
                # print(" ", end="")
        print("\n" + "-" * self.state_space.n)


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
        """A simple function to print the state of the environment, given the agent's location."""
        grid = ""
        for i in range(4):
            row = ""
            for j in range(4):
                if (j, i) == self._state_to_grid[self._agent_location]:
                    row += "A"
                elif (j, i) in self.terminal_grids:
                    row += "T"
                else:
                    # print unicode for light box
                    row += "\u25A1"
            grid = row + "\n" + grid
        print(grid)


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
            "G111111G",
            "00010001",
            "01110001",
            "01011111",
            "11110101",
            "11110101",
            "11111111",
            "S111110G",
        ][::-1]

        self.maze = [
            "G111111G",
            "00010001",
            "01110001",
            "01011111",
            "11110101",
            "11110101",
            "11111111",
            "S111110G",
        ][::-1]

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
        """A simple function to print the state of the environment, given the agent's location."""
        grid = ""
        for i in range(8):
            row = ""
            for j in range(8):
                if (j, i) == self._agent_location:
                    row += "A"
                elif (j, i) in self.terminal_grids:
                    row += "T"
                elif (j, i) in self._state_to_grid.values():
                    # print unicode for light box
                    row += "\u25A1"
                else:
                    row += " "
            grid = row + "\n" + grid
        print(grid)
