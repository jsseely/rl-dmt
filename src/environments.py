import gymnasium as gym
import numpy as np


class CustomMDP(gym.Env):
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


class GridMDP(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Discrete(4)  # Actions: 0, 1, 2, 3
        self.state_space = gym.spaces.Discrete(16)  # States: 0, 1, ..., 15
        self.terminal_states = [0, 15]
        self._agent_location = None

        self._state_to_grid = {i: (i // 4, i % 4) for i in range(self.state_space.n)}
        self._grid_to_state = dict(
            zip(self._state_to_grid.values(), self._state_to_grid.keys())
        )
        self._action_to_direction = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._agent_location = 14
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
        print("-" * 4)
        for i in range(16):
            if i == self._agent_location:
                print("A", end="")
            elif i in self.terminal_states:
                print("T", end="")
            else:
                # print unicode for light box
                print("\u25A1", end="")
            if (i + 1) % 4 == 0:
                print()
        print("-" * 4)
