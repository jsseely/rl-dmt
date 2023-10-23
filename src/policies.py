import gymnasium as gym
import numpy as np


class HardCodedAgent:
    def __init__(self, env, gamma=0.95):
        self.env = env
        self.gamma = gamma
        # self.probabilities = [1.0, 0.0, 0.0]
        self.probabilities = [
            1 / self.env.action_space.n for _ in range(self.env.action_space.n)
        ]
        self.policy = np.zeros(self.env.state_space.n, dtype=int)
        self.value_function = np.zeros(self.env.state_space.n)

    def select_action(self, state):
        return np.random.choice(self.env.action_space.n, p=self.probabilities)

    def evaluate_policy(self, threshold=0.01, max_iterations=1000):
        V = np.zeros(self.env.state_space.n)
        for _ in range(max_iterations):
            delta = 0
            for state in range(self.env.state_space.n):
                v = V[state]
                V[state] = self.calculate_state_value(state, V)
                delta = max(delta, abs(v - V[state]))
            if delta < threshold:
                break
        return V

    def calculate_state_value(self, state, V):
        if state in self.env.terminal_states:
            return 0
        value = 0
        for action in range(self.env.action_space.n):
            next_state, reward = self.env.get_next_state_and_reward(state, action)
            value += self.probabilities[action] * (reward + self.gamma * V[next_state])
        return value

    def value_iteration(self, threshold=0.01, max_iterations=1000):
        for _ in range(max_iterations):
            delta = 0
            for state in range(self.env.state_space.n):
                v = self.value_function[state]
                self.value_function[state] = self.calculate_max_value(state)
                delta = max(delta, abs(v - self.value_function[state]))
            if delta < threshold:
                break
        self.update_policy()

    def calculate_max_value(self, state):
        max_value = float("-inf")
        for action in range(self.env.action_space.n):
            next_state, reward = self.env.get_next_state_and_reward(state, action)
            value = reward + self.gamma * self.value_function[next_state]
            max_value = max(max_value, value)
        return max_value

    def update_policy(self):
        for state in range(self.env.state_space.n):
            max_value = float("-inf")
            best_action = None
            for action in range(self.env.action_space.n):
                next_state, reward = self.env.get_next_state_and_reward(state, action)
                value = reward + self.gamma * self.value_function[next_state]
                if value > max_value:
                    max_value = value
                    best_action = action
            self.policy[state] = best_action
