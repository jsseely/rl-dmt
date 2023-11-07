import gymnasium as gym
import numpy as np


class BaseAgent:
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


class MonteCarloAgent:
    def __init__(self, env, gamma=0.95):
        self.env = env
        self.gamma = gamma
        self.policy = np.random.choice(self.env.action_space.n, self.env.state_space.n)
        self.V = np.zeros(self.env.state_space.n)
        self.returns = {s: [] for s in range(self.env.state_space.n)}

    def select_action(self, state):
        # Follow the current policy
        return self.policy[state]

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False
        while not done:
            action = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def evaluate_policy(self, num_episodes=1000):
        for _ in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            for t in reversed(range(len(episode))):
                state, _, reward = episode[t]
                G = self.gamma * G + reward
                # Check if the state is visited for the first time in this episode
                if state not in [x[0] for x in episode[:t]]:
                    self.returns[state].append(G)
                    self.V[state] = np.mean(self.returns[state])

    def improve_policy(self):
        for state in range(self.env.state_space.n):
            action_values = []
            for action in range(self.env.action_space.n):
                next_state, reward = self.env.get_next_state_and_reward(state, action)
                action_value = reward + self.gamma * self.V[next_state]
                action_values.append(action_value)
            self.policy[state] = np.argmax(action_values)

    def train(self, num_episodes=1000):
        self.evaluate_policy(num_episodes)
        self.improve_policy()
