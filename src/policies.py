import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


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


class QLearningAgent:
    """
    Q-learning agent for environments with discrete state and action spaces.
    """

    def __init__(self, env, gamma=0.95, alpha=0.1, epsilon=1.0, epsilon_decay=0.995):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((self.env.state_space.n, self.env.action_space.n))
        self.reward_history = []

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def train(self, episodes, max_steps_per_episode):
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            for step in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                if done:
                    break
                total_reward += reward
            self.reward_history.append(total_reward)
            self.epsilon *= self.epsilon_decay

    def compute_optimal_q_function(self, threshold=0.01, max_iterations=1000):
        new_q_table = np.copy(self.q_table)
        for _ in range(max_iterations):
            delta = 0
            for state in range(self.env.state_space.n):
                for action in range(self.env.action_space.n):
                    # Get the next state and reward for this state-action pair
                    next_state, reward = self.env.get_next_state_and_reward(
                        state, action
                    )
                    # Bellman Optimality Update
                    best_future_q = np.max(self.q_table[next_state])
                    updated_q_value = reward + self.gamma * best_future_q
                    delta = max(
                        delta, np.abs(updated_q_value - new_q_table[state, action])
                    )
                    new_q_table[state, action] = updated_q_value

            if delta < threshold:
                break
            self.q_table = new_q_table

        print("Optimal Q-function computed.")

    def plot_reward_history(self):
        plt.plot(self.reward_history)
        plt.title("Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()

    def plot_epsilon(self, episodes):
        epsilons = self.epsilon * np.power(self.epsilon_decay, np.arange(episodes))
        plt.plot(epsilons)
        plt.title("Epsilon Decay Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.show()

    def plot_q_table_heatmap(self):
        # Initialize a grid with NaN for impassable areas
        grid_shape = (8, 8)
        q_value_grids = np.full((4, *grid_shape), np.nan)

        # Fill in the Q-values for passable areas
        for state in range(self.env.state_space.n):
            x, y = self.env._state_to_grid[state]
            for action in range(self.env.action_space.n):
                q_value_grids[action, y, x] = self.q_table[state, action]

        # Plotting the heatmaps
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        actions = ["Up", "Right", "Down", "Left"]

        for i, ax in enumerate(axs.flat):
            heatmap = ax.imshow(q_value_grids[i], cmap="hot", interpolation="nearest")
            ax.set_title(f"Action: {actions[i]}")
            fig.colorbar(heatmap, ax=ax)

        plt.tight_layout()
        plt.show()

    def plot_best_action_q_values(self):
        best_actions = np.argmax(self.q_table, axis=1)
        best_q_values = np.max(self.q_table, axis=1)

        best_q_value_grid = np.full((8, 8), np.nan)
        action_symbols = ["↑", "→", "↓", "←"]  # Flipped up and down

        for state in range(self.env.state_space.n):
            x, y = self.env._state_to_grid[state]
            best_q_value_grid[7 - y, x] = best_q_values[state]  # Flipping the y-axis

        plt.imshow(best_q_value_grid, cmap="Spectral", interpolation="nearest")
        for y in range(8):
            for x in range(8):
                if not np.ian(best_q_value_grid[y, x]):
                    plt.text(
                        x,
                        y,
                        action_symbols[
                            best_actions[self.env._grid_to_state[(x, 7 - y)]]
                        ],
                        ha="center",
                        va="center",
                        color="black",
                    )  # Adjust for flipped y-axis
        plt.colorbar()
        plt.title("Best Q-value and Corresponding Action per State")
        plt.show()

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def simulate_with_policy(self, policy):
        state = self.env.reset()
        self.env.render()

        done = False
        while not done:
            action = policy[state]
            state, _, done, _ = self.env.step(action)
            self.env.render()
