import numpy as np
from environments.base_env import FinancialRLBaseEnv
from gymnasium import spaces

class PortfolioManagementEnv(FinancialRLBaseEnv):
    """
    OpenAI Gym-compatible environment for portfolio management using reinforcement learning.
    The agent learns to allocate funds across multiple assets to maximize returns.
    """

    def __init__(self, config):
        # Extract config parameters
        self.n_assets = config.get("n_assets", 5)
        self.window_size = config.get("window_size", 10)
        self.timesteps = config.get("timesteps", 10000)
        self.initial_balance = config.get("initial_balance", 100000)
        self.transaction_cost = config.get("transaction_cost", 0.001)
        self.max_drawdown = config.get("max_drawdown", 0.3)
        self.algorithm = config.get("algorithm", "ppo")  # Default to PPO

        self.price_data = self._generate_synthetic_data()

        super().__init__(config)

        # Define observation space (historical prices + portfolio state)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_assets * self.window_size + self.n_assets,), dtype=np.float32
        )

        # Define action space dynamically based on algorithm
        if self.algorithm == "dqn":
            self.action_space = spaces.Discrete(self.n_assets)  # DQN requires discrete actions
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)  # PPO/DDPG

        # Reset environment
        self.reset()

    def _generate_synthetic_data(self):
        """ Generates synthetic price data for assets. """
        np.random.seed(42)
        return np.cumprod(1 + np.random.normal(0, 0.01, size=(self.timesteps, self.n_assets)), axis=0)

    def _get_observation(self):
        """ Returns the current observation: historical price movements and portfolio state. """
        price_history = self.price_data[self.current_step - self.window_size : self.current_step].flatten()
        return np.concatenate([price_history, self.portfolio_allocation])

    def step(self, action):
        """ Executes a portfolio rebalance and computes the reward. """

        if self.algorithm == "dqn":
            # Convert discrete action into one-hot vector for portfolio allocation
            action_vector = np.zeros(self.n_assets)
            action_vector[action] = 1.0
            action = action_vector
        else:
            # Normalize actions for PPO/DDPG
            action = np.clip(action, -1, 1)
            action = (action + 1) / 2  # Convert from [-1,1] to [0,1]
            action /= np.sum(action) + 1e-8  

        # Calculate portfolio returns
        returns = self.price_data[self.current_step] / self.price_data[self.current_step - 1] - 1
        portfolio_return = np.dot(action, returns)

        # Apply transaction cost
        cost = np.sum(np.abs(action - self.portfolio_allocation)) * self.transaction_cost
        net_return = portfolio_return - cost

        # Update portfolio value
        self.portfolio_value *= (1 + net_return)

        # Store new allocation
        self.portfolio_allocation = action

        # Check termination conditions
        terminated = self.current_step >= self.timesteps - 1 
        truncated = self.portfolio_value < self.initial_balance * (1 - self.max_drawdown)

        # Compute reward (log return)
        reward = np.log(1 + net_return)

        # Move to the next time step
        self.current_step += 1

        return self._get_observation(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """ Resets the environment to the initial state. """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.portfolio_allocation = np.ones(self.n_assets) / self.n_assets  # Equal-weighted allocation
        return self._get_observation(), {}

    def render(self, mode="human"):
        """ Displays the current portfolio state. """
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}")