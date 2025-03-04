import numpy as np
from environments.base_env import FinancialRLBaseEnv
from gym import spaces

class PortfolioManagementEnv(FinancialRLBaseEnv):
    """
    OpenAI Gym-compatible environment for portfolio management using reinforcement learning.
    The agent learns to allocate funds across multiple assets to maximize returns.
    """

    def __init__(self, config):
        # Extract config parameters
        self.n_assets = config.get("n_assets", 5)
        self.window_size = config.get("window_size", 10)
        self.time_steps = config.get("time_steps", 1000)

        self.initial_balance = config.get("initial_balance", 100000)
        self.transaction_cost = config.get("transaction_cost", 0.001)
        self.max_drawdown = config.get("max_drawdown", 0.3)

        self.price_data = self._generate_synthetic_data()

        super().__init__(config)

        # Define observation space (historical prices + portfolio state)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_assets * self.window_size + self.n_assets,), dtype=np.float32
        )

        # Define action space (continuous portfolio allocation)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        # Reset environment
        self.reset()

    def _generate_synthetic_data(self):
        """
        Generates synthetic price data for assets. This can be replaced with real historical data.
        """
        np.random.seed(42)
        prices = np.cumprod(1 + np.random.normal(0, 0.01, size=(self.time_steps, self.n_assets)), axis=0)
        return prices

    def _get_observation(self):
        """
        Returns the current observation: historical price movements and portfolio state.
        """
        price_history = self.price_data[self.current_step - self.window_size : self.current_step].flatten()
        return np.concatenate([price_history, self.portfolio_allocation])

    def step(self, action):
        """
        Executes a portfolio rebalance and computes the reward.
        """
        # Normalize actions (sum to 1 for valid portfolio weights)
        action = np.clip(action, 0, 1)
        action /= np.sum(action) + 1e-8  # Ensure sum is 1

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
        done = self.current_step >= self.time_steps - 1 or (self.portfolio_value < self.initial_balance * (1 - self.max_drawdown))

        # Compute reward (log return)
        reward = np.log(1 + net_return)

        # Move to the next time step
        self.current_step += 1

        return self._get_observation(), reward, done, {}

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.portfolio_allocation = np.ones(self.n_assets) / self.n_assets  # Equal-weighted allocation
        return self._get_observation()

    def render(self, mode="human"):
        """
        Displays the current portfolio state.
        """
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}")
