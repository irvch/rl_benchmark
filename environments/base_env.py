import gymnasium as gym
import numpy as np
from gym import spaces

class FinancialRLBaseEnv(gym.Env):
    """
    Base class for financial reinforcement learning environments.
    Implements OpenAI Gym interface for compatibility with RL algorithms.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(FinancialRLBaseEnv, self).__init__()

        # Extract parameters from config
        self.config = config
        self.time_steps = config.get("time_steps", 1000)
        self.initial_balance = config.get("initial_balance", 10000)
        self.transaction_cost = config.get("transaction_cost", 0.001)

        # Define observation space (example: prices + portfolio state)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(config["obs_dim"],), dtype=np.float32
        )

        # Define action space (example: continuous action for portfolio weights)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(config["action_dim"],), dtype=np.float32
        )

        # Initialize environment state
        self.reset()

    def step(self, action):
        """
        Executes one step in the environment.
        """
        raise NotImplementedError("Subclasses must implement step()")

    def reset(self):
        """
        Resets the environment to an initial state.
        """
        raise NotImplementedError("Subclasses must implement reset()")

    def render(self, mode='human'):
        """
        Render the current environment state.
        """
        pass

    def close(self):
        """
        Cleanup resources if needed.
        """
        pass


# Factory function to create the desired environment dynamically
def create_env(domain, config):
    """
    Creates an instance of the appropriate financial RL environment.

    Parameters:
        domain (str): The financial domain ('portfolio_management', 'algorithmic_trading', etc.)
        config (dict): Environment configuration.

    Returns:
        gym.Env: An instance of the chosen environment.
    """
    if domain == "portfolio_management":
        from environments.portfolio_management import PortfolioManagementEnv
        return PortfolioManagementEnv(config)
    elif domain == "algorithmic_trading":
        from environments.algorithmic_trading import AlgorithmicTradingEnv
        return AlgorithmicTradingEnv(config)
    elif domain == "options_pricing":
        from environments.options_pricing import OptionsPricingEnv
        return OptionsPricingEnv(config)
    elif domain == "market_making":
        from environments.market_making import MarketMakingEnv
        return MarketMakingEnv(config)
    else:
        raise ValueError(f"Unknown domain: {domain}")