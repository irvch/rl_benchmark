import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG, DQN
from environments.base_env import create_env

# Mapping for available models
ALGO_MAP = {
    "ppo": PPO,
    "ddpg": DDPG,
    "dqn": DQN,
}

def backtest(domain, algorithm, config, model_path):
    """
    Runs a backtest using a trained RL model on the given domain.
    """
    # Create environment
    env = create_env(domain, config)

    # Load trained model
    if algorithm not in ALGO_MAP:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model = ALGO_MAP[algorithm].load(model_path)

    # Run backtest
    obs = env.reset()
    portfolio_values = [env.portfolio_value]

    for _ in range(config["time_steps"]):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        portfolio_values.append(env.portfolio_value)
        if done:
            break

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_values, label=f"{algorithm.upper()} Portfolio Value")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value")
    plt.title(f"Backtest Results for {algorithm.upper()} on {domain}")
    plt.legend()
    plt.show()

    return portfolio_values

if __name__ == "__main__":
    # Example backtest
    config = {
        "n_assets": 5,
        "window_size": 10,
        "initial_balance": 100000,
        "transaction_cost": 0.001,
        "time_steps": 1000
    }
    backtest("{domain}", "{algorithm}", config, "models/{algorithm}_{domain}.zip")