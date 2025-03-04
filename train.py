import argparse
import yaml
from stable_baselines3 import PPO, DDPG, DQN
from environments.base_env import create_env
from backtest import backtest
from plotting import plot_portfolio_performance

ALGO_MAP = {
    "ppo": PPO,
    "ddpg": DDPG,
    "dqn": DQN,
}

def load_config():
    with open("cfgs\config.yaml", "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("args", nargs="*", help="Specify arguments like domain=portfolio algorithm=ppo")
    parsed_args = parser.parse_args()
    user_args = {}
    for arg in parsed_args.args:
        if "=" in arg:
            key, value = arg.split("=")
            user_args[key] = value
    return user_args

def main():
    # Load default config
    config = load_config()
    
    # Parse user inputs
    user_args = parse_args()
    domain = user_args.get("domain")
    algorithm = user_args.get("algorithm")

    if domain not in config["domains"]:
        raise ValueError(f"Invalid domain '{domain}'. Choose from {list(config['domains'].keys())}.")
    if algorithm not in config["algorithms"]:
        raise ValueError(f"Invalid algorithm '{algorithm}'. Choose from {list(ALGO_MAP.keys())}.")

    # Get environment and algorithm configs
    env_config = config["domains"][domain]
    algo_config = config["algorithms"][algorithm]

    # Create environment
    env = create_env(domain, env_config)

    # Select RL algorithm
    AlgoClass = ALGO_MAP[algorithm]

    if not AlgoClass:
        raise NotImplementedError(f"Algorithm '{algorithm}' is not implemented yet.")

    # Extract policy and remove it from algo_config
    policy = algo_config.pop("policy")

    # Train model
    model = AlgoClass(policy, env, **algo_config)
    model.learn(total_timesteps=config["defaults"]["timesteps"])

    # Save model
    model_path = f"models/{algorithm}_{domain}.zip"
    model.save(model_path)
    print(f"✅ Model saved: {model_path}")

    # Run backtest
    print("Running Backtest...")
    portfolio_values = backtest(domain, algorithm, env_config, model_path)
    
    # Plot performance
    print("Plotting Results...")
    plot_portfolio_performance(portfolio_values, title=f"Backtest Results for {algorithm.upper()} on {domain}")

if __name__ == "__main__":
    main()