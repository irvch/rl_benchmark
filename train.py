import argparse
import yaml
import os
from stable_baselines3 import PPO, DDPG, DQN
from environments.base_env import create_env
from backtest import backtest
from plotting import plot_portfolio_performance

# Algorithm mapping for SB3
ALGO_MAP = {
    "ppo": PPO,
    "ddpg": DDPG,
    "dqn": DQN,
}

def load_config():
    """Load the configuration file."""
    with open("cfgs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "backtest", "plot"],
                        help="Mode: train, backtest, or plot.")
    parser.add_argument("--domain", type=str, required=True, help="Financial domain (e.g., portfolio_management, trading)")
    parser.add_argument("--algorithm", type=str, required=True, choices=ALGO_MAP.keys(), 
                        help="RL algorithm (ppo, ddpg, dqn).")
    return parser.parse_args()

def main():
    # Load configuration
    config = load_config()
    args = parse_args()

    domain = args.domain
    algorithm = args.algorithm
    mode = args.mode

    if domain not in config["domains"]:
        raise ValueError(f"Invalid domain '{domain}'. Choose from {list(config['domains'].keys())}.")
    if algorithm not in config["algorithms"]:
        raise ValueError(f"Invalid algorithm '{algorithm}'. Choose from {list(ALGO_MAP.keys())}.")

    # Get environment and algorithm configs
    env_config = config["domains"][domain]
    env_config["algorithm"] = algorithm
    algo_config = config["algorithms"][algorithm]
    model_path = f"models/{algorithm}_{domain}.zip"

    # Create environment
    env = create_env(domain, env_config)

    if mode == "train":
        # Train a new model
        AlgoClass = ALGO_MAP[algorithm]
        policy = algo_config.pop("policy")

        print(f"🔨 Training model for {domain} using {algorithm.upper()}...")
        model = AlgoClass(
            policy, 
            env, 
            **algo_config, 
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )
        model.learn(total_timesteps=config["defaults"]["timesteps"])

        # Save model
        model.save(model_path)
        print(f"✅ Model saved: {model_path}")

    elif mode == "backtest":
        # Check if model exists before running backtest
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}. Train the model first.")
        
        print(f"⚡ Running Backtest for {domain} using {algorithm.upper()}...")
        portfolio_values = backtest(domain, algorithm, env_config, model_path)

    # elif mode == "plot":
    #     # Check if backtest results exist before plotting
    #     print(f"📈 Plotting Backtest Results for {domain} using {algorithm.upper()}...")
    #     plot_portfolio_performance(portfolio_values, title=f"Backtest Results for {algorithm.upper()} on {domain}")

if __name__ == "__main__":
    main()