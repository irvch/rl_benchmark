from stable_baselines3 import PPO
from environments.base_env import create_env

# Load environment (same config as during training)
config = {
    "n_assets": 5,
    "obs_dim": 10,
    "action_dim": 3,
    "window_size": 10,
    "initial_balance": 100000,
    "transaction_cost": 0.001,
    "max_drawdown": 0.3    
}
env = create_env("portfolio_management", config)

# Load the trained model
model = PPO.load("models\ppo_portfolio_management.zip")

# Test the model
obs = env.reset()
for _ in range(10):  # Run for 10 steps
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        break