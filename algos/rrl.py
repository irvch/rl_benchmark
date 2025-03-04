import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm.notebook import tqdm, trange
from helpers import read_file, save_results, read_results
from metrics import get_metrics, add_columns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print(device)

results_path = "./results"

# Define the RRL Model
class RRLModel(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.neuron = nn.Linear(m + 1, 1, bias=True)
        nn.init.xavier_uniform_(self.neuron.weight)
        nn.init.constant_(self.neuron.bias, 0)

    def forward(self, features):
        return torch.tanh(self.neuron(features))

class RRLTrainer:
    def __init__(self, model: RRLModel, lr: float, delta: float, T: int, lamb: float, early_stop: bool):
        self.model = model
        self.lr = lr
        self.delta = delta
        self.T = T
        self.lamb = lamb
        self.early_stop = early_stop
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lamb)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def sharpe_ratio(self, returns: torch.Tensor, bias: int = 1e-6):
        mean = torch.mean(returns, dim=-1)
        std = torch.std(returns, dim=-1, correction=0) + bias
        return mean / std

    def reward_function(self, rt: torch.Tensor, Ft: torch.Tensor):
        returns = Ft[:-1] * rt - self.delta * (torch.abs(Ft[1:] - Ft[:-1]))
        sharpe = self.sharpe_ratio(returns)
        return returns, sharpe

    def predict_Ft(self, X: torch.Tensor, prev: torch.Tensor):
        Ft1_values = torch.zeros(X.shape[0] + 1).to(self.device)
        Ft1_values[0] = prev
        for i in range(len(X) - 1):
            x_i = torch.cat([X[i], Ft1_values[i].detach().unsqueeze(0)], 0).float()
            Ft1_values[i + 1] = self.model(x_i)

        Ft1_adjusted = Ft1_values.clone()
        Ft1_adjusted[Ft1_adjusted == 0] = -1
        return torch.sign(Ft1_adjusted)

    def gradient_ascent(self, X: torch.Tensor, rt: torch.Tensor):
        batches = int(X.shape[0] / self.T)
        self.last_reward = None
        consecutive_no_gain = 0

        for epoch in trange(self.epochs, unit="iteration"):
            prevFt = torch.tensor([1]).float()
            for batch in range(batches):
                index = batch * self.T
                self.optimizer.zero_grad()
                Ft1 = self.predict_Ft(X[index: index + self.T], prevFt)
                returns, reward = self.reward_function(rt[index: index + self.T], Ft1.clone())
                (-1 * reward).backward(retain_graph=True)
                self.optimizer.step()
                prevFt = Ft1[-1].clone().unsqueeze(0)

            if self.early_stop:
                rewards_, _, _ = self.test(X, rt)
                if self.last_reward is not None and rewards_ - self.last_reward <= 0:
                    consecutive_no_gain += 1
                    if consecutive_no_gain >= 2:
                        print(f"Early stopping activated at epoch {epoch + 1}.")
                        return
                else:
                    consecutive_no_gain = 0
                self.last_reward = rewards_

    def train(self, X: torch.Tensor, rt: torch.Tensor, epochs: int = 100):
        self.epochs = epochs
        self.gradient_ascent(X, rt)

    def test(self, X_test: torch.Tensor, rt_test: torch.Tensor):
        prevFt = torch.tensor([1]).float()
        test_Ft = self.predict_Ft(X_test, prevFt)
        test_returns, test_rewards = self.reward_function(rt_test, test_Ft)
        return test_rewards, test_returns, test_Ft[1:]

# Rolling Training and Testing (Using Only Log Returns)
def rolling_training_testing(df, returns, df_name: str, model_name: str, delta, N_train=500, N_test=500, T=5, lr=0.1, epochs=10, lamb=0.01, early_stop=True):
    X = torch.tensor(df.values).float().to(device)
    rt = torch.tensor(returns.values).float().to(device)

    model = RRLModel(X.shape[-1]).float().to(device)
    torch.autograd.set_detect_anomaly(True)
    trainer = RRLTrainer(model, lr, delta, T, lamb, early_stop)

    rewards_list = []
    returns_list = []
    ft_list = []

    for i in range(10):
        start = i * N_train
        print(f"Training on Batch {i + 1}")
        X_train = X[start:start + N_train]
        rt_train = rt[start:start + N_train]
        trainer.train(X_train, rt_train, epochs)

        print(f"Testing on Batch {i + 2}: ", end="")
        X_test = X[start + N_train:start + N_train + N_test]
        rt_test = rt[start + N_train:start + N_train + N_test]
        test_rewards, test_returns, test_ft = trainer.test(X_test, rt_test)
        print(f"Returns= {torch.sum(test_returns):.2f}")

        rewards_list.append(test_rewards.detach().cpu().numpy().tolist())
        returns_list.extend(test_returns.detach().cpu().numpy().tolist())
        ft_list.extend(test_ft.detach().cpu().numpy().tolist())
        print()

    path = f'./model/{df_name}_{model_name}_eStop_{early_stop}_RRL.pth'
    torch.save(model.state_dict(), path)
    print(f'Successfully saved {model_name} RRL model for {df_name} with early stop {early_stop}')

    return rewards_list, returns_list, ft_list

# Train the model (Using Log Returns Instead of TIs)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.1
lamb = 0.01  # L2 regularization parameter
N_train = 500
N_test = 500
epochs = 10
seed = 42
T = 5
early_stop_flag = False
df_names = ['Corn']
model_names = ["LogReturns"]
deltas = [0.001]

# Benchmarking
metrics_df = pd.DataFrame(columns=["Dataset", "Model", "Metrics", "Values"])

for i, df_name in enumerate(df_names):
    df = read_file(f"data/{df_name}_20yr_OHLC.csv")

    # Compute Log Returns Instead of Technical Indicators
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df[['Log_Returns']].dropna()
    returns = df['Log_Returns']

    for model_name in model_names:
        print(f"Running {model_name} for {df_name} for T={T}, Epochs={epochs}, delta={deltas[i]}")
        torch.manual_seed(seed)
        test_rewards, test_returns, test_ft = rolling_training_testing(df, returns, df_name, model_name, delta=deltas[i], epochs=epochs, lamb=lamb, T=T, early_stop=early_stop_flag)
        save_results(test_returns, test_rewards, test_ft, df_name, model_name, T, epochs, "xavier-uniform", early_stop_flag)

        metrics = get_metrics(test_returns, df['Log_Returns'].iloc[500])
        metrics = add_columns(metrics, df_name, model_name)
        metrics_df = pd.concat([metrics_df, metrics])

metrics_df.Values = metrics_df.Values.round(2)
metrics_df_show = pd.pivot_table(metrics_df, index=["Dataset", "Metrics"], columns="Model", values="Values", sort=False)
print(metrics_df_show)