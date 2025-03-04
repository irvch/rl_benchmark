import matplotlib.pyplot as plt

def plot_portfolio_performance(portfolio_values, title="Portfolio Performance"):
    """
    Plots portfolio value over time.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_values, label="Portfolio Value")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_multiple_backtests(results, labels):
    """
    Plots multiple backtest results for comparison.
    """
    plt.figure(figsize=(12, 6))
    for portfolio_values, label in zip(results, labels):
        plt.plot(portfolio_values, label=label)
    
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value")
    plt.title("Comparison of Backtest Results")
    plt.legend()
    plt.grid()
    plt.show()