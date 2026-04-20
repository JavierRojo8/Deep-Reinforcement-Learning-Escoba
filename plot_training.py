import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_monitor_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Monitor CSV has first line as json metadata
    df = pd.read_csv(path, skiprows=1)
    return df

train_path = Path("logs/train_monitor.csv")
eval_path = Path("logs/evaluations.npz")  # created by EvalCallback

if train_path.exists():
    df = load_monitor_csv(train_path)

    df["episode"] = range(1, len(df) + 1)
    df["reward_ma20"] = df["r"].rolling(20, min_periods=1).mean()
    df["length_ma20"] = df["l"].rolling(20, min_periods=1).mean()

    plt.figure(figsize=(8, 5))
    plt.plot(df["episode"], df["r"], alpha=0.4, label="Episode reward")
    plt.plot(df["episode"], df["reward_ma20"], label="Reward MA20")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training reward")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(df["episode"], df["l"], alpha=0.4, label="Episode length")
    plt.plot(df["episode"], df["length_ma20"], label="Length MA20")
    plt.xlabel("Episode")
    plt.ylabel("Episode length")
    plt.title("Training episode length")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("train_monitor.csv not found")