# utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history: Dict[str, List], title: str = "Training History"):
    plt.figure(figsize=(12, 4))
    plt.plot(history["loss"], label="Negative Log Likelihood")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_experiment_data(experiment_data):
    choices, rewards = experiment_data
    n_trials = len(choices)
    plt.figure(figsize=(12, 2))
    plt.scatter(np.arange(n_trials)[rewards==1], choices[rewards==1], color='orange', marker='o', s=10)
    plt.scatter(np.arange(n_trials)[rewards==0], choices[rewards==0], color='red', marker='x', s=20)
    window = 10
    rolling_avg = np.convolve(choices, np.ones(window), 'full')[:n_trials] / window
    plt.plot(rolling_avg, color='firebrick', linewidth=2)
    reward_0 = np.convolve(np.logical_and(choices == 0, rewards == 1), np.ones(window), 'full')[:n_trials] / window
    reward_1 = np.convolve(np.logical_and(choices == 1, rewards == 1), np.ones(window), 'full')[:n_trials] / window
    reward_ratio = np.where(reward_0 + reward_1 > 0, reward_1 / (reward_0 + reward_1), 0.5)
    plt.plot(reward_ratio, color='black', linestyle='--')
    plt.xlabel('Trial')
    plt.ylabel('Choice')
    plt.title('Experiment Data')
    plt.show()
