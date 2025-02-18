import os
import pandas as pd
import numpy as np

def fetch_choices_and_rewards(data_folder, minimum_trials):
    # List files in the data folder
    files = os.listdir(data_folder)
    
    # Load choices and rewards data
    choices = np.loadtxt(data_folder + 'choices.csv', delimiter=',')
    rewards = np.loadtxt(data_folder + 'rewards.csv', delimiter=',')
    
    # Remove the nan values to make a staggered array
    cleaned_choices = []
    cleaned_rewards = []
    for i in range(len(choices)):
        c = choices[i][~np.isnan(choices[i])]
        r = rewards[i][~np.isnan(rewards[i])]
        if len(c) < minimum_trials:
            continue
        cleaned_choices.append(c)
        cleaned_rewards.append(r)
    
    cleaned_choices = np.array(cleaned_choices, dtype=object)
    cleaned_rewards = np.array(cleaned_rewards, dtype=object)
    
    return cleaned_choices, cleaned_rewards