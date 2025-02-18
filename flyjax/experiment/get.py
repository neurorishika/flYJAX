import os
import pandas as pd
import numpy as np

def fetch_behavioral_data(data_folder, minimum_trials, remove_control=False, max_per_experiment=None):
    # List files in the data folder
    files = os.listdir(data_folder)

    # verify that the data folder contains the required files
    if 'choices.csv' not in files or 'rewards.csv' not in files and 'metadata.csv' not in files:
        raise FileNotFoundError('The data folder is missing important files')
    
    # Load choices and rewards data
    choices = np.loadtxt(data_folder + 'choices.csv', delimiter=',')
    rewards = np.loadtxt(data_folder + 'rewards.csv', delimiter=',')
    metadata = pd.read_csv(data_folder + 'metadata.csv')
    
    # Remove the nan values to make a staggered array
    cleaned_choices = []
    cleaned_rewards = []
    cleaned_metadata = pd.DataFrame()

    for i in range(len(choices)):
        c = choices[i][~np.isnan(choices[i])]
        r = rewards[i][~np.isnan(rewards[i])]
        if len(c) < minimum_trials:
            continue
        cleaned_choices.append(c)
        cleaned_rewards.append(r)
        cleaned_metadata = pd.concat([cleaned_metadata, metadata.iloc[i]], axis=1)
    
    cleaned_choices = np.array(cleaned_choices, dtype=object)
    cleaned_rewards = np.array(cleaned_rewards, dtype=object)
    cleaned_metadata = cleaned_metadata.T.reset_index(drop=True)

    # sort dataset by experiment start time
    cleaned_metadata['Experiment Start Time'] = pd.to_datetime(cleaned_metadata['Experiment Start Time'], format='%Y-%m-%d %H:%M:%S')
    cleaned_metadata['Starvation Time'] = pd.to_datetime(cleaned_metadata['Starvation Time'], format='%Y-%m-%d %H:%M:%S')

    order = cleaned_metadata.sort_values(by='Experiment Start Time').index
    cleaned_choices = cleaned_choices[order]
    cleaned_rewards = cleaned_rewards[order]
    cleaned_metadata = cleaned_metadata.iloc[order].reset_index(drop=True)

    if remove_control:
        cleaned_choices = cleaned_choices[cleaned_metadata['Fly Experiment'].str.contains('control') == False]
        cleaned_rewards = cleaned_rewards[cleaned_metadata['Fly Experiment'].str.contains('control') == False]
        cleaned_metadata = cleaned_metadata[cleaned_metadata['Fly Experiment'].str.contains('control') == False].reset_index(drop=True)
    
    if max_per_experiment is not None:
        temp_choices = []
        temp_rewards = []
        temp_metadata = pd.DataFrame()
        # for each experiment, keep only the first max_per_experiment trials
        unique_experiments = cleaned_metadata['Fly Experiment'].unique()
        # sort the experiments by name
        unique_experiments = sorted(unique_experiments, key=lambda x: int(x.replace('_reciprocal', '').split('.')[0][3:]))
        for exp in unique_experiments:
            idxs = list(cleaned_metadata[cleaned_metadata['Fly Experiment'] == exp].index)
            for i in range(min(max_per_experiment, len(idxs))):
                temp_choices.append(cleaned_choices[idxs[i]])
                temp_rewards.append(cleaned_rewards[idxs[i]])
                temp_metadata = pd.concat([temp_metadata, cleaned_metadata.iloc[idxs[i]]], axis=1)

        cleaned_choices = np.array(temp_choices, dtype=object)
        cleaned_rewards = np.array(temp_rewards, dtype=object)
        cleaned_metadata = temp_metadata.T.reset_index(drop=True)

    return cleaned_choices, cleaned_rewards, cleaned_metadata

def get_experiments(data_folder):
    # List files in the data folder
    files = os.listdir(data_folder)
    
    # Make sure the experiments folder exists
    if 'experiments' not in files:
        raise FileNotFoundError('The data folder is missing the experiments folder')
    
    # get the files inside the experiments folder
    experiments_folder = data_folder + 'experiments/'
    experiments = os.listdir(experiments_folder)

    # keep only csv files
    experiments = [f for f in experiments if f.endswith('.csv')]

    # load the experiments in a dictionary
    experiments_dict = {}
    for exp in experiments:
        experiments_dict[exp.split('.')[0]] = pd.read_csv(experiments_folder + exp)

    return experiments_dict
