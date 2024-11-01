### Load pkl files that start with "adult"

import os
import pickle

import pandas as pd


def load_adult_data():
    # Load adult_train.csv dataset
    csv_path = "../data/adult/adult_test.csv"
    try:
        adult_train_data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        adult_train_data = None
    adult_train_data.rename(columns={"Unnamed: 0": "instance_id"}, inplace=True)
    return adult_train_data


def load_adult_pkl_files():
    # Get the list of all files in the current directory
    files_in_directory = os.listdir('.')

    # Filter files that start with "adult" and end with ".pkl"
    adult_pkl_files = [file for file in files_in_directory if file.startswith('adult') and file.endswith('.pkl')]

    # Load each pickle file and store the data_static_interactive in a dictionary
    data_dict = {}
    for file in adult_pkl_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            data_dict[file] = data

    return data_dict


data = load_adult_pkl_files()

# load adult_train data_static_interactive
adult_data = load_adult_data()
instance_ids = data['adult-diverse-instances.pkl']

# Filter adult data_static_interactive by ids
adult_train_data = adult_data[adult_data['instance_id'].isin(instance_ids)]
print(data)
