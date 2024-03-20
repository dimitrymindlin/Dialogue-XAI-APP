# Import necessary libraries
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data import load_config, create_folder_if_not_exists
from data.ml_utilities import label_encode_and_save_classes, construct_pipeline, train_model

DATASET_NAME = "adult"
config_path = f"./{DATASET_NAME}_model_config.json"
save_path = f"./{DATASET_NAME}"


def map_education_levels(data):
    new_bins = {
        0: ('Primary Education', [1, 2, 3, 4, 5]),
        1: ('Secondary Education', [6, 7, 8, 9]),
        2: ('Postsecondary Education', [10, 11, 12, 13]),
        3: ('Graduate Education', [14, 15, 16])
    }

    def transform_num(edu_num):
        for rank, (bin_name, levels) in new_bins.items():
            if edu_num in levels:
                return rank
        return -1

    data['Education.num'] = data['Education.num'].apply(transform_num)


def standardize_column_names(data):
    data.columns = [col.capitalize() for col in data.columns]


def add_work_life_balance(data):
    categories = ['Poor', 'Fair', 'Good', 'Excellent']
    # Assign random categories for demonstration
    data['WorkLifeBalance'] = np.random.choice(categories, size=len(data))
    # Create a mapping from string values to integers
    mapping = {category: i for i, category in enumerate(categories)}
    # Apply the mapping to the 'WorkLifeBalance' column
    data['WorkLifeBalance'] = data['WorkLifeBalance'].map(mapping)
    return mapping


def clean_data(data):
    data[data == '?'] = np.nan
    for col in ['Workclass', 'Native.country', 'Age']:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Check outliars in work hours (working with std dev)
    mean = data['Hours.per.week'].mean()
    std = data['Hours.per.week'].std()
    data_cleaned = data[(data['Hours.per.week'] > mean - 2 * std) & (data['Hours.per.week'] < mean + 2 * std)]
    print(f"Removed {len(data) - len(data_cleaned)} outliars from the data")
    return data_cleaned


def map_ordinal_columns(data, config):
    for col, mapping in config["ordinal_mapping"].items():
        if col == "WorkLifeBalance":
            continue
        data[col] = data[col].map(mapping)


def bin_age_column(data):
    bins = [17, 25, 40, 60, 100]
    labels = ["Young (18-25)", "Adult (26-40)", "Middle Aged (41-60)", "Senior (61-100)"]
    data["Age"] = pd.cut(data["Age"], bins=bins, labels=labels)


def bin_workclass_col(data):
    # Define the mapping from original labels to new categories
    workclass_map = {
        'Private': 'Private Sector',
        'State-gov': 'Government Employment',
        'Federal-gov': 'Government Employment',
        'Self-emp-not-inc': 'Self-Employed / Entrepreneurial',
        'Self-emp-inc': 'Private Sector',
        'Local-gov': 'Government Employment',
        'Without-pay': 'Unemployed / Other',
        'Never-worked': 'Unemployed / Other'
    }

    # Apply the mapping to the "Workclass" column
    data["Workclass"] = data["Workclass"].map(workclass_map)

    return data


def map_occupation_col(data):
    # Mapping of professions to their respective subgroups
    profession_to_subgroup = {
        'Exec-managerial': 'White-Collar Professions',
        'Prof-specialty': 'White-Collar Professions',
        'Adm-clerical': 'White-Collar Professions',
        'Tech-support': 'White-Collar Professions',
        'Sales': 'White-Collar Professions',
        'Machine-op-inspct': 'Blue-Collar Professions',
        'Craft-repair': 'Blue-Collar Professions',
        'Transport-moving': 'Blue-Collar Professions',
        'Handlers-cleaners': 'Blue-Collar Professions',
        'Farming-fishing': 'Blue-Collar Professions',
        'Other-service': 'Service Industry Professions',
        'Protective-serv': 'Service Industry Professions',
        'Priv-house-serv': 'Service Industry Professions',
        'Armed-Forces': 'Specialized and Miscellaneous',
        '?': 'Specialized and Miscellaneous'
    }

    # Apply mapping to the profession column
    data['Occupation'] = data['Occupation'].map(profession_to_subgroup)
    return data


def preprocess_marital_status(data):
    data["Marital.status"] = data["Marital.status"].replace(['Never-married', 'Divorced', 'Separated', 'Widowed'],
                                                            'Single')
    data["Marital.status"] = data["Marital.status"].replace(
        ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')


def preprocess_data_specific(data, config):
    standardize_column_names(data)
    # Following functions assume capitalized col names
    config["ordinal_mapping"]['WorkLifeBalance'] = add_work_life_balance(data)
    map_ordinal_columns(data, config)
    data = clean_data(data)

    map_education_levels(data)
    preprocess_marital_status(data)
    bin_age_column(data)
    bin_workclass_col(data)
    map_occupation_col(data)

    target_col = config["target_col"]
    X = data.drop(columns=[target_col])
    y = data[target_col]

    if "drop_columns" in config:
        X.drop(columns=config["drop_columns"], inplace=True)

    if "rename_columns" in config:
        X = X.rename(columns=config["rename_columns"])

    if "columns_to_encode" in config:
        X, encoded_classes = label_encode_and_save_classes(X, config)
    else:
        encoded_classes = {}

    return X, y, encoded_classes


def main():
    config = load_config(config_path)
    config["save_path"] = save_path

    data = pd.read_csv(config["dataset_path"])
    create_folder_if_not_exists(save_path)
    X, y, encoded_classes = preprocess_data_specific(data, config)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    target_col = config["target_col"]
    # Add labels to X and save train and test data
    X_train[target_col] = y_train
    X_test[target_col] = y_test
    X_train.to_csv(os.path.join(save_path, f"{DATASET_NAME}_train.csv"))
    X_test.to_csv(os.path.join(save_path, f"{DATASET_NAME}_test.csv"))
    X_train.drop(columns=[target_col], inplace=True)
    X_test.drop(columns=[target_col], inplace=True)

    # Check if there are nan values in the data and raise an error if there are
    if X_train.isna().sum().sum() > 0:
        raise ValueError("There are NaN values in the training data.")

    # Copy the config file to the save path
    with open(os.path.join(save_path, f"{DATASET_NAME}_model_config.json"), 'w') as file:
        json.dump(config, file)

    columns_to_encode = config["columns_to_encode"]
    # Change list of column names to be encoded to a list of column indices
    columns_to_encode = [X_train.columns.get_loc(col) for col in columns_to_encode]
    pipeline = construct_pipeline(columns_to_encode, RandomForestClassifier())

    model_params = {("model__" + key if not key.startswith("model__") else key): value for key, value in
                    config["model_params"].items()}
    search_params = config.get("random_search_params", {"n_iter": 10, "cv": 5, "random_state": 42})

    best_model, best_params = train_model(X_train, y_train, pipeline, model_params, search_params)

    # Print evaluation metrics on train and test
    print("Best Model Score Train:", best_model.score(X_train, y_train))
    print("Best Model Score Test:", best_model.score(X_test, y_test))
    print("Best Parameters:", best_params)

    # Save the best model
    pickle.dump(best_model, open(os.path.join(save_path, f"{DATASET_NAME}_model_rf.pkl"), 'wb'))


if __name__ == "__main__":
    main()
