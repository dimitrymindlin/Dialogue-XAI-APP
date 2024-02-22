import json

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml_utilities import label_encode_and_save_classes, construct_pipeline, train_model
import os

config_path = "./data/sf_model_config.json"
save_path = "./data/sf_crime"

# create folder if it does not exist
if not os.path.exists(save_path):
    os.makedirs(save_path)


# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)


# Updated dataset-specific preprocessing function
def preprocess_data_specific(data, config):
    columns_to_encode = config["columns_to_encode"]
    ordinal_mapping = config["ordinal_mapping"]["Crimes"]

    data.drop(columns=["Dates"], inplace=True)
    rename_col_dict = config.get("rename_columns", {})
    if rename_col_dict:
        data = data.rename(columns=rename_col_dict)

    bins = [-np.inf, 33, 48, 65, np.inf]
    data["Crimes"] = pd.cut(data["Crimes"], bins=bins, labels=["low", "medium", "high", "very high"])
    data["Crimes"] = data["Crimes"].map(ordinal_mapping)

    data, encoded_classes = label_encode_and_save_classes(data, columns_to_encode, save_path)
    X = data.drop(columns=["Crimes", "Year"])
    y = data["Crimes"]

    return X, y, encoded_classes


def main():
    config = load_config(config_path)

    data = pd.read_csv(config["dataset_path"])
    X, y, encoded_classes = preprocess_data_specific(data, config)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add labels to X and save train and test data
    X_train["Crimes"] = y_train
    X_test["Crimes"] = y_test
    X_train.to_csv(os.path.join(save_path, "sf_crime_train.csv"))
    X_test.to_csv(os.path.join(save_path, "sf_crime_test.csv"))
    X_train.drop(columns=["Crimes"], inplace=True)
    X_test.drop(columns=["Crimes"], inplace=True)

    # Copy the config file to the save path
    with open(os.path.join(save_path, "sf_model_config.json"), 'w') as file:
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
    print("Best Model Score:", best_model.score(X_train, y_train))
    print("Best Model Score:", best_model.score(X_test, y_test))
    print("Best Parameters:", best_params)

    # Save the best model
    pickle.dump(best_model, open(os.path.join(save_path, "sf_crime_model_rf.pkl"), 'wb'))


if __name__ == "__main__":
    main()
