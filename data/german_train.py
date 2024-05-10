# Import necessary libraries
import json
import os
import pickle

import shap
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data import load_config, create_folder_if_not_exists
from data.ml_utilities import label_encode_and_save_classes, construct_pipeline, train_model

DATASET_NAME = "german"
config_path = f"./{DATASET_NAME}_model_config.json"
save_path = f"./{DATASET_NAME}"


def standardize_column_names(data):
    data.columns = [col.capitalize() for col in data.columns]


def clean_data(data, cols_with_nan_list):
    data[data == '?'] = np.nan
    for col in cols_with_nan_list:
        if col in ['Saving accounts', 'Checking account']:  # Fill missing account values with 0
            data[col].fillna('little', inplace=True)
        else:
            data[col].fillna(data[col].mode()[0], inplace=True)
    return data


def bin_age_column(data):
    column_name = "AgeGroup"
    bins = [18, 25, 40, 60, 90]
    labels = ["Young (18-25 years)", "Adult (26-40 years)", "Middle Aged (41-60 years)", "Senior (61-90 years)"]
    data[column_name] = pd.cut(data[column_name], bins=bins, labels=labels)


def map_ordinal_columns(data, config):
    for col, mapping in config["ordinal_mapping"].items():
        data[col] = data[col].map(mapping)


def map_job_col(config, data):
    """
    Maps numeric job level values to descriptive categories in the 'Job Level' column of a DataFrame.

    Parameters:
    - data: pandas.DataFrame containing a 'Job Level' column with numeric values to be mapped.

    Returns:
    - data: pandas.DataFrame with 'Job Level' values replaced with corresponding string labels.
    """
    col_name = "JobLevel"
    job_level_mapping = config["ordinal_mapping"][col_name]
    # reverse the mapping
    job_level_mapping = {v: k for k, v in job_level_mapping.items()}

    if col_name in data.columns:
        data[col_name] = data[col_name].map(job_level_mapping)
    else:
        print(f"Warning: '{col_name}' column not found in the provided DataFrame.")


def map_duration_col(config, data):
    """
    Categorizes the 'Duration' column into 'Short-term', 'Medium-term', and 'Long-term' based on credit duration in months.

    Parameters:
    - data: pandas.DataFrame containing a 'Duration' column with credit duration in months.

    Returns:
    - data: pandas.DataFrame with a new column 'CreditDurationCategory' indicating the categorized credit duration.
    """
    # Define bins and labels for the duration categories
    """ Add this to model_config ordinal_mapping if this function is used.
    "CreditDuration": {
      "Short-term (3-12 months)": 0,
      "Medium-term (13-36 months)": 1,
      "Long-term (37-72 months)": 2
    },
    """
    col_name = "CreditDuration"
    bins = [3, 12, 36, 72]  # Bin edges
    labels = list(config['ordinal_mapping'][col_name].keys())  # Category labels

    # Check if the 'Duration' column exists
    if col_name in data.columns:
        # Categorize 'Duration' into bins
        data[col_name] = pd.cut(data[col_name], bins=bins, labels=labels, right=True)
    else:
        print(f"Warning: '{col_name}' column not found in the provided DataFrame.")


def preprocess_data_specific(data, config):
    standardize_column_names(data)
    # Following functions assume capitalized col names
    columns_with_nan = data.columns[data.isna().any()].tolist()
    data = clean_data(data, columns_with_nan)

    # remove 'Unnamed: 0' column
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)

    if "rename_columns" in config:
        data = data.rename(columns=config["rename_columns"])

    map_job_col(config, data)
    # map_duration_col(config, data)
    bin_age_column(data)
    map_ordinal_columns(data, config)
    target_col = config["target_col"]
    X = data.drop(columns=[target_col])
    y = data[target_col]

    if "drop_columns" in config:
        X.drop(columns=config["drop_columns"], inplace=True)

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
        # print the nan columns
        print(X_train.columns[X_train.isna().any()].tolist())
        raise ValueError("There are NaN values in the training data.")

    # Copy the config file to the save path
    with open(os.path.join(save_path, f"{DATASET_NAME}_model_config.json"), 'w') as file:
        json.dump(config, file)

    # Change list of column names to be encoded to a list of column indices
    columns_to_encode = [X_train.columns.get_loc(col) for col in config["columns_to_encode"]]
    pipeline = construct_pipeline(columns_to_encode, RandomForestClassifier())

    model_params = {("model__" + key if not key.startswith("model__") else key): value for key, value in
                    config["model_params"].items()}
    search_params = config.get("random_search_params", {"n_iter": 10, "cv": 5, "random_state": 42})

    best_model, best_params = train_model(X_train, y_train, pipeline, model_params, search_params)

    best_model.fit(X_train, y_train)

    # Predict probabilities for the train set
    y_train_pred = best_model.predict_proba(X_train)[:, 1]

    # Compute ROC AUC score for the train set
    train_score = roc_auc_score(y_train, y_train_pred)
    print("Best Model Score Train:", train_score)

    # Predict probabilities for the test set
    y_test_pred = best_model.predict_proba(X_test)[:, 1]

    # Compute ROC AUC score for the test set
    test_score = roc_auc_score(y_test, y_test_pred)
    print("Best Model Score Test:", test_score)

    # Get global shapley feature importance
    # Transform the data to a numpy array with preprocessor from the pipeline
    explainer = shap.Explainer(best_model.predict, X)
    shap_values = explainer(X)
    shap.plots.bar(shap_values)


    # Print evaluation metrics on train and test
    # print("Best Model Score Train:", best_model.score(X_train, y_train))
    # print("Best Model Score Test:", best_model.score(X_test, y_test))
    # print("Best Parameters:", best_params)

    # Save the best model
    # pickle.dump(best_model, open(os.path.join(save_path, f"{DATASET_NAME}_model_rf.pkl"), 'wb'))


if __name__ == "__main__":
    main()
