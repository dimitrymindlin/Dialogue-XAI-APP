# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from data.ml_utilities import label_encode_and_save_classes, construct_pipeline, train_model
from data.train_utils import (
    setup_training_environment,
    save_train_test_data,
    setup_feature_display_names,
    apply_display_names,
    save_config_file,
    save_categorical_mapping,
    save_model_and_features,
    print_feature_importances
)

DATASET_NAME = "diabetes"
config_path = f"./{DATASET_NAME}_model_config.json"
save_path = f"./{DATASET_NAME}"


def standardize_column_names(data):
    data.columns = [col.capitalize() for col in data.columns]


def clean_data(data, cols_with_nan_list):
    data[data == '?'] = np.nan
    for col in cols_with_nan_list:
        # Replace NaN with "Not Disclosed / Missing" for categorical columns
        if data[col].dtype == 'object':
            data[col].fillna("Not Disclosed / Missing", inplace=True)
    return data


def map_ordinal_columns(data, config, exclude_columns):
    for col, mapping in config["ordinal_mapping"].items():
        if col in exclude_columns:
            continue
        data[col] = data[col].map(mapping)


def add_control_variable(data, variable_name):
    # Define the categories and their corresponding probabilities
    categories = ["O", "A", "B", "AB"]
    probabilities = [0.45, 0.40, 0.11, 0.04]

    # Generate random blood group assignments based on the defined probabilities
    data[variable_name] = np.random.choice(categories, size=len(data), p=probabilities)

    # Create a mapping from string values to integers
    mapping = {category: i for i, category in enumerate(categories)}

    # Apply the mapping to the new variable
    data[variable_name] = data[variable_name].map(mapping)

    return mapping


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
    control_variable_name = "BloodGroup"
    config["ordinal_mapping"][control_variable_name] = add_control_variable(data, control_variable_name)

    # Following functions assume capitalized col names
    columns_with_nan = data.columns[data.isna().any()].tolist()
    if len(columns_with_nan) > 0:
        print(f"Columns with NaN values: {columns_with_nan}")
        data = clean_data(data, columns_with_nan)
    else:
        print("No columns with NaN values found.")

    if "rename_columns" in config:
        data = data.rename(columns=config["rename_columns"])

    map_ordinal_columns(data, config, exclude_columns=[control_variable_name])
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
    # Initialize environment and paths
    config, save_path = setup_training_environment(DATASET_NAME, config_path)

    # Load and preprocess data
    data = pd.read_csv(config["dataset_path"])
    X, y, encoded_classes = preprocess_data_specific(data, config)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
        stratify=y
    )

    target_col = config["target_col"]
    # Attach target and save CSVs
    X_train[target_col] = y_train
    X_test[target_col] = y_test
    save_train_test_data(X_train, X_test, DATASET_NAME, save_path)

    # Set up feature display names
    feature_display_names, config = setup_feature_display_names(X_train, config, target_col)
    X_train, X_test = apply_display_names(X_train, X_test, feature_display_names)

    # Label-encode categorical features and update config
    _, encoded_classes = label_encode_and_save_classes(
        X_train.drop(columns=[target_col]),
        config
    )
    save_config_file(config, save_path, DATASET_NAME)
    save_categorical_mapping(encoded_classes, save_path)

    # Prepare arrays for training
    X_train.drop(columns=[target_col], inplace=True)
    X_test.drop(columns=[target_col], inplace=True)
    feature_names = list(X_train.columns)

    # Train model using existing utilities
    from sklearn import __version__ as sklearn_version
    model_params = {("model__" + k if not k.startswith("model__") else k): v
                    for k, v in config["model_params"].items()}
    search_params = config.get("random_search_params", {"n_iter": 10, "cv": 5, "random_state": 42})
    best_model, best_params = train_model(
        X_train.values, y_train,
        construct_pipeline(
            [X_train.columns.get_loc(c) for c in config["columns_to_encode"]],
            RandomForestClassifier()
        ),
        model_params,
        search_params
    )

    # Evaluate and record metrics
    train_score = roc_auc_score(y_train, best_model.predict_proba(X_train.values)[:, 1])
    test_score = roc_auc_score(y_test, best_model.predict_proba(X_test.values)[:, 1])
    config["evaluation_metrics"] = {
        "train_score_auc_roc": train_score,
        "test_score_auc_roc": test_score
    }
    config["sklearn_version"] = sklearn_version
    save_config_file(config, save_path, DATASET_NAME)

    # Save model and features, then print importances
    save_model_and_features(best_model, feature_names, save_path, DATASET_NAME)
    print_feature_importances(best_model, feature_names)


if __name__ == "__main__":
    main()
