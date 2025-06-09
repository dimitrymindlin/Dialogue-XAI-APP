# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data.ml_utilities import label_encode_and_save_classes, construct_pipeline, train_model
from data.train_utils import (
    setup_training_environment,
    save_train_test_data,
    setup_feature_display_names,
    apply_display_names,
    save_config_file,
    save_categorical_mapping,
    save_model_and_features,
    evaluate_model,
    update_config_with_metrics,
    get_and_store_feature_importances,
    generate_feature_names_output,
    check_nan_values
)

DATASET_NAME = "diabetes"
config_path = f"./{DATASET_NAME}_model_config.json"
save_path = f"./{DATASET_NAME}"
save_flag = True


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


def preprocess_data_specific(data, config):
    if "drop_columns" in config:
        data.drop(columns=config["drop_columns"], inplace=True)

    standardize_column_names(data)
    target_col = config["target_col"]
    control_variable_name = "BloodGroup"
    # Add BloodGroup as a categorical variable (not ordinal)
    add_control_variable(data, control_variable_name)
    # Note: BloodGroup should be in columns_to_encode, not ordinal_mapping

    # Following functions assume capitalized col names
    columns_with_nan = data.columns[data.isna().any()].tolist()
    if len(columns_with_nan) > 0:
        print(f"Columns with NaN values: {columns_with_nan}")
        data = clean_data(data, columns_with_nan)
    else:
        print("No columns with NaN values found.")

    if "rename_columns" in config:
        data = data.rename(columns=config["rename_columns"])

    map_ordinal_columns(data, config, exclude_columns=[control_variable_name, target_col])

    # Check which column has "Other" values
    for col in data.columns:
        if "Other" in data[col].unique():
            print(f"Column '{col}' has 'Other' values.")

    # Debug: Check target column information
    print(f"Target column name: '{target_col}'")
    print(f"Available columns: {list(data.columns)}")
    if target_col in data.columns:
        print(f"Target column data type: {data[target_col].dtype}")
        print(f"Target column unique values: {data[target_col].unique()}")
        print(f"Target column value counts:\n{data[target_col].value_counts(dropna=False)}")
    else:
        print(f"ERROR: Target column '{target_col}' not found in data!")
        return None, None, {}

    # Check for NaN values in target column and remove rows with NaN target values
    if data[target_col].isna().any():
        print(f"Found {data[target_col].isna().sum()} NaN values in target column '{target_col}'. Removing these rows.")
        data = data.dropna(subset=[target_col])
        print(f"Remaining data shape: {data.shape}")

        # If no data remains, this is a critical error
        if len(data) == 0:
            print("ERROR: All rows have NaN target values. Check your data and target column configuration.")
            return None, None, {}

    X = data.drop(columns=[target_col])
    y = data[target_col]

    if "columns_to_encode" in config:
        X, encoded_classes = label_encode_and_save_classes(X, config)
    else:
        encoded_classes = {}

    return X, y, encoded_classes


def main():
    # Set up environment and load config
    config, save_path = setup_training_environment(DATASET_NAME, config_path)

    data = pd.read_csv(config["dataset_path"])

    # Store original column names before any preprocessing
    original_columns = list(data.columns)

    # Perform preprocessing without applying rename_columns
    saved_rename_columns = None
    if "rename_columns" in config:
        # Temporarily save and remove rename_columns to prevent premature renaming
        saved_rename_columns = config["rename_columns"]
        config["rename_columns"] = {}

    X, y, encoded_classes = preprocess_data_specific(data, config)

    # Check if preprocessing failed
    if X is None or y is None:
        print("Preprocessing failed. Exiting.")
        return

    # Restore rename_columns
    if saved_rename_columns is not None:
        config["rename_columns"] = saved_rename_columns

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
        stratify=y
    )

    target_col = config["target_col"]
    # Add labels to X and save train and test data
    X_train[target_col] = y_train
    X_test[target_col] = y_test

    # Save training and test data
    save_train_test_data(X_train, X_test, DATASET_NAME, save_path)

    # Fix columns_to_encode to use actual column names after standardization
    if "columns_to_encode" in config:
        original_columns_to_encode = config["columns_to_encode"].copy()
        config["columns_to_encode"] = []
        for col in original_columns_to_encode:
            # Map original names to standardized names
            if col == "BloodGroup" and "BloodGroup" in X_train.columns:
                config["columns_to_encode"].append("BloodGroup")

    # Set up feature display names and update config
    feature_display_names, config = setup_feature_display_names(X_train, config, target_col)

    # Apply display names to the dataframes
    X_train, X_test = apply_display_names(X_train, X_test, feature_display_names)

    # Update columns_to_encode to use the renamed columns after display names are applied
    updated_columns_to_encode = []
    for feature in config["columns_to_encode"]:
        # Map original column names to display names
        display_name = feature_display_names.get_display_name(feature)
        if display_name in X_train.columns:
            updated_columns_to_encode.append(display_name)
        else:
            # Fallback to original name if display name not found
            updated_columns_to_encode.append(feature)

    # Save the updated config
    save_config_file(config, save_path, DATASET_NAME)

    # Create a copy of dataframes for later reference
    X_train_with_names = X_train.copy()
    X_test_with_names = X_test.copy()

    # Note: categorical_mapping.json is already created by label_encode_and_save_classes in ml_utilities.py
    # No need to create a duplicate mapping here that would overwrite the correct format

    # Remove target column for model training
    X_train.drop(columns=[target_col], inplace=True)
    X_test.drop(columns=[target_col], inplace=True)

    # Check for NaN values
    check_nan_values(X_train)

    # Change list of column names to be encoded to a list of column indices (using updated names)
    columns_to_encode = [X_train.columns.get_loc(col) for col in updated_columns_to_encode]
    pipeline = construct_pipeline(columns_to_encode, RandomForestClassifier())

    # Set up model parameters
    model_params = {("model__" + key if not key.startswith("model__") else key): value
                    for key, value in config["model_params"].items()}
    search_params = config.get("random_search_params", {"n_iter": 10, "cv": 5, "random_state": 42})

    # Train model
    best_model, best_params = train_model(X_train, y_train, pipeline, model_params, search_params)
    best_model.fit(X_train, y_train)

    # Evaluate model
    train_score, test_score = evaluate_model(best_model, X_train, X_test, y_train, y_test)
    print("Best Parameters:", best_params)

    # Update config with metrics
    config = update_config_with_metrics(config, train_score, test_score)

    # Save updated config with metrics
    save_config_file(config, save_path, DATASET_NAME)

    # Save model and feature information
    if save_flag:
        save_model_and_features(best_model, X_train, config, save_path, DATASET_NAME)

    # Generate feature names output after transformation
    # Create feature mapping for generate_feature_names_output (different format than categorical_mapping.json)
    # Use original column names (not display names) for the mapping
    feature_mapping_for_names = {}
    original_columns_to_encode = ["BloodGroup"]  # Use original column names
    for feature in original_columns_to_encode:
        if feature in encoded_classes:
            feature_mapping_for_names[feature] = {str(v): k for k, v in encoded_classes[feature].items()}
    
    # Create a temporary config with original column names for generate_feature_names_output
    temp_config = config.copy()
    temp_config["columns_to_encode"] = original_columns_to_encode
    
    feature_names_out = generate_feature_names_output(X_train, temp_config, feature_mapping_for_names)

    # Get and store feature importances in config
    feature_importances = get_and_store_feature_importances(best_model, feature_names_out, config, save_path,
                                                            DATASET_NAME)

    print("Saved model!")


if __name__ == "__main__":
    main()
