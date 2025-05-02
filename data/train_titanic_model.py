"""Train titanic model."""
import sys
import os
import json
from os.path import dirname, abspath

import numpy as np
import pickle as pkl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier as clf

from data import load_config, create_folder_if_not_exists
from data.train_utils import (
    setup_training_environment, save_train_test_data, setup_feature_display_names,
    apply_display_names, save_config_file, save_categorical_mapping,
    save_model_and_features, print_feature_importances, generate_feature_names_output
)

np.random.seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from data.processing_functions import get_and_preprocess_titanic
from data.ml_utilities import label_encode_and_save_classes

# Define dataset name and paths
DATASET_NAME = "titanic"
save_path = f"./{DATASET_NAME}"
config_path = f"./{DATASET_NAME}_model_config.json"
save_flag = True
target_col = "Survived"

def main():
    # Set up environment and load config
    config, save_path = setup_training_environment(DATASET_NAME, config_path)
    
    # Load and preprocess data
    titanic_data, categorical_mapping = get_and_preprocess_titanic()
    
    # Split df in x and y values
    y_train = titanic_data['train'][target_col].values
    X_train = titanic_data['train']
    y_test = titanic_data['test'][target_col].values
    X_test = titanic_data['test']
    
    # Save to CSV in the dedicated folder
    save_train_test_data(X_train, X_test, DATASET_NAME, save_path)
    
    # Store original column names before any renaming
    original_columns = list(X_train.columns)
    
    # Set up feature display names and update config
    feature_display_names, config = setup_feature_display_names(X_train, config, target_col)
    
    # Apply display names to the dataframes
    X_train, X_test = apply_display_names(X_train, X_test, feature_display_names)
    
    # Add encoded mappings to config for saving
    X_train_for_encoding = X_train.copy()
    X_train_for_encoding.pop(target_col)
    
    # Call label_encode_and_save_classes to generate encoded_col_mapping.json
    _, encoded_classes = label_encode_and_save_classes(X_train_for_encoding, config)
    
    # Save config with updated information
    save_config_file(config, save_path, DATASET_NAME)
    
    # Save categorical mapping from preprocess function (different from encoded_classes)
    save_categorical_mapping(categorical_mapping, save_path)
    
    # Create a copy of dataframes before removing target
    X_train_with_names = X_train.copy()
    X_test_with_names = X_test.copy()
    
    X_train.pop(target_col)
    X_test.pop(target_col)
    
    # Store feature names for later use
    feature_names = list(X_train.columns)
    
    # Convert to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    
    # Get column indices for categorical features
    column_indices = {col: i for i, col in enumerate(feature_names)}
    categorical_indices = [column_indices[col] for col in config["columns_to_encode"]]
    
    # Create a ColumnTransformer with better feature names
    preprocessor = ColumnTransformer(
        transformers=[
            ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_indices)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False  # Prevents prefixing feature names
    )
    
    # Define the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', clf())
    ])
    
    # Set the feature names to be used
    preprocessor.set_output(transform="pandas")
    
    # Extract hyperparameter search configuration
    model_params = config["model_params"]
    search_params = config["random_search_params"]
    
    # Randomized search for hyperparameter tuning
    random_search = RandomizedSearchCV(pipeline, param_distributions=model_params, **search_params)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Fit the best model
    best_model.fit(X_train, y_train)
    
    print("Train Score:", best_model.score(X_train, y_train))
    print("Test Score:", best_model.score(X_test, y_test))
    print("Portion y==1:", np.sum(y_test == 1) * 1. / y_test.shape[0])
    
    # Calculate ROC AUC
    y_test_pred = best_model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred, pos_label=1)
    test_auc = metrics.auc(fpr, tpr)
    print("AUC_test", test_auc)
    print("Best Parameters:", best_params)
    
    # Update config with evaluation metrics
    config["evaluation_metrics"] = {
        "train_score": best_model.score(X_train, y_train),
        "test_score": best_model.score(X_test, y_test),
        "test_auc": test_auc
    }
    from sklearn import __version__ as sklearn_version
    config["sklearn_version"] = sklearn_version
    
    # Save the updated config
    save_config_file(config, save_path, DATASET_NAME)
    
    # Save model and feature names
    if save_flag:
        with open(os.path.join(save_path, f"{DATASET_NAME}_model_rf.pkl"), "wb") as f:
            pkl.dump(best_model, f)
        
        # Also save the feature names for later interpretation
        with open(os.path.join(save_path, f"{DATASET_NAME}_feature_names.json"), "w") as f:
            json.dump({
                "feature_names": feature_names,
                "categorical_features": config["columns_to_encode"]
            }, f)
    
    # Generate feature names output after transformation
    feature_names_out = []
    # Get categorical feature names (after one-hot encoding)
    for feature in config["columns_to_encode"]:
        # Get unique values for this feature
        unique_values = categorical_mapping.get(feature, {}).keys()
        for value in unique_values:
            feature_names_out.append(f"{feature}_{value}")
    # Add numeric features
    for feature in feature_names:
        if feature not in config["columns_to_encode"]:
            feature_names_out.append(feature)
    
    # Print feature importances
    print_feature_importances(best_model, feature_names_out)
    
    print("Saved model!")

if __name__ == "__main__":
    main()
