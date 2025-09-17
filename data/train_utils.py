# train_utils.py

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn import __version__ as sklearn_version

def setup_training_environment(dataset_name, config_path):
    """Initialize training environment with config and paths."""
    from data import load_config, create_folder_if_not_exists
    
    save_path = f"./{dataset_name}"
    config = load_config(config_path)
    config["save_path"] = save_path
    create_folder_if_not_exists(save_path)
    
    return config, save_path

def save_train_test_data(X_train, X_test, dataset_name, save_path):
    """Save training and test data to CSV files."""
    X_train.to_csv(os.path.join(save_path, f"{dataset_name}_train.csv"))
    X_test.to_csv(os.path.join(save_path, f"{dataset_name}_test.csv"))

def setup_feature_display_names(X_train, config, target_col):
    """Setup feature display names and update config accordingly."""
    from data.response_templates.feature_display_names import FeatureDisplayNames
    
    # Get current column names excluding target
    current_columns = list(X_train.columns)
    if target_col in current_columns:
        current_columns.remove(target_col)
    
    # Create FeatureDisplayNames instance and update config
    feature_display_names = FeatureDisplayNames(config=config, feature_names=current_columns)
    feature_display_names.update_config(current_columns, target_col=target_col)
    config = feature_display_names.save_to_config(config)
    
    # DON'T convert columns_to_encode to display names - they should remain as original column names
    # The columns_to_encode list should contain the actual column names used for machine learning
    
    return feature_display_names, config

def apply_display_names(X_train, X_test, feature_display_names):
    """Apply column renaming to dataframes."""
    X_train = feature_display_names.apply_column_renaming(X_train)
    X_test = feature_display_names.apply_column_renaming(X_test)
    return X_train, X_test

def save_config_file(config, save_path, dataset_name):
    """Save config to a JSON file."""
    config_path = os.path.join(save_path, f"{dataset_name}_model_config.json")
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)
    return config_path

def save_categorical_mapping(categorical_mapping, save_path):
    """Save categorical mapping to a JSON file."""
    with open(os.path.join(save_path, "categorical_mapping.json"), 'w') as f:
        json.dump(categorical_mapping, f)

def save_model_and_features(best_model, X_train, config, save_path, dataset_name):
    """Save model, feature names and metadata."""
    import pickle
    
    # Save the model
    pickle.dump(best_model, open(os.path.join(save_path, f"{dataset_name}_model_rf.pkl"), 'wb'))
    
    # Save feature names
    with open(os.path.join(save_path, f"{dataset_name}_feature_names.json"), "w") as f:
        json.dump({
            "feature_names": list(X_train.columns),
            "categorical_features": config["columns_to_encode"]
        }, f)

def evaluate_model(best_model, X_train, X_test, y_train, y_test):
    """Evaluate model performance on train and test sets."""
    # Predict probabilities for train set
    y_train_pred = best_model.predict_proba(X_train)[:, 1]
    train_score = roc_auc_score(y_train, y_train_pred)
    
    # Predict probabilities for test set
    y_test_pred = best_model.predict_proba(X_test)[:, 1]
    test_score = roc_auc_score(y_test, y_test_pred)
    
    print("Best Model Score Train:", train_score)
    print("Best Model Score Test:", test_score)
    
    return train_score, test_score

def update_config_with_metrics(config, train_score, test_score):
    """Add evaluation metrics to config."""
    config["evaluation_metrics"] = {
        "train_score_auc_roc": train_score,
        "test_score_auc_roc": test_score
    }
    config["sklearn_version"] = sklearn_version
    return config

def print_feature_importances(best_model, feature_names_out):
    """Print feature importances."""
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        feature_importances = best_model.named_steps['model'].feature_importances_
        print("\nFeature importances:")
        for name, importance in zip(feature_names_out, feature_importances):
            print(f"{name} = {importance}")


def get_and_store_feature_importances(best_model, feature_names_out, config, save_path, dataset_name):
    """Get feature importances and store them in the model config."""
    feature_importances_dict = {}
    
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        feature_importances = best_model.named_steps['model'].feature_importances_
        
        # Create dictionary of feature names and their importance values
        for name, importance in zip(feature_names_out, feature_importances):
            feature_importances_dict[name] = float(importance)
        
        # Sort by importance (descending order)
        feature_importances_dict = dict(sorted(feature_importances_dict.items(), 
                                             key=lambda x: x[1], reverse=True))
        
        # Store in config
        config["feature_importances"] = feature_importances_dict
        
        # Print feature importances (maintain existing behavior)
        print("\nFeature importances:")
        for name, importance in feature_importances_dict.items():
            print(f"{name} = {importance}")
        
        # Save updated config with feature importances
        save_config_file(config, save_path, dataset_name)
        
        return feature_importances_dict
    else:
        print("\nModel does not have feature_importances_ attribute")
        return {}

def generate_feature_names_output(X_train, config, categorical_mapping):
    """Generate feature names after one-hot encoding."""
    feature_names_out = []
    
    # Add categorical feature names (one-hot encoded)
    for feature in config["columns_to_encode"]:
        if feature in categorical_mapping:
            unique_values = categorical_mapping[feature].values()
            for value in unique_values:
                feature_names_out.append(f"{feature}_{value}")
    
    # Add numeric features
    for feature in X_train.columns:
        if feature not in config["columns_to_encode"]:
            feature_names_out.append(feature)
            
    return feature_names_out

def standardize_column_names(data):
    """Standardize column names by capitalizing them."""
    data.columns = [col.capitalize() for col in data.columns]
    return data

def check_nan_values(X_train):
    """Check for NaN values in the data."""
    if X_train.isna().sum().sum() > 0:
        print(X_train.isna().sum())
        raise ValueError("There are NaN values in the training data.")