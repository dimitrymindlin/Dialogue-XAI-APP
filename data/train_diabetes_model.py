import numpy as np
import pandas as pd
import pickle as pkl
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from data.feature_mapping_utils import create_feature_name_mapping

np.random.seed(3)

# Define dataset name and paths
DATASET_NAME = "diabetes"
save_path = f"./data/{DATASET_NAME}"
target_col = "y"

# Create folder if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

X_values = pd.read_csv("diabetesSmall.csv")

# Store original column names before any processing
original_columns = list(X_values.columns)

y_values = X_values.pop("y")

# Create feature name mapping
create_feature_name_mapping(
    original_columns=original_columns,
    save_path=save_path,
    dataset_name="feature",
    target_col=target_col
)

# Create Pipeline
model_name = "Linear Regression"
model_instance = LogisticRegression()

lr_pipeline = Pipeline([('scaler', StandardScaler()),
                        ('lr', model_instance)])

# KFold instance with k = 5
kf = KFold(n_splits=5, shuffle=True, random_state=3)

train_scores = []
validation_scores = []

# Loop through each fold
for train_index, val_index in kf.split(X_values, y_values):
    X_train, X_val = X_values.iloc[train_index], X_values.iloc[val_index]
    y_train, y_val = y_values.iloc[train_index], y_values.iloc[val_index]

    # Turn data to numpy arrays
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()

    lr_pipeline.fit(X_train, y_train)

    # Evaluate and store scores
    train_scores.append(lr_pipeline.score(X_train, y_train))
    validation_scores.append(lr_pipeline.score(X_val, y_val))

# Print average scores for the Ridge Classifier
print(f"Model: {model_name}")
print("Average Train Score:", np.mean(train_scores))
print("Average Validation Score:", np.mean(validation_scores))
print("-" * 50)

# Sort and print feature importances (coefficients) for the Logistic Regression model
importances = lr_pipeline.named_steps['lr'].coef_[0]  # Access the first row for binary classification
sorted_indices = np.argsort(np.abs(importances))[::-1]  # Sort by absolute value

print(f"Sorted feature importances for {model_name}:")
for i in sorted_indices:
    print(f"    {X_values.columns[i]}: {importances[i]}")

#with open("./data/diabetes_model_logistic_regression.pkl", "wb") as f:
#    pkl.dump(lr_pipeline, f)

print("Saved model!")
