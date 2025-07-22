### Load pkl files that start with "adult"

import os
import pickle
import pandas as pd

from explain.explanation import MegaExplainer
from explain.logic import ExplainBot

root = "/Users/dimitrymindlin/UniProjects/Dialogue-XAI-APP"
def load_data(dataset):
    # Load adult_train.csv dataset
    csv_path = f"{root}/data/{dataset}/{dataset}_test.csv"
    try:
        test_data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        test_data = None
    test_data.rename(columns={"Unnamed: 0": "instance_id"}, inplace=True)
    return test_data


def load_pkl_files(dataset):
    # Get the list of all files in the current directory
    files_in_directory = os.listdir('.')

    # Filter files that start with "adult" and end with ".pkl"
    adult_pkl_files = [file for file in files_in_directory if file.startswith(f"{dataset}") and file.endswith('.pkl')]

    # Load each pickle file and store the data_static_interactive in a dictionary
    data_dict = {}
    for file in adult_pkl_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            data_dict[file] = data

    return data_dict


dataset = "adult"
data = load_pkl_files(dataset)

# load adult_train data_static_interactive
adult_data = load_data(dataset)
instance_ids = data[f'{dataset}-diverse-instances.pkl']

#instance_ids = list()
"""for key, value in instance_ids_dict.items():
    print(f"{key}: {len(value)} instances")
    # Convert the list of instance_ids to a set for faster lookup
    instance_ids.extend(value)"""

# Filter adult data_static_interactive by ids
test_data_instances = adult_data[adult_data['instance_id'].isin(instance_ids)]

# Print distribution of test_data_instances labels
print(test_data_instances['Income'].value_counts())
print(data)

# Check the average feature importances by mega-explainer
mega_explainer = data[f'{dataset}-mega-explainer-tabular.pkl']
feature_importances_dict = {}
# Calculate average feature importance across all instances
for instance_id, exp in mega_explainer.items():
    for feature_id, importance in exp.list_exp:
        # Turn negative feature importances into positive values
        importance = abs(importance)
        feature_importances_dict.setdefault(feature_id, []).append(importance)

# Calculate the average importance for each feature
average_feature_importances = {feature_id: sum(importances) / len(importances)
                                for feature_id, importances in feature_importances_dict.items()}
# Sort features by average importance
sorted_feature_importances = sorted(average_feature_importances.items(), key=lambda x: x[1], reverse=True)
# Print the sorted feature importances
print("Average Feature Importances:")
for feature_id, avg_importance in sorted_feature_importances:
    print(f"Feature: {feature_id}, Average Importance: {avg_importance:.4f}")

# Check which variables the counterfactual explanations are based on most often
#Create ExplainBot instance
bot = ExplainBot("chat", "1", "TEST")
question_id = "counterfactualAnyChange"
feature_id = None
exp = bot.update_state_new(question_id, feature_id)
print(exp)
