"""
analysis_config.py

Central configuration for experiment data pipeline. Set all experiment-specific variables here.
All pipeline steps should import this config to ensure DRY and consistent folder usage.
"""

import os

# Set your experiment-specific parameters here
EXPERIMENT_HANDLE = "thesis"
EXPERIMENT_DATE = "09_2025"
DATASET = "diabetes"
GROUP = "mapek"
EXPLICITELY_REPLACE_GROUP_NAME = {
    "prolific_export_68aefeb0905f55201f3d1143.csv": {
        "adult-chat-thesis-08-2025": "adult-thesis-mapek-08-2025"
    },
    "prolific_export_68b547f459f1a30dd204d743.csv": {
        "diabetes-thesis-conversational-08-2025": "diabetes-thesis-mapek-09-2025"
    }
}

"""EXPERIMENT_HANDLE = "measuring"
EXPERIMENT_DATE = "04_2024"
DATASET = "adult"
GROUP = "interactive"
EXPLICITELY_REPLACE_GROUP_NAME = {"adult-chat-thesis-08-2025": "adult-thesis-mapek-08-2025"}"""

PROLIFIC_CSV_FOLDER = f"data_{DATASET}_{EXPERIMENT_HANDLE}_{GROUP}_{EXPERIMENT_DATE}/"
RESULTS_BASE = "results"
INTRO_TEST_CYCLES = 10
TEACHING_CYCLES = 10
FINAL_TEST_CYCLES = 10

# The results subfolder will match the prolific folder name
PROLIFIC_FOLDER_NAME = os.path.basename(os.path.normpath(PROLIFIC_CSV_FOLDER))
RESULTS_DIR = os.path.join(RESULTS_BASE, PROLIFIC_FOLDER_NAME)

DB_CONFIG = {
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432",
    "dbname": "portainer_latest"
}

# Dummy variable configuration for analysis
DUMMY_VAR_NAME = "worklifebalance"
DUMMY_VAR_COLUMN = "dummy_var_mention"
DUMMY_VAR_THRESHOLD = 80  # Similarity threshold for fuzzy matching

