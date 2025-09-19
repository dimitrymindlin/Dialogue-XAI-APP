"""
2_extract_information_to_columns.py

This script loads raw experiment data (CSV files) produced by 1_get_raw_experiment_data.py,
performs extraction and transformation steps (e.g., unpacking JSON columns, creating new columns),
and saves the resulting DataFrames as new CSVs in the next pipeline stage folder.

Expected Inputs:
- results/1_raw/prolific_merged.csv
- results/1_raw/users_raw.csv
- results/1_raw/events_raw.csv
- results/1_raw/user_completed_raw.csv

Outputs:
- results/2_unpacked/events_unpacked.csv
- (add more as needed for further extraction steps)
"""

import analysis_config as config
from analysis_utils import (
    unpack_questions,
    extract_all_questionnaires,
    get_study_group_and_events,
    extract_questions,
    add_dummy_var_mention_column,
    add_prolific_times,
    create_understanding_over_time_df,
    add_understanding_question_analysis,
    add_subjective_understanding_score
)
from experiment_analysis.calculations import create_predictions_df
from ml_knowledge_utils import ML_KNOWLEDGE_TEXT_TO_NUM
import os
import pandas as pd

RAW_DIR = os.path.join(config.RESULTS_DIR, "1_raw")
UNPACKED_DIR = os.path.join(config.RESULTS_DIR, "2_unpacked")
os.makedirs(UNPACKED_DIR, exist_ok=True)

# Load raw data
users = pd.read_csv(os.path.join(RAW_DIR, "users_raw.csv"))
events = pd.read_csv(os.path.join(RAW_DIR, "events_raw.csv"))
user_completed = pd.read_csv(os.path.join(RAW_DIR, "user_completed_raw.csv"))
prolific_merged = pd.read_csv(os.path.join(RAW_DIR, "prolific_merged.csv"))

# Add Prolific start and end times to user_completed
print("Adding Prolific start and end times to user_completed...")
user_completed, non_prolific_user_ids = add_prolific_times(user_completed, prolific_merged, events)
print(f"Added Prolific times. {len(non_prolific_user_ids)} users not found in Prolific data.")
user_completed.to_csv(os.path.join(UNPACKED_DIR, "user_completed_with_times.csv"), index=False)

# Unpack details column
events_unpacked = unpack_questions(events)
events_unpacked.to_csv(os.path.join(UNPACKED_DIR, "events_unpacked.csv"), index=False)

# Extract questions over time and feedback for each user
questions_over_time_list = []
all_questionnaires_list = []
initial_test_preds_list = []
learning_test_preds_list = []
final_test_preds_list = []
for user_id in users["id"]:
    study_group, user_events = get_study_group_and_events(users, events, user_id)
    questions_df = extract_questions(user_events, study_group)
    if questions_df is not None:
        questions_df["user_id"] = user_id
        questions_over_time_list.append(questions_df)
    questionnaire_df = extract_all_questionnaires(users, user_id)
    if questionnaire_df is not None:
        all_questionnaires_list.append(questionnaire_df)
    # Aggregate prediction DataFrames for each user
    intro_test_preds, preds_learning, final_test_preds, _ = create_predictions_df(
        users,
        user_events,
        exclude_incomplete=False,
        teaching_cycles=config.TEACHING_CYCLES,
        final_test_cycles=config.FINAL_TEST_CYCLES,
    )
    if intro_test_preds is not None:
        initial_test_preds_list.append(intro_test_preds)
    if preds_learning is not None:
        learning_test_preds_list.append(preds_learning)
    if final_test_preds is not None:
        final_test_preds_list.append(final_test_preds)

# Add quesionnaires to users DataFrame
if all_questionnaires_list:
    all_questionnaires_df = pd.concat(all_questionnaires_list, ignore_index=True)
    # Fix column name mismatch: user_id in questionnaires matches id in users
    users = users.merge(all_questionnaires_df, left_on="id", right_on="user_id", how="left")
    # Drop the duplicate user_id column to avoid confusion
    if "user_id" in users.columns and "id" in users.columns:
        users = users.drop(columns=["user_id"])

# Concatenate and save
if questions_over_time_list:
    questions_over_time_df = pd.concat(questions_over_time_list, ignore_index=True)
    questions_over_time_df.to_csv(os.path.join(UNPACKED_DIR, "questions_over_time.csv"), index=False)

# Add dummy variable mention column to final DataFrame
final_test_preds_df = pd.concat(final_test_preds_list, ignore_index=True) if final_test_preds_list else pd.DataFrame()
final_test_preds_df = add_dummy_var_mention_column(
    final_test_preds_df,
    dummy_var_name=config.DUMMY_VAR_NAME,
    column_name=config.DUMMY_VAR_COLUMN,
    similarity_threshold=config.DUMMY_VAR_THRESHOLD,
)
initial_test_preds_df = pd.concat(initial_test_preds_list,
                                  ignore_index=True) if initial_test_preds_list else pd.DataFrame()
learning_test_preds_df = pd.concat(learning_test_preds_list,
                                   ignore_index=True) if learning_test_preds_list else pd.DataFrame()

# Combine all prediction dataframes into a single dataframe
print("Creating combined predictions dataframe...")
all_preds_dfs = [df for df in [initial_test_preds_df, learning_test_preds_df, final_test_preds_df] if not df.empty]

if all_preds_dfs:
    # Combine all prediction dataframes
    all_preds_df = pd.concat(all_preds_dfs, ignore_index=True)

    # Save the combined predictions dataframe
    all_preds_df.to_csv(os.path.join(UNPACKED_DIR, "all_preds_df.csv"), index=False)
    print(f"Saved all combined predictions with {len(all_preds_df)} rows")

    # Create understanding over time dataframe
    print("Creating understanding over time dataframe...")
    understanding_df = create_understanding_over_time_df(learning_test_preds_df)
    understanding_df.to_csv(os.path.join(UNPACKED_DIR, "understanding_over_time.csv"), index=False)
    print(f"Saved understanding over time data with {len(understanding_df)} rows")

    # Add accuracy_over_time to users DataFrame
    if 'accuracy_over_time' in understanding_df.columns:
        users = users.merge(
            understanding_df[['user_id', 'accuracy_over_time']],
            left_on='id',
            right_on='user_id',
            how='left'
        )
        # Drop the duplicate user_id column if it exists
        if 'user_id' in users.columns and 'id' in users.columns:
            users = users.drop(columns=['user_id'])

# Add understanding question analysis to users DataFrame
print("Analyzing understanding questions...")
try:
    users = add_understanding_question_analysis(users, dataset_name=config.DATASET)
    print(
        f"Added understanding question analysis columns to users DataFrame for {config.DATASET} dataset"
    )
except FileNotFoundError as e:
    print(f"WARNING: {str(e)}")
    print(
        "Understanding question analysis skipped. Please create a config file at "
        f"configs/{config.DATASET}_understanding.json"
    )
except Exception as e:
    print(f"ERROR analyzing understanding questions: {str(e)}")
    print("Understanding question analysis failed, but continuing with pipeline...")

# Add profile variables (ML knowledge, studies participation) to users DataFrame
def extract_profile_variables(users_df):
    """Extract and convert profile variables to numeric columns."""
    import json
    
    # Define mappings
    fam_ml_mapping = ML_KNOWLEDGE_TEXT_TO_NUM
    
    ml_studies_mapping = {
        "never": 0,
        "a few times": 1,
        "regularly": 2
    }
    
    xai_studies_mapping = {
        "never": 0,
        "a few times": 1,
        "regularly": 2
    }
    
    def extract_and_convert(profile_json, field_name, mapping):
        """Extract a field from profile JSON and convert using mapping."""
        try:
            profile_data = json.loads(profile_json)
            value = profile_data.get(field_name, "")
            
            if value == "":
                return None
                
            # Try to cast to int first (for backward compatibility)
            try:
                value = int(value)
                # For ml_knowledge, ensure it's in valid range
                if field_name == "fam_ml_val":
                    return value if 0 <= value <= 4 else None
                return value
            except ValueError:
                pass
                
            # If it's a string, use mapping
            if isinstance(value, str):
                return mapping.get(value, None)
                
        except (json.JSONDecodeError, TypeError):
            return None
        
        return None
    
    # Extract all three variables
    users_df["ml_knowledge"] = users_df["profile"].apply(
        lambda x: extract_and_convert(x, "fam_ml_val", fam_ml_mapping)
    )
    
    users_df["ml_studies_participation_num"] = users_df["profile"].apply(
        lambda x: extract_and_convert(x, "ml_studies_participation", ml_studies_mapping)
    )
    
    users_df["xai_studies_participation_num"] = users_df["profile"].apply(
        lambda x: extract_and_convert(x, "xai_studies_participation", xai_studies_mapping)
    )
    
    # Print extraction summary
    ml_knowledge_count = users_df["ml_knowledge"].notna().sum()
    ml_studies_count = users_df["ml_studies_participation_num"].notna().sum()
    xai_studies_count = users_df["xai_studies_participation_num"].notna().sum()
    
    print(f"Profile variable extraction summary:")
    print(f"  ml_knowledge: {ml_knowledge_count}/{len(users_df)} users")
    print(f"  ml_studies_participation: {ml_studies_count}/{len(users_df)} users")  
    print(f"  xai_studies_participation: {xai_studies_count}/{len(users_df)} users")
    
    return users_df

users = extract_profile_variables(users)

# Add subjective understanding variable to users DataFrame
users = add_subjective_understanding_score(users)

# Save the processed users DataFrame
users.to_csv(os.path.join(UNPACKED_DIR, "users_unpacked.csv"), index=False)
print(f"Saved processed users data with {len(users)} rows")

print("Extraction complete. Ready for further processing and filtering.")
print(f"Found {len(questions_over_time_list)} users with questions over time.")
print(f"Found questionnaire results for {len(all_questionnaires_list)} users.")
