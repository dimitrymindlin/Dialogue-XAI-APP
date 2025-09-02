"""
3_filter_and_clean_data.py

This script loads extracted/unpacked experiment data from results/{PROLIFIC_FOLDER_NAME}/2_unpacked/ 
and applies all filtering and cleaning steps. It produces filtered and cleaned DataFrames for downstream analysis.

Expected Inputs:
- results/{PROLIFIC_FOLDER_NAME}/2_unpacked/events_unpacked.csv
- results/{PROLIFIC_FOLDER_NAME}/2_unpacked/users_raw.csv
- results/{PROLIFIC_FOLDER_NAME}/2_unpacked/user_completed_with_times.csv
- results/{PROLIFIC_FOLDER_NAME}/2_unpacked/questions_over_time.csv
- results/{PROLIFIC_FOLDER_NAME}/2_unpacked/all_questionnaires.csv
- results/{PROLIFIC_FOLDER_NAME}/2_unpacked/all_predictions.csv

Outputs:
- results/{PROLIFIC_FOLDER_NAME}/3_filtered/users_filtered.csv
- results/{PROLIFIC_FOLDER_NAME}/3_filtered/events_filtered.csv
- results/{PROLIFIC_FOLDER_NAME}/3_filtered/questions_filtered.csv
- results/{PROLIFIC_FOLDER_NAME}/3_filtered/predictions_filtered.csv
"""

from experiment_analysis.analysis_config import RESULTS_DIR
import os
import pandas as pd
from experiment_analysis.filter_users import (
    filter_completed_users,
    filter_by_dataset_and_condition,
    filter_by_time,
    filter_by_negative_times,
    remove_outliers_by_attention_check,
    filter_by_work_life_balance,
    filter_by_missing_ml_kowledge,
    filter_by_self_defined_attention_check,
    filter_by_wanted_count,
    filter_users_that_didnt_ask_questions_from_df,
    filter_by_fatal_understanding_error,
    filter_by_invalid_scores
)
from experiment_analysis.analysis_utils import (
    count_dummy_var_users,
    filter_by_dummy_var_mentions,
    filter_dataframes_by_user_ids,
    save_dataframes_to_csv
)
from experiment_analysis.analysis_config import DUMMY_VAR_NAME, DUMMY_VAR_COLUMN, DATASET, GROUP

UNPACKED_DIR = os.path.join(RESULTS_DIR, "2_unpacked")
FILTERED_DIR = os.path.join(RESULTS_DIR, "3_filtered")
os.makedirs(FILTERED_DIR, exist_ok=True)

# Load unpacked data
print("Loading unpacked data from", UNPACKED_DIR)
users = pd.read_csv(os.path.join(UNPACKED_DIR, "users_unpacked.csv"))
events = pd.read_csv(os.path.join(UNPACKED_DIR, "events_unpacked.csv"))
user_completed = pd.read_csv(os.path.join(UNPACKED_DIR, "user_completed_with_times.csv"))
all_preds_df = pd.read_csv(os.path.join(UNPACKED_DIR, "all_preds_df.csv"))

# Load pre-computed questions_over_time_df if it exists
questions_over_time_df = None
questions_over_time_path = os.path.join(UNPACKED_DIR, "questions_over_time.csv")
if os.path.exists(questions_over_time_path):
    try:
        questions_over_time_df = pd.read_csv(questions_over_time_path)
        print(f"\nLoaded pre-computed questions_over_time_df with {len(questions_over_time_df)} rows")
    except Exception as e:
        print(f"Failed to load questions_over_time.csv: {e}")
else:
    print("\nNo pre-computed questions_over_time.csv found. Will generate on-the-fly if needed.")

# Debug: Print column names to check for time columns
print("\nAvailable columns in users DataFrame:")
print(", ".join(users.columns))
print("\nAvailable columns in user_completed DataFrame:")
print(", ".join(user_completed.columns))

# Check for time columns
time_columns = ['total_exp_time', 'exp_instruction_time', 'total_learning_time']
missing = [col for col in time_columns if col not in users.columns]
if missing:
    print(f"\nWarning: These time columns are missing in users DataFrame: {', '.join(missing)}")
    print("Some filters that depend on these columns will be skipped.")

    # Check if time columns exist in user_completed
    time_cols_in_completed = [col for col in missing if col in user_completed.columns]
    if time_cols_in_completed:
        # merge the missing time columns from user_completed into users
        print(f"Found missing time columns in user_completed: {', '.join(time_cols_in_completed)}")
        users = users.merge(
            user_completed[time_cols_in_completed + ['user_id']],
            left_on='id',
            right_on='user_id',
            how='left'
        )

# List of tuples: (filter_function, kwargs_dict)
ENABLED_FILTERS = [
    (filter_completed_users, {"user_df": users}),
    (filter_by_dataset_and_condition, {"user_df": users, "dataset_name": DATASET, "condition": GROUP}),
    (filter_by_invalid_scores, {"user_df": users}),
    (filter_by_time, {"user_df": users, "time_df": user_completed}),
    (filter_by_missing_ml_kowledge, {"user_df": users}),
]

# Only add the filter_by_negative_times if total_exp_time column exists
if 'total_exp_time' in users.columns:
    ENABLED_FILTERS.append((filter_by_negative_times, {"user_df": users}))
else:
    print("Warning: Skipping filter_by_negative_times because 'total_exp_time' column is missing")

# Add the remaining filters
ENABLED_FILTERS.extend([
    # Uncomment filters as needed:
    # (filter_by_work_life_balance, {"user_df": users, "wlb_users": wlb_users}),
    # (filter_by_dummy_var_mentions, {"predictions_df": all_preds_df, "dummy_var_column": DUMMY_VAR_COLUMN}),
    (filter_by_fatal_understanding_error,
     {"user_df": users, "study_group": "all", "question_type": "either"}),
    (remove_outliers_by_attention_check, {"user_df": users, "user_completed_df": user_completed}),
    (filter_users_that_didnt_ask_questions_from_df,
     {"users_df": users, "questions_over_time_df": questions_over_time_df}),
])

# --- RUN FILTERS AND COLLECT USER IDS TO EXCLUDE ---
exclude_user_ids = set()
filter_results = []  # Store (filter_name, count, ids)
for filter_func, kwargs in ENABLED_FILTERS:
    try:
        print("_________________________")
        print(f"Running filter: {filter_func.__name__}")
        ids = filter_func(**kwargs)
        if ids is not None and len(ids) > 0:
            ids_set = set(ids)
            exclude_user_ids.update(ids_set)
            filter_results.append((filter_func.__name__, len(ids_set), ids_set))
        else:
            filter_results.append((filter_func.__name__, 0, set()))
    except KeyError as e:
        print(f"Filter {filter_func.__name__} failed: Missing column {e}")
        filter_results.append((filter_func.__name__, 'ERROR-MissingColumn', set()))
    except Exception as e:
        print(f"Filter {filter_func.__name__} failed: {e}")
        filter_results.append((filter_func.__name__, 'ERROR', set()))

# Print summary of filter results
print("\nFilter exclusion summary:")
for name, count, ids in filter_results:
    print(f"{name}: {count} users excluded")

print(f"\nTotal users to exclude: {len(exclude_user_ids)}")
print(f"Original users count: {len(users)}")

# --- CREATE DETAILED FILTER SUMMARY TABLE ---
# Create a comprehensive table showing which filters flagged each user
all_flagged_users = set()
for name, count, ids in filter_results:
    if isinstance(count, int) and count > 0:
        all_flagged_users.update(ids)

# Create detailed summary DataFrame
detailed_summary_data = []
for user_id in all_flagged_users:
    user_row = {"user_id": user_id}

    # Check which filters flagged this user
    for filter_name, count, user_ids in filter_results:
        if isinstance(count, int) and count > 0:
            user_row[filter_name] = 1 if user_id in user_ids else 0
        else:
            user_row[filter_name] = 0

    detailed_summary_data.append(user_row)

if detailed_summary_data:
    detailed_summary_df = pd.DataFrame(detailed_summary_data)

    # Add a column showing total filters failed per user
    filter_columns = [col for col in detailed_summary_df.columns if col != 'user_id']
    detailed_summary_df['total_filters_failed'] = detailed_summary_df[filter_columns].sum(axis=1)

    # Sort by total filters failed (descending), then by user_id for consistency
    detailed_summary_df = detailed_summary_df.sort_values(['total_filters_failed', 'user_id'], ascending=[False, True])

    # Drop the helper column for the final output
    detailed_summary_df = detailed_summary_df.drop(columns=['total_filters_failed'])

    # Add summary row showing totals
    summary_row = {"user_id": "TOTAL_UNIQUE_USERS"}
    for filter_name, count, user_ids in filter_results:
        if isinstance(count, int):
            summary_row[filter_name] = count
        else:
            summary_row[filter_name] = 0

    # Add total unique users count
    summary_row["user_id"] = f"TOTAL: {len(all_flagged_users)} unique users"

    # Create a DataFrame with the summary row
    summary_df = pd.DataFrame([summary_row])

    # Combine detailed data with summary
    filter_detail_df = pd.concat([detailed_summary_df, summary_df], ignore_index=True)

    # Save detailed filter summary
    filter_detail_df.to_csv(os.path.join(FILTERED_DIR, "filter_detail_summary.csv"), index=False)

    print(f"\nDetailed filter summary (showing first 10 users):")
    print(filter_detail_df.head(10).to_string(index=False))
    if len(filter_detail_df) > 11:  # 10 users + 1 summary row
        print("...")
        print(filter_detail_df.tail(1).to_string(index=False, header=False))

    print(f"\nSaved detailed filter breakdown to filter_detail_summary.csv")
else:
    print("\nNo users were flagged by any filters.")
    # Still create an empty summary file for consistency
    empty_summary = pd.DataFrame({"user_id": ["TOTAL: 0 unique users"], **{name: [0] for name, _, _ in filter_results}})
    empty_summary.to_csv(os.path.join(FILTERED_DIR, "filter_detail_summary.csv"), index=False)

# --- FILTER DATAFRAMES USING UTILITY FUNCTION ---
dataframes_to_filter = {
    'users': users,
    'events': events,
    'user_completed': user_completed,
    'predictions': all_preds_df,
    'questions': questions_over_time_df
}

filtered_dataframes = filter_dataframes_by_user_ids(dataframes_to_filter, exclude_user_ids)

# Extract filtered DataFrames
users_filtered = filtered_dataframes['users']
events_filtered = filtered_dataframes['events']
user_completed_filtered = filtered_dataframes['user_completed']
all_preds_filtered = filtered_dataframes['predictions']
questions_filtered = filtered_dataframes['questions']

# --- SAVE FILTERED DATA USING UTILITY FUNCTION ---
print(f"\nSaving filtered data to {FILTERED_DIR}")

# Prepare DataFrames for saving (exclude None values)
dataframes_to_save = {
    'users_filtered': users_filtered,
    'events_filtered': events_filtered,
    'user_completed_filtered': user_completed_filtered,
    'predictions_filtered': all_preds_filtered
}

# Only add questions if it exists
if questions_filtered is not None and not questions_filtered.empty:
    dataframes_to_save['questions_filtered'] = questions_filtered

print("Saved filtered data files:")
# Check for duplicated user_ids and print those DataFrames and user_ids (only for user_df and user_completed_df)
for name, df in dataframes_to_save.items():
    if name in ['users_filtered', 'user_completed_filtered'] and df is not None and not df.empty:
        if 'user_id' in df.columns:
            duplicated_user_ids = df[df['user_id'].duplicated()]['user_id'].unique()
            if len(duplicated_user_ids) > 0:
                print(f"Duplicated user_ids found in {name}: {duplicated_user_ids}")
        print(f"- {name}.csv with {len(df)} rows")
save_dataframes_to_csv(dataframes_to_save, FILTERED_DIR)

# Save simple filter summary for quick reference
filter_summary = pd.DataFrame([
    {"filter_name": name, "excluded_count": count if count != 'ERROR' and count != 'ERROR-MissingColumn' else 0,
     "status": "SUCCESS" if count not in ['ERROR', 'ERROR-MissingColumn'] else count}
    for name, count, ids in filter_results
])
filter_summary.to_csv(os.path.join(FILTERED_DIR, "filter_summary.csv"), index=False)
print(f"- filter_summary.csv (simple filter execution log)")
print(f"- filter_detail_summary.csv (detailed user-by-filter breakdown)")

print(f"\nFiltering complete. Excluded {len(exclude_user_ids)} users from {len(users)} original users.")
