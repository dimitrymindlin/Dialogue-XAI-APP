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

import os
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import analysis_config as config
from experiment_analysis.filter_users import (
    filter_completed_users,
    filter_by_dataset_and_condition,
    filter_by_time,
    filter_by_negative_times,
    remove_outliers_by_attention_check,
    filter_by_work_life_balance,
    filter_by_missing_ml_kowledge,
    filter_by_high_ml_knowledge,
    filter_by_self_defined_attention_check,
    filter_by_wanted_count,
    filter_users_that_didnt_ask_questions_from_df,
    filter_by_fatal_understanding_error,
    filter_by_invalid_scores,
    filter_users_not_in_original_data,
    filter_users_missing_self_assessment,
    filter_users_with_sorry_responses,
)
from experiment_analysis.analysis_utils import (
    filter_by_dummy_var_mentions,
    filter_dataframes_by_user_ids,
    save_dataframes_to_csv
)

UNPACKED_DIR = os.path.join(config.RESULTS_DIR, "2_unpacked")
FILTERED_DIR = os.path.join(config.RESULTS_DIR, "3_filtered")


DEFAULT_STEP3_FILTERS: List[str] = [
    "filter_completed_users",
    "filter_by_dataset_and_condition",
    "filter_by_invalid_scores",
    "filter_by_time",
    "filter_by_missing_ml_kowledge",
    "filter_by_negative_times",
    "filter_by_fatal_understanding_error",
    "remove_outliers_by_attention_check",
    "filter_users_that_didnt_ask_questions_from_df",
    "filter_users_missing_self_assessment",
]
def _normalise_filter_entries(entries: Iterable[Any]) -> List[Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            normalised.append({"name": entry, "options": {}})
        elif isinstance(entry, dict) and "name" in entry:
            options = entry.get("options") or {}
            normalised.append({"name": entry["name"], "options": options})
        else:
            print(f"Warning: Skipping invalid filter definition: {entry}")
    return normalised


def _build_filter(name: str, context: Dict[str, Any], options: Dict[str, Any]):
    users_df = context["users"]

    if name == "filter_completed_users":
        return filter_completed_users, {"user_df": users_df}

    if name == "filter_by_dataset_and_condition":
        dataset_name = options.get("dataset_name", config.DATASET)
        condition = options.get("condition", config.GROUP)
        return filter_by_dataset_and_condition, {"user_df": users_df, "dataset_name": dataset_name, "condition": condition}

    if name == "filter_by_invalid_scores":
        return filter_by_invalid_scores, {"user_df": users_df}

    if name == "filter_by_time":
        kwargs = {"user_df": users_df, "time_df": context["user_completed"]}
        if "std_dev_threshold" in options:
            kwargs["std_dev_threshold"] = options["std_dev_threshold"]
        return filter_by_time, kwargs

    if name == "filter_by_missing_ml_kowledge":
        return filter_by_missing_ml_kowledge, {"user_df": users_df}

    if name == "filter_by_high_ml_knowledge":
        column_name = options.get("column_name", "ml_knowledge")
        threshold = options.get("threshold", 2)
        return filter_by_high_ml_knowledge, {
            "user_df": users_df,
            "column_name": column_name,
            "threshold": threshold,
        }

    if name == "filter_by_negative_times":
        if "total_exp_time" not in users_df.columns:
            print("Warning: Skipping filter_by_negative_times because 'total_exp_time' column is missing")
            return None
        return filter_by_negative_times, {"user_df": users_df}

    if name == "filter_by_fatal_understanding_error":
        study_group = options.get("study_group", "all")
        question_type = options.get("question_type", "most_important_feature")
        return filter_by_fatal_understanding_error, {
            "user_df": users_df,
            "study_group": study_group,
            "question_type": question_type,
        }

    if name == "remove_outliers_by_attention_check":
        return remove_outliers_by_attention_check, {
            "user_df": users_df,
            "user_completed_df": context["user_completed"],
        }

    if name == "filter_users_that_didnt_ask_questions_from_df":
        questions_df = context.get("questions_over_time")
        if questions_df is None or questions_df.empty:
            print("Warning: Skipping filter_users_that_didnt_ask_questions_from_df because questions data is unavailable")
            return None
        return filter_users_that_didnt_ask_questions_from_df, {
            "users_df": users_df,
            "questions_over_time_df": questions_df,
        }

    if name == "filter_users_not_in_original_data":
        return filter_users_not_in_original_data, {"user_df": users_df}

    if name == "filter_users_missing_self_assessment":
        column_name = options.get("column_name", "self_assessment_answers")
        return filter_users_missing_self_assessment, {
            "user_df": users_df,
            "column_name": column_name,
        }

    if name == "filter_users_with_sorry_responses":
        events_df = context.get("events")
        if events_df is None or events_df.empty:
            print("Warning: Skipping filter_users_with_sorry_responses because events data is unavailable")
            return None
        return filter_users_with_sorry_responses, {
            "events_df": events_df,
            "user_column": options.get("user_column", "user_id"),
            "message_column": options.get("message_column", "message"),
            "actor_column": options.get("actor_column", "actor"),
        }

    if name == "filter_by_dummy_var_mentions":
        predictions_df = context.get("all_preds")
        if predictions_df is None or predictions_df.empty:
            print("Warning: Skipping filter_by_dummy_var_mentions because predictions dataframe is empty")
            return None
        dummy_column = options.get(
            "dummy_var_column",
            getattr(config, "DUMMY_VAR_COLUMN", "dummy_var_mention"),
        )
        return filter_by_dummy_var_mentions, {
            "predictions_df": predictions_df,
            "dummy_var_column": dummy_column,
        }

    if name == "filter_by_work_life_balance":
        predictions_df = context.get("all_preds")
        dummy_column = options.get(
            "dummy_var_column",
            getattr(config, "DUMMY_VAR_COLUMN", "dummy_var_mention"),
        )
        kwargs = {
            "user_df": users_df,
            "predictions_df": predictions_df,
            "dummy_var_column": dummy_column,
            "fallback_users": options.get("wlb_users"),
        }
        return filter_by_work_life_balance, kwargs

    if name == "filter_by_self_defined_attention_check":
        predictions_df = context.get("all_preds")
        dummy_column = options.get(
            "dummy_var_column",
            getattr(config, "DUMMY_VAR_COLUMN", "dummy_var_mention"),
        )
        kwargs = {
            "user_df": users_df,
            "predictions_df": predictions_df,
            "dummy_var_column": dummy_column,
            "wlb_users": options.get("wlb_users"),
        }
        return filter_by_self_defined_attention_check, kwargs

    if name == "filter_by_wanted_count":
        count = options.get("count", 70)
        return filter_by_wanted_count, {"user_df": users_df, "count": count}

    print(f"Warning: Unknown filter '{name}' requested; skipping")
    return None


def _build_filter_pipeline(context: Dict[str, Any]) -> List[Tuple[Any, Dict[str, Any]]]:
    configured_pipeline = getattr(config, "STEP3_FILTER_PIPELINE", None)
    additional_filters = getattr(config, "STEP3_ADDITIONAL_FILTERS", None)

    if configured_pipeline is not None:
        raw_entries: Iterable[Any] = configured_pipeline
    else:
        raw_entries = list(DEFAULT_STEP3_FILTERS)
        if additional_filters:
            raw_entries = list(raw_entries) + list(additional_filters)

    normalised_entries = _normalise_filter_entries(raw_entries)

    exclude_filters = {
        entry.lower()
        for entry in getattr(config, "STEP3_EXCLUDE_FILTERS", [])
        if isinstance(entry, str)
    }
    if exclude_filters:
        normalised_entries = [
            entry
            for entry in normalised_entries
            if entry["name"].lower() not in exclude_filters
        ]

    pipeline: List[Tuple[Any, Dict[str, Any]]] = []
    for entry in normalised_entries:
        filter_tuple = _build_filter(entry["name"], context, entry["options"])
        if filter_tuple:
            pipeline.append(filter_tuple)
    return pipeline

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

filter_context: Dict[str, Any] = {
    "users": users,
    "events": events,
    "user_completed": user_completed,
    "questions_over_time": questions_over_time_df,
    "all_preds": all_preds_df,
}

ENABLED_FILTERS = _build_filter_pipeline(filter_context)

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
