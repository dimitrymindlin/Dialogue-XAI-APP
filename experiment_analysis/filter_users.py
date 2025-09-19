import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from experiment_analysis.plot_overviews import print_feedback_json
import numpy as np
from experiment_analysis.analysis_utils import add_prolific_times, count_dummy_var_users
from experiment_analysis.analysis_config import DATASET, INTRO_TEST_CYCLES, TEACHING_CYCLES, FINAL_TEST_CYCLES
from experiment_analysis import analysis_config as config


def filter_by_time(user_df, time_df, std_dev_threshold=2):
    """
    Removes outliers based on total_time by excluding participants who are less than x standard deviations from the mean of their study group.
    This version only filters out participants who complete the study too quickly, ignoring those who take longer than average.

    Parameters:
    - time_df: DataFrame with columns 'user_id', 'start_time', 'end_time', 'study_group', and 'total_time'.

    Returns:
    - List of user IDs considered as outliers for being too fast.
    """
    # Add study group col to time_df if not present
    if 'study_group' not in time_df.columns:
        time_df['study_group'] = user_df.set_index('id').reindex(time_df['user_id'])['study_group'].values

    # Calculate mean and standard deviation for total_learning_time by study_group
    mean_time_learning = time_df.groupby('study_group')['total_learning_time'].transform('mean')
    std_time_learning = time_df.groupby('study_group')['total_learning_time'].transform('std')

    # Calculate mean and standard deviation for total_exp_time by study_group
    mean_time_exp = time_df.groupby('study_group')['total_exp_time'].transform('mean')
    std_time_exp = time_df.groupby('study_group')['total_exp_time'].transform('std')

    # Determine if each participant's total_time is less than the lower bound of allowed standard deviation range
    is_too_fast_learning = time_df['total_learning_time'] < (mean_time_learning - std_dev_threshold * std_time_learning)
    is_too_fast_exp = time_df['total_exp_time'] < (mean_time_exp - std_dev_threshold * std_time_exp)

    # Identify too fast participants
    too_fast_participants_learning = time_df[is_too_fast_learning]
    too_fast_participants_exp = time_df[is_too_fast_exp]

    # Get the IDs of users considered too fast
    too_fast_learning_ids = too_fast_participants_learning['user_id'].tolist()
    too_fast_exp_ids = too_fast_participants_exp['user_id'].tolist()

    if len(too_fast_learning_ids) > 0:
        print("Users too fast in learning phase: ", len(too_fast_learning_ids))
    if len(too_fast_exp_ids) > 0:
        print("Users too fast in whole experiment: ", len(too_fast_exp_ids))

    for user_id in too_fast_exp_ids:
        total_time = time_df.loc[time_df["user_id"] == user_id, "total_exp_time"].values[0]
        print(f"       User ID: {user_id}, Experiment Time: {total_time} minutes")

    for user_id in too_fast_learning_ids:
        total_time = time_df.loc[time_df["user_id"] == user_id, "total_learning_time"].values[0]
        print(f"       User ID: {user_id}, Learning Time: {total_time} minutes")

    exclude_user_ids = too_fast_learning_ids + too_fast_exp_ids
    return exclude_user_ids


def remove_outliers_by_attention_check(user_df, user_completed_df):
    # Add new column "failed_checks" to user_df with value 0
    user_df["failed_checks"] = 0

    for user_id in user_completed_df["user_id"]:
        attention_checks = user_completed_df[user_completed_df["user_id"] == user_id]["attention_checks"].values[0]
        # Turn attention_checks (strings) into a dict
        if isinstance(attention_checks, str):
            try:
                attention_checks = json.loads(attention_checks)
            except json.JSONDecodeError:
                # If it's not valid JSON, try ast.literal_eval
                try:
                    import ast
                    attention_checks = ast.literal_eval(attention_checks)
                except (SyntaxError, ValueError):
                    print(f"Warning: Could not parse attention_checks for user {user_id}: {attention_checks}")
                    continue  # Skip this user if we can't parse the checks

        # Skip users with no attention checks
        if not attention_checks or not isinstance(attention_checks, dict):
            continue

        for check_id, check_result in attention_checks.items():
            # don't count first check because its comprehension check
            if check_id == "1":
                continue
            if isinstance(check_result['correct'], str):
                if check_result['correct'] != check_result['selected']:
                    user_df.loc[user_df["id"] == user_id, "failed_checks"] += 1
            else:  # is a list
                if check_result['selected'] not in check_result['correct']:
                    user_df.loc[user_df["id"] == user_id, "failed_checks"] += 1

    # Get the IDs of users that failed 2 attention_checks
    failed_attention_check_ids = user_df.loc[user_df["failed_checks"] >= 2, 'id'].tolist()
    print("Users failed attention check: ", len(failed_attention_check_ids))

    # Get the prolific IDs of users that failed 2 attention_checks
    failed_attention_check_prolific_ids = user_df.loc[user_df["failed_checks"] >= 2, 'prolific_id'].tolist()

    # Print the id with their answers to the attention checks
    for user_id in failed_attention_check_ids:
        attention_checks = user_completed_df[user_completed_df["user_id"] == user_id]["attention_checks"].values[0]
        # Convert string to dict if needed
        if isinstance(attention_checks, str):
            try:
                attention_checks = json.loads(attention_checks)
            except json.JSONDecodeError:
                try:
                    import ast
                    attention_checks = ast.literal_eval(attention_checks)
                except (SyntaxError, ValueError):
                    pass  # Keep as string if parsing fails
        print(f"User ID: {user_id}, Attention Checks: {attention_checks}")

    return failed_attention_check_ids


def filter_completed_users(user_df):
    incomplete_user_ids = user_df.loc[user_df["completed"] == False, 'id'].tolist()
    print("Users that did not complete the study: ", len(incomplete_user_ids))
    return incomplete_user_ids


def filter_by_prolific_users(analysis, prolific_file_name):
    prolific_df = pd.read_csv(prolific_file_name)
    include_users_list = []
    non_prolific_user_ids = add_prolific_times(analysis.user_completed_df, prolific_df)
    exclude_users_list = ["a4e2fba8-6e9f-4001-88d2-f109a1f9acc6", "9a84f941-a140-4d47-8766-01dc4e70993e",
                          "8d9f18d9-db90-4ce6-ba44-719e242e51bf",
                          "5fab286a-90b7-4eab-93c0-c033e3667a8b",  # From here, because of knowledge too high
                          "f383429a-10ad-4986-8069-d00ee92865c1",
                          "0d0cac57-1b99-4437-9d52-bcdd4a8d607e",
                          "72f2ade2-58ef-4415-a949-bf4864ac8181",
                          # From here because "Sorry i cannot understand"
                          'ba51a07b-0d9b-4533-a6f6-0c15ebd12191',
                          'e7a5c102-8c1a-4d1e-a9e9-1bd4b0356fdf',
                          'f3ca4b1c-dbcc-455d-a2a5-a3f95dd8d9ae',
                          '7056ef0d-a408-4a28-b539-c08e5823f8ad',
                          'b59e936f-311e-40a1-a793-3fed5cd3e878',
                          'ea960cf7-6cfa-4d43-a48f-d59f8d069b6c',
                          '1b2551b8-d14b-4ea7-9039-c6b2673eadef',
                          '7ba157ae-753b-4169-9d2b-7f05c84dc79e',
                          'c800795b-c050-457d-8af3-29d1e30cf0e0',
                          '24cf145e-c985-4423-becd-bcb98e1373aa',
                          '45535fbf-68a3-4432-bea1-75c78271ecc9',
                          '3c30f681-7215-487c-abfa-61ef56c5b352',
                          '5b0ff77e-3ad2-4aa7-9a77-83ffae20daf3',
                          'f8cb2154-037e-4519-b569-e1648f82a8d1',
                          '61be6d32-5b0c-473c-abcc-bdc768432fd5',
                          'c87eae3b-359c-4723-8c3b-4a7bb334f01c',
                          '91dbbedb-9d53-4a93-8cf2-96f3929efac9',
                          '5cfa78c8-2505-4c3b-b5b1-185b1bf0e0d2']
    non_prolific_user_ids.extend(exclude_users_list)
    non_prolific_user_ids = [user_id for user_id in non_prolific_user_ids if user_id not in include_users_list]
    return non_prolific_user_ids


def _normalise_wlb_entries(entries: Iterable) -> List[Tuple[str, str]]:
    normalised: List[Tuple[str, str]] = []
    for entry in entries:
        if isinstance(entry, str):
            normalised.append((entry, ""))
        elif isinstance(entry, Sequence):
            if not entry:
                continue
            user_id = entry[0]
            feedback = entry[1] if len(entry) > 1 else ""
            normalised.append((str(user_id), str(feedback) if feedback is not None else ""))
        elif isinstance(entry, dict) and "user_id" in entry:
            normalised.append((str(entry["user_id"]), str(entry.get("feedback", ""))))
    return normalised


def _resolve_wlb_file(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    if candidate.is_absolute():
        return candidate

    experiment_dir = Path(getattr(config, "ACTIVE_EXPERIMENT_PATH", Path(__file__).parent))
    for base in (experiment_dir, experiment_dir.parent, Path.cwd()):
        resolved = base / path_str
        if resolved.exists():
            return resolved

    return candidate


def _load_wlb_users_from_file(path_str: str) -> List[Tuple[str, str]]:
    path = _resolve_wlb_file(path_str)
    if not path.exists():
        print(f"Warning: Work-life balance users file not found at {path}.")
        return []

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - fallback logging
        print(f"Warning: Failed to read work-life balance users file {path}: {exc}")
        return []

    if "user_id" not in df.columns:
        print(f"Warning: Work-life balance users file {path} missing 'user_id' column.")
        return []

    feedback_col = "feedback" if "feedback" in df.columns else None
    entries = []
    for _, row in df.iterrows():
        user_id = row.get("user_id")
        if pd.isna(user_id):
            continue
        feedback = row.get(feedback_col, "") if feedback_col else ""
        if pd.isna(feedback):
            feedback = ""
        entries.append((str(user_id), str(feedback)))
    return entries


def _get_configured_wlb_users(source_override: Iterable | None = None) -> List[Tuple[str, str]]:
    if source_override is not None:
        return _normalise_wlb_entries(source_override)

    configured_entries = getattr(config, "WORK_LIFE_BALANCE_USERS", [])
    if configured_entries:
        return _normalise_wlb_entries(configured_entries)

    source_file = getattr(config, "WORK_LIFE_BALANCE_USERS_FILE", None)
    if source_file:
        if isinstance(source_file, (list, tuple)):
            return _normalise_wlb_entries(source_file)
        return _load_wlb_users_from_file(source_file)

    return []


def filter_by_work_life_balance(
    user_df,
    predictions_df=None,
    dummy_var_column: str = "dummy_var_mention",
    fallback_users=None,
):
    entries: List[Tuple[str, str, str, Optional[int]]] = []  # (user_id, feedback, source, mention_count)

    if predictions_df is not None and dummy_var_column in predictions_df.columns:
        dummy_var_users = count_dummy_var_users(predictions_df, dummy_var_column)
        for user_id, feedback, mention_count in dummy_var_users:
            if pd.isna(user_id):
                continue
            entries.append(
                (str(user_id), str(feedback) if feedback is not None else "", "prediction", int(mention_count))
            )

    configured_entries = _get_configured_wlb_users(fallback_users)
    for user_id, feedback in configured_entries:
        entries.append((str(user_id), feedback, "config", None))

    if not entries:
        print("Work-life balance filter enabled but no users were identified. Skipping filter.")
        return []

    available_ids = set(user_df["id"].astype(str))
    filtered_entries = [entry for entry in entries if entry[0] in available_ids]
    if not filtered_entries:
        print("Work-life balance filter: identified users not present in dataset. Skipping filter.")
        return []

    augmented_entries = []
    for user_id, feedback, source, mention_count in filtered_entries:
        matching_rows = user_df[user_df["id"].astype(str) == user_id]
        learning_time = None
        if not matching_rows.empty and "total_learning_time" in matching_rows.columns:
            learning_time = matching_rows["total_learning_time"].iloc[0]
        augmented_entries.append((user_id, feedback, source, learning_time, mention_count))

    wlb_df = pd.DataFrame(
        augmented_entries,
        columns=["user_id", "feedback", "source", "learning_time", "mention_count"],
    )
    if wlb_df["learning_time"].notna().any():
        wlb_df = wlb_df.sort_values(by="learning_time", ascending=False, na_position="last")

    output_dir = Path(config.RESULTS_DIR) / "3_filtered"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "wlb_users.csv"
    wlb_df.to_csv(output_path, index=False)
    print(f"Saved work-life balance filter details to {output_path}")
    print("Unique remaining WLB users: ", wlb_df["user_id"].nunique())
    return wlb_df["user_id"].tolist()


def filter_users_not_in_original_data(user_df):
    """Exclude users that are not present in experiment `user_df.csv` if available."""

    experiment_dir = Path(getattr(config, "ACTIVE_EXPERIMENT_PATH", Path(".")))
    source_file = experiment_dir / "user_df.csv"

    if not source_file.exists():
        print(f"filter_users_not_in_original_data skipped: {source_file} not found")
        return []

    try:
        original_df = pd.read_csv(source_file)
    except Exception as exc:  # pragma: no cover - I/O safeguard
        print(f"Warning: Unable to read {source_file}: {exc}")
        return []

    if "id" not in original_df.columns:
        print(f"Warning: '{source_file}' missing required 'id' column. Skipping filter.")
        return []

    allowed_ids = set(original_df["id"].dropna().astype(str))
    current_ids = user_df["id"].dropna().astype(str)

    missing_mask = ~current_ids.isin(allowed_ids)
    missing_ids = current_ids[missing_mask].unique()

    if len(missing_ids) == 0:
        return []

    print(
        f"filter_users_not_in_original_data: excluding {len(missing_ids)} users not present in {source_file.name}"
    )
    return missing_ids.tolist()


def filter_users_missing_self_assessment(user_df, column_name="self_assessment_answers"):
    """Exclude users lacking self-assessment answers."""

    if column_name not in user_df.columns:
        print(
            f"filter_users_missing_self_assessment skipped: '{column_name}' column not found in user data"
        )
        return []

    column = user_df[column_name]
    if column.notna().sum() == 0:
        print(
            "filter_users_missing_self_assessment skipped: no self-assessment data available for any user"
        )
        return []

    def has_answers(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        if isinstance(value, str):
            return value.strip() not in ("", "[]", "null", "None")
        if isinstance(value, (list, tuple)):
            return len(value) > 0
        return False

    missing_mask = ~column.apply(has_answers)
    missing_ids = user_df.loc[missing_mask, "id"].dropna().astype(str).tolist()

    print(
        f"filter_users_missing_self_assessment: excluding {len(missing_ids)} users missing self assessment"
    )
    return missing_ids


def filter_users_with_sorry_responses(
    events_df,
    user_column="user_id",
    message_column="message",
    actor_column="actor",
    phrases=None,
):
    """Exclude users whose system messages contain specific fallback apologies."""

    required_columns = {user_column, message_column}
    if not required_columns.issubset(events_df.columns):
        print(
            "filter_users_with_sorry_responses skipped: events dataframe missing required columns"
        )
        return []

    df = events_df[[user_column, message_column]].copy()

    actor_series = None
    for col in events_df.columns:
        if col.lower() == actor_column.lower():
            actor_series = events_df[col].astype(str).str.lower()
            break
    if actor_series is not None:
        df = df[actor_series.str.contains("system", na=False)]

    df["__message_lower"] = df[message_column].astype(str).str.lower()
    df["__message_lower"] = df["__message_lower"].str.replace("’", "'", regex=False)

    if phrases is None:
        phrases = ["sorry! i couldn't understand that."]
    phrases = [phrase.lower().replace("’", "'") for phrase in phrases]

    mask = False
    for phrase in phrases:
        mask = mask | df["__message_lower"].str.contains(phrase, na=False)

    flagged_users = df.loc[mask, user_column].dropna().astype(str).unique().tolist()

    if flagged_users:
        print(
            f"filter_users_with_sorry_responses: excluding {len(flagged_users)} users with apology messages"
        )

    return flagged_users


def filter_by_missing_ml_kowledge(user_df):
    """
    Filter users with missing ML knowledge values.
    
    Note: ml_knowledge column should already exist from step 2 (profile variable extraction).
    This function now only filters based on the existing column.
    """
    # Check if ml_knowledge column exists (it should from step 2)
    if "ml_knowledge" not in user_df.columns:
        print("Warning: ml_knowledge column not found. Skipping ML knowledge filter.")
        return []
    
    # Filter users with missing ml_knowledge values
    missing_count = user_df["ml_knowledge"].isnull().sum()
    if missing_count > 0:
        users_to_remove = user_df[user_df["ml_knowledge"].isnull()]["id"]
        if not users_to_remove.empty:
            print(f"Missing ml_knowledge values: {missing_count}")
            return users_to_remove.tolist()
    
    print(f"ML knowledge distribution:")
    if user_df["ml_knowledge"].notna().sum() > 0:
        value_counts = user_df["ml_knowledge"].value_counts().sort_index()
        for value, count in value_counts.items():
            print(f"  {value}: {count} users")
    
    return []


def filter_by_high_ml_knowledge(user_df, column_name="ml_knowledge", threshold=4):
    """Exclude users whose ML knowledge score exceeds ``threshold``."""

    if column_name not in user_df.columns:
        print(
            f"filter_by_high_ml_knowledge skipped: '{column_name}' column not found in user data"
        )
        return []

    try:
        scores = pd.to_numeric(user_df[column_name], errors="coerce")
    except Exception as exc:  # pragma: no cover - conversion safeguard
        print(f"filter_by_high_ml_knowledge: unable to parse '{column_name}' values ({exc}). Skipping filter")
        return []

    high_mask = scores > threshold
    high_ids = user_df.loc[high_mask, "id"].dropna().astype(str).tolist()

    if not high_ids:
        return []

    print(
        f"filter_by_high_ml_knowledge: excluding {len(high_ids)} users with {column_name} > {threshold}"
    )
    return high_ids


def filter_by_negative_times(user_df):
    """
    Checks if there are rows where total_time is negative in the DataFrame.

    :param user_df: DataFrame containing user data.
    """
    remove_user_ids = []

    # Simply check if the columns exist before trying to filter
    if "total_exp_time" not in user_df.columns:
        print("Warning: 'total_exp_time' column not found in user DataFrame. Skipping this check.")
        return []

    # Check if there are rows where total_exp_time is negative
    negative_exp_times = user_df[user_df["total_exp_time"] < 0]["id"]
    if not negative_exp_times.empty:
        remove_user_ids.extend(negative_exp_times.tolist())
        print(f"Found {len(negative_exp_times)} users with negative total_exp_time.")

    # Check if there are rows where exp_instruction_time is negative (if column exists)
    if "exp_instruction_time" in user_df.columns:
        negative_instruct_times = user_df[user_df["exp_instruction_time"] < 0]["id"]
        if not negative_instruct_times.empty:
            remove_user_ids.extend(negative_instruct_times.tolist())
            print(f"Found {len(negative_instruct_times)} users with negative exp_instruction_time.")

    # Remove duplicates
    remove_user_ids = list(set(remove_user_ids))
    if len(remove_user_ids) > 0:
        print(f"Total {len(remove_user_ids)} unique users with negative times.")

    return remove_user_ids


def filter_by_broken_variables(user_df, user_completed_df, event_df=None):
    """
    Exclude incomplete records where some variables cannot be inferred
    """
    exclude_user_ids = set()

    incomplete_user_ids = filter_completed_users(user_df)
    exclude_user_ids.update(incomplete_user_ids)

    failed_attention_check_ids = remove_outliers_by_attention_check(user_df, user_completed_df)
    exclude_user_ids.update(failed_attention_check_ids)

    too_fast_ids = filter_by_time(user_df, user_completed_df)
    exclude_user_ids.update(too_fast_ids)

    negative_times_ids = filter_by_negative_times(user_df)
    exclude_user_ids.update(negative_times_ids)

    missing_ml_knowledge_ids = filter_by_missing_ml_kowledge(user_df)
    exclude_user_ids.update(missing_ml_knowledge_ids)

    return list(exclude_user_ids)


def filter_by_self_defined_attention_check(user_df, predictions_df=None, dummy_var_column="dummy_var_mention", wlb_users=None):
    # Filter by users who used work-life balance as an important variable
    remove_user_ids = filter_by_work_life_balance(
        user_df,
        predictions_df=predictions_df,
        dummy_var_column=dummy_var_column,
        fallback_users=wlb_users,
    )
    return remove_user_ids


def filter_by_wanted_count(user_df, count=70):
    # Filter participants by time (Take first count participants) per study group, sorted by prolific_start_time
    sorted_user_ids = user_df.sort_values("prolific_start_time")["id"]
    # First, 70 interactive users
    interactive_ids = sorted_user_ids[
        sorted_user_ids.isin(user_df[user_df["study_group"] == "interactive"]["id"])]
    interactive_ids = interactive_ids[:count]
    # Then, 70 static users
    static_ids = sorted_user_ids[
        sorted_user_ids.isin(user_df[user_df["study_group"] == "static"]["id"])]
    static_ids = static_ids[:count]
    user_ids_to_remove = sorted_user_ids[~sorted_user_ids.isin(interactive_ids) & ~sorted_user_ids.isin(static_ids)]
    print("Amount of users per study group after time filter:")
    print(user_df.groupby("study_group").size())
    return user_ids_to_remove.tolist()


def filter_users_that_didnt_ask_questions_from_df(users_df, questions_over_time_df):
    """
    Identifies users who didn't ask any questions based on a pre-computed questions_over_time_df.

    Args:
        users_df (pd.DataFrame): DataFrame containing user information
        questions_over_time_df (pd.DataFrame): DataFrame containing user questions over time

    Returns:
        list: List of user IDs who didn't ask any questions
    """
    # Get all user IDs
    all_user_ids = set(users_df["id"].unique())

    # Get user IDs that have questions
    users_with_questions = set()
    if questions_over_time_df is not None and not questions_over_time_df.empty and "user_id" in questions_over_time_df.columns:
        users_with_questions = set(questions_over_time_df["user_id"].unique())

    # Find users without questions
    users_without_questions = list(all_user_ids - users_with_questions)

    if users_without_questions:
        print(f"Found {len(users_without_questions)} users who didn't ask questions:")
        for user_id in users_without_questions:
            print(f"       User ID: {user_id} did not ask any questions")
    else:
        print("All users asked at least one question")

    return users_without_questions


def filter_users_that_didnt_ask_questions(users_df, events_df):
    """
    Identifies users who didn't ask any questions during the experiment.

    Args:
        users_df (pd.DataFrame): DataFrame containing user information
        events_df (pd.DataFrame): DataFrame containing user events

    Returns:
        list: List of user IDs who didn't ask any questions
    """
    from experiment_analysis.analysis_utils import extract_questions, get_study_group_and_events

    # Build a questions_over_time_df for all users
    questions_over_time_list = []
    user_ids_processed = set()

    for user_id in users_df["id"].unique():
        try:
            study_group, user_events = get_study_group_and_events(users_df, events_df, user_id)
            user_questions_df = extract_questions(user_events, study_group)

            if user_questions_df is not None and not user_questions_df.empty:
                # Ensure user_id is in the DataFrame
                if "user_id" not in user_questions_df.columns:
                    user_questions_df["user_id"] = user_id
                questions_over_time_list.append(user_questions_df)
                user_ids_processed.add(user_id)
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")

    # Create consolidated questions_over_time_df
    questions_over_time_df = pd.DataFrame()
    if questions_over_time_list:
        questions_over_time_df = pd.concat(questions_over_time_list, ignore_index=True)

    # Use the new function to filter users
    return filter_users_that_didnt_ask_questions_from_df(users_df, questions_over_time_df)


def filter_by_fatal_understanding_error(user_df, study_group="all", question_type="both"):
    """
    Filters users who selected the fatal error option for understanding questions.
    
    Args:
        user_df (pd.DataFrame): DataFrame containing user information with understanding answer columns
        study_group (str): Study group to filter by, or "all" for all groups
        question_type (str): Type of question to check for fatal errors:
            - "most_important_feature" - only check most important feature question
            - "least_important_feature" - only check least important feature question
            - "both" - check both questions (default)
            - "either" - include users who failed either question
    
    Returns:
        list: List of user IDs who made fatal errors in the specified questions
    """
    # Filter by study group if specified
    if study_group != "all":
        df = user_df[user_df["study_group"] == study_group].copy()
    else:
        df = user_df.copy()

    # Column names for understanding question answers
    most_important_col = "understanding_q_most_important_feature_answer"
    least_important_col = "understanding_q_least_important_feature_answer"

    # Filter based on question type
    if question_type == "most_important_feature":
        fatal_users = df[df[most_important_col] == "fatal_wrong"]["id"].tolist()
    elif question_type == "least_important_feature":
        fatal_users = df[df[least_important_col] == "fatal_wrong"]["id"].tolist()
    elif question_type == "both":
        # Users who made fatal errors in BOTH questions
        fatal_users = df[(df[most_important_col] == "fatal_wrong") &
                         (df[least_important_col] == "fatal_wrong")]["id"].tolist()
    elif question_type == "either":
        # Users who made fatal errors in EITHER question
        fatal_users = df[(df[most_important_col] == "fatal_wrong") |
                         (df[least_important_col] == "fatal_wrong")]["id"].tolist()
    else:
        raise ValueError(
            f"Invalid question_type: {question_type}. Must be 'most_important_feature', 'least_important_feature', 'both', or 'either'")

    print(f"Found {len(fatal_users)} users who made fatal errors in {question_type} question(s)")
    return fatal_users


def filter_by_dataset_and_condition(user_df, dataset_name, condition="all"):
    """
    Return user_ids that are not in the specified dataset and need to be removed.

    Args:
        user_df (pd.DataFrame): DataFrame containing user information with 'profile' column.
        dataset_name (str): Name of the dataset to filter by.
        condition (str): The study group to filter by (e.g., "interactive", "static").

    Returns:
        list: List of user IDs who are not in the specified dataset.
    """
    # Convert 'profile' column to dictionary if it's a string
    profile_col = user_df["profile"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    # Extract 'study_group_name' from the profile column and split to get the dataset name
    user_df["dataset"] = profile_col.apply(
        lambda x: x.get("study_group_name", DATASET).split("-")[0] if isinstance(x, dict) else DATASET)

    # Extract condition from study group name (adult-thesis-mapek -> mapek)
    if "study_group" in user_df.columns:
        existing_study_groups = user_df["study_group"].copy()
    else:
        existing_study_groups = pd.Series([pd.NA] * len(user_df), index=user_df.index, dtype="object")

    user_df["study_group"] = [
        profile.get("study_group_name", "").split("-")[2]
        if isinstance(profile, dict)
        and "study_group_name" in profile
        and len(profile.get("study_group_name", "").split("-")) > 2
        else existing_study_groups.get(idx)
        for idx, profile in profile_col.items()
    ]

    # Identify users not in the specified dataset
    users_not_in_dataset = user_df[user_df['dataset'] != dataset_name]['id'].tolist()

    # Users not in group
    users_not_in_group = []
    if condition != "all":
        users_not_in_group = user_df[user_df['study_group'] != condition]['id'].tolist()

    # Print the count of users not in the dataset
    print(f"Found {len(users_not_in_dataset)} users who are not in the {dataset_name} dataset.")
    print(f"Found {len(users_not_in_group)} users who are not in the {condition} group.")

    # Combine both lists and remove duplicates
    users_not_in_dataset.extend(users_not_in_group)

    return users_not_in_dataset


def filter_by_invalid_scores(user_df):
    """
    Filter users with scores exceeding the maximum possible values defined in analysis_config.
    
    Checks:
    - intro_score <= INTRO_TEST_CYCLES
    - learning_score <= TEACHING_CYCLES  
    - final_score <= FINAL_TEST_CYCLES
    
    Args:
        user_df (pd.DataFrame): DataFrame containing user scores
        
    Returns:
        list: List of user IDs with invalid scores
    """
    invalid_users = []
    score_checks = [
        ('intro_score', INTRO_TEST_CYCLES),
        ('learning_score', TEACHING_CYCLES),
        ('final_score', FINAL_TEST_CYCLES)
    ]
    
    for score_col, max_score in score_checks:
        if score_col in user_df.columns:
            # Find users where score exceeds maximum
            invalid_mask = user_df[score_col] > max_score
            if invalid_mask.any():
                invalid_score_users = user_df[invalid_mask]
                invalid_user_ids = invalid_score_users['id'].tolist()
                
                print(f"Found {len(invalid_user_ids)} users with {score_col} > {max_score}:")
                for _, user in invalid_score_users.iterrows():
                    score_val = user[score_col]
                    if pd.notna(score_val):
                        print(f"  User {user['id']}: {score_col} = {score_val} (max allowed: {max_score})")
                
                invalid_users.extend(invalid_user_ids)
        else:
            print(f"Warning: {score_col} column not found in user_df")
    
    # Remove duplicates while preserving order
    invalid_users = list(dict.fromkeys(invalid_users))
    
    print(f"Total users with invalid scores: {len(invalid_users)}")
    return invalid_users
