import json
from datetime import datetime
import pandas as pd
from experiment_analysis.plot_overviews import print_feedback_json
import numpy as np
from experiment_analysis.analysis_utils import add_prolific_times
from experiment_analysis.analysis_config import DATASET


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


def filter_by_work_life_balance(user_df, wlb_users):
    wlb_users = [user for user in wlb_users if user[0] in user_df["id"].values]
    # Add learning time to wlb_df based on user_id
    wlb_users = [
        (user[0], user[1], user_df[user_df["id"] == user[0]]["total_learning_time"].values[0])
        for user in wlb_users]
    # Save wlb users to a csv file
    wlb_df = pd.DataFrame(wlb_users, columns=["user_id", "feedback", "learning_time"])
    wlb_df = wlb_df.sort_values(by="learning_time", ascending=False)
    wlb_df.to_csv("wlb_users.csv", index=False)
    # Print unique remaining wlb users
    print("Unique remaining WLB users: ", wlb_df["user_id"].nunique())
    return wlb_df["user_id"].tolist()


def filter_by_missing_ml_kowledge(user_df):
    fam_ml_mapping = {
        "very low": 0,
        "low": 1,
        "moderate": 2,
        "high": 3,
        "very high": 4,
        "anonymous": None
    }

    def parse_ml_knowledge(profile):
        """Safely extract and map ML knowledge level from profile."""
        try:
            profile_data = json.loads(profile)
            fam_ml_val = profile_data.get("fam_ml_val", "")
            if fam_ml_val == "":
                return None
            # try to cast to int
            try:
                fam_ml_val = int(fam_ml_val)
            except ValueError:
                pass
            if isinstance(fam_ml_val, int):
                return fam_ml_val if 0 <= fam_ml_val <= 4 else None  # Ensure it's in range
            elif isinstance(fam_ml_val, str):  # String mapping needed
                return fam_ml_mapping.get(fam_ml_val, None)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None  # Handle unexpected cases

    # Apply function to extract ML knowledge
    user_df["ml_knowledge"] = user_df["profile"].apply(parse_ml_knowledge)

    # Filter users with missing ml_knowledge values
    missing_count = user_df["ml_knowledge"].isnull().sum()
    if missing_count > 0:
        users_to_remove = user_df[user_df["ml_knowledge"].isnull()]["id"]
        if not users_to_remove.empty:
            print(f"Missing ml_knowledge values: {missing_count}")
            return users_to_remove.tolist()
    return []


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


def filter_by_self_defined_attention_check(user_df, wlb_users):
    # Filter by users who used work-life balance as an important variable
    remove_user_ids = filter_by_work_life_balance(user_df, wlb_users)
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
