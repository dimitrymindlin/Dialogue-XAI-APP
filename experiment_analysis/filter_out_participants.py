import json

from experiment_analysis.plot_overviews import print_feedback_json


def remove_outliers_by_time(time_df, user_df, event_df):
    """
    Removes outliers based on total_time by excluding participants who are more than 3 standard deviations from the mean of their study group.
    Filters `user_df` and `event_df` to only include remaining participants.

    Parameters:
    - time_df: DataFrame with columns 'user_id', 'start_time', 'end_time', and 'study_group'.
    - user_df: DataFrame of users with at least 'id'.
    - event_df: DataFrame of events with at least 'user_id'.

    Returns:
    - Tuple of DataFrames (time_df, user_df, event_df) after removing outliers.
    """
    # Calculate mean and standard deviation for total_time by study_group
    mean_time = time_df.groupby('study_group')['total_time'].transform('mean')
    std_time = time_df.groupby('study_group')['total_time'].transform('std')

    # Filter out outliers beyond 2 standard deviations
    within_3_std = ((time_df['total_time'] > (mean_time - 2 * std_time)) &
                    (time_df['total_time'] < (mean_time + 2 * std_time)))
    filtered_time_df = time_df[within_3_std]

    # Filter user_df and event_df based on remaining users in filtered_time_df
    remaining_users = filtered_time_df["user_id"].unique()
    filtered_user_df = user_df[user_df["id"].isin(remaining_users)]
    filtered_event_df = event_df[event_df["user_id"].isin(remaining_users)]

    print(f"Mean time per study group: {time_df.groupby('study_group')['total_time'].mean()}")
    print(f"Standard deviation per study group: {time_df.groupby('study_group')['total_time'].std()}")
    print("Remaining users after removing by time: ", len(filtered_user_df))

    return filtered_time_df, filtered_user_df, filtered_event_df



def remove_outliers_by_attention_check(user_df, user_completed_df):
    # Add new column "failed_checks" to user_df with value 0
    user_df["failed_checks"] = 0

    for user_id in user_df["id"]:
        attention_checks = user_completed_df[user_completed_df["user_id"] == user_id]["attention_checks"].values[0]
        for check_id, check_result in attention_checks.items():
            # don't count first check because its comprehension check
            if check_id == "1":
                continue
            if check_result['correct'] != check_result['selected']:
                user_df.loc[user_df["id"] == user_id, "failed_checks"] += 1

    # Remove users that failed 2 attention_checks
    user_df = user_df[user_df["failed_checks"] < 2]
    print("Remaining users after removing by attention check: ", len(user_df))
    return user_df
