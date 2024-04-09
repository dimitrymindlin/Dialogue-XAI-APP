import json

from experiment_analysis.plot_overviews import print_feedback_json


def remove_outliers_by_time(time_df, user_df, event_df):
    """
    Removes outliers based on total_time and filters user_df and event_df to only include remaining users.

    Parameters:
    - df: DataFrame with start_time, end_time, and study_group columns.
    - user_df: DataFrame of users.
    - event_df: DataFrame of events.

    Returns:
    - Tuple of DataFrames (df, user_df, event_df) after removing outliers.
    """
    # Calculate mean and standard deviation for total_time by study_group
    mean_time = time_df.groupby('study_group')['total_time'].transform('mean')
    std_time = time_df.groupby('study_group')['total_time'].transform('std')

    # Filter out outliers
    time_df = time_df[((time_df['total_time'] > (mean_time - 2 * std_time)) &
                       (time_df['total_time'] < (mean_time + 2 * std_time)))]

    # Filter user_df and event_df based on remaining users in df
    remaining_users = time_df["user_id"].unique()
    user_df = user_df[user_df["id"].isin(remaining_users)]
    event_df = event_df[event_df["user_id"].isin(remaining_users)]
    print("Remaining users after removing by time: ", len(remaining_users))
    return time_df, user_df, event_df


def remove_outliers_by_attention_check(user_df):
    # Add new column "attention_1_passed" to user_df
    user_df = user_df.assign(attention_1_passed=False)

    # Add new column "attention_2_passed" to user_df
    user_df = user_df.assign(attention_2_passed=False)

    for user_id in user_df["id"]:
        # Get questionnaires list
        questionnaires = user_df[user_df["id"] == user_id]["questionnaires"].values[0]
        self_assesment_attention_check = json.loads(questionnaires[2])
        # check if -2 is selected for question ''I can select minus two for this question.'
        attention_check1_question_id = 3
        attention_check2_question_id = 13
        if self_assesment_attention_check['self_assessment']['answers'][attention_check1_question_id] == -2:
            user_df.loc[user_df["id"] == user_id, "attention_1_passed"] = True
        try:
            exit_questionnaire_attention_check = json.loads(questionnaires[3])
            if exit_questionnaire_attention_check['exit']['answers'][attention_check2_question_id] == -2:
                user_df.loc[user_df["id"] == user_id, "attention_2_passed"] = True
        except IndexError:
            # TODO: This is hacky. They did not pass it because they didn't answer the questionnaire at all.
            user_df.loc[user_df["id"] == user_id, "attention_2_passed"] = True

    print("Users that passed attention check1: ", len(user_df[user_df["attention_1_passed"] == True]), ", ",
          len(user_df[user_df["attention_1_passed"] == False]), "failed")
    print("Users that passed attention check2: ", len(user_df[user_df["attention_2_passed"] == True]), ", ",
          len(user_df[user_df["attention_2_passed"] == False]), "failed")
    # Remove users that failed attention check 1 or 2 or have "None" in attention_2_passed
    user_df = user_df[(user_df["attention_1_passed"] == True) & (user_df["attention_2_passed"] == True)]
    print("Remaining users after removing by attention check: ", len(user_df))
    return user_df
