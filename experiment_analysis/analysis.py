import json

import psycopg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from experiment_analysis.calculations import calculate_user_score, get_user_feedback_per_data_point
from experiment_analysis.correlation_analysis import prepare_correlation_df
from experiment_analysis.feedback_buttons import get_button_feedback
from experiment_analysis.filter_out_participants import remove_outliers_by_attention_check, remove_outliers_by_time
from experiment_analysis.plot_overviews import plot_chatbot_feedback, plot_understanding_over_time, \
    plot_asked_questions_per_user, plot_understanding_with_questions, plot_question_raking
from experiment_analysis.process_mining import ProcessMining

POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "example"
POSTGRES_DB = "final_study_results"
POSTGRES_HOST = "localhost"

analysis_steps = ["filter_completed_users",
                  "filter_by_attention_check",
                  "filter_by_time"
                  ]


def connect_to_db():
    return psycopg2.connect(
        f"dbname={POSTGRES_DB} user={POSTGRES_USER} host={POSTGRES_HOST} password={POSTGRES_PASSWORD}"
    )


def fetch_data_as_dataframe(query, connection):
    cur = connection.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    column_names = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=column_names)


def append_user_data_to_df(df, user_id, study_group, data, data_type="time"):
    if data_type == "time":
        df = df.append({"user_id": user_id, "study_group": study_group, "total_time": data}, ignore_index=True)
    else:  # score
        df = df.append({"user_id": user_id, "study_group": study_group, "score": data}, ignore_index=True)
    return df


def plot_data(df, x_label, y_label, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_label, y=y_label, data=df)
    sns.stripplot(x=x_label, y=y_label, data=df, color=".25")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def prepare_time_dataframe(user_df, event_df):
    """
    Prepares a DataFrame with start and end times, and total time spent, for each user.

    :param user_df: DataFrame containing user data, including 'id' and 'created_at' columns.
    :param event_df: DataFrame containing event data, with 'user_id' and 'created_at' columns.
    :return: DataFrame with user IDs, start and end times, study group, and total time spent.
    """
    # Apply the provided function to remove outliers based on attention check
    remaining_users = user_df["id"].unique()
    print(f"Remaining users after removing by attention check: {len(remaining_users)}")

    # Prepare initial time_df
    time_df = pd.DataFrame({
        "user_id": user_df["id"],
        "start_time": user_df["created_at"],
        "study_group": user_df["study_group"]
    })

    # Get end time for each user from event_df
    end_time_df = event_df.groupby("user_id")["created_at"].max()
    time_df["end_time"] = time_df["user_id"].map(end_time_df)

    # Ensure start_time and end_time are in datetime format
    time_df['start_time'] = pd.to_datetime(time_df['start_time'])
    time_df['end_time'] = pd.to_datetime(time_df['end_time'])

    # Calculate total_time in minutes
    time_df["total_time"] = (time_df["end_time"] - time_df["start_time"]).dt.total_seconds() / 60

    return time_df


def extract_exit_feedback(user_df, user_id, exit_feedback_answers_dicts):
    exit_questionnaire = json.loads(user_df[user_df["id"] == user_id]["questionnaires"].values[0][3])['exit']
    exit_feedback_answers_dicts.append(exit_questionnaire)


def main():
    conn = connect_to_db()
    user_df = fetch_data_as_dataframe("SELECT * FROM users", conn)
    event_df = fetch_data_as_dataframe("SELECT * FROM events", conn)
    print("Found users: ", len(user_df))
    print("Found events: ", len(event_df))

    if "filter_completed_users" in analysis_steps:
        user_df = user_df[user_df["completed"] == True]
        print("Completed users: ", len(user_df))

    if "filter_by_attention_check" in analysis_steps:
        user_df = remove_outliers_by_attention_check(user_df)

    if "filter_by_time" in analysis_steps:
        time_df = prepare_time_dataframe(user_df, event_df)
        time_df, user_df, event_df = remove_outliers_by_time(time_df, user_df, event_df)

    if "print_question_feedback" in analysis_steps:
        # Get All Feedback button feedback per question
        print(get_button_feedback(event_df))

    score_df = pd.DataFrame(columns=["user_id", "study_group", "score"])
    final_test_feedback_df = pd.DataFrame(columns=["user_id", "study_group", "feedback"])

    exit_feedback_answers_dicts = []
    user_predictions_over_time_list_dict = {}
    user_questions_over_time_list_dict = {}
    for user_id in user_df["id"]:
        # Extract Study Group and User Events
        study_group = user_df[user_df["id"] == user_id]["study_group"].values[0]
        user_events = event_df[event_df["user_id"] == user_id]

        # Extract user predictions and feedback
        user_predictions = user_events[
            (user_events["action"] == "user_prediction") & (user_events["source"] == "final-test")]
        user_final_test_feedback_per_data_point = get_user_feedback_per_data_point(user_predictions)
        score = calculate_user_score(user_predictions)
        user_predictions_over_time = user_events[
            (user_events["action"] == "user_prediction") & (user_events["source"] == "test")]
        if study_group == "interactive":
            user_questions_over_time = user_events[
                (user_events["action"] == "question") & (user_events["source"] == "teaching")]
            try:
                user_questions_over_time_list_dict[study_group].append(user_questions_over_time)
            except KeyError:
                user_questions_over_time_list_dict[study_group] = [user_questions_over_time]
        try:
            user_predictions_over_time_list_dict[study_group].append(user_predictions_over_time)
        except KeyError:
            user_predictions_over_time_list_dict[study_group] = [user_predictions_over_time]
        score_df = append_user_data_to_df(score_df, user_id, study_group, score, "score")
        if not len(user_final_test_feedback_per_data_point) == 0:
            final_test_feedback_df = append_user_data_to_df(final_test_feedback_df, user_id, study_group,
                                                            user_final_test_feedback_per_data_point, "feedback")
        print(final_test_feedback_df)

        # Get User Chatbot Feedback from interactive group
        if study_group == "interactive":
            extract_exit_feedback(user_df, user_id, exit_feedback_answers_dicts)

    # plot_chatbot_feedback(exit_feedback_answers_dicts)
    # Save final_test_feedback_df["score"] as csv
    final_test_feedback_df['score'].to_csv("final_test_feedback.csv", index=False)
    plot_data(time_df, "study_group", "total_time", 'Boxplot of Time Spent per Study Group')
    # plot_data(score_df, "study_group", "score", 'Barplot of User Scores per Study Group')
    user_end_score_dict = score_df.groupby("user_id")["score"].max().to_dict()
    # plot_understanding_over_time(user_predictions_over_time_list_dict['static'], "static", user_end_score_dict)
    understanding_df, understanding_matrix, user_ids_u = plot_understanding_over_time(
        user_predictions_over_time_list_dict['interactive'],
        "interactive", user_end_score_dict)

    questions_matrix, user_ids_q = plot_asked_questions_per_user(user_questions_over_time_list_dict['interactive'],
                                                                 user_end_score_dict)

    pm = ProcessMining()
    pm.create_pm_csv(event_df, understanding_df, datapoint_count=5, filter_by_score="best")

    # plot_understanding_with_questions(understanding_matrix, questions_matrix, user_ids_u, user_ids_q)

    # plot_question_raking(questions_matrix)

    ### Create Dataframe with Personal information, questions asked and the final understanding score
    # TODO:
    #corr_df = prepare_correlation_df(user_df, event_df, understanding_df)


if __name__ == "__main__":
    main()
