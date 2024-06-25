import psycopg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz

from experiment_analysis.analyse_explanation_ranking import extract_questionnaires
from experiment_analysis.analysis_data_holder import AnalysisDataHolder
from experiment_analysis.calculations import create_predictions_df
from experiment_analysis.filter_out_participants import filter_by_prolific_users, filter_by_broken_variables
from experiment_analysis.plot_overviews import plot_understanding_over_time, \
    plot_asked_questions_per_user
from experiment_analysis.process_mining import ProcessMining
import json

from parsing.llm_intent_recognition.prompts.explanations_prompt import question_to_id_mapping

POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "example"
POSTGRES_DB = "prolific_study"
POSTGRES_HOST = "localhost"

analysis_steps = ["filter_by_prolific_users",
                  "filter_completed_users",
                  "filter_by_attention_check",
                  "filter_by_time"]


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


def append_user_data_to_df(df, user_id, study_group, data, column_name="time"):
    if (df['user_id'] == user_id).any():
        # If the user exists, update the existing row
        df.loc[df['user_id'] == user_id, column_name] = data
    else:
        # If the user does not exist, create a new row
        new_row = pd.DataFrame([{"user_id": user_id, "study_group": study_group, column_name: data}])
        df = pd.concat([df, new_row], ignore_index=True)
    return df


def plot_pairplot(df, vars, hue, title):
    sns.pairplot(df, vars=vars, hue=hue, diag_kind="hist", kind="hist")
    plt.suptitle(title, y=1.02)  # y is a float that adjusts the position of the title vertically
    plt.show()


def plot_heatmap(df, cols, title):
    df = df[cols]  # Select only the columns in cols
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def extract_exit_feedback(user_df, user_id):
    exit_q = None
    try:
        questionnaires = user_df[user_df["id"] == user_id]["questionnaires"].values[0]
        # Remove redundant keys and values
        if len(questionnaires) > 4:
            found_exit = False
            for questionnaire in questionnaires:
                if isinstance(questionnaire, dict):
                    continue
                questionnaire = json.loads(questionnaire)
                if "exit" in questionnaire.keys():
                    exit_q = questionnaire['exit']
                    found_exit = True
            if not found_exit:
                print("no exit q.", len(questionnaires), user_id)
        else:
            exit_questionnaire = json.loads(questionnaires[3])['exit']
            exit_q = exit_questionnaire
    except (KeyError, IndexError):
        print(f"{user_id} did not complete the exit questionnaire.")

    # Append the user_id and exit_q to the DataFrame
    new_row = pd.DataFrame([{"user_id": user_id, "exit_q_answers": exit_q['answers']}])
    return new_row


def get_study_group_and_events(user_df, event_df, user_id):
    study_group = user_df[user_df["id"] == user_id]["study_group"].values[0]
    user_events = event_df[event_df["user_id"] == user_id]
    return study_group, user_events


def extract_questions(user_events):
    """
    Process the interactive group by extracting user questions over time.
    """
    user_questions_over_time = user_events[
        (user_events["action"] == "question") & (user_events["source"] == "teaching")]
    # Check how many users have not asked any questions (i.e. all cols contain nan)
    if user_questions_over_time.empty:
        return None

    return user_questions_over_time


def analyse_user_questions(questions_df):
    id_to_question_mapping = {v: k for k, v in question_to_id_mapping.items()}
    # For each user_id, get the rows and inspect the details column
    for user_id in questions_df["user_id"].unique():
        print(user_id)
        user_questions = questions_df[questions_df["user_id"] == user_id]
        print(f"User: {user_id}")
        datapoint = 0
        for index, row in user_questions.iterrows():
            details_dict = json.loads(row["details"])
            current_datapoint = details_dict["datapoint_count"]
            if datapoint != current_datapoint:
                print(f"Datapoint: {current_datapoint}")
                datapoint = current_datapoint
            print(f"Question: {details_dict['question']}")
            print(f"Answer: {id_to_question_mapping[int(details_dict['question_id'].split(',')[0])]}")
    print()


def main():
    conn = connect_to_db()
    analysis = AnalysisDataHolder(user_df=fetch_data_as_dataframe("SELECT * FROM users", conn),
                                  event_df=fetch_data_as_dataframe("SELECT * FROM events", conn),
                                  user_completed_df=fetch_data_as_dataframe("SELECT * FROM user_completed", conn))
    # if "filter_by_prolific_users" in analysis_steps:
    filter_by_prolific_users(analysis)
    analysis.create_time_columns()

    print("Found users: ", len(analysis.user_df))
    print("Found events: ", len(analysis.event_df))

    ### Filtering
    filter_by_broken_variables(analysis)
    print("Amount of users per study group after broken variables filter:")
    print(analysis.user_df.groupby("study_group").size())

    user_questions_over_time_list = []
    initial_test_preds_list = []
    learning_test_preds_list = []
    final_test_preds_list = []
    final_q_feedback_list = []
    exclude_user_ids = []
    wlb_users = []
    for user_id in analysis.user_df["id"]:
        study_group, user_events = get_study_group_and_events(analysis.user_df, analysis.event_df, user_id)
        # Create predictions dfs and if there is a user with missing or broken data, exclude them
        # check if user_events is empty
        if user_events.empty:
            exclude_user_ids.append(user_id)
            continue

        # check if source teaching handle next is 10 times
        if len(user_events[user_events["source"] == "teaching"]) < 10:
            exclude_user_ids.append(user_id)
            continue

        if study_group == "interactive" or study_group == "chat":
            user_questions_over_time_df = extract_questions(user_events)
            if user_questions_over_time_df is not None:
                user_questions_over_time_list.append(user_questions_over_time_df)
            else:
                exclude_user_ids.append(user_id)

        intro_test_preds, preds_learning, final_test_preds, exclude = create_predictions_df(analysis.user_df,
                                                                                            user_events,
                                                                                            exclude_incomplete=True)
        if exclude:
            exclude_user_ids.append(user_id)
            continue
        initial_test_preds_list.append(intro_test_preds)
        learning_test_preds_list.append(preds_learning)
        final_test_preds_list.append(final_test_preds)

        # Check if the user has indicated work-life balance as an important variable for the prediction
        feedback = final_test_preds["feedback"].values
        for f in feedback:
            # Check if the similarity score is above a certain threshold, e.g., 80
            if fuzz.partial_ratio("worklifebalance", f.lower()) > 80 or fuzz.partial_ratio("work life balance",
                                                                                           f.lower()) > 80:
                wlb_users.append((user_id, f))

        # Get User Final Q. Feedback from interactive group
        exit_q_df = extract_exit_feedback(analysis.user_df, user_id)
        final_q_feedback_list.append(exit_q_df)

    # Update analysis dfs and exclude users
    print(f"Users excluded: {len(exclude_user_ids)}")
    analysis.update_dfs(exclude_user_ids)
    analysis.add_self_assessment_value_column()

    # Save wlb_users to a csv file for manual analysis
    wlb_users_df = pd.DataFrame(wlb_users, columns=["user_id", "feedback"])
    wlb_users_df.to_csv("data/wlb_users.csv", index=False)
    # get wlb user ids and exclude them
    wlb_user_ids = wlb_users_df["user_id"].values
    analysis.update_dfs(wlb_user_ids)

    print("Amount of users per study group after first filters:")
    print(analysis.user_df.groupby("study_group").size())

    # perform_power_analysis()
    # filter_by_wanted_count(analysis, 70)

    ### Add Additional Columns to analysis.user_df
    # Calculate total score improvement and confidence improvement
    # analysis.user_df["score_improvement"] = analysis.user_df["final_score"] - analysis.user_df["intro_score"]
    analysis.user_df["confidence_avg_improvement"] = analysis.user_df["final_avg_confidence"] - analysis.user_df[
        "intro_avg_confidence"]

    # Merge final_q_feedback_list to analysis.user_df on user_id
    analysis.user_df = analysis.user_df.merge(pd.concat(final_q_feedback_list), left_on="id", right_on="user_id",
                                              how="left")

    initial_test_preds_df = pd.concat(initial_test_preds_list)
    learning_test_preds_df = pd.concat(learning_test_preds_list)
    analysis.add_initial_test_preds_df(initial_test_preds_df)
    analysis.add_learning_test_preds_df(learning_test_preds_df)
    analysis.add_final_test_preds_df(pd.concat(final_test_preds_list))
    analysis.add_questions_over_time_df(pd.concat(user_questions_over_time_list))

    # Update all dfs
    analysis.update_dfs()

    assert len(analysis.user_df) == len(analysis.initial_test_preds_df['user_id'].unique())
    assert len(analysis.user_df) == len(analysis.learning_test_preds_df['user_id'].unique())

    user_accuracy_over_time_df = plot_understanding_over_time(analysis.learning_test_preds_df, analysis)
    assert len(analysis.user_df) == len(user_accuracy_over_time_df['user_id'].unique())

    # Extract questionnaires into columns
    #analysis.user_df = extract_questionnaires(analysis.user_df)

    # Save the dfs to csv
    folder_path = "data/"
    """analysis.user_df.to_csv(folder_path + "user_df.csv", index=False)
    analysis.event_df.to_csv(folder_path + "event_df.csv", index=False)
    analysis.user_completed_df.to_csv(folder_path + "user_completed_df.csv", index=False)
    analysis.initial_test_preds_df.to_csv(folder_path + "initial_test_preds_df.csv", index=False)
    analysis.learning_test_preds_df.to_csv(folder_path + "learning_test_preds_df.csv", index=False)
    analysis.final_test_preds_df.to_csv(folder_path + "final_test_preds_df.csv", index=False)
    analysis.questions_over_time_df.to_csv(folder_path + "questions_over_time_df.csv", index=False)
    user_accuracy_over_time_df.to_csv(folder_path + "user_accuracy_over_time_df.csv", index=False)"""

    print("Amount of users per study group after All filters:")
    print(analysis.user_df.groupby("study_group").size())

    # Get demographics of the users
    user_prolific_ids = analysis.user_df["prolific_id"].values
    prolific_df = pd.read_csv("prolific_export.csv")
    prolific_df = prolific_df[prolific_df["Participant id"].isin(user_prolific_ids)]
    # Get Age statistics
    # ignore revoked from analysis for now
    prolific_df = prolific_df[prolific_df["Age"] != "CONSENT_REVOKED"]
    # turn age to int
    prolific_df["Age"] = prolific_df["Age"].astype(int)
    print(prolific_df["Age"].describe())
    print()
    # Get Sex statistics
    print(prolific_df["Sex"].value_counts())
    print(prolific_df["Nationality"].value_counts())
    # Turn mL knowledge to int
    analysis.user_df["ml_knowledge"] = analysis.user_df["ml_knowledge"].astype(int)
    print(analysis.user_df["ml_knowledge"].value_counts())
    print()

    # Get top 10 people based on final score with their prolific id
    # print(analysis.user_df[["prolific_id", "final_score"]].sort_values("final_score", ascending=False).head(10))

    ### LOOK AT ANALYSIS
    if "plot_question_raking" in analysis_steps:
        plot_asked_questions_per_user(analysis.questions_over_time_df, analysis)

    def print_correlation_ranking(target_var, group=None):
        # Make correlation df for user_df with each column against score_improvement
        if group is not None:
            correlation_df = analysis.user_df[analysis.user_df["study_group"] == group]
        correlation_df = correlation_df.corr()
        correlation_df = correlation_df[target_var].reset_index()
        correlation_df.columns = ["column", "correlation"]
        correlation_df = correlation_df.sort_values("correlation", ascending=False)
        print(f"Correlation ranking for {group}", target_var)
        print(correlation_df)

    def get_wort_and_best_users():
        score_name = "final_irt_score"
        top_5_percent_threshold = analysis.user_df[score_name].quantile(0.85)
        bottom_5_percent_threshold = analysis.user_df[score_name].quantile(0.15)

        # Filter best users and worst users
        best_users = analysis.user_df[analysis.user_df["final_score"] >= top_5_percent_threshold]
        worst_users = analysis.user_df[analysis.user_df["final_score"] <= bottom_5_percent_threshold]

        # Count best and worst users (unique user_ids)
        best_users_count = len(best_users["user_id"].unique())
        worst_users_count = len(worst_users["user_id"].unique())

        # Make sure to have same amount of best and worst users
        if worst_users_count > best_users_count:
            # Order by final_score and take the worst users
            worst_users = worst_users.sort_values("final_score", ascending=True)
            worst_users = worst_users.head(best_users_count)
        else:
            best_users = best_users.sort_values("final_score", ascending=False)
            best_users = best_users.head(worst_users_count)
        print("Best users: ", len(best_users["user_id"].unique()))
        print("Worst users: ", len(worst_users["user_id"].unique()))
        best_users_ids = best_users[["user_id"]]
        worst_users_ids = worst_users[["user_id"]]
        return best_users_ids, worst_users_ids

    analysis.questions_over_time_df = analysis.questions_over_time_df.merge(
        analysis.user_df[["id", "final_score"]],
        left_on="user_id", right_on="id")
    best_user_ids, worst_user_ids = get_wort_and_best_users()
    # Get questions_over_time_df for best and worst users
    best_users = analysis.questions_over_time_df[
        analysis.questions_over_time_df["user_id"].isin(best_user_ids["user_id"])]
    worst_users = analysis.questions_over_time_df[
        analysis.questions_over_time_df["user_id"].isin(worst_user_ids["user_id"])]

    pm = ProcessMining()
    pm.create_pm_csv(analysis,
                     datapoint_count=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     target_user_ids=best_users,
                     target_group_name="best")


if __name__ == "__main__":
    main()
