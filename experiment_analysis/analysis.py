import os

import psycopg2
import pandas as pd
from fuzzywuzzy import fuzz

from experiment_analysis.analyse_final_understanding_q import get_users_failed_final_understanding_check
from experiment_analysis.analysis_data_holder import AnalysisDataHolder
from experiment_analysis.calculations import create_predictions_df
from experiment_analysis.filter_out_participants import filter_by_prolific_users, filter_by_broken_variables
from experiment_analysis.plot_overviews import plot_understanding_over_time
import json

from parsing.llm_intent_recognition.prompts.explanations_prompt import question_to_id_mapping

POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "example"
#POSTGRES_DB = "prolific_chat_final"
POSTGRES_DB = "adult"
POSTGRES_HOST = "localhost"
prolific_file_name_combined = "prolific_export_llm_combined.csv"
prolific_file_name1 = "prolific_export_llm_combined.csv"
#prolific_file_name2 = "prolific_export_chat_01.11.24.csv"
prolific_file_name2 = "prolific_export_very_last.csv"
result_folder_path = "data_chat/"


def merge_prolific_files():
    # Read the CSV files into dataframes
    df1 = pd.read_csv(prolific_file_name1)
    df2 = pd.read_csv(prolific_file_name2)

    # Add df2 rows to df1 without duplicate prolific_id
    df_combined = pd.concat([df1, df2]).drop_duplicates(subset="Participant id")

    """# Delete old combined file
    try:
        os.remove(prolific_file_name_combined)
    except FileNotFoundError:
        pass"""

    # Save the combined dataframe back to a CSV file
    df_combined.to_csv(prolific_file_name_combined, index=False)


analysis_steps = ["filter_by_prolific_users",
                  "filter_completed_users",
                  "filter_by_attention_check",
                  "filter_by_time",
                  "remove_users_that_didnt_ask_questions",
                  "remove_users_with_high_ml_knowledge",
                  "remove_users_with_fatal_error"]


# Possible steps: remove_users_with_high_ml_knowledge


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


def extract_all_feedback(user_df, user_id):
    try:
        questionnaires = user_df[user_df["id"] == user_id]["questionnaires"].values[0]
        feedback_dict = {"user_id": user_id}
        for questionnaire in questionnaires:
            if isinstance(questionnaire, dict):
                continue
            questionnaire = json.loads(questionnaire)
            for key, value in questionnaire.items():
                feedback_dict[f"{key}"] = value['questions']
                feedback_dict[f"{key}_answers"] = value['answers']
    except (KeyError, IndexError):
        print(f"{user_id} did not complete the questionnaires.")

    # Create a DataFrame from the feedback dictionary
    feedback_df = pd.DataFrame([feedback_dict])
    return feedback_df


def get_study_group_and_events(user_df, event_df, user_id):
    study_group = user_df[user_df["id"] == user_id]["study_group"].values[0]
    user_events = event_df[event_df["user_id"] == user_id]
    return study_group, user_events


def extract_questions(user_events, study_group):
    """
    Process the interactive group by extracting user questions over time.
    """
    user_questions_over_time = user_events[
        (user_events["action"] == "question") & (user_events["source"] == "teaching")]

    # Check how many users have not asked any questions (i.e. all cols contain nan)
    if user_questions_over_time.empty:
        return None

    ## UNPACK DETAILS COLUMN
    # Details is a string that contains a dictionary
    user_questions_over_time = user_questions_over_time.copy()
    user_questions_over_time.loc[:, 'details'] = user_questions_over_time['details'].apply(json.loads)
    details_df = user_questions_over_time['details'].apply(pd.Series)

    # Check is study group is chat or active chat
    if study_group in ["chat", "active_chat"]:
        if study_group == "chat":
            # Message is a string that contains a dictionary
            try:
                message_details = details_df['message'].apply(pd.Series)
                # Delete feedback column and isUser column
                message_details = message_details.drop(columns=['feedback', 'isUser', 'feature_id', 'id', 'followup'])
                # Concatenate the original DataFrame with the new columns
                details_df = details_df.drop(columns=['message'])
                user_questions_over_time = pd.concat(
                    [user_questions_over_time.drop(columns=['details']), details_df, message_details], axis=1)
            except KeyError:
                # No 'message' column found in details_df
                user_questions_over_time = pd.concat([user_questions_over_time.drop(columns=['details']), details_df],
                                                     axis=1)
                # Add 'user_question' column with the value "follow_up"
                user_questions_over_time.loc[:, "user_question"] = "follow_up"
        else:
            try:  # user used some free chat
                message_details = details_df['message'].apply(pd.Series)
                # Only keep columns "reasoning" and "text"
                message_details = message_details[['reasoning', 'text']]
                # Concatenate the original DataFrame with the new columns
                details_df = details_df.drop(columns=['message'])
                # for rows with methods ceterisParibus and featureStatistics, set "feature_id" to "infer_from_question"
                details_df.loc[details_df['question_id'].isin(
                    ["ceterisParibus", "featureStatistics"]), "feature_id"] = "infer_from_question"
                user_questions_over_time = pd.concat(
                    [user_questions_over_time.drop(columns=['details']), details_df, message_details], axis=1)
            except KeyError:  # User did not use free chat at all
                user_questions_over_time = pd.concat(
                    [user_questions_over_time.drop(columns=['details']), details_df], axis=1)
                # Add text col with None values
                user_questions_over_time.loc[:, "text"] = ""
            # rename question to clicked_question
            user_questions_over_time.rename(columns={"question": "clicked_question"}, inplace=True)

        # rename user_question to typed_question
        user_questions_over_time.rename(columns={"user_question": "typed_question"}, inplace=True)
        # Set 'id' as the index
        user_questions_over_time.set_index('id', inplace=True)
        # Replace nan in text col with ""
        try:
            user_questions_over_time["text"].fillna("", inplace=True)
        except KeyError:
            raise KeyError(
                "No 'text' column found in user_questions_over_time. Maybe you forgot to change change 'active_users' in pg admin?")
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
    #merge_prolific_files()
    conn = connect_to_db()
    analysis = AnalysisDataHolder(user_df=fetch_data_as_dataframe("SELECT * FROM users", conn),
                                  event_df=fetch_data_as_dataframe("SELECT * FROM events", conn),
                                  user_completed_df=fetch_data_as_dataframe("SELECT * FROM user_completed", conn))
    # if "filter_by_prolific_users" in analysis_steps:
    filter_by_prolific_users(analysis, prolific_file_name_combined)
    analysis.create_time_columns()

    """uuid_list = [
        "5b20033c-8c81-4fcf-8f98-1b57a0ab2e8f",
        "e9633076-9982-41cb-9e9d-ccc941ec7487",
        "5e8881d1-a3ae-49d6-8220-60926a3a8bc6",
        "64771ca5-17dd-4ac2-b302-1e9341fb3f83",
        "16a79c62-55f1-4e68-b4a4-b3e86609dad6",
        "35114667-d73c-4271-9b79-a61ba96f2099",
        "ea8992e7-4192-4f5e-bd81-92baab08effa",
        "9402f154-9297-4af6-b890-d25d44f0fa9c",
        "a09c056a-bfe7-401c-ba33-a776cffb74e7",
        "9a1d48f8-97d3-4c87-9026-a287fd53ff7f",
        "d117806b-b981-4eb1-ba45-265146526335",
        "fcb0b9b7-f6e9-43f1-a9c9-560681f52c89",
        "9f320cc2-8e2c-445c-af82-5dba8ae665ca",
        "34da3fab-5da8-4717-9cf3-98fa310398c1",
        "0036b2c7-7a21-4fd6-bdcf-d2ed542211bd"
    ]

    print(len(uuid_list))

    # Get prolific ids of the users in uuid_list
    prolific_ids = analysis.user_df[analysis.user_df["id"].isin(uuid_list)]["prolific_id"].values
    # print them line by line with a comma after the id and a 1 after the comma
    for id in prolific_ids:
        if id in ["64401708880c30ec28b9aef4",
                  "5f406d28899a9b1a9ddd609a",
                  "5f653cb18aad310a9ee7c32d",
                  "57acc170c6bab4000172a42e",
                  "665a086659c5d96abbd29687",
                  "65636b8039e7d18bb07ff7b9",
                  "5e9fd9d1ae42fb18d841f570"]:
            print(f"{id},1")"""

    print("Found users: ", len(analysis.user_df))
    print("Found events: ", len(analysis.event_df))
    print(analysis.user_df.groupby("study_group").size())

    ### Filtering
    filter_by_broken_variables(analysis)
    print("Amount of users per study group after broken variables filter:")
    print(analysis.user_df.groupby("study_group").size())

    user_questions_over_time_list = []
    initial_test_preds_list = []
    learning_test_preds_list = []
    final_test_preds_list = []
    all_q_list = []
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

        if study_group != "static":
            user_questions_over_time_df = extract_questions(user_events, study_group)
            if user_questions_over_time_df is not None:
                user_questions_over_time_list.append(user_questions_over_time_df)
            else:
                if "remove_users_that_didnt_ask_questions" in analysis_steps:
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
        # exit_q_df = extract_exit_feedback(analysis.user_df, user_id)

        all_questionnaires_df = extract_all_feedback(analysis.user_df, user_id)
        all_q_list.append(all_questionnaires_df)

    """# Check how many rows are there where "text" column starts with "Sorry" in user_questions_over_time_list
    count = 0
    user_id_set = set()
    for df in user_questions_over_time_list:
        try:
            total_rows = len(df)
            sorry_rows = df[df["text"].str.startswith("Sorry")]
            amount = len(sorry_rows)
            if amount / total_rows > 0.15:
                count += amount
                # get user id
                if amount > 0:
                    user_id_set.update(sorry_rows["user_id"])
        except ValueError:
            print("ValueError occurred in checking 'Sorry' rows.")

    print(f"Amount of questions starting with 'Sorry': {count}")
    print(f"Amount of users with 'Sorry': {len(user_id_set)}")

    # Get latest "created_at" of users in user_id_set
    latest_created_at = analysis.user_df[analysis.user_df["id"].isin(user_id_set)]["created_at"].max()
    # Get the user id of the user with the latest "created_at"
    latest_user_id = analysis.user_df[analysis.user_df["created_at"] == latest_created_at]["id"].values[0]

    # Find that event of the user that has "Sorry" in the text
    for df in user_questions_over_time_list:
        if latest_user_id in df["user_id"].values:
            print(df[df["user_id"] == latest_user_id][df["text"].str.startswith("Sorry")])
            break
    # Remove these users from the analysis
    exclude_user_ids.extend(list(user_id_set))"""

    # Update analysis dfs and exclude users
    print(f"Users excluded: {len(exclude_user_ids)}")
    analysis.update_dfs(exclude_user_ids)
    analysis.add_self_assessment_value_column()

    """# Save wlb_users to a csv file for manual analysis
    wlb_users_df = pd.DataFrame(wlb_users, columns=["user_id", "feedback"])
    wlb_users_df.to_csv(f"wlb_users_{prolific_file_name_combined}.csv", index=False)
    # get wlb user ids and exclude them
    wlb_user_ids = wlb_users_df["user_id"].values
    # analysis.update_dfs(wlb_user_ids)"""

    print("Amount of users per study group after first filters:")
    print(analysis.user_df.groupby("study_group").size())

    # perform_power_analysis()
    # filter_by_wanted_count(analysis, 70)

    ### Add Additional Columns to analysis.user_df
    # Calculate total score improvement and confidence improvement
    # analysis.user_df["score_improvement"] = analysis.user_df["final_score"] - analysis.user_df["intro_score"]

    # Merge final_q_feedback_list to analysis.user_df on user_id
    analysis.user_df = analysis.user_df.merge(pd.concat(all_q_list), left_on="id", right_on="user_id", how="left")

    initial_test_preds_df = pd.concat(initial_test_preds_list)
    learning_test_preds_df = pd.concat(learning_test_preds_list)
    analysis.add_initial_test_preds_df(initial_test_preds_df)
    analysis.add_learning_test_preds_df(learning_test_preds_df)
    analysis.add_final_test_preds_df(pd.concat(final_test_preds_list))
    analysis.add_questions_over_time_df(pd.concat(user_questions_over_time_list))

    # Update all dfs
    analysis.update_dfs()

    print(analysis.user_df.loc[analysis.user_df['id'] == "5470e036-7300-4de0-bd37-088a0a7816e5"])

    assert len(analysis.user_df) == len(analysis.initial_test_preds_df['user_id'].unique())
    assert len(analysis.user_df) == len(analysis.learning_test_preds_df['user_id'].unique())

    user_accuracy_over_time_df = plot_understanding_over_time(analysis.learning_test_preds_df, analysis)
    assert len(analysis.user_df) == len(user_accuracy_over_time_df['user_id'].unique())

    # Extract questionnaires into columns
    # analysis.user_df = extract_questionnaires(analysis.user_df)

    # Remove users with too high ML knowledge (>3)
    # Turn mL knowledge to int
    analysis.user_df["ml_knowledge"] = analysis.user_df["ml_knowledge"].astype(int)
    print(analysis.user_df["ml_knowledge"].value_counts())
    # Print how many > 3 per study group
    print(analysis.user_df[analysis.user_df["ml_knowledge"] > 3].groupby("study_group").size())

    if "remove_users_with_high_ml_knowledge" in analysis_steps:
        users_to_remove = analysis.user_df[analysis.user_df["ml_knowledge"] > 3]["id"].values
        # Sort user_ids by created_at and save id and created_at to a df
        # users_to_remove_df = analysis.user_df[analysis.user_df["id"].isin(users_to_remove)][["id", "created_at", "study_group"]]
        analysis.update_dfs(users_to_remove)

    if "remove_users_with_fatal_error" in analysis_steps:
        # Filter final check failed.
        failure_report_active_chat, fatal_error_users_active, understanding_scores_df_active = get_users_failed_final_understanding_check(
            analysis.user_df, "active_chat")
        failure_report_chat, fatal_error_users, understanding_scores_df_chat = get_users_failed_final_understanding_check(
            analysis.user_df, "chat")

        # combine understanding_scores_df_active and understanding_scores_df_chat
        understanding_scores_df = pd.concat([understanding_scores_df_active, understanding_scores_df_chat])
        # merge with user_df on id
        analysis.user_df = analysis.user_df.merge(understanding_scores_df, on='id', how='left')

        analysis.user_df = analysis.user_df[~analysis.user_df["id"].isin(fatal_error_users)]
        analysis.user_df = analysis.user_df[~analysis.user_df["id"].isin(fatal_error_users_active)]

    print("Amount of users per study group after All filters:")
    print(analysis.user_df.groupby("study_group").size())

    """# Save the dfs to csv
    analysis.user_df.to_csv(result_folder_path + "user_df.csv", index=False)
    analysis.event_df.to_csv(result_folder_path + "event_df.csv", index=False)
    analysis.user_completed_df.to_csv(result_folder_path + "user_completed_df.csv", index=False)
    analysis.initial_test_preds_df.to_csv(result_folder_path + "initial_test_preds_df.csv", index=False)
    analysis.learning_test_preds_df.to_csv(result_folder_path + "learning_test_preds_df.csv", index=False)
    analysis.final_test_preds_df.to_csv(result_folder_path + "final_test_preds_df.csv", index=False)
    analysis.questions_over_time_df.to_csv(result_folder_path + "questions_over_time_df.csv", index=False)
    user_accuracy_over_time_df.to_csv(result_folder_path + "user_accuracy_over_time_df.csv", index=False)"""

    # Define the save_unique_to_csv method
    def save_unique_to_csv(df, file_path, subset=None):
        """
        Saves a DataFrame to a CSV file, appending only unique rows if the file exists.

        Parameters:
        - df (pd.DataFrame): The DataFrame to save.
        - file_path (str): The path to the CSV file.
        - subset (list, optional): List of columns to consider for deduplication. If not specified,
          converts complex columns to strings for deduplication on all columns.
        """
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)

            # Convert complex columns to strings if no subset is specified
            if subset is None:
                df = df.applymap(lambda x: str(x) if isinstance(x, dict) or isinstance(x, list) else x)
                existing_df = existing_df.applymap(
                    lambda x: str(x) if isinstance(x, dict) or isinstance(x, list) else x)

            # Concatenate with the new DataFrame and remove duplicates
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=subset)
        else:
            combined_df = df

        combined_df.to_csv(file_path, index=False)

    """# Use the method to save each DataFrame
    save_unique_to_csv(analysis.user_df, result_folder_path + "user_df.csv")
    save_unique_to_csv(analysis.event_df, result_folder_path + "event_df.csv")
    save_unique_to_csv(analysis.user_completed_df, result_folder_path + "user_completed_df.csv")
    save_unique_to_csv(analysis.initial_test_preds_df, result_folder_path + "initial_test_preds_df.csv")
    save_unique_to_csv(analysis.learning_test_preds_df, result_folder_path + "learning_test_preds_df.csv")
    save_unique_to_csv(analysis.final_test_preds_df, result_folder_path + "final_test_preds_df.csv")
    save_unique_to_csv(analysis.questions_over_time_df, result_folder_path + "questions_over_time_df.csv")
    save_unique_to_csv(user_accuracy_over_time_df, result_folder_path + "user_accuracy_over_time_df.csv")"""

    # Get prolific ids of the users in analysis.user_df
    # load user_df from csv
    analysis.user_df = pd.read_csv(result_folder_path + "user_df.csv")
    # print number of people per study group
    print(analysis.user_df.groupby("study_group").size())
    # Get demographics of the users
    user_prolific_ids = analysis.user_df["prolific_id"].values
    prolific_df = pd.read_csv(prolific_file_name_combined)
    prolific_df = prolific_df[prolific_df["Participant id"].isin(user_prolific_ids)]
    # Get Age statistics
    # ignore revoked from analysis for now
    prolific_df = prolific_df[prolific_df["Age"] != "CONSENT_REVOKED"]
    # turn age to int and fill nan with median
    prolific_df["Age"] = prolific_df["Age"].astype(int)
    prolific_df["Age"].fillna(prolific_df["Age"].median(), inplace=True)
    print(prolific_df["Age"].describe())
    print()
    # Get Sex statistics
    print(prolific_df["Sex"].value_counts())
    print(prolific_df["Nationality"].value_counts())
    # Turn mL knowledge to int
    analysis.user_df["ml_knowledge"] = analysis.user_df["ml_knowledge"].astype(int)
    print(analysis.user_df["ml_knowledge"].value_counts())
    # Print ml knowledge categories
    print(analysis.user_df["ml_knowledge"].value_counts())


    print()

    # Get top 10 people based on final score with their prolific id
    # print(analysis.user_df[["prolific_id", "final_score"]].sort_values("final_score", ascending=False).head(10))


if __name__ == "__main__":
    main()
