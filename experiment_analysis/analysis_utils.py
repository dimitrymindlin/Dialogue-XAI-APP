import os
import glob
import json
import pandas as pd
import psycopg2
import ast
from typing import Dict
from fuzzywuzzy import fuzz


def connect_db(db_cfg: Dict) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        dbname=db_cfg["dbname"],
        user=db_cfg["user"],
        password=db_cfg["password"],
        host=db_cfg["host"],
        port=db_cfg["port"]
    )
    return conn


def fetch_table(conn, table: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {table}", conn)


def unpack_questions(events_df: pd.DataFrame) -> pd.DataFrame:
    # Example: unpack 'details' JSON column
    if "details" in events_df.columns:
        details_df = events_df["details"].apply(json.loads).apply(pd.Series)
        events_df = pd.concat([events_df.drop(columns=["details"]), details_df], axis=1)
    return events_df


def make_stage_dirs(base_folder, stages=None):
    """
    Create enumerated subdirectories for pipeline stages.
    Args:
        base_folder (str): Top-level results folder.
        stages (list, optional): List of stage names. Defaults to ["raw", "unpacked"].
    Returns:
        dict: Mapping of stage name to directory path.
    """
    if stages is None:
        stages = ["raw", "unpacked"]
    stage_dirs = {name: os.path.join(base_folder, f"{i + 1}_{name}") for i, name in enumerate(stages)}
    for d in stage_dirs.values():
        os.makedirs(d, exist_ok=True)
    return stage_dirs


def fetch_data_as_dataframe(query, connection):
    cur = connection.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    column_names = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=column_names)


def merge_prolific_csvs(folder: str, output_file: str) -> pd.DataFrame:
    """
    Concisely merge all CSVs in a folder, deduplicate by 'Participant id', and save to output_file.
    Skips files that do not have 'Participant id' column.
    Returns the merged DataFrame.
    """
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    csv_files = [f for f in csv_files if "prolific_" in os.path.basename(f).lower()]
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if "Participant id" in df.columns:
                dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    if not dataframes:
        raise ValueError(f"No valid Prolific CSVs with 'Participant id' found in {folder}")
    merged = pd.concat(dataframes, ignore_index=True)
    merged = merged.drop_duplicates(subset=["Participant id"])
    merged.to_csv(output_file, index=False)
    return merged


def extract_all_questionnaires(user_df, user_id):
    try:
        questionnaires = user_df[user_df["id"] == user_id]["questionnaires"].values[0]
        # if string, convert to list robustly
        if isinstance(questionnaires, str):
            try:
                questionnaires = json.loads(questionnaires)
            except json.JSONDecodeError:
                questionnaires = ast.literal_eval(questionnaires)
        feedback_dict = {"user_id": user_id}
        for questionnaire in questionnaires:
            if isinstance(questionnaire, dict):
                continue
            questionnaire = json.loads(questionnaire)
            for key, value in questionnaire.items():
                feedback_dict[f"{key}_questions"] = value['questions']
                feedback_dict[f"{key}_answers"] = value['answers']
    except (KeyError, IndexError, Exception) as e:
        print(f"{user_id} did not complete the questionnaires or error: {e}")
        return None
    feedback_df = pd.DataFrame([feedback_dict])
    return feedback_df


def get_study_group_and_events(user_df, event_df, user_id):
    study_group = user_df[user_df["id"] == user_id]["study_group"].values[0]
    user_events = event_df[event_df["user_id"] == user_id]
    return study_group, user_events


def get_study_name_description_if_possible(user_df):
    """
    Return a dictionary with unique 'study_group_name' values from the 'profile' JSON and their counts.
    If not available, fallback to 'study_group' values and their counts.
    """
    user_profiles = user_df["profile"].dropna().unique()
    study_name_counts = {}

    for profile in user_profiles:
        try:
            profile_data = json.loads(profile)
            if "study_group_name" in profile_data:
                study_name = profile_data["study_group_name"]
                study_name_counts[study_name] = study_name_counts.get(study_name, 0) + 1
        except (json.JSONDecodeError, TypeError):
            continue

    if not study_name_counts:
        study_group_counts = user_df["study_group"].dropna().value_counts().to_dict()
        return study_group_counts

    return study_name_counts


def extract_questions(user_events, study_group):
    user_questions_over_time = user_events[
        (user_events["action"] == "question") & (user_events["source"] == "teaching")]
    if user_questions_over_time.empty:
        return None
    user_questions_over_time = user_questions_over_time.copy()

    # Check if details col still exists and unpack if so
    if not "details" in user_questions_over_time.columns:
        return user_questions_over_time

    user_questions_over_time.loc[:, 'details'] = user_questions_over_time['details'].apply(json.loads)
    details_df = user_questions_over_time['details'].apply(pd.Series)
    result_df = user_questions_over_time.drop(columns=['details'])
    overlapping_cols = set(details_df.columns) & set(result_df.columns)
    if overlapping_cols:
        details_df = details_df.drop(columns=list(overlapping_cols))

    if study_group in ["chat", "active_chat"]:
        message_df = _process_message_column(details_df, study_group)
        if message_df is not None:
            details_df = details_df.drop(columns=['message'])
            # Check duplicated columns in result_df and message_df and details_df
            overlapping_cols = set(message_df.columns) & set(details_df.columns)
            if overlapping_cols:
                message_df = message_df.drop(columns=list(overlapping_cols))
            result_df = pd.concat([result_df, details_df, message_df], axis=1)
        else:
            result_df = pd.concat([result_df, details_df], axis=1)
            if study_group == "chat":
                result_df.loc[:, "user_question"] = "follow_up"
            elif study_group == "active_chat":
                result_df.loc[:, "text"] = ""
        _apply_chat_transformations(result_df, study_group)
        if "text" in result_df.columns:
            result_df["text"] = result_df["text"].fillna("")
        elif study_group == "active_chat":
            raise KeyError(
                "No 'text' column found in user_questions_over_time. Maybe you forgot to change 'active_users' in pg admin?")
    else:
        result_df = pd.concat([result_df, details_df], axis=1)
    result_df = result_df.reset_index(drop=True)
    return result_df


def _process_message_column(details_df, study_group):
    if 'message' not in details_df.columns:
        return None
    try:
        message_details = details_df['message'].apply(pd.Series)
        if study_group == "chat":
            columns_to_drop = ['feedback', 'isUser', 'feature_id', 'id', 'followup']
            existing_cols = [col for col in columns_to_drop if col in message_details.columns]
            if existing_cols:
                message_details = message_details.drop(columns=existing_cols)
        elif study_group == "active_chat":
            if set(['reasoning', 'text']).issubset(message_details.columns):
                message_details = message_details[['reasoning', 'text']]
        return message_details
    except (KeyError, TypeError):
        return None


def _apply_chat_transformations(df, study_group):
    if study_group == "active_chat":
        if 'question_id' in df.columns and 'feature_id' in df.columns:
            df.loc[
                df['question_id'].isin(["ceterisParibus", "featureStatistics"]), "feature_id"] = "infer_from_question"
        if "question" in df.columns:
            df.rename(columns={"question": "clicked_question"}, inplace=True)
    if "user_question" in df.columns:
        df.rename(columns={"user_question": "typed_question"}, inplace=True)


def get_data_for_prolific_users(users_df, prolific_csv_path, user_id_col_db="prolific_id",
                                user_id_col_csv="Participant id"):
    """
    Filter a DataFrame to only include users present in the Prolific export.
    Args:
        users_df (pd.DataFrame): DataFrame with a column for Prolific IDs.
        prolific_csv_path (str): Path to merged Prolific CSV.
        user_id_col_db (str): Column in users_df with Prolific IDs.
        user_id_col_csv (str): Column in CSV with Prolific IDs.
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    prolific_df = pd.read_csv(prolific_csv_path)
    prolific_ids = set(prolific_df[user_id_col_csv].astype(str))
    filtered = users_df[users_df[user_id_col_db].astype(str).isin(prolific_ids)]
    return filtered


def add_dummy_var_mention_column(predictions_df,
                                 dummy_var_name="worklifebalance",
                                 column_name="dummy_var_mention",
                                 similarity_threshold=80):
    """
    Adds a column counting how often a dummy variable was mentioned in the feedback.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing user predictions with 'feedback' column
        dummy_var_name (str): The dummy variable name to search for (default: "worklifebalance")
        column_name (str): The name of the column to add (default: "dummy_var_mention")
        similarity_threshold (int): The fuzzy matching threshold for considering a match (default: 80)
        
    Returns:
        pd.DataFrame: DataFrame with added column counting dummy variable mentions
    """
    # Create a copy to avoid modifying the original
    predictions_df = predictions_df.copy()

    # Initialize the column with zero counts
    predictions_df[column_name] = 0

    # Handle different formats of the dummy variable (with spaces, without spaces)
    dummy_var_with_spaces = " ".join([word for word in dummy_var_name if word])

    # Go through each row and count dummy variable mentions
    for idx, row in predictions_df.iterrows():
        if pd.notna(row['feedback']):
            feedback = str(row['feedback']).lower()
            # Check both formats (with and without spaces)
            if (fuzz.partial_ratio(dummy_var_name.lower(), feedback) > similarity_threshold or
                    fuzz.partial_ratio(dummy_var_with_spaces.lower(), feedback) > similarity_threshold):
                predictions_df.at[idx, column_name] += 1

    return predictions_df


def count_dummy_var_users(predictions_df, dummy_var_column="dummy_var_mention", user_id_column='user_id'):
    """
    Identifies users who mention a dummy variable in their feedback.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with predictions that has been processed 
                                      with add_dummy_var_mention_column
        dummy_var_column (str): Name of the column containing dummy variable mention counts
        user_id_column (str): Name of the column containing user IDs
        
    Returns:
        list: List of tuples with (user_id, feedback, count) for users who mentioned the dummy variable
    """
    dummy_var_users = []

    # Group by user_id to check if any feedback from this user mentions the dummy variable
    for user_id, user_df in predictions_df.groupby(user_id_column):
        mentions = user_df[user_df[dummy_var_column] > 0]
        if len(mentions) > 0:
            # Get the feedback where the dummy variable was mentioned
            for _, row in mentions.iterrows():
                dummy_var_users.append((user_id, row['feedback'], row[dummy_var_column]))

    return dummy_var_users


def filter_by_dummy_var_mentions(predictions_df, dummy_var_column="dummy_var_mention", user_id_column='user_id'):
    """
    Filter function that returns user IDs who mentioned the dummy variable in their feedback.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with predictions processed with add_dummy_var_mention_column
        dummy_var_column (str): Name of the column containing dummy variable mention counts
        user_id_column (str): Name of the column containing user IDs
        
    Returns:
        list: List of user IDs to exclude from analysis
    """
    dummy_var_users = count_dummy_var_users(predictions_df, dummy_var_column, user_id_column)
    # Extract just the user IDs from the tuples
    return [user[0] for user in dummy_var_users]


def add_prolific_times(user_completed_df, prolific_df, event_df=None):
    """
    Adds prolific_start_time and prolific_end_time columns to the user_completed_df
    by matching prolific_id with the Prolific export data.
    If event_df is provided, also calculates additional time columns similar to create_time_columns.
    Returns list of user_ids not found in Prolific data.
    """
    from datetime import datetime
    import pandas as pd

    user_completed_df["prolific_start_time"] = None
    user_completed_df["prolific_end_time"] = None
    non_prolific_user_ids = []
    strange_cases = []
    no_end_time = []

    # Prepare Prolific lookup dicts for fast access
    prolific_lookup = prolific_df.set_index("Participant id")
    completed_at = prolific_lookup["Completed at"].to_dict()
    started_at = prolific_lookup["Started at"].to_dict()
    status_at = prolific_lookup["Status"].to_dict()

    # Helper: try to parse any reasonable datetime string
    def try_parse(dt):
        if pd.isnull(dt):
            return None
        if isinstance(dt, datetime):
            return dt
        for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
            try:
                return datetime.strptime(str(dt), fmt)
            except Exception:
                continue
        return None

    # Vectorized matching
    for idx, row in user_completed_df.iterrows():
        pid = row.get("prolific_id")
        if pd.isnull(pid) or pid not in prolific_lookup.index:
            non_prolific_user_ids.append(row["user_id"])
            continue

        start = try_parse(started_at.get(pid))
        end = try_parse(completed_at.get(pid))
        db_created = try_parse(row.get("created_at"))
        if end is None:
            no_end_time.append((pid, status_at.get(pid)))
            continue
        if end and db_created and end < db_created:
            strange_cases.append((pid, start, end, row.get("created_at")))
            non_prolific_user_ids.append(row["user_id"])
            continue

        user_completed_df.at[idx, "prolific_start_time"] = start
        user_completed_df.at[idx, "prolific_end_time"] = end

    # Fill missing end times for completed users using median duration
    user_completed_df["prolific_start_time"] = pd.to_datetime(user_completed_df["prolific_start_time"])
    user_completed_df["prolific_end_time"] = pd.to_datetime(user_completed_df["prolific_end_time"])

    # Check if we need to impute missing end times
    valid = user_completed_df.dropna(subset=["prolific_start_time", "prolific_end_time"])
    if not valid.empty:
        median_duration = (valid["prolific_end_time"] - valid["prolific_start_time"]).median()
        mask = user_completed_df["completed"] & user_completed_df["prolific_end_time"].isnull() & user_completed_df[
            "prolific_start_time"].notnull()

        # Only perform imputation if there are users with missing end times
        if mask.sum() > 0:
            user_completed_df.loc[mask, "prolific_end_time"] = user_completed_df.loc[
                                                                   mask, "prolific_start_time"] + median_duration
            print("Added end time to", mask.sum(), "users who completed the experiment but had missing end time data.")
    else:
        print("Warning: No valid prolific_start_time and prolific_end_time pairs found for imputation reference.")

    # If event_df is provided, calculate additional time-related metrics
    # This adds event timestamps and derived time measurements needed for analysis
    if event_df is not None:
        # Create a DataFrame with event start and end times for each user
        event_times = pd.DataFrame()

        # Get start event for each user from event_df
        start_time_df = event_df.groupby("user_id")["created_at"].min()
        event_times["event_start_time"] = start_time_df

        # Get end time for each user from event_df
        end_time_df = event_df.groupby("user_id")["created_at"].max()
        event_times["event_end_time"] = end_time_df

        # Reset index to make user_id a column
        event_times.reset_index(inplace=True)

        # Merge event times with user_completed_df
        user_completed_df = user_completed_df.merge(
            event_times,
            left_on="user_id",
            right_on="user_id",
            how="left"
        )

        # Ensure datetime format
        user_completed_df["event_start_time"] = pd.to_datetime(user_completed_df["event_start_time"])
        user_completed_df["event_end_time"] = pd.to_datetime(user_completed_df["event_end_time"])

        # Calculate time columns in minutes
        # Time spent in the learning phase (from first event to last event)
        user_completed_df["total_learning_time"] = (
                                                           user_completed_df["event_end_time"] - user_completed_df[
                                                       "event_start_time"]
                                                   ).dt.total_seconds() / 60

        # Time spent in experiment instructions (prolific start - first event)
        user_completed_df["exp_instruction_time"] = (
                                                            user_completed_df["event_start_time"] - user_completed_df[
                                                        "prolific_start_time"]
                                                    ).dt.total_seconds() / 60

        # Total experiment time (from prolific start to prolific end)
        user_completed_df["total_exp_time"] = (
                                                      user_completed_df["prolific_end_time"] - user_completed_df[
                                                  "prolific_start_time"]
                                              ).dt.total_seconds() / 60

    print("Non prolific users:", len(non_prolific_user_ids))
    print("Strange cases:", len(strange_cases))
    return user_completed_df, non_prolific_user_ids


def create_understanding_over_time_df(user_predictions_over_time_df) -> pd.DataFrame:
    """
    Create a DataFrame with each row representing a user and columns for prediction accuracy
    over time. This extracts the DF creation logic from plot_understanding_over_time.
    
    Args:
        user_predictions_over_time_df: DataFrame with user predictions, including columns:
            - user_id: the user identifier
            - prediction: the user's prediction
            - true_label: the ground truth label
            - datapoint_count: the order of the prediction
            
    Returns:
        tuple: (DataFrame with user prediction accuracy over time, list of user_ids to exclude)
    """
    # Create a new column 'accuracy' to reflect Correct/Wrong
    user_predictions_over_time_df['accuracy'] = user_predictions_over_time_df.apply(
        lambda row: 'Correct' if row['prediction'] == row['true_label'] else 'Wrong', axis=1)

    # Print unique datapoint counts
    unique_counts = user_predictions_over_time_df['datapoint_count'].unique()
    print(f"Unique datapoint counts: {unique_counts}")

    # Pivot the table with 'accuracy' as the values
    result_df = user_predictions_over_time_df.pivot_table(
        index='user_id',
        columns='datapoint_count',
        values='accuracy',
        aggfunc=lambda x: x
    )

    # Flattening the multi-index in columns
    result_df.columns = [f'accuracy_{col}' for col in result_df.columns]
    result_df.reset_index(inplace=True)

    return result_df


def add_understanding_question_analysis(user_df, study_group=None, dataset_name=None):
    """
    Analyze understanding questions and add columns with results.
    
    Args:
        user_df (pd.DataFrame): User dataframe with questions and answers
        study_group (str, optional): Study group to filter by
        dataset_name (str, required): Dataset name for configuration
        
    Returns:
        pd.DataFrame: User dataframe with added columns for question analysis
        
    Raises:
        ValueError: If dataset_name is not provided or if configuration file cannot be loaded
    """
    # Dataset name is required
    if not dataset_name:
        raise ValueError("dataset_name parameter is required to load the appropriate configuration")

    # Attempt to load dataset-specific config
    config_path = os.path.join("../configs", f"{dataset_name}_understanding.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. Please create a configuration file for the {dataset_name} dataset.")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file {config_path}: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading configuration file {config_path}: {str(e)}")

    # Make a copy to avoid modifying the original
    df = user_df.copy()

    # Filter by study group if provided
    if study_group is not None:
        df = df[df['study_group'] == study_group]

    # Get column names from config
    id_col = config["column_mapping"]["user_id"]
    questions_col = config["column_mapping"]["questions"]
    answers_col = config["column_mapping"]["answers"]

    # Create new column names for results
    most_important_col = "understanding_q_most_important_feature_answer"
    least_important_col = "understanding_q_least_important_feature_answer"
    decision_rules_col = "understanding_q_decision_rules_text"

    # Initialize the new columns
    df[most_important_col] = None
    df[least_important_col] = None
    df[decision_rules_col] = None

    # Process each user's answers
    for idx, row in df.iterrows():
        user_id = row[id_col]

        # Get questions and answers
        try:
            questions = row[questions_col]
            answers = row[answers_col]

            # Process the answers
            for q_idx, (question, answer) in enumerate(zip(questions, answers)):
                answer = str(answer).lower() if answer else ""

                # Most important feature (first question)
                if q_idx == 0:
                    correct = config["most_important_feature"]["correct_answer"]
                    fatal = config["most_important_feature"]["fatal_error"]

                    if answer == correct:
                        df.loc[idx, most_important_col] = "correct"
                    elif answer == fatal:
                        df.loc[idx, most_important_col] = "fatal_wrong"
                    elif answer in config["non_dataset_features"]:
                        df.loc[idx, most_important_col] = "non_dataset_wrong"
                    else:
                        df.loc[idx, most_important_col] = "wrong"

                # Least important feature (second question)
                elif q_idx == 1:
                    correct = config["least_important_feature"]["correct_answer"]
                    fatal = config["least_important_feature"]["fatal_error"]

                    if answer == correct:
                        df.loc[idx, least_important_col] = "correct"
                    elif answer == fatal:
                        df.loc[idx, least_important_col] = "fatal_wrong"
                    elif answer in config["non_dataset_features"]:
                        df.loc[idx, least_important_col] = "non_dataset_wrong"
                    else:
                        df.loc[idx, least_important_col] = "wrong"

                # Decision rules text (third question)
                elif q_idx == 2:
                    df.loc[idx, decision_rules_col] = answer

        except Exception as e:
            print(f"Error processing user {user_id}: {e}")

    return df


def filter_dataframes_by_user_ids(dataframes_dict, exclude_user_ids, user_id_columns=None):
    """
    Filter multiple DataFrames by excluding specified user IDs.
    
    Args:
        dataframes_dict (dict): Dictionary of {name: dataframe} to filter
        exclude_user_ids (set or list): User IDs to exclude
        user_id_columns (dict, optional): Dictionary of {dataframe_name: user_id_column_name}
                                        Defaults to 'user_id' for all except 'users' which uses 'id'
    
    Returns:
        dict: Dictionary of filtered DataFrames with same keys as input
    """
    if user_id_columns is None:
        user_id_columns = {}

    filtered_dfs = {}

    for name, df in dataframes_dict.items():
        if df is None or df.empty:
            filtered_dfs[name] = df
            continue

        # Determine the user ID column name
        user_id_col = user_id_columns.get(name, 'user_id' if name != 'users' else 'id')

        # Filter the DataFrame
        filtered_df = df[~df[user_id_col].isin(exclude_user_ids)].copy()
        filtered_dfs[name] = filtered_df

        print(f"{name.capitalize()} before filtering: {len(df)}, after filtering: {len(filtered_df)}")

    return filtered_dfs


def save_dataframes_to_csv(dataframes_dict, output_dir, file_suffix=""):
    """
    Save multiple DataFrames to CSV files.
    
    Args:
        dataframes_dict (dict): Dictionary of {name: dataframe} to save
        output_dir (str): Directory to save CSV files
        file_suffix (str): Suffix to add to filenames (e.g., "_filtered")
    
    Returns:
        list: List of saved file paths
    """
    saved_files = []

    for name, df in dataframes_dict.items():
        if df is not None:
            filename = f"{name}{file_suffix}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            saved_files.append(filepath)
            print(f"- {filename} ({len(df)} rows)")

    return saved_files


def add_subjective_understanding_score(user_df):
    """
    Adds a subjective understanding score based on user's self assessment.
    """
    # Make a copy to avoid modifying the original
    df = user_df.copy()

    # Check if 'subjective_understanding' column exists
    if 'self_assessment_answers' not in df.columns:
        print("No 'self_assessment_answers' column found. Skipping addition of subjective understanding score.")
        return df

    # Calculate subjective understanding from likert scale results. Users answered following questions on 5 point scale from -2 to 2
    # [\'I understand the important attributes for a decision.\', \'I cannot distinguish between the possible prediction.\', \'I understand how the Machine Learning model works.\', \'Select "Strongly Disagree" for this selection.\', \'I did not understand how the Machine Learning model makes decisions.\']
    # Combine the answers into a single score, where the third question is an attention check and should be ignored and the second question is inverted.

    def calculate_understanding_score(x):
        """Step-by-step calculation for debugging"""

        # Handle both list and string inputs
        if isinstance(x, list):
            answers = x
        elif isinstance(x, str):
            try:
                # Parse the string to get the list of answers
                answers = ast.literal_eval(x)
            except Exception as e:
                print(f"Error parsing string: {e}")
                return 0
        else:
            print(f"Input is neither string nor list, returning 0")
            return 0

        if not isinstance(answers, list) or len(answers) < 4:
            print(f"Invalid answers format or length, returning 0")
            return 0

        # Extract individual answers for debugging
        q1_answer = answers[0]  # I understand the important attributes for a decision
        q2_answer = answers[1]  # I cannot distinguish between the possible prediction (will be inverted)
        q3_answer = answers[2]  # Attention check (ignored)
        q4_answer = answers[3]  # I did not understand how the ML model makes decisions (will be inverted)

        # Calculate score: Q1 + (-Q2) + (-Q4)
        # Note: Q2 and Q4 are negative statements, so we invert them
        score = q1_answer + (-q2_answer) + (-q4_answer)
        return score

    subj_understanding = df['self_assessment_answers'].apply(calculate_understanding_score)

    # Average the scores to get a single understanding score per user (divide by 3 questions used)
    subj_understanding = subj_understanding / 3.0
    print(f"Understanding scores after averaging: {subj_understanding.tolist()}")

    # Create a new column with added subj_understanding
    df['subjective_understanding'] = subj_understanding
    return df
