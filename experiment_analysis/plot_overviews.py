import json

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def plot_chatbot_feedback(feedback_dict_list):
    # Accumulate the data into a dictionary
    feedback_dict = {}
    for feedback in feedback_dict_list:
        for question, answer in zip(feedback['questions'], feedback['answers']):
            if question not in feedback_dict:
                feedback_dict[question] = []
            feedback_dict[question].append(answer)

    # Create a dataframe from the dictionary
    feedback_df = pd.DataFrame(feedback_dict)

    # Sort the ratings by average rating
    feedback_df = feedback_df[feedback_df.mean().sort_values().index]

    # Adjust figure size to increase width
    plt.figure(figsize=(20, 6))  # Increase the first value to widen the plot

    sns.boxplot(data=feedback_df)
    sns.stripplot(data=feedback_df, color=".25")

    plt.title("Chatbot Feedback")
    plt.xlabel("Questions")
    plt.ylabel("Answers")

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90)  # Adjust rotation as needed

    # Use subplots_adjust to fine-tune the layout, especially the bottom margin
    plt.subplots_adjust(bottom=0.5, top=0.9)  # Adjust bottom as needed to fit the x-axis labels

    plt.show()


def get_user_id_predictions_over_time_matrix(user_predictions_over_time_list):
    """
    Prepare a matrix where each row represents a user and each column represents the correctness of one of the 5 predictions.

    :param user_predictions_over_time_list: List of user predictions over time
    :return: A 2D list with user correctness data, and a list of user_ids in the order they appear in the matrix.
    """
    user_correctness = {}

    for user_predictions_over_time in user_predictions_over_time_list:
        user_id = user_predictions_over_time["user_id"].unique()[0]
        details_col = user_predictions_over_time["details"]
        details_list = details_col.apply(json.loads).tolist()

        sorted_predictions = sorted(details_list, key=lambda x: x['datapoint_count'])

        # if there are duplicate predictions, keep the first one
        seen = set()
        sorted_predictions = [x for x in sorted_predictions if
                              x['datapoint_count'] not in seen and not seen.add(x['datapoint_count'])]

        user_correctness[user_id] = []
        for prediction in sorted_predictions:
            is_correct = prediction["true_label"].lower() == prediction["prediction"].lower()
            user_correctness[user_id].append(is_correct)

    """# Ensure all users have entries for each of the 5 predictions
    for user_id in user_correctness:
        while len(user_correctness[user_id]) < 5:
            user_correctness[user_id].append(False)  # Assuming missing predictions are incorrect"""

    # Convert to a matrix format suitable for plotting
    matrix_data = [user_correctness[user_id] for user_id in sorted(user_correctness)]
    user_ids = sorted(user_correctness.keys())

    return matrix_data, user_ids


def plot_user_predictions(matrix_data, user_ids, study_group_name, users_end_score_dict=None):
    """
    Plot a matrix where each row represents a user and each column represents the correctness of one of the 5 predictions.
    Optionally includes an end score for each user as the last column.

    :param matrix_data: A 2D list with user correctness data (expected to be numeric)
    :param user_ids: List of user IDs corresponding to rows in matrix_data
    :param study_group_name: Name of the study group for the title
    :param users_end_score_dict: Optional dictionary mapping user_id to end score (expected to be numeric or None for missing)
    """
    # Convert boolean 'True'/'False' to 1/0 and handle missing end scores
    for i, row in enumerate(matrix_data):
        for j, val in enumerate(row):
            if isinstance(val, str):  # Convert 'True'/'False' strings to 1/0
                matrix_data[i][j] = 1 if val == 'True' else 0

        if users_end_score_dict:
            user_id = user_ids[i]
            end_score = users_end_score_dict.get(user_id, np.nan)
            matrix_data[i].append(end_score)

    xticklabels = [str(i) for i in range(1, 6)] + ['objective score'] if users_end_score_dict else [str(i) for i in
                                                                                                    range(1, 6)]

    fig, ax = plt.subplots(figsize=(10, len(user_ids) / 2 + 1))
    sns.heatmap(matrix_data, annot=True, fmt=".1f", cmap="YlGn", ax=ax, cbar=False, xticklabels=xticklabels,
                yticklabels=user_ids)

    ax.set_xlabel("Prediction Number")
    ax.set_ylabel("User ID")
    ax.set_title(f"{study_group_name} - User Prediction Correctness Matrix")

    plt.tight_layout()
    plt.show()


def plot_understanding_over_time(user_predictions_over_time_list, study_group_name, users_end_score_dict=None):
    """
    Plot the proportion of correct and incorrect predictions for each prediction order.
    """
    matrix_data, user_ids = get_user_id_predictions_over_time_matrix(user_predictions_over_time_list)
    plot_user_predictions(matrix_data, user_ids, study_group_name, users_end_score_dict)
    return matrix_data, user_ids


def print_feedback_json(user_df):
    for user_id in user_df["id"]:
        # Retrieve feedback JSON for the current user
        feedback_list = user_df[user_df["id"] == user_id]["feedback"].values[0]

        # Check if feedback_list is not empty and contains JSON string
        if feedback_list and isinstance(feedback_list[1], str):
            feedback_json = feedback_list[1]

            # Convert JSON string to Python dictionary
            feedback_dict = json.loads(feedback_json)

            # Add study group to the dictionary
            feedback_dict['study_group'] = user_df[user_df["id"] == user_id]["study_group"].values[0]

            # Convert dictionary back to JSON string
            modified_feedback_json = json.dumps(feedback_dict)

            # Print the modified JSON string
            print(modified_feedback_json)
            print(",")


def get_user_id_questions_over_time_matrix(user_questions_dict_list):
    user_questions = {}

    for single_user_questions_df in user_questions_dict_list:
        user_id = single_user_questions_df["user_id"].unique()[0]
        details_col = single_user_questions_df["details"]

        # Assuming details_col is a Series, convert JSON strings to dictionaries
        details_list = details_col.apply(json.loads).tolist()

        sorted_questions_by_datapoint = sorted(details_list, key=lambda x: x['datapoint_count'])

        if user_id not in user_questions:
            user_questions[user_id] = {}

        # Save questions per datapoint in dictionary
        for question in sorted_questions_by_datapoint:
            datapoint_count = question['datapoint_count']

            if datapoint_count not in user_questions[user_id]:
                user_questions[user_id][datapoint_count] = [question['question_id']]
            else:
                user_questions[user_id][datapoint_count].append(question['question_id'])

    # The conversion to matrix_data below assumes a list of lists for each user, sorted by datapoint_count.
    matrix_data = [[user_questions[user_id][dp] for dp in sorted(user_questions[user_id])] for user_id in
                   sorted(user_questions)]
    user_ids = sorted(user_questions.keys())

    return matrix_data, user_ids


def plot_user_questions(matrix, user_ids, study_group_name):
    """
    Plots a matrix where each cell contains the list of question IDs asked by each user at each datapoint.

    :param matrix: A nested list containing question IDs for each user at each datapoint.
    :param user_ids: List of user IDs corresponding to the rows in the matrix.
    :param study_group_name: Name of the study group for the plot title.
    """
    # Determine the matrix size
    num_rows = len(matrix)
    num_cols = max(len(row) for row in matrix)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(num_cols * 1.5, num_rows * 0.5))  # Adjust figure size as needed

    # Create an empty matrix for plotting
    plot_matrix = np.full((num_rows, num_cols), "", dtype=object)

    # Fill the plot matrix with question IDs (or a summary)
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            # Here, we join question IDs with a comma, or you can customize this part
            plot_matrix[i, j] = ", ".join(map(str, cell))

    # Use a table to display the matrix since it may contain text
    table = ax.table(cellText=plot_matrix, rowLabels=user_ids, colLabels=[f"DP {i + 1}" for i in range(num_cols)],
                     cellLoc='center', loc='center')

    # Adjust layout
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # Adjust font size as needed
    table.scale(1, 1.5)  # Adjust cell size as needed
    ax.axis('off')

    plt.title(f"{study_group_name} - Questions Asked Per User Per Datapoint")
    plt.show()


def plot_asked_questions_per_user(event_df, end_score_dict):
    """
    Plot the question ids per user per datapoint count.
    """
    matrix, user_ids = get_user_id_questions_over_time_matrix(event_df)
    plot_user_questions(matrix, user_ids, end_score_dict)
    return matrix, user_ids


def plot_understanding_with_questions(matrix_understanding, matrix_questions, user_ids_u, user_ids_q):
    """
    Plots a matrix where the cell color represents the understanding score (0 or 1),
    and the cell text displays the questions asked.

    :param matrix_understanding: A nested list with understanding scores for each user at each datapoint.
    :param matrix_questions: A nested list with questions IDs for each user at each datapoint.
    :param user_ids_u: List of user IDs corresponding to the rows in the matrix_understanding.
    :param user_ids_q: List of user IDs corresponding to the rows in the matrix_questions.
    """
    assert user_ids_u == user_ids_q, "User IDs must match between understanding and questions matrices"

    num_rows = len(matrix_understanding)
    num_cols = max(max(len(row) for row in matrix_understanding), max(len(row) for row in matrix_questions))

    # Initialize a figure
    fig, ax = plt.subplots(figsize=(num_cols * 1.5, num_rows * 0.5))  # Adjust size as needed

    # Define colors for understanding scores
    colors = {0: "tomato", 1: "lightgreen"}  # Red for 0, Green for 1

    # Plot each cell
    for i, (scores_row, questions_row) in enumerate(zip(matrix_understanding, matrix_questions)):
        for j in range(num_cols):
            score = scores_row[j] if j < len(scores_row) else None  # Handle different row lengths
            questions = ", ".join(map(str, questions_row[j])) if j < len(questions_row) and j < len(scores_row) else ""

            # Set cell color based on understanding score
            cell_color = colors.get(score, "lightgrey")  # Default color for missing scores

            # Create a rectangle as the cell background
            rect = plt.Rectangle((j, num_rows - i - 1), 1, 1, color=cell_color)
            ax.add_patch(rect)

            # Annotate the cell with question IDs
            ax.text(j + 0.5, num_rows - i - 0.5, questions, ha='center', va='center', fontsize=8)  # Adjust text alignment and size as needed

    # Set up the plot axes
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_xticks(np.arange(num_cols) + 0.5)
    ax.set_yticks(np.arange(num_rows) + 0.5)
    ax.set_xticklabels([f"DP {i + 1}" for i in range(num_cols)], rotation=45, ha='right')
    ax.set_yticklabels(reversed(user_ids_u))  # Reverse the order to match the top-to-bottom plotting
    ax.grid(False)  # Turn off the grid

    plt.title("Interactive - Understanding and Questions")
    plt.xlabel("Datapoint")
    plt.ylabel("User ID")

    # Rename last x tick label to "Objective Score"
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[-1] = "Objective Score"
    ax.set_xticklabels(labels)

    plt.tight_layout()  # Adjust layout to fit everything
    plt.show()

def plot_question_raking(question_matrix):
    """
    Print a list of questions asked ranked by the number of times they were asked across all users
    """
    # Flatten the matrix and count the occurrences of each question
    all_questions = [question for user_questions in question_matrix for questions in user_questions for question in questions]
    question_counts = pd.Series(all_questions).value_counts()

    # Plot the question counts
    plt.figure(figsize=(10, 6))
    question_counts.plot(kind='bar')
    plt.title("Question Ranking")
    plt.xlabel("Question ID")
    plt.ylabel("Number of Occurrences")
    plt.show()

