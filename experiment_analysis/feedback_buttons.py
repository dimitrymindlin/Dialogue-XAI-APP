import ast


def get_button_feedback(event_df):
    """Filter the events for feedback button clicks and collect feedback per question"""

    question_feedback_dict = {}
    # Filter for feedback button clicks
    feedback_button_click_rows = event_df[event_df["action"] == "feedback"]
    feedback_button_click_rows['details'] = feedback_button_click_rows['details'].apply(ast.literal_eval)

    # Collect feedback per question
    for index, row in feedback_button_click_rows.iterrows():
        try:
            question = row["details"]["question_id"]
        except KeyError: # TODO: Datapoint like forgot to log question_id...
            continue
        feedback = row["details"]["user_comment"]
        try:
            question_feedback_dict[question].append(feedback)
        except KeyError:
            question_feedback_dict[question] = [feedback]


    return question_feedback_dict
