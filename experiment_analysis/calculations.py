import json


def calculate_user_score(user_predictions):
    user_score = 0
    for prediction_json in user_predictions["details"]:
        prediction = json.loads(prediction_json)
        if prediction["true_label"].lower() == prediction["prediction"].lower():
            user_score += 1
    return user_score

def get_user_feedback_per_data_point(user_predictions):
    user_feedback = []
    for prediction_json in user_predictions["details"]:
        prediction = json.loads(prediction_json)
        user_feedback.append(prediction["feedback"])
    return user_feedback


def calculate_total_time_per_user(user_events):
    start_time = user_events["created_at"].min()
    end_time = user_events["created_at"].max()
    total_time = (end_time - start_time).total_seconds() / 60  # Convert to minutes
    return total_time
