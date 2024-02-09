import json

POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "example"
POSTGRES_DB = "prestudy"
POSTGRES_HOST = "localhost"

import psycopg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Connect to your postgres DB
conn = psycopg2.connect(
    f"dbname={POSTGRES_DB} user={POSTGRES_USER} host={POSTGRES_HOST} password={POSTGRES_PASSWORD}"
)


# Get user table
def get_user_table_as_df():
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")
    rows = cur.fetchall()
    # Get column names from cursor description
    column_names = [desc[0] for desc in cur.description]
    # Turn the rows into a pandas dataframe with column names
    df = pd.DataFrame(rows, columns=column_names)
    return df


def get_event_table_as_df():
    cur = conn.cursor()
    cur.execute("SELECT * FROM events")
    rows = cur.fetchall()
    # Get column names from cursor description
    column_names = [desc[0] for desc in cur.description]
    # Turn the rows into a pandas dataframe with column names
    df = pd.DataFrame(rows, columns=column_names)
    return df


def get_user_score(user_predictions):
    user_score = 0
    predictions_list = user_predictions["details"].tolist()  # list of json strings
    for prediction in predictions_list:
        prediction = json.loads(prediction)
        if prediction["true_label"].lower() == prediction["prediction"].lower():
            user_score += 1
    return user_score


user_df = get_user_table_as_df()
event_df = get_event_table_as_df()

# get users where completed is true
completed_users = user_df[user_df["completed"] == True]

# Create an empty DataFrame to store user_id, study_group, and total_time
time_df = pd.DataFrame(columns=["user_id", "study_group", "total_time"])
score_df = pd.DataFrame(columns=["user_id", "study_group", "score"])

# Loop over the user_id in the completed_users and get the events for each user
for user_id in completed_users["id"]:
    # Check study group of user
    user_study_group = user_df[user_df["id"] == user_id]["study_group"].values[0]
    user_events = event_df[event_df["user_id"] == user_id]
    # Calculate total time spent on the platform
    start_time = user_events["created_at"].min()
    end_time = user_events["created_at"].max()
    total_time = end_time - start_time
    # Convert Timestamp to minutes
    total_time = total_time.total_seconds() / 60

    # Get the score of the user
    ## Get the actions of "user_prediction" where "source" is "final_test"
    user_predictions = user_events[(user_events["action"] == "user_prediction")]
    user_predictions = user_predictions[user_predictions["source"] == "final-test"]

    # Get the time of final test

    score = get_user_score(user_predictions)
    score_df = score_df.append({"user_id": user_id, "study_group": user_study_group, "score": score}, ignore_index=True)

    # Append the data to the DataFrame
    time_df = time_df.append({"user_id": user_id, "study_group": user_study_group, "total_time": total_time},
                             ignore_index=True)

# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x="study_group", y="total_time", data=time_df)
sns.stripplot(x="study_group", y="total_time", data=time_df, color=".25")
plt.title('Boxplot of Time Spent per Study Group')
plt.xlabel('Study Group')
plt.ylabel('Time Spent (in minutes)')
plt.show()

# Create a plot
plt.figure(figsize=(10, 6))
sns.boxplot(x="study_group", y="score", data=score_df)
sns.stripplot(x="study_group", y="score", data=score_df, color=".25")
plt.title('Barplot of User Scores per Study Group')
plt.xlabel('Study Group')
plt.ylabel('User Score')
plt.show()
