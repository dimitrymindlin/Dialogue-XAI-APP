import ast
import pandas as pd


class ProcessMining:
    def __init__(self):
        self.pm_event_df = None

    def _filter_by_score(self, user_scores_df, filter_by_score):
        user_scores_df = user_scores_df.reset_index()
        user_scores_df.rename(columns={"index": "user_id"}, inplace=True)
        final_scores_dict = user_scores_df.set_index("user_id")["Objective Score"].to_dict()

        if filter_by_score == "best":
            # Only keep users with score > 3-5, directly assign the filtered DataFrame
            user_scores_df = user_scores_df[user_scores_df["Objective Score"] >= 3]
        elif filter_by_score == "worst":
            # Only keep users with score > 0-2, directly assign the filtered DataFrame
            user_scores_df = user_scores_df[user_scores_df["Objective Score"] <= 2]
        else:
            raise ValueError("filter_by_score must be 'best' or 'worst'")

        return user_scores_df, final_scores_dict

    def create_pm_csv(self,
                      event_df: pd.DataFrame,
                      user_scores_df,
                      datapoint_count=5,
                      filter_by_score: str = None):
        """
        Preprocesses experiment event df into a format suitable for process mining.
        Format should be table of [CaseID, Activity, Timestamp]
        :param event_df: DataFrame with event data
        :param user_scores_df: DataFrame with user scores, used to filter users by score
        :param datapoint_count: Datapoint count to filter by datapoint (conversation turn)
        :param filter_by_score: Filter users by score, e.g. "best" or "worst"
        """
        if filter_by_score:
            # Keep only users with best or worst scores
            user_scores_df, final_scores_dict = self._filter_by_score(user_scores_df, filter_by_score)
            # filter event_df by user_scores_df
            event_df = event_df[event_df["user_id"].isin(user_scores_df["user_id"])]

        pm_event_df = pd.DataFrame(columns=["Case id", "Activity", "Timestamp"])

        # collect questions per user in the teaching phase per datapoint count
        event_df_tmp = event_df.copy()

        questions = event_df_tmp[event_df_tmp["action"] == "question"]
        # Convert 'details' from JSON strings to dictionaries (if not already done)
        questions['details'] = questions['details'].apply(ast.literal_eval)

        process_mining_rows = []  # List to collect new rows for process mining

        for user_id in questions["user_id"].unique():
            user_questions = questions[questions["user_id"] == user_id]
            user_questions = user_questions.sort_values("created_at")
            # Filter rows where details['datapoint_count'] == datapoint_count
            user_questions_filtered = user_questions[
                user_questions['details'].apply(lambda x: x['datapoint_count'] == datapoint_count)]

            # Collect rows for process mining
            for index, row in user_questions_filtered.iterrows():
                question = row["details"]["question"]
                # if question is a feature question, mask the feature name to treat all feature questions as one
                if question.strip().startswith("What if I changed the value of"):
                    question = "What if I changed the value of feature"
                elif question.strip().startswith("Is the current value of"):
                    question = "Is the current value of feature high or low?"
                user_id = row["user_id"]
                timestamp = row["created_at"].strftime("%Y-%m-%d %H:%M:%S")
                process_mining_rows.append({"Case id": user_id, "Activity": question, "Timestamp": timestamp})

        # Append new rows to pm_event_df in one operation
        if process_mining_rows:
            pm_event_df = pd.DataFrame(process_mining_rows) if self.pm_event_df is None else pd.concat(
                [self.pm_event_df, pd.DataFrame(process_mining_rows)], ignore_index=True)

        self.pm_event_df = pm_event_df

        # Save as csv file with header
        if filter_by_score is not None:
            self.pm_event_df.to_csv(f"process_mining_{datapoint_count}_{filter_by_score}.csv", index=False, header=True)
        else:
            self.pm_event_df.to_csv(f"process_mining_{datapoint_count}_all.csv", index=False, header=True)
