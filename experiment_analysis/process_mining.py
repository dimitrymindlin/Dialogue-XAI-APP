import pandas as pd

question_text = {23: "Most Important Features",
                 27: "Least Important Features",
                 24: "Feature Attributions",
                 7: "Counterfactuals",
                 11: "Anchors",
                 25: "Ceteris Paribus",
                 13: "Feature Ranges"}


class ProcessMining:
    def __init__(self):
        self.pm_event_df = None

    def _filter_by_score(self, user_scores_df, filter_by_score):
        user_scores_df = user_scores_df.reset_index()
        user_scores_df.rename(columns={"index": "user_id"}, inplace=True)
        final_scores_dict = user_scores_df.set_index("user_id")["Aggr. Final Score"].to_dict()

        if filter_by_score == "best":
            # Only keep users with score > 3-5, directly assign the filtered DataFrame
            user_scores_df = user_scores_df[user_scores_df["Aggr. Final Score"] >= 5]
        elif filter_by_score == "worst":
            # Only keep users with score > 0-2, directly assign the filtered DataFrame
            user_scores_df = user_scores_df[user_scores_df["Aggr. Final Score"] <= 5]
        else:
            raise ValueError("filter_by_score must be 'best' or 'worst'")

        return user_scores_df, final_scores_dict

    def create_pm_csv(self,
                      analysis,
                      datapoint_count=[5],
                      filter_by_score: str = None):
        """
        Preprocesses experiment event df into a format suitable for process mining.
        Format should be table of [CaseID, Activity, Timestamp]
        :param event_df: DataFrame with event data
        :param user_scores_df: DataFrame with user scores, used to filter users by score
        :param datapoint_count: Datapoint count to filter by datapoint (conversation turn)
        :param filter_by_score: Filter users by score, e.g. "best" or "worst"
        """
        interactive_users = analysis.user_df[analysis.user_df["study_group"] == "interactive"]
        if filter_by_score:
            # Keep only users with best or worst scores
            best_users = interactive_users[interactive_users["score_improvement"] >= 3]
            worst_users = interactive_users[interactive_users["score_improvement"] <= 0]

            best_users_ids = best_users["id"].to_list()
            worst_users_ids = worst_users["id"].to_list()

            # filter event_df by user_scores_df
            if filter_by_score == "best":
                questions_df = analysis.questions_over_time_df[
                    analysis.questions_over_time_df["user_id"].isin(best_users_ids)]
            elif filter_by_score == "worst":
                questions_df = analysis.questions_over_time_df[
                    analysis.questions_over_time_df["user_id"].isin(worst_users_ids)]
            else:
                # Take all users
                questions_df = analysis.questions_over_time_df

        pm_event_df = pd.DataFrame(columns=["Case id", "Activity", "Timestamp"])

        process_mining_rows = []  # List to collect new rows for process mining

        for user_id in questions_df["user_id"].unique():
            user_questions = questions_df[questions_df["user_id"] == user_id]
            user_questions = user_questions.sort_values("created_at")

            # Collect rows for process mining
            for index, row in user_questions.iterrows():
                # Filter by datapoint count
                if int(row["datapoint_count"]) not in datapoint_count:
                    continue
                question_id = row["question_id"]
                question = question_text[question_id]
                case_id = row["user_id"] + "_" + str(row["datapoint_count"])
                timestamp = row["created_at"].strftime("%Y-%m-%d %H:%M:%S")
                process_mining_rows.append({"Case id": case_id, "Activity": question, "Timestamp": timestamp})
            break
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
