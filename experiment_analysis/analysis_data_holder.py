import pandas as pd

class AnalysisDataHolder:
    def __init__(self, user_df, event_df, user_completed_df):
        self.user_df = user_df
        self.event_df = event_df
        self.user_completed_df = user_completed_df
        self.initial_test_preds_df = None
        self.learning_test_preds_df = None
        self.final_test_preds_df = None
        self.questions_over_time_df = None

    def update_dfs(self, user_ids_to_remove=None):
        if user_ids_to_remove is None:
            # Update based on ids in self.user_df
            user_ids_to_keep = self.user_df["id"]
            self.event_df = self.event_df[self.event_df["user_id"].isin(user_ids_to_keep)]
            self.user_completed_df = self.user_completed_df[self.user_completed_df["user_id"].isin(user_ids_to_keep)]
            if self.initial_test_preds_df is not None:
                self.initial_test_preds_df = self.initial_test_preds_df[
                    self.initial_test_preds_df["user_id"].isin(user_ids_to_keep)]
            if self.learning_test_preds_df is not None:
                self.learning_test_preds_df = self.learning_test_preds_df[
                    self.learning_test_preds_df["user_id"].isin(user_ids_to_keep)]
            if self.final_test_preds_df is not None:
                self.final_test_preds_df = self.final_test_preds_df[
                    self.final_test_preds_df["user_id"].isin(user_ids_to_keep)]
            if self.questions_over_time_df is not None:
                self.questions_over_time_df = self.questions_over_time_df[
                    self.questions_over_time_df["user_id"].isin(user_ids_to_keep)]
            return
        self.user_df = self.user_df[~self.user_df["id"].isin(user_ids_to_remove)]
        self.event_df = self.event_df[~self.event_df["user_id"].isin(user_ids_to_remove)]
        self.user_completed_df = self.user_completed_df[~self.user_completed_df["user_id"].isin(user_ids_to_remove)]
        if self.initial_test_preds_df is not None:
            self.initial_test_preds_df = self.initial_test_preds_df[
                ~self.initial_test_preds_df["user_id"].isin(user_ids_to_remove)]
        if self.learning_test_preds_df is not None:
            self.learning_test_preds_df = self.learning_test_preds_df[
                ~self.learning_test_preds_df["user_id"].isin(user_ids_to_remove)]
        if self.final_test_preds_df is not None:
            self.final_test_preds_df = self.final_test_preds_df[
                ~self.final_test_preds_df["user_id"].isin(user_ids_to_remove)]
        if self.questions_over_time_df is not None:
            self.questions_over_time_df = self.questions_over_time_df[
                ~self.questions_over_time_df["user_id"].isin(user_ids_to_remove)]

    def add_initial_test_preds_df(self, initial_test_preds_df):
        self.initial_test_preds_df = initial_test_preds_df

    def add_learning_test_preds_df(self, learning_test_preds_df):
        self.learning_test_preds_df = learning_test_preds_df

    def add_final_test_preds_df(self, final_test_preds_df):
        self.final_test_preds_df = final_test_preds_df

    def add_questions_over_time_df(self, questions_over_time_df):
        """
        Adds the questions_over_time_df to the AnalysisDataHolder.
        Extracts the 'datapoint_count', 'question', and 'question_id' columns from the 'details' column.
        """
        """questions_over_time_df['details'] = questions_over_time_df['details'].apply(json.loads)
        details_df = questions_over_time_df['details'].apply(pd.Series)
        # Concatenate the original DataFrame with the new columns
        questions_over_time_df = pd.concat([questions_over_time_df.drop(columns=['details']), details_df], axis=1)"""
        self.questions_over_time_df = questions_over_time_df

    def create_time_columns(self):
        """
        Prepares a DataFrame with start and end times, and total time spent, for each user.

        :param user_df: DataFrame containing user data, including 'id' and 'created_at' columns.
        :param event_df: DataFrame containing event data, with 'user_id' and 'created_at' columns.
        :return: DataFrame with user IDs, start and end times, study group, and total time spent.
        """

        # Prepare initial time_df
        time_df = pd.DataFrame({
            "user_id": self.user_df["id"],
        })

        # Get start event for each user from event_df
        start_time_df = self.event_df.groupby("user_id")["created_at"].min()
        time_df["event_start_time"] = time_df["user_id"].map(start_time_df)

        # Get Experiment end time and start time
        experiemnt_end_time_df = self.user_completed_df.groupby("user_id")["prolific_end_time"].max()
        time_df["prolific_end_time"] = time_df["user_id"].map(experiemnt_end_time_df)
        experiment_start_time_df = self.user_completed_df.groupby("user_id")["prolific_start_time"].min()
        time_df["prolific_start_time"] = time_df["user_id"].map(experiment_start_time_df)

        # Get end time for each user from event_df
        end_time_df = self.event_df.groupby("user_id")["created_at"].max()
        time_df["event_end_time"] = time_df["user_id"].map(end_time_df)

        # Ensure datetime format
        time_df["prolific_start_time"] = pd.to_datetime(time_df["prolific_start_time"])
        time_df["event_start_time"] = pd.to_datetime(time_df["event_start_time"])
        time_df["event_end_time"] = pd.to_datetime(time_df["event_end_time"])
        time_df["prolific_end_time"] = pd.to_datetime(time_df["prolific_end_time"])

        # Calculate total_time in minutes
        # Time spent in the learning phase (from first event to last event)
        time_df["total_learning_time"] = (time_df["event_end_time"] - time_df[
            "event_start_time"]).dt.total_seconds() / 60

        # Time spent in experiment instructions (prolific start - first event)
        time_df["exp_instruction_time"] = (time_df["event_start_time"] - time_df[
            "prolific_start_time"]).dt.total_seconds() / 60

        # Total experiment time (from prolific start to prolific end)
        time_df["total_exp_time"] = (time_df["prolific_end_time"] - time_df[
            "prolific_start_time"]).dt.total_seconds() / 60

        self.user_df = self.user_df.merge(time_df, left_on="id", right_on="user_id", how="left")