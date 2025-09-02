"""
1_get_raw_experiment_data.py

This script collects and saves the raw experiment data required for further analysis.
It merges Prolific CSV exports, pulls tables from the database, and saves all raw outputs as CSVs.

Expected Inputs:
- Prolific CSV exports in the folder specified by CONFIG["prolific_csv_folder"]
- Access to the experiment database (see CONFIG["db"])

Outputs:
- results/1_raw/prolific_merged.csv
- results/1_raw/users_raw.csv
- results/1_raw/events_raw.csv
- results/1_raw/user_completed_raw.csv

Run this script before any extraction or filtering steps.
"""

import os
import pandas as pd
from analysis_config import PROLIFIC_CSV_FOLDER, RESULTS_DIR, DB_CONFIG, EXPLICITELY_REPLACE_GROUP_NAME, GROUP, PROLIFIC_FOLDER_NAME
from analysis_utils import merge_prolific_csvs, connect_db, make_stage_dirs, fetch_data_as_dataframe, \
    get_data_for_prolific_users, get_study_name_description_if_possible


def main():
    # Create stage dirs inside the experiment-specific results folder
    stage_dirs = make_stage_dirs(RESULTS_DIR)

    # 1. Merge Prolific CSVs
    merged_csv = os.path.join(stage_dirs["raw"], "prolific_merged.csv")
    merge_prolific_csvs(PROLIFIC_CSV_FOLDER, merged_csv)

    # 2. Pull from DB (using fetch_data_as_dataframe for flexibility)
    conn = connect_db(DB_CONFIG)
    users = fetch_data_as_dataframe("SELECT * FROM users", conn)
    events = fetch_data_as_dataframe("SELECT * FROM events", conn)
    user_completed = fetch_data_as_dataframe("SELECT * FROM user_completed", conn)

    # 3. Filter by Prolific users
    users = get_data_for_prolific_users(users, merged_csv)
    # For events and user_completed, filter by user_id column matching users['id']
    user_ids = set(users['id'].astype(str))
    events = events[events['user_id'].astype(str).isin(user_ids)]
    user_completed = user_completed[user_completed['user_id'].astype(str).isin(user_ids)]

    # Apply group name replacement if configured
    if EXPLICITELY_REPLACE_GROUP_NAME:
        import json
        # Load prolific data to get source_file information
        prolific_df = pd.read_csv(merged_csv)
        
        # Merge source_file info into users dataframe based on prolific_id
        if 'source_file' in prolific_df.columns:
            prolific_source_map = prolific_df.set_index('Participant id')['source_file'].to_dict()
            users['source_file'] = users['prolific_id'].map(prolific_source_map)
        
        for idx, row in users.iterrows():
            try:
                profile_data = json.loads(row['profile'])
                if 'study_group_name' in profile_data:
                    current_name = profile_data['study_group_name']
                    source_file = row.get('source_file', '')
                    
                    # Check if we have replacement mappings for this source file
                    if source_file and source_file in EXPLICITELY_REPLACE_GROUP_NAME:
                        source_replacements = EXPLICITELY_REPLACE_GROUP_NAME[source_file]
                        if current_name in source_replacements:
                            profile_data['study_group_name'] = source_replacements[current_name]
                            users.at[idx, 'profile'] = json.dumps(profile_data)
                            print(f"Replaced '{current_name}' with '{source_replacements[current_name]}' for user {row.get('id', idx)} from {source_file}")
            except (json.JSONDecodeError, TypeError):
                # Skip rows with invalid JSON
                continue

    # 4. Filter by study group
    users.to_csv(os.path.join(stage_dirs["raw"], "users_raw.csv"), index=False)
    events.to_csv(os.path.join(stage_dirs["raw"], "events_raw.csv"), index=False)
    user_completed.to_csv(os.path.join(stage_dirs["raw"], "user_completed_raw.csv"), index=False)

    user_experiment_name = get_study_name_description_if_possible(users, GROUP)
    print(f"Raw data collection complete. Results saved in: {RESULTS_DIR}.")
    print(f"Found data for {len(users)} users with experiment names: {user_experiment_name}")

if __name__ == "__main__":
    main()
