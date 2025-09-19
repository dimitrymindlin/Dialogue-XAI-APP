"""
1_get_raw_experiment_data.py

This script collects and saves the raw experiment data required for further analysis.
It merges Prolific CSV exports, pulls tables from the database, and saves all raw outputs as CSVs.

Expected Inputs:
- Prolific CSV exports inside the active ``data_*`` experiment folder
- Access to the experiment database defined by ``analysis_config.DB_CONFIG``

Outputs:
- results/1_raw/prolific_merged.csv
- results/1_raw/users_raw.csv
- results/1_raw/events_raw.csv
- results/1_raw/user_completed_raw.csv
- results/1_raw/user_prolific_ml_mapping.csv - Maps user_ids to prolific info and ML knowledge

Run this script before any extraction or filtering steps.
"""

import os
import json
import pandas as pd
import analysis_config as config
from analysis_utils import merge_prolific_csvs, connect_db, make_stage_dirs, fetch_data_as_dataframe, \
    get_data_for_prolific_users, get_study_name_description_if_possible
from ml_knowledge_utils import normalize_ml_knowledge_value, ML_KNOWLEDGE_TEXT_TO_NUM


def create_user_prolific_mapping(users_df, prolific_df, output_path):
    """
    Create a CSV mapping internal user_ids to prolific information and ML knowledge.

    Args:
        users_df: DataFrame with user data including id, prolific_id, and profile
        prolific_df: DataFrame with prolific export data
        output_path: Path where the mapping CSV should be saved

    Returns:
        DataFrame with the mapping
    """
    # Initialize the mapping dataframe with user_id and prolific_id
    mapping_data = []

    for idx, user_row in users_df.iterrows():
        user_id = user_row['id']
        prolific_id = user_row.get('prolific_id')

        # Extract ML knowledge from profile
        ml_knowledge_num = None
        ml_knowledge_text = None
        profile_data = user_row.get('profile')

        if pd.notna(profile_data):
            try:
                profile_dict = json.loads(profile_data) if isinstance(profile_data, str) else profile_data
                fam_ml_val = profile_dict.get('fam_ml_val')

                if fam_ml_val is not None:
                    # Normalize the ML knowledge value (handles both text and numeric formats)
                    ml_knowledge_text = normalize_ml_knowledge_value(fam_ml_val)
                    if ml_knowledge_text:
                        ml_knowledge_num = ML_KNOWLEDGE_TEXT_TO_NUM.get(ml_knowledge_text)
            except (json.JSONDecodeError, TypeError):
                pass

        # Get prolific information if available
        prolific_info = {}
        if pd.notna(prolific_id) and prolific_id in prolific_df['Participant id'].values:
            prolific_row = prolific_df[prolific_df['Participant id'] == prolific_id].iloc[0]
            prolific_info = {
                'age': prolific_row.get('Age'),
                'sex': prolific_row.get('Sex'),
                'ethnicity': prolific_row.get('Ethnicity simplified'),
                'country_of_birth': prolific_row.get('Country of birth'),
                'country_of_residence': prolific_row.get('Country of residence'),
                'nationality': prolific_row.get('Nationality'),
                'language': prolific_row.get('Language'),
                'student_status': prolific_row.get('Student status'),
                'employment_status': prolific_row.get('Employment status'),
                'status': prolific_row.get('Status'),
                'time_taken_seconds': prolific_row.get('Time taken'),
            }
        else:
            prolific_info = {
                'age': None,
                'sex': None,
                'ethnicity': None,
                'country_of_birth': None,
                'country_of_residence': None,
                'nationality': None,
                'language': None,
                'student_status': None,
                'employment_status': None,
                'status': None,
                'time_taken_seconds': None,
            }

        # Combine user_id, ml_knowledge, and prolific info
        mapping_row = {
            'user_id': user_id,
            'ml_knowledge_num': ml_knowledge_num,
            'ml_knowledge_text': ml_knowledge_text,
            **prolific_info
        }
        mapping_data.append(mapping_row)

    # Create DataFrame and save
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(output_path, index=False)

    print(f"\nCreated user-prolific mapping with {len(mapping_df)} users")
    print(f"ML knowledge distribution:")
    if 'ml_knowledge_text' in mapping_df.columns:
        print(mapping_df['ml_knowledge_text'].value_counts(dropna=False))

    return mapping_df


def main():
    # Create stage dirs inside the experiment-specific results folder
    stage_dirs = make_stage_dirs(config.RESULTS_DIR, stages=["raw"])

    # 1. Merge Prolific CSVs
    merged_csv = os.path.join(stage_dirs["raw"], "prolific_merged.csv")
    merge_prolific_csvs(config.PROLIFIC_CSV_FOLDER, merged_csv)

    # 2. Pull from DB (using fetch_data_as_dataframe for flexibility)
    conn = connect_db(config.DB_CONFIG)
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
    if config.EXPLICITELY_REPLACE_GROUP_NAME:
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
                source_file = row.get('source_file', '')
                if not source_file:
                    continue

                if source_file not in config.EXPLICITELY_REPLACE_GROUP_NAME:
                    continue

                source_replacements = config.EXPLICITELY_REPLACE_GROUP_NAME[source_file]
                current_name = None
                has_study_group_name = False

                if isinstance(profile_data, dict) and 'study_group_name' in profile_data:
                    current_name = profile_data['study_group_name']
                    has_study_group_name = True
                else:
                    current_name = str(row.get('study_group', '')).strip()

                if current_name and current_name in source_replacements:
                    new_name = source_replacements[current_name]

                    if has_study_group_name:
                        profile_data['study_group_name'] = new_name
                        users.at[idx, 'profile'] = json.dumps(profile_data)
                    else:
                        users.at[idx, 'profile'] = json.dumps(profile_data)

                    if not has_study_group_name:
                        users.at[idx, 'study_group'] = new_name
                    elif not row.get('study_group'):
                        # Fallback: derive study group from new_name if original value missing
                        parts = new_name.split('-')
                        if len(parts) > 1:
                            users.at[idx, 'study_group'] = parts[1]

                    print(
                        f"Replaced '{current_name}' with '{new_name}' for user {row.get('id', idx)} from {source_file}"
                    )
            except (json.JSONDecodeError, TypeError):
                # Skip rows with invalid JSON
                continue

    # 4. Filter by study group
    users.to_csv(os.path.join(stage_dirs["raw"], "users_raw.csv"), index=False)
    events.to_csv(os.path.join(stage_dirs["raw"], "events_raw.csv"), index=False)
    user_completed.to_csv(os.path.join(stage_dirs["raw"], "user_completed_raw.csv"), index=False)

    # 5. Create user_id to prolific mapping with ML knowledge
    prolific_df = pd.read_csv(merged_csv)
    user_prolific_mapping_path = os.path.join(stage_dirs["raw"], "user_prolific_ml_mapping.csv")
    create_user_prolific_mapping(users, prolific_df, user_prolific_mapping_path)

    user_experiment_name = get_study_name_description_if_possible(users, config.GROUP)
    print(f"Raw data collection complete. Results saved in: {config.RESULTS_DIR}.")
    print(f"Found data for {len(users)} users with experiment names: {user_experiment_name}")

if __name__ == "__main__":
    main()
