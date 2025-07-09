import pandas as pd
import json

# Will hold IDs of users to exclude
exclude_users = []

# Helper to deduplicate questionnaire entries
def _dedupe(lst):
    """Skip the first empty dict and drop consecutive duplicate questionnaire entries."""
    seen = set()
    uniq = []
    for item in lst[1:]:  # skip initial {}
        key = item if isinstance(item, str) else json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq

from collections import defaultdict

interactive_question_ids = [23, 27, 24, 7, 11, 25, 13]
static_explanation_ids = [23, 27, 24, 7, 11]


def _borda_count_ranking(rankings):
    # Initialize the score dictionary for question IDs
    score_dict = defaultdict(int)

    # Find the highest rank that will be used to calculate points
    max_rank = max(rank for ranking in rankings for _, rank in ranking if rank != "")

    # Calculate Borda count for each ranking list
    for ranking in rankings:
        for question_id, rank in ranking:
            if rank != "":
                # The points are now max_rank + 1 - rank because lower rank means higher importance
                score_dict[question_id] += (max_rank + 1 - rank)

    # Sort the question IDs based on their Borda score in descending order
    overall_ranking = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)

    return [question_id for question_id, _ in overall_ranking]


def get_interactive_ranking(user_df):
    # Get questionnaires
    questionnaires = user_df[['id', 'questionnaires']]
    # Normalize the questionnaires column
    questionnaires = pd.concat([questionnaires.drop(['questionnaires'], axis=1),
                                questionnaires['questionnaires'].apply(pd.Series)], axis=1)
    # Get the interactive ranking (col 1)
    ranking_list = list(questionnaires[1].apply(lambda x: json.loads(x)))
    all_rankings = []
    for ranking_dict in ranking_list:
        ranking = ranking_dict['question_ranking']["answers"]
        # Zip the question IDs with the ranking
        ranking = list(zip(interactive_question_ids, ranking))
        all_rankings.append(ranking)
    overall_ranking = _borda_count_ranking(all_rankings)
    print(overall_ranking)
    # Replace the question IDs with the corresponding question text
    question_text = {23: "Most Important Features",
                     27: "Least Important Features",
                     24: "Feature Attributions",
                     7: "Counterfactuals",
                     11: "Anchors",
                     25: "Ceteris Paribus",
                     13: "Feature Ranges"}
    overall_ranking = [(question_text[question_id]) for question_id in overall_ranking]
    print(overall_ranking)
    return overall_ranking


def get_static_ranking(user_df):
    # Get questionnaires
    questionnaires = user_df[['id', 'questionnaires']]
    # Normalize the questionnaires column
    questionnaires = pd.concat([questionnaires.drop(['questionnaires'], axis=1),
                                questionnaires['questionnaires'].apply(pd.Series)], axis=1)
    # Get the interactive ranking (col 1)
    ranking_list = list(questionnaires[1].apply(lambda x: json.loads(x)))
    all_rankings = []
    for ranking_dict in ranking_list:
        ranking = ranking_dict['question_ranking']["answers"]
        # Zip the question IDs with the ranking
        ranking = list(zip(static_explanation_ids, ranking))
        all_rankings.append(ranking)
    overall_ranking = _borda_count_ranking(all_rankings)
    print(overall_ranking)
    # Replace the question IDs with the corresponding question text
    question_text = {23: "Feature Attributions",
                     27: "Counterfactuals",
                     24: "Anchors",
                     7: "Ceteris Paribus",
                     11: "Feature Ranges"}
    overall_ranking = [(question_text[question_id]) for question_id in overall_ranking]
    print(overall_ranking)
    return overall_ranking


def extract_questionnaires(user_df, dataset_name):
    # 1. Parse JSON strings into lists of dicts if necessary
    parsed = user_df['questionnaires'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    # 2. Remove duplicates and skip the initial empty dict, then flatten into a single dict per row
    flat_dicts = parsed.apply(lambda lst: {
        k: v
        for d in _dedupe(lst)
        for k, v in (json.loads(d) if isinstance(d, str) else d).items()
    })

    # 3. Build a tidy DataFrame directly from the flat dicts
    questionnaires = pd.DataFrame({
        'id': user_df['id'],
        'self_assessment_q': flat_dicts.map(lambda d: d.get('self_assessment')),
        'understanding_q':          flat_dicts.map(lambda d: d.get('understanding')),
        'exit_q':             flat_dicts.map(lambda d: d.get('exit')),
    })

    # 4. Merge back into user_df
    merged = user_df.merge(questionnaires, on='id', how='left')
    # 5. Unpack 'understanding_q' dict into separate question and answer columns
    merged['understanding_q_questions'] = merged['understanding_q'].map(lambda d: d.get('questions', []) if isinstance(d, dict) else [])
    merged['understanding_q_answers']   = merged['understanding_q'].map(lambda d: d.get('answers',   []) if isinstance(d, dict) else [])

    # Check attantion checks
    if dataset_name == "adult":
        exclude_users = merged.loc[
            merged['understanding_q_answers'].apply(lambda lst: bool(lst and lst[0] in ['workLifeBalance', 'yearOfCareerStart', 'gender'])),
            'id'
        ].tolist()
    elif dataset_name == "diabetes":
        exclude_users = merged.loc[
            merged['understanding_q_answers'].apply(
                lambda lst: bool(lst and lst[0] in ['gender', 'height', 'bloodgGroup'])),
            'id'
        ].tolist()

    # Print the IDs of users to exclude
    print(f"Users to exclude: {exclude_users}")
    print(f"Total users to exclude: {len(exclude_users)}")

    return merged, exclude_users
