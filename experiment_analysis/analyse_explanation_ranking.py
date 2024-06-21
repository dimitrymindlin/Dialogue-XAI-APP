import pandas as pd
import json

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


def extract_questionnaires(user_df):
    # Get questionnaires
    questionnaires = user_df[['id', 'questionnaires']]
    # Normalize the questionnaires column
    questionnaires = pd.concat([questionnaires.drop(['questionnaires'], axis=1),
                                questionnaires['questionnaires'].apply(pd.Series)], axis=1)
    # drop empty columns
    questionnaires.drop([0], axis=1, inplace=True)
    # Change column names
    questionnaires.columns = ['id', 'ranking_q', 'self_assessment_q', 'exit_q']
    # Apply to user_df and drop id column
    user_df = user_df.merge(questionnaires, on='id', how='left')
    return user_df
