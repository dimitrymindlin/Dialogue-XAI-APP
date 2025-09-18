import json
import re
import os
import pandas as pd


def load_config(dataset_name=None):
    """Load dataset-specific configuration or use default."""
    default_config = {
        "most_important_feature": {
            "correct_answer": "maritalstatus",
            "fatal_error": "worklifebalance"
        },
        "least_important_feature": {
            "correct_answer": "worklifebalance",
            "fatal_error": "investmentoutcome"
        },
        "non_dataset_features": ["workexperience", "gender"],
        "column_mapping": {
            "user_id": "id",
            "questions": "understanding",
            "answers": "understanding_answers"
        }
    }
    
    if dataset_name:
        config_path = os.path.join("configs", f"{dataset_name}_understanding.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except:
                pass
    
    return default_config


def fix_json_string(s):
    """Replace single quotes around JSON keys/values with double quotes."""
    return re.sub(r'(?<!\w)\'|\'(?!\w)', '"', s)


def add_understanding_question_analysis(user_df, study_group=None, dataset_name=None):
    """
    Analyze understanding questions and add columns with results.
    
    Args:
        user_df (pd.DataFrame): User dataframe with questions and answers
        study_group (str, optional): Study group to filter by
        dataset_name (str, optional): Dataset name for configuration
        
    Returns:
        pd.DataFrame: User dataframe with added columns for question analysis
    """
    config = load_config(dataset_name)
    
    # Make a copy to avoid modifying the original
    df = user_df.copy()
    
    # Filter by study group if provided
    if study_group is not None:
        df = df[df['study_group'] == study_group]
    
    # Get column names from config
    id_col = config["column_mapping"]["user_id"]
    questions_col = config["column_mapping"]["questions"]
    answers_col = config["column_mapping"]["answers"]
    
    # Create new column names for results
    most_important_col = "understanding_q_most_important_feature_answer"
    least_important_col = "understanding_q_least_important_feature_answer"
    decision_rules_col = "understanding_q_decision_rules_text"
    
    # Initialize the new columns
    df[most_important_col] = None
    df[least_important_col] = None
    df[decision_rules_col] = None
    
    # Process each user's answers
    for idx, row in df.iterrows():
        user_id = row[id_col]
        
        # Get questions and answers
        try:
            questions = row[questions_col]
            answers = row[answers_col]
            
            # Check if we have questions and answers
            if pd.isna(questions) or pd.isna(answers):
                continue
                
            # Process the answers
            for q_idx, (question, answer) in enumerate(zip(questions, answers)):
                answer = str(answer).lower() if answer else ""
                
                # Most important feature (first question)
                if q_idx == 0:
                    correct = config["most_important_feature"]["correct_answer"]
                    fatal = config["most_important_feature"]["fatal_error"]
                    
                    if answer == correct:
                        df.loc[idx, most_important_col] = "correct"
                    elif answer == fatal:
                        df.loc[idx, most_important_col] = "fatal_wrong"
                    elif answer in config["non_dataset_features"]:
                        df.loc[idx, most_important_col] = "non_dataset_wrong"
                    else:
                        df.loc[idx, most_important_col] = "wrong"
                
                # Least important feature (second question)
                elif q_idx == 1:
                    correct = config["least_important_feature"]["correct_answer"]
                    fatal = config["least_important_feature"]["fatal_error"]
                    
                    if answer == correct:
                        df.loc[idx, least_important_col] = "correct"
                    elif answer == fatal:
                        df.loc[idx, least_important_col] = "fatal_wrong"
                    elif answer in config["non_dataset_features"]:
                        df.loc[idx, least_important_col] = "non_dataset_wrong"
                    else:
                        df.loc[idx, least_important_col] = "wrong"
                
                # Decision rules text (third question)
                elif q_idx == 2:
                    df.loc[idx, decision_rules_col] = answer
        
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
    
    return df


def plot_failing_users(failing_users):
    # Separate the 'failed' categories and other categories
    failed_users = {'failed_1': failing_users.pop('failed_1'), 'failed_2': failing_users.pop('failed_2')}

    # Sort the remaining categories alphabetically
    sorted_categories = sorted(failing_users.keys())
    sorted_occurrences = [failing_users[category] for category in sorted_categories]

    # Extract 'failed' occurrences
    failed_1_occurrences = [failed_users['failed_1']]
    failed_2_occurrences = [failed_users['failed_2']]

    # Combine sorted categories with 'failed' categories
    combined_categories = sorted_categories + ['failed']
    combined_occurrences = sorted_occurrences + [failed_users['failed_1'] + failed_users['failed_2']]

    # Create indices for the categories
    indices = np.arange(len(combined_categories))

    # Create the bar chart
    plt.figure(figsize=(12, 8))

    # Plot the sorted categories
    plt.bar(indices[:len(sorted_categories)], sorted_occurrences, color='blue', label='Other Categories')

    # Plot the stacked 'failed_1' and 'failed_2'
    plt.bar(indices[-1], failed_1_occurrences, color='orange', label='Failed 1')
    plt.bar(indices[-1], failed_2_occurrences, bottom=failed_1_occurrences, color='red', label='Failed 2')

    # Add title and labels
    plt.title('Occurrences of Failing Users by Category')
    plt.xlabel('Category')
    plt.ylabel('Occurrences')

    # Rotate x-axis labels for better readability
    plt.xticks(indices, combined_categories, rotation=45, ha='right')

    # Add legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()
    plt.ylabel('Occurrences')

    # Rotate x-axis labels for better readability
    plt.xticks(indices, combined_categories, rotation=45, ha='right')

    # Add legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


def get_failing_conditions(user_id, questions_list, answers_list, config=None):
    if config is None:
        config = load_config()
        
    understanding_result = {}
    understanding_points = 0
    fatal_errors = 0
    
    question_types = config["question_types"]
    
    for idx, (question, answer) in enumerate(zip(questions_list, answers_list)):
        answer = answer.lower()
        
        # Skip if question index is out of defined question types
        if idx not in question_types:
            continue
            
        question_type = question_types[idx]
        
        # Handle decision rules text separately
        if question_type == "decision_rules_text":
            understanding_result[question_type] = answer
            continue
            
        # Get question config
        q_config = config["questions"].get(question_type, {})
        correct_answer = q_config.get("correct_answer", "")
        fatal_error = q_config.get("fatal_error", "")
        non_dataset_features = config.get("non_dataset_features", [])
        
        if answer != correct_answer:
            understanding_result[question_type] = {"correct": "false"}
            if answer in non_dataset_features:
                # Non dataset feature
                understanding_result[question_type]['reason'] = f"{question_type}_non_dataset_{answer}"
                understanding_result[question_type]['category'] = 'non_dataset'
                understanding_points += q_config["error_penalties"]["non_dataset"]
            elif answer == fatal_error:
                # Fatal error
                understanding_result[question_type]['reason'] = f"{question_type}_fatal_error_{answer}"
                understanding_result[question_type]['category'] = 'fatal_error'
                fatal_errors += 1
                understanding_points += q_config["error_penalties"]["fatal_error"]
            else:
                # Wrong feature
                understanding_result[question_type]['reason'] = f"{question_type}_wrong_feature:_{answer}"
                understanding_result[question_type]['category'] = 'wrong_feature'
                understanding_points += q_config["error_penalties"]["wrong_feature"]
        else:
            understanding_result[question_type] = {"correct": "true"}
            understanding_result[question_type]['category'] = 'correct'
            understanding_points += q_config.get("weight", 1.0)
            
    understanding_result['understanding_points'] = understanding_points
    return understanding_result, fatal_errors


# Replace single quotes around JSON keys/values with double quotes
def fix_json_string(s):
    # Replace single quotes around JSON keys and values with double quotes
    s = re.sub(r'(?<!\w)\'|\'(?!\w)', '"', s)
    return s


def get_users_failed_final_understanding_check(user_df, study_group, config=None):
    if config is None:
        config = load_config()
    
    # Filter the user_df by study_group
    user_df = user_df[user_df['study_group'] == study_group]
    
    # Get column names from config
    id_col = config["column_names"]["user_id"]
    questions_col = config["column_names"]["questions"]
    answers_col = config["column_names"]["answers"]
    
    understanding_questions = user_df[[id_col, questions_col]]
    understanding_answers = user_df[[id_col, answers_col]]

    # Turn dfs to lists
    question_id_tuples = list(zip(understanding_questions[questions_col], understanding_questions[id_col]))
    understanding_answers_list = list(understanding_answers[answers_col])

    failing_users = {}
    fatal_errors_users = set()
    understanding_scores = []
    
    for questions_tuple, answers in zip(question_id_tuples, understanding_answers_list):
        user_id = questions_tuple[1]
        questions = questions_tuple[0]
        failed = 0

        report_dict, fatal_errors_count = get_failing_conditions(user_id, questions, answers, config)

        understanding_score = report_dict['understanding_points']
        
        # Get question types for key extraction
        question_types = config["question_types"]
        q1_type = question_types.get(0)
        q2_type = question_types.get(1)
        
        score_data = {
            'id': user_id, 
            'understanding_questions_score': understanding_score
        }
        
        # Add categories if available
        if q1_type in report_dict and 'category' in report_dict[q1_type]:
            score_data['q1_category'] = report_dict[q1_type]['category']
        if q2_type in report_dict and 'category' in report_dict[q2_type]:
            score_data['q2_category'] = report_dict[q2_type]['category']
            
        understanding_scores.append(score_data)

        if fatal_errors_count > 0:
            try:
                failing_users[f'fatal_errors_{fatal_errors_count}'] += 1
            except KeyError:
                failing_users[f'fatal_errors_{fatal_errors_count}'] = 1
                
            fatal_errors_users.add(user_id)
            
        # Check if q1 and q2 are correct
        for key, value in report_dict.items():
            if isinstance(value, dict) and value.get('correct') == "false":
                try:
                    failing_users[value["reason"]] += 1
                except KeyError:
                    failing_users[value["reason"]] = 1
                failed += 1

        try:
            failing_users['failed_1'] += 1 if failed == 1 else 0
        except KeyError:
            failing_users['failed_1'] = 1 if failed == 1 else 0
        try:
            failing_users['failed_2'] += 1 if failed == 2 else 0
        except KeyError:
            failing_users['failed_2'] = 1 if failed == 2 else 0
        try:
            if failed == 0:
                failing_users['correct'] += 1
        except KeyError:
            if failed == 0:
                failing_users['correct'] = 1
                
    understanding_scores_df = pd.DataFrame(understanding_scores)
    return failing_users, list(fatal_errors_users), understanding_scores_df