# Unit test to test prediction task complexity
import copy

import pandas as pd

from explain.explanations.test_instances import calculate_prediction_task_complexity


def test_prediction_task_complexity():
    original_instance = {
        'Age Group': 3,
        'Gender': 0,
        'Job Level': 1,
        'Housing Type': 2,
        'Saving accounts': 1,
        'Checking account': 0,
        'Credit amount': 1864,
        'Credit Duration': 18,
        'Credit Purpose': 3
    }

    original_instance = pd.DataFrame.from_dict(original_instance, orient='index', columns=['331'])

    # Create feature importances dict
    feature_importances = {'Age Group': -0.4,
                           'Credit Duration': 0.1,
                           'Credit Purpose': 0.1,
                           'Credit amount': 0.1,
                           'Checking account': 0.01,
                           'Saving accounts': 0.01,
                           'Housing Type': 0.001,
                           'Job Level': 0.001,
                           'Gender': 0.0005}

    # Create model certainties
    model_certainty_old = (0.1, 0.9)
    model_certainty_new = (0.1, 0.9)

    identical_instance = copy.deepcopy(original_instance)
    assert calculate_prediction_task_complexity(original_instance, identical_instance, feature_importances,
                                                model_certainty_old, model_certainty_new) == 0.0

    # Change most important feature
    changed_most_important = copy.deepcopy(original_instance)
    changed_most_important['331']['Age Group'] = 2
    changed_most_important_complexity = calculate_prediction_task_complexity(original_instance,
                                                                             changed_most_important,
                                                                             feature_importances,
                                                                             model_certainty_old, model_certainty_new)
    assert changed_most_important_complexity > 0.0

    # change least important feature
    changed_least_important = copy.deepcopy(original_instance)
    changed_least_important['331']['Gender'] = 1
    changed_least_important_complexity = calculate_prediction_task_complexity(original_instance,
                                                                              changed_least_important,
                                                                              feature_importances,
                                                                              model_certainty_old, model_certainty_new)
    assert changed_least_important_complexity > 0.0
    assert changed_least_important_complexity < changed_most_important_complexity

    # change all features slightly
    changed_all_features = copy.deepcopy(original_instance)
    changed_all_features['331']['Age Group'] = 2

