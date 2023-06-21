"""Explanation action.

This action controls the explanation generation operations.
"""
from explain.actions.utils import gen_parse_op_text


def explain_operation(conversation, parse_text, i, **kwargs):
    """The explanation operation."""
    # TODO(satya): replace explanation generation code here

    # Example code loading the model
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    regen = conversation.temp_dataset.contents['ids_to_regenerate']
    parse_op = gen_parse_op_text(conversation)

    # Note, do we want to remove parsing for lime -> mega_explainer here?
    if parse_text[i + 1] == 'features' or parse_text[i + 1] == 'lime':
        # mega explainer explanation case
        return explain_feature_importances(conversation, data, parse_op, regen)
    if parse_text[i + 1] == 'cfe':
        return explain_cfe(conversation, data, parse_op, regen)
    if parse_text[i + 1] == 'shap':
        # This is when a user asks for a shap explanation
        raise NotImplementedError
    raise NameError(f"No explanation operation defined for {parse_text}")


def explain_feature_importances(conversation, data, parse_op, regen):
    """Get Lime or SHAP explanation, considering fidelity (mega explainer functionality)"""
    mega_explainer_exp = conversation.get_var('mega_explainer').contents
    explanation_text = mega_explainer_exp.summarize_explanations(data,
                                                                 filtering_text=parse_op,
                                                                 ids_to_regenerate=regen)
    conversation.store_followup_desc(explanation_text)
    return explanation_text, 1


def get_feature_importance_by_feature_id(conversation,
                                         data,
                                         regen,
                                         feature_id):
    """Get Lime or SHAP explanation for a specific feature, considering fidelity (mega explainer functionality)"""
    feature_name = data.columns[feature_id]
    mega_explainer_exp = conversation.get_var('mega_explainer').contents
    feature_importances, _ = mega_explainer_exp.get_feature_importances(data=data, ids_to_regenerate=regen)
    label = list(feature_importances.keys())[0]  # TODO: Currently only working for 1 instance in data.
    # Get ranking of feature importance (position in feature_importances)
    feature_importance_ranking = list(feature_importances[label].keys()).index(feature_name)
    feature_importance_value = feature_importances[label][feature_name]
    feature_importance_value = round(feature_importance_value[0], 3)

    # If the feature importance value is 0, then the feature is not important for the prediction
    if feature_importance_value == 0:
        output_text = f"The feature <em>{feature_name}</em> is not important for the prediction."
        return output_text, 1, 0

    feature_importance_ranking_name = feature_importance_ranking + 1
    if feature_importance_ranking_name == 1:
        feature_importance_ranking_name = 'most'
    elif feature_importance_ranking_name == 2:
        feature_importance_ranking_name = 'second most'

    output_text = f"The feature <em>{feature_name}</em> is the <em>{feature_importance_ranking_name}</em>. important feature with a value of {str(feature_importance_value)}. "
    output_text += f"This means that if the feature didn't have the current value, the prediction probability would change by the given amount."
    return output_text, 1, feature_importance_value


def explain_cfe(conversation, data, parse_op, regen):
    """Get CFE explanation"""
    dice_tabular = conversation.get_var('tabular_dice').contents
    out = dice_tabular.summarize_explanations(data,
                                              filtering_text=parse_op,
                                              ids_to_regenerate=regen)
    additional_options, short_summary = out
    conversation.store_followup_desc(additional_options)
    return short_summary, 1


def explain_cfe_by_given_features(conversation,
                                  data,
                                  feature_names_list):
    """Get CFE explanation when changing the features in the feature_names_list
    Args:
        conversation: Conversation object
        data: Dataframe of data to explain
        feature_names_list: List of feature names to change
    """
    dice_tabular = conversation.get_var('tabular_dice').contents
    cfes = dice_tabular.run_explanation(data, "opposite", features_to_vary=feature_names_list)
    for instance_id, cfe in cfes.items():
        change_string = dice_tabular.summarize_cfe(cfe, data)
    conversation.store_followup_desc(change_string[0])
    return change_string


def explain_anchor_changeable_attributes_without_effect(conversation, data, parse_op, regen):
    """Get Anchor explanation"""
    anchor_exp = conversation.get_var('tabular_anchor').contents
    out = anchor_exp.summarize_explanations(data,
                                            filtering_text=parse_op,
                                            ids_to_regenerate=regen)
    additional_options, short_summary = out
    conversation.store_followup_desc(additional_options)
    return short_summary, 1
