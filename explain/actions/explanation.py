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


def explain_feature_importances(conversation, data, parse_op, regen, return_full_summary=False):
    """Get Lime or SHAP explanation, considering fidelity (mega explainer functionality)"""
    mega_explainer_exp = conversation.get_var('mega_explainer').contents
    full_summary, short_summary = mega_explainer_exp.summarize_explanations(data,
                                                                            filtering_text=parse_op,
                                                                            ids_to_regenerate=regen)
    conversation.store_followup_desc(full_summary)
    if return_full_summary:
        return full_summary, 1
    return short_summary, 1


def get_feature_importance_by_feature_id(conversation, data, regen, feature_name):
    """Get Lime or SHAP explanation, considering fidelity (mega explainer functionality)"""
    mega_explainer_exp = conversation.get_var('mega_explainer').contents
    feature_importances, scores = mega_explainer_exp.get_feature_importances(data=data, ids_to_regenerate=regen)
    label = list(feature_importances.keys())[0]  # TODO: Currently only working for 1 instance in data.
    # Get ranking of feature importance (position in feature_importances)
    feature_importance_ranking = list(feature_importances[label].keys()).index(feature_name)
    feature_importance_value = feature_importances[label][feature_name]
    output_text = f"The feature {feature_name} is the {feature_importance_ranking}. important feature with a value of {feature_importance_value[0]}."
    return output_text, 1


def explain_cfe(conversation, data, parse_op, regen):
    """Get CFE explanation"""
    dice_tabular = conversation.get_var('tabular_dice').contents
    out = dice_tabular.summarize_explanations(data,
                                              filtering_text=parse_op,
                                              ids_to_regenerate=regen)
    additional_options, short_summary = out
    conversation.store_followup_desc(additional_options)
    return short_summary, 1


def explain_cfe_single_feature(conversation, data, parse_op, feature_name):
    dice_tabular = conversation.get_var('tabular_dice').contents
    out = dice_tabular.get_cfe_with_single_feature_change(data, parse_op, feature_name)
    additional_options, short_summary = out
    conversation.store_followup_desc(additional_options)
    return short_summary, 1


def explain_anchor(conversation, data, parse_op, regen):
    """Get Anchor explanation"""
    anchor_exp = conversation.get_var('tabular_anchor').contents
    out = anchor_exp.summarize_explanations(data,
                                            filtering_text=parse_op,
                                            ids_to_regenerate=regen)
    additional_options, short_summary = out
    conversation.store_followup_desc(additional_options)
    return short_summary, 1