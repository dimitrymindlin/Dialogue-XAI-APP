"""Explanation action.

This action controls the explanation generation operations.
"""
import base64
import io
import matplotlib.pyplot as plt
import shap

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
        return explain_local_feature_importances(conversation, data, parse_op, regen)
    if parse_text[i + 1] == 'cfe':
        return explain_cfe(conversation, data, parse_op, regen)
    if parse_text[i + 1] == 'shap':
        # This is when a user asks for a shap explanation
        raise NotImplementedError
    raise NameError(f"No explanation operation defined for {parse_text}")


def explain_local_feature_importances(conversation, data, parse_op, regen, as_text=True):
    """Get Lime or SHAP explanation, considering fidelity (mega explainer functionality)"""
    mega_explainer_exp = conversation.get_var('mega_explainer').contents
    if as_text:
        explanation_text = mega_explainer_exp.summarize_explanations(data,
                                                                     filtering_text=parse_op,
                                                                     ids_to_regenerate=regen)
        conversation.store_followup_desc(explanation_text)
        return explanation_text, 1
    else:
        feature_importances = mega_explainer_exp.get_feature_importances(data=data, ids_to_regenerate=regen)
        # Extract inner dict ... TODO: This is a hacky way to do this.
        top_features_dict = feature_importances[0]
        predicted_label = list(top_features_dict.keys())[0]
        top_features_dict = top_features_dict[predicted_label]
        return top_features_dict, 1


def explain_global_feature_importances(conversation, as_plot=True):
    global_shap_explainer = conversation.get_var('global_shap').contents
    # get shap explainer
    explanation = global_shap_explainer.get_explanations()
    if as_plot:
        shap.plots.bar(explanation, show=False)

        # Get current figure and axis
        fig = plt.gcf()
        ax = plt.gca()

        # Get x-tick labels
        xticks = ax.get_xticklabels()

        # Show only every second x-tick
        for i in range(len(xticks)):
            if i % 2 == 1:  # Skip every second tick
                xticks[i].set_visible(False)

        # Change x label
        ax.set_xlabel('Average Importance', fontsize=18)

        # Increase the y-axis label size
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.tight_layout()

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Clear the current plot to free memory
        plt.close()

        html_string = f'<img src="data:image/png;base64,{image_base64}" alt="Your Plot">' \
                      f'<span>This is a general trend and might change for individual instances.</span>'
        return html_string, 1


def explain_feature_importances_as_plot(conversation, data, parse_op, regen):
    data_dict, _ = explain_local_feature_importances(conversation, data, parse_op, regen, as_text=False)
    labels = list(data_dict.keys())
    values = [val[0] for val in data_dict.values()]

    # Reverse the order
    labels = labels[::-1]
    values = values[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color=['red' if v < 0 else 'blue' for v in values])

    # Increase the y-axis label size
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # Clear the current plot to free memory
    plt.close()

    html_string = f'<img src="data:image/png;base64,{image_base64}" alt="Your Plot">' \
                  f'<span>The blue bars show that the variable contributes to the current prediction,' \
                  f'and the red bars show attributes that count towards the opposite prediction.</span>'

    return html_string, 1


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

    # TODO: Outsource text generation to answer_templates and get relationship to first feature and previous feature as
    #  answer type.

    # If the feature importance value is 0, then the feature is not important for the prediction
    if feature_importance_value == 0:
        output_text = f"The feature <b>{feature_name}</b> is not important for the prediction."
        return output_text, 1, 0

    feature_importance_ranking_name = feature_importance_ranking + 1
    if feature_importance_ranking_name == 1:
        feature_importance_ranking_name = 'most'
    elif feature_importance_ranking_name == 2:
        feature_importance_ranking_name = 'second most'
    elif feature_importance_ranking_name == 3:
        feature_importance_ranking_name = 'third most'
    else:
        feature_importance_ranking_name = str(feature_importance_ranking_name) + "th most"

    output_text = f"The feature <b>{feature_name}</b> is the {feature_importance_ranking_name} important feature."
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
                                  feature_names_list,
                                  top_features):
    """Get CFE explanation when changing the features in the feature_names_list
    Args:
        conversation: Conversation object
        data: Dataframe of data to explain
        feature_names_list: List of feature names to change
        top_features: dict sorted by most important feature
    """
    dice_tabular = conversation.get_var('tabular_dice').contents
    cfes = dice_tabular.run_explanation(data, "opposite", features_to_vary=feature_names_list)
    initial_feature_to_vary = feature_names_list[0]  # TODO: So far expecting that user only selects one feature.
    change_string_prefix = ""
    if cfes[data.index[0]].cf_examples_list[0].final_cfs_df is None:
        change_string_prefix = f"The attribute {initial_feature_to_vary} cannot be changed <b>by itself</b> to alter the prediction. <br><br>"
        # Find cfs with more features than just one feature by iterating over the top features and adding them
        for new_feature, importance in top_features.items():
            if new_feature not in feature_names_list:
                feature_names_list.append(new_feature)
                cfes = dice_tabular.run_explanation(data, "opposite", features_to_vary=feature_names_list)
                if cfes[data.index[0]].cf_examples_list[0].final_cfs_df is not None:
                    break
    change_string, _ = dice_tabular.summarize_cfe_for_given_attribute(cfes, data, initial_feature_to_vary)
    conversation.store_followup_desc(change_string)
    change_string = change_string_prefix + change_string
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


def explain_feature_statistic(conversation, feature_name):
    feature_stats_exp = conversation.get_var('feature_statistics_explainer').contents
    explanation = feature_stats_exp.get_single_feature_statistic(feature_name)
    return explanation


def explain_ceteris_paribus(conversation, data, feature_name):
    ceteris_paribus_exp = conversation.get_var('ceteris_paribus').contents

    #ceteris_paribus_exp.get_simplified_explanation(data, feature_name, as_plot=False)


    # Plotly figure
    fig = ceteris_paribus_exp.get_explanation(data, feature_name)

    # Convert the figure to PNG as a BytesIO object
    buf = io.BytesIO()
    fig.write_image(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # Create the HTML string with the base64 image
    html_string = f'<img src="data:image/png;base64,{image_base64}" alt="Your Plot">' \
                  f'<span>When the prediction value is on the Y-axis above 0.5 , ' \
                  f'the model would predict likely to have diabetes and,' \
                  f'when it is below 0.5, unlikely to have diabetes..</span>'

    return html_string, 1
