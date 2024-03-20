"""Explanation action.

This action controls the explanation generation operations.
"""
import base64
import io
import matplotlib.pyplot as plt
import shap
from matplotlib.ticker import FuncFormatter

from data.response_templates.ceteris_paribus_template import cp_categorical_template, cp_numerical_template
from explain.actions.utils import gen_parse_op_text


def fig_to_base64(fig):
    """
    Converts a matplotlib or plotly figure to a base64 string.

    Parameters:
    - fig: A matplotlib or plotly Figure object.

    Returns:
    - A base64-encoded string representation of the figure.
    """
    buf = io.BytesIO()

    # Check if the figure is a plotly Figure
    if 'plotly.graph_objs._figure.Figure' in str(type(fig)):
        fig.write_image(buf, format='png')
    else:  # Assume matplotlib
        fig.savefig(buf, format='png')

    buf.seek(0)  # Go to the beginning of the buffer
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')  # Encode as base64
    buf.close()  # Close the buffer
    return image_base64


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


def explain_local_feature_importances(conversation,
                                      data,
                                      parse_op,
                                      regen,
                                      as_text=True,
                                      template_manager=None):
    """Get Lime or SHAP explanation, considering fidelity (mega explainer functionality)"""
    mega_explainer_exp = conversation.get_var('mega_explainer').contents
    if as_text:
        explanation_text = mega_explainer_exp.summarize_explanations(data,
                                                                     filtering_text=parse_op,
                                                                     ids_to_regenerate=regen,
                                                                     template_manager=template_manager)
        conversation.store_followup_desc(explanation_text)
        return explanation_text, 1
    else:
        feature_importances = mega_explainer_exp.get_feature_importances(data=data, ids_to_regenerate=regen)
        # Extract inner dict ... TODO: This is a hacky way to do this.
        top_features_dict = feature_importances[0]
        predicted_label = list(top_features_dict.keys())[0]
        top_features_dict = top_features_dict[predicted_label]
        return top_features_dict, 1


def get_fi_follow_up(conversation, data, parse_op, regen, template_manager):
    pass


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


def explain_feature_importances_as_plot(conversation,
                                        data,
                                        parse_op,
                                        regen,
                                        current_prediction_string: str,
                                        prediction_id: int,
                                        target_class=None):
    explanation_dict, _ = explain_local_feature_importances(conversation, data, parse_op, regen, as_text=False)
    labels = list(explanation_dict.keys())
    values = [val[0] for val in explanation_dict.values()]

    # get current attribute values
    tm = conversation.get_var('template_manager').contents
    df_with_feature = tm.decode_numeric_columns_to_names(data)

    # Reverse the order
    labels = labels[::-1]
    values = values[::-1]

    # Turn labels to display names
    template_manager = conversation.get_var('template_manager').contents
    feature_name_to_display_name = template_manager.feature_display_names.feature_name_to_display_name
    for feature, display_name in feature_name_to_display_name.items():
        feature_value_name = df_with_feature[feature].iloc[0]
        if feature in labels:
            labels[labels.index(feature)] = f"{display_name} ({feature_value_name})"

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color=['red' if v < 0 else 'blue' for v in values])

    # Increase the y-axis label size
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.tight_layout()

    # Format x-ticks to have only two decimal places
    ax.xaxis.set_major_formatter(FuncFormatter('{:.2f}'.format))

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # Clear the current plot to free memory
    plt.close()

    # TODO: Change to multilabel and get correct class name.
    class_0_label = conversation.class_names[0]

    # TODO: Only for binary case?
    html_string = f'<img src="data:image/png;base64,{image_base64}" alt="Your Plot">' \
                  f'<span>Blue bars = attributes in favor of predicting <b>{current_prediction_string}</b>. <br>' \
                  f'Red bars = attributes against current prediction.</span>'

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
    out, desired_class = dice_tabular.summarize_explanations(data,
                                                             filtering_text=parse_op,
                                                             ids_to_regenerate=regen,
                                                             template_manager=conversation.get_var(
                                                                 'template_manager').contents)
    short_summary = out
    return short_summary, desired_class


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


def explain_anchor_changeable_attributes_without_effect(conversation, data, parse_op, regen, template_manager):
    """Get Anchor explanation"""
    anchor_exp = conversation.get_var('tabular_anchor').contents
    return anchor_exp.summarize_explanations(data, ids_to_regenerate=regen, template_manager=template_manager)


def explain_feature_statistic(conversation,
                              feature_name=None,
                              as_string=False):
    """
    Get feature statistics explanation for a single feature or all features.
    """
    feature_stats_exp = conversation.get_var('feature_statistics_explainer').contents
    if feature_name is not None:
        explanation = feature_stats_exp.get_single_feature_statistic(feature_name, as_string=as_string)
    else:
        explanation = feature_stats_exp.get_all_feature_statistics(as_string=as_string)
    if not as_string:
        # Convert the figure to PNG as a BytesIO object
        image_base64 = fig_to_base64(explanation)
        # Create the HTML string with the base64 image
        html_string = f'<img src="data:image/png;base64,{image_base64}" alt="Your Plot">' \
                      f'<span>Distribution of the possible values for {feature_name}.</span>'
        return html_string
    return explanation


def explain_ceteris_paribus(conversation, data, feature_name, instance_type_name, opposite_class, as_text=False):
    def write_tipping_point_cp_categorical(feature_id):
        ### Simplified Text version
        x_flip_value_list = ceteris_paribus_exp.get_feature_values_flipping_prediction(data, feature_name)
        """if x_flip_value is None:
            return f"For the given {instance_type_name}, variations only in <b>{feature_name}</b> have no impact on the model prediction and cannot change it to", 1"""
        # get current feature value
        explanations = []
        for x_flip_value in x_flip_value_list:
            x_flip_categorical_value = ceteris_paribus_exp.categorical_mapping[feature_id][x_flip_value]
            explanations.append(x_flip_categorical_value)
        return explanations

    def write_tipping_point_cp_numerical():
        x_flip_value_list = ceteris_paribus_exp.get_feature_values_flipping_prediction(data, feature_name)
        if len(x_flip_value_list) == 0:
            return None, None
        if isinstance(x_flip_value_list, list):
            x_flip_value = x_flip_value_list[0]  # TODO: What if multiple x tipping points exist?
        current_feature_value = data[feature_name].iloc[0]
        # get the difference
        difference = current_feature_value - x_flip_value
        sign = "decreased" if difference > 0 else "increased"
        return sign, x_flip_value

    ceteris_paribus_exp = conversation.get_var('ceteris_paribus').contents
    feature_id = data.columns.get_loc(feature_name)
    if as_text:
        # Check if categorical or numerical
        if feature_id in ceteris_paribus_exp.categorical_mapping.keys():
            tipping_categories = write_tipping_point_cp_categorical(feature_id)
            return cp_categorical_template(feature_name, opposite_class, tipping_categories)
        else:
            sign, x_flip_value = write_tipping_point_cp_numerical()
            return cp_numerical_template(feature_name, opposite_class, sign, x_flip_value,
                                         template_manager=conversation.get_var('template_manager').contents)

    """# plot the figure
    pyo.plot(fig, filename='ceteris_paribus.html', auto_open=True)"""

    # Plotly figure
    fig = ceteris_paribus_exp.get_explanation(data, feature_name)
    fig.show()
    # Convert the figure to PNG as a BytesIO object
    image_base64 = fig_to_base64(fig)
    if feature_id in ceteris_paribus_exp.categorical_mapping.keys():
        axis = 'X-axis'
    else:
        axis = 'Y-axis'

    # Create the HTML string with the base64 image
    html_string = f'<img src="data:image/png;base64,{image_base64}" alt="Your Plot">' \
                  f'<span>When the prediction probability crosses 0.5 on the {axis}, ' \
                  f'the model would change the prediction. This is not always possible, since the selected attribute might' \
                  f' not be able to change the probability enough.</span>'

    return html_string, 1
