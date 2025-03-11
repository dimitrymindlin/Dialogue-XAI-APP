# Function to flatten nested dictionaries using dot notation
def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


# Function to format feature importance dictionaries into readable lists
def format_feature_importance(favor_dict):
    return "\n".join([f"- {v['description']}" for k, v in favor_dict.items()])


# Function to prepare data for substitution
def prepare_substitution_data(xai_explanations):
    flat_xai_explanations = flatten_dict(xai_explanations)

    # Convert lists to formatted strings
    flat_xai_explanations['ceteris_paribus_text'] = "\n".join(
        [f"- {item}" for item in xai_explanations['ceteris_paribus']])
    flat_xai_explanations[
        'impact_of_multiple_features_text'] = "Understanding the combined impact of multiple features on the counterfactual plan."
    flat_xai_explanations[
        'diagnostic_query_text'] = "What is the most important feature in favor of the current prediction?"
    flat_xai_explanations[
        'wrong_assumptions_text'] = "Feature importance does not imply causation; it only indicates correlation."

    # Format feature_importance_in_favor_of_under 50k and feature_importance_in_favour_ofover 50k into lists
    flat_xai_explanations['feature_importance.most_important_features_text'] = format_feature_importance(
        xai_explanations['feature_importance_in_favor_of_under 50k'])
    flat_xai_explanations['feature_importance.least_important_features_text'] = format_feature_importance(
        xai_explanations['feature_importance_in_favour_ofover 50k'])

    # Add the plot URL (assuming you have a plot saved and hosted)
    flat_xai_explanations[
        'feature_influences_plot_url'] = "https://yourdomain.com/path/to/feature_influences_plot.png"  # Replace with actual URL

    # Add diagnostic queries and wrong assumptions if needed
    flat_xai_explanations[
        'feature_importance.diagnostic_query_text'] = "Which feature has the highest importance in favor of the current prediction?"
    flat_xai_explanations[
        'feature_importance.wrong_assumptions_text'] = "Feature importance does not imply causation; it only indicates correlation."

    return flat_xai_explanations
