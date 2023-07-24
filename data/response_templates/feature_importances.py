import pandas
import numpy as np
from typing import Dict
import io

def textual_fi_with_values(sig_coefs: Dict[str, float],
                           num_features_to_show: int = None):
    """Formats dict of label -> feature name -> feature_importance dicts to string.

    Arguments:
        sig_coefs: Dict of label -> feature name -> feature_importance dicts with textual feature names.
        num_features_to_show: Number of features to show in the output. If None, all features are shown.
    Returns:
        output_text: String with the formatted feature importances to show in the dialog.
    """
    output_text = "Here is a list of the attributes that are most important for the current prediction, starting with " \
                  "the most important one: <br></br>"

    for i, (current_feature_value, feature_importance_value) in enumerate(sig_coefs):
        if i == 0:
            position = "most"
        else:
            position = f"{i + 1}."
        increase_decrease = "increases" if feature_importance_value > 0 else "decreases"
        new_text = (f"<b>{current_feature_value}</b> is the <b>{position}</b> important attribute and it"
                    f" <em>{increase_decrease}</em> the likelihood of the current prediction by <b>{str(feature_importance_value)}.</b>")
        if new_text != "":
            output_text += "<li>" + new_text + "</li>"
        if num_features_to_show:
            if i == num_features_to_show:
                break
    return output_text


def textual_fi_relational(sig_coefs: Dict[str, float],
                          num_features_to_show: int = None,
                          print_unimportant_features: bool = False,
                          show_only_most_important: bool = True):
    """Formats dict of label -> feature name -> feature_importance dicts to string.

    Arguments:
        sig_coefs: Dict of label -> feature name -> feature_importance dicts with textual feature names.
    Returns:

    """

    def relational_percentage_to_comparative_language(percentage_number):
        comparative_string = ""
        if percentage_number > 95:
            comparative_string = "as important as"
        elif percentage_number > 80:
            comparative_string = "almost as important as"
        elif percentage_number > 60:
            comparative_string = "three fourth as important as"
        elif percentage_number > 40:
            comparative_string = "half as important as"
        elif percentage_number > 20:
            comparative_string = "one fourth as important as"
        elif percentage_number > 5:
            comparative_string = "almost unimportant compared to"
        else:
            comparative_string = "unimportant compared to"
        return comparative_string

    output_text = ""

    previous_feature_importance = None
    for i, (current_feature_value, feature_importance) in enumerate(sig_coefs):
        if show_only_most_important:
            if i > 3:
                break
        if i == 0:
            position = "most"
        else:
            position = f"{i + 1}"
        increase_decrease = "increases" if feature_importance > 0 else "decreases"

        ### Get importance comparison strings
        # Most important can be printed as it is
        new_text = (f"<b>{current_feature_value}</b> is the <b>{position}.</b> important attribute and it"
                    f" <em>{increase_decrease}</em> the likelihood of the current prediction.")
        previous_feature_importance = feature_importance
        # For the rest, we need to check the relation to the previous feature
        if i > 0:
            relation_to_previous_in_percentage = np.abs(
                round((feature_importance / previous_feature_importance) * 100, 2))
            relation_to_top_in_percentage = np.abs(round((feature_importance / sig_coefs[0][1]) * 100, 2))
            if not print_unimportant_features:
                if relation_to_top_in_percentage < 5:
                    break
            comparitive_string_to_top_feature = relational_percentage_to_comparative_language(
                relation_to_top_in_percentage)
            new_text += f" It is {comparitive_string_to_top_feature} the top feature"
            if i > 1:
                comparative_string_to_previous = relational_percentage_to_comparative_language(
                    relation_to_previous_in_percentage)
                new_text += f" and {comparative_string_to_previous} the previous feature"
            new_text += "."

        if new_text != "":
            output_text += "<li>" + new_text + "</li>"
        if num_features_to_show:
            if i == num_features_to_show:
                break
    return output_text


def visual_feature_importance_list(sig_coefs):
    import matplotlib.pyplot as plt

    def plot_values(names, values):
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Set the y-axis labels
        ax.set_yticklabels(names)

        # Set the y-axis ticks
        ax.set_yticks(range(len(names)))

        # Set the x-axis range
        ax.set_xlim(min(values), max(values))

        plt.subplots_adjust(left=0.5)

        # Plot the bars
        for i, val in enumerate(values):
            color = 'green' if val > 0 else 'red'
            ax.barh(i, val, color=color)

        # Save the plot to a bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Clear the current plot
        plt.clf()

        # Close the figure to release memory
        plt.close(fig)

        # Send the buffer as a file attachment
        return buffer

    names = [sig_coefs[i][0] for i in range(len(sig_coefs))]
    names.reverse()
    values = [sig_coefs[i][1] for i in range(len(sig_coefs))]
    values.reverse()
    plot_values(names, values)