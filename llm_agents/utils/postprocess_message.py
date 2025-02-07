import logging
import re

logger = logging.getLogger(__name__)


def replace_plot_placeholders(response, visual_explanations_dict):
    """
    Replaces all plot placeholders in the execute_result.response with their corresponding plots.

    Args:
        response (str): The response to the user's question, possibly containing plot placeholders.
        visual_explanations_dict (dict): A dictionary mapping plot names to their corresponding HTML representations

    Returns:
        str: The response with all plot placeholders replaced by their corresponding plots
    """

    # Define a regex pattern to find all placeholders within curly braces
    pattern = re.compile(r'##([^#]+)##')

    # Find all unique plot names in the response
    plot_names = re.findall(pattern, response)

    if not plot_names:
        # No placeholders found; no action needed
        return

    # Iterate over each plot name and attempt to replace its placeholder
    for plot_name in set(plot_names):  # Using set to avoid redundant replacements
        placeholder = f'##{plot_name}##'
        try:
            # Retrieve the corresponding plot from the dictionary
            plot = visual_explanations_dict[plot_name]
            # Replace all instances of the current placeholder with the plot
            response = response.replace(placeholder, plot)
        except KeyError:
            # Log an error if the plot name is not found in the dictionary
            available_plots = ', '.join(visual_explanations_dict.keys())
            logger.error(
                f"Could not find plot with name '{plot_name}' in visual explanations dictionary. "
                f"Available plots: {available_plots}"
            )

    return response
