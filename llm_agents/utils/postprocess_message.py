import logging
import re

logger = logging.getLogger(__name__)


def remove_html_plots_and_restore_placeholders(response, visual_explanations_dict):
    """
    Removes HTML plot content from the response and replaces it with placeholders.
    This is the reverse operation of replace_plot_placeholders.

    Args:
        response (str): The response containing HTML plot content
        visual_explanations_dict (dict): A dictionary mapping plot names to their corresponding HTML representations

    Returns:
        str: The response with HTML plots replaced by placeholders
    """
    if not visual_explanations_dict or not response:
        return response
    
    # Pattern to match HTML img tags with base64 content
    img_pattern = re.compile(r'<img src="data:image/[^"]*"[^>]*>', re.IGNORECASE)
    
    # Iterate over each plot name and replace its HTML content with placeholder
    for plot_name, plot_html in visual_explanations_dict.items():
        if plot_html and plot_html in response:
            placeholder = f'##{plot_name}##'
            response = response.replace(plot_html, placeholder)
    
    # Also remove any remaining base64 img tags that might not be in the dictionary
    # This is a fallback to catch any HTML plots that weren't properly mapped
    remaining_imgs = img_pattern.findall(response)
    for img_tag in remaining_imgs:
        # Replace with a generic placeholder
        response = response.replace(img_tag, "##PlotRemoved##")
        logger.warning(f"Removed unmapped HTML plot from conversation history: {img_tag[:100]}...")
    
    return response


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
        return response

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
