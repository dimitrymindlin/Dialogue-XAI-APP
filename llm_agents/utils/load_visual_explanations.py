"""
Utility functions for loading and generating visual explanations.
"""

def generate_dummy_explanations():
    """
    Generate dummy visual explanations for testing.
    
    Returns:
        dict: A dictionary of dummy visual explanations
    """
    return {
        "FeatureInfluencesPlot": "<div class='visualization'>Feature Influences Plot Placeholder</div>",
        "CounterfactualPlot": "<div class='visualization'>Counterfactual Plot Placeholder</div>",
        "GlobalFeatureImportancePlot": "<div class='visualization'>Global Feature Importance Plot Placeholder</div>",
        "DecisionTreePlot": "<div class='visualization'>Decision Tree Plot Placeholder</div>"
    } 