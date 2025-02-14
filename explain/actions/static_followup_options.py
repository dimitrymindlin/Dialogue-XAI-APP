def explainConceptOfFeatureImportance():
    """The user saw important features but wonders why a specific one is most important. Explain how importances are
    determined and they might contradict the user's intuition."""
    return """These attributes had the biggest impact because the model found them most relevant for this person, 
    based on learned patterns. Models sometimes spot patterns humans miss, which can lead to unexpected results. 
    This is where Explainable AI and explanations like looking at the most important features helps detect why the 
    model makes decisions and uncover biases or errors in these decisions."""


def explainConceptOfLocalImportance(instance_type_name):
    """The user saw important features for the last instance but wonders why they change with every instance"""
    return f"""This explains why the model made its decision for this specific {instance_type_name}. Instead of 
    showing general trends, it highlights which features mattered most in this case. Since each {instance_type_name}
    has different data, the model weighs the features differently each time, leading to varying importance. For example,
    while in general the working hours matter a lot, for a specific person their investments might be more important."""


def explainWhyFeaturesAreConsideredAndOthersNot():
    """The user wonders why some features are considered at all or is wondering why some features are not considered"""
    return """Models can only use the attributes they were trained on. Data is collected with certain limitations, meaning 
    some useful attributes may be unavailable. If certain information wasn’t collected, the model can’t 
    consider it. It makes predictions based only on the available data, even if other factors seem important to you. 
    Also, this model cannot access external information beyond what it was trained on. The 
    model looks at these attribute to make it's prediction, but doesn’t decide their values."""


def get_mapping():
    return {
        "top3Features": ("followupWhyThisFeatureImportant", "Why are these the most important attributes?"),
        "least3Features": None,
        "shapAllFeatures": ("followupWhyFeatureImportancesChange", "Why do most important factors change?"),
        "counterfactualAnyChange": None,
        "anchor": None,
        "ceterisParibus": None,
        "featureStatistics": None,
        "globalPdp": ("followupWhyAreTheseFeaturesConsidered", "Why are all these attributes considered?"),
    }