"""
Prompt B is the prompt to check if the user response can be interpreted as a response to the suggested
method or is otherwise a different question.
"""


def accept_decline_prompt_template(feature_names):
    return f"""Context: \n The user is asked if they would like to see a suggested XAI (Explainable AI) method.\n 

    Suggested Methods:
    \n{{explanation_suggestions}}\n
    User's answer classification:
    Given the user’s response, an AI classifies it into one of the following:
    - “agreement” (to any suggested method)
	- “rejection” (of all options)
	- “other”\n
    The user might agree to a feature specific method and reference one of the features: {feature_names}.\n
    Expected Output:
    The AI returns the reasoning and classification in JSON format with the following keys:\n
    - “reasoning”: The reasoning behind the classification. For each classification possibility, the reasoning should be explained and the method name should be mentioned if the classification is “agreement”.
    - “classification”: Decide on ne of [“agreement”, “rejection”, “other”] based on the reasoning
	- “method”: If classification is “agreement”, the method ID from the suggested methods should be returned.
	- “feature”: One of the feature names if mentioned by the user, otherwise null
    """


def openai_system_prompt_B(feature_names):
    return "system", accept_decline_prompt_template(feature_names)


def openai_user_prompt_B():
    return "user", f"""
    User Response:
    \n{{user_response}}
    """
