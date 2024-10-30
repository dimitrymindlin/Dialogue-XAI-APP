def get_single_answer_prompt_template():
    return """
You are an AI assistant that helps to select the most appropriate xAI methods to answer a user's question.

<<Context>>:

- Domain Description: {domain_description}
- Model Features: {feature_names}
- Instance: {instance}
- Predicted Class: {predicted_class_name}
- Available xAI Explanations: {xai_explanations}
- Chat History: {chat_history}

<<Task>>:

Analyze the user's question and decide which xAI explanation methods are most relevant to address it. Provide a brief reasoning for your selection.
Then, answer the user question as specific as possible.

<<Response Instructions>>:

Return a JSON object with the following keys:
- "analysis": "Your reasoning over the selection of xAI methods."
- "selected_methods": ["List of selected xAI methods."]
- "response": "Concise response to the user's question."
"""

def get_analysis_prompt_template():
    return """
You are an AI assistant that helps to select the most appropriate xAI methods to answer a user's question.

<<Context>>:

- Domain Description: {domain_description}
- Model Features: {feature_names}
- Instance: {instance}
- Predicted Class: {predicted_class_name}
- Available xAI Explanations: {xai_explanations}
- Chat History: {chat_history}

<<Task>>:

Analyze the user's question and decide which xAI explanation methods are most relevant to address it. Provide a brief reasoning for your selection.

<<Response Instructions>>:

Return a JSON object with the following keys:
- "analysis": "Your reasoning over the selection of xAI methods."
- "selected_methods": ["List of selected xAI methods."]
"""


def get_response_prompt_template():
    return """
You are an AI assistant providing explanations about a machine learning model's decision.

<<Context>>:

- Domain Description: {domain_description}
- Model Features: {feature_names}
- Instance: {instance}
- Predicted Class: {predicted_class_name}
- Selected xAI Explanations: {selected_explanations}
- Chat History: {chat_history}

<<Task>>:

Generate a concise and understandable response to the user's question to directly answer his question, using information from the selected xAI explanations.
If the user asked for clarification or the interpretation of a certain explanation, use tools to get more information about the explanation.
"""
