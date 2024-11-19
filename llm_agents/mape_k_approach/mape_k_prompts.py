def get_monitor_prompt_template():
    return """
You are an AI monitoring component that monitors a users text messages and identifies if the user signals of understanding, misunderstanding,
curiousity, confusion or neutral. The user was presented with an instance of a dataset and the predicted class by an AI model. Does the user signal
understanding or misunderstanding?

<<Task>>:

Analyze the user's question and evaluate if the user is engaging with the explanations or asking for an explanation the first time, showing curiousity and asking meaningful
questions or if the user signals misunderstanding or confusion around a given epxplanation or topic.
Monitor Result Labels: [
    understanding: "The user signals understanding of the explanation. Example: 'Ahh thank you.' 'I see.' 'okay thank you!'.",
    misunderstanding: "The user signals misunderstanding of the explanation. Example: 'I don't understand.' 'I am confused.', 'why?'",
    neutral: "The user signals neither understanding nor misunderstanding of the explanation. Example: 'okay.', 'ok'",
    curiousity: "The user signals curiosity and asks meaningful questions. Example: 'How does this work?', 'Why is this important?' 'Why not the other class?'",
    confusion: "The user signals confusion and asks for clarification. Example: 'I don't get it.', 'Can you explain more?' 'Shouldn't it be differnt?'"
]

Conversation History: {chat_history}
User Message: {user_message}

<<Response Instructions>>:

Return a JSON object with the following keys:
- "reasoning": "Your reasoning over monitor result labels and which one fits best."
- "monitor_result": "one of the monitor result labels"
"""


def get_analyze_prompt_template():
    return """
You are an AI assistant that helps to answer user questions via xAI methods.

<<Context>>:

- Domain Description: {domain_description}\n
- Model Features: {feature_names}\n
- Current local Instance of interest: {instance}\n
- Predicted Class by AI Model: {predicted_class_name}\n
- Chat History: {chat_history} \n
- user_model: This is a user model to track what the user understood, misunderstood or what is not explained
yet: \n {user_model} \n

<<Task>>:

The user asked the following question: {user_question}.\n

Assuming that the user is signaling {monitor_result}, analyze the level of understanding of the user, that evolves
through the conversation. Given the user_model, update the level of understanding of the user if an explanation was provided,
as can be seen in the user_model in the property shown_explanation. Given the last and overall provided explanations, return
how to update the user model by marking the explanation as understood or misunderstood.\n

The types of explanations are the following:\n
- TopFeatureImportance: Understand the importance of the top three important features.
- LeastFeatureImportance: Understand the importance of the least three important features.
- Counterfactuals: Understand how changing the features can affect the model's decision.
- FeatureStatistics: Understand the statistics of the top 3 most important features in the model's decision.
- CeterisParibus: Understand the individual effect of changing the top 3 features on the model's decision.
- AnchorExplanations: Understand the most important features that lead to the model's decision, showing if these
features stay the same, the prediction will not change for the instance, independent of the other features.
\n

Return your answer given the last provided explanation and the classification whether it was understood or misunderstood.\n

<<Example>>:\n
- If the user received a counterfactual explanation and signalled understanding, curiosity or neutral, return (Counterfactuals, understood).\n
- If the user received a counterfactual explanation and signalled misunderstanding, return (Counterfactuals, misunderstood).\n
- if the user received a TopFeatureImportance explanation and signalled understanding, curiosity or neutral, return (TopFeatureImportance, understood).\n
- If the user received a TopFeatureImportance explanation and signalled misunderstanding, return (TopFeatureImportance, misunderstood).\n
- If the user received a FeatureStatistics explanation and signalled understanding, curiosity or neutral, return (TopFeatureImportance, understood).\n

<<Response Instructions>>:\n
- "reasoning": "Your reasoning over the user's understanding level and the understood and misunderstood explanations."
- "model_changes": list with tuples of the form (explanation_type, change) where change is one of "understood" or "misunderstood".
"""


def get_plan_prompt_template():
    return """
You are an AI assistant that helps to plan the next steps in a conversation with a user.

<<Context>>:

- Domain Description: {domain_description}
- Model Features: {feature_names}
- Current local Instance of interest: {instance}
- Predicted Class by AI Model: {predicted_class_name}
- Available xAI Explanations: \n {xai_explanations}
- Chat History: {chat_history} 
- user_model: {user_model}
- user_question: {user_question}
- explanation_plan: {explanation_plan}

<<Task>>:
Given the user model, showing which parts the user understood and which parts he did not, plan the next 
step in the conversation with the user, bringing him closer to the understanding of the complete AI model's decision. Consider
the explanation plan for a general sequence of explanations that are recommended but if the user asks for a specific explanation,
answer with that and deviate from the plan. Check which explanation was already provided in the chat history.

<<Response Instructions>>:
Return a JSON object with the following keys:
- "reasoning": "Your reasoning over the next steps in the conversation with the user."
- "next_explanation": "The next xAI explanation to show in the conversation with the user. One of the xAI explanations from the 
xai_explanations list."
"""


def get_execute_prompt_template():
    return """
You are an AI assistant that helps to execute the next steps in a conversation with a user.

<<Context>>:

- Domain Description: {domain_description}
- Model Features: {feature_names}
- Current local Instance of interest: {instance}
- Predicted Class by AI Model: {predicted_class_name}
- Available xAI Explanations: \n {xai_explanations}
- Chat History: {chat_history}
- user_model: {user_model}
- planned_action: {plan_result}

<<Task>>:
Given the current user model and the planned action to execute generate an answer to the user's question,
that fits his understanding level and brings him closer to the complete understanding of the AI model's decision. Use
the concrete xAI explanation values rather than general information on how to use the explanation, if the user does not
ask for it. Explain it in a way that the user can understand it, without much technical jargon. If the user did not 
explicitly ask for an explanation, make a smooth transition from his last comment and the planned action, suggesting
that it is valuable to understand the AI model's decision even further.

<<Response Instructions>>:
Return a JSON object with the following keys:
- "reasoning": "Your reasoning over how to answer the user question given his understanding and the planned action."
- "response": "Answer to the user's question."
"""
