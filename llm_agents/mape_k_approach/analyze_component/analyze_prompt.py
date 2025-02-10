from pydantic import BaseModel, Field


class AnalyzeResult(BaseModel):
    reasoning: str = Field(..., description="The reasoning behind the classification of the user message.")
    model_changes: list = Field(..., description="List of changes to the user model with dicts with keys `(explanation_name, step, state)` where `state` is one of the possible states and `step` indicates the step in the explanation plan that was provided to the user.")


def get_analyze_prompt_template():
    return """
You are an AI assistant that helps answer user questions via Explainable AI (XAI) methods.

<<Context>>:

- **Domain Description:** {domain_description}
- **Model Features:** {feature_names}
- **Current Local Instance of Interest:** {instance}
- **Predicted Class by AI Model:** {predicted_class_name}

<<Possible Understanding Display Labels>>:
{understanding_displays}

<<Explanation Plan>>:
{explanation_plan}

<<Possible States of Explanations>>:
"not_yet_explained": The explanation has not been shown to the user yet.
"partially_understood": The explanation was shown to the user and is only partially understood.
"understood": The user has understood the explanation.
"not_understood": The user has not understood the explanation.

<<Chat History>>:
{chat_history}

<<User Model>>:
This is the user model, indicating which explanations were understood, not yet explained or being shown: {user_model}.

<<Task>>:
The user sent the following message: "{user_message}".

1.	Suggest updates the states of the last shown explanations with Name,Step,Content: {last_shown_explanations}\n, based  on the user’s message, cognitive engagement and understanding displays. Take into account that the user might say 'yes' or 'okay' to indicate accepting a suggested explanation, rather than explicitly stating that they understood it. Check  each explanation individually and assess if the user understood it or not. If the user asks a followup question resulting from the latest given explanation, provide a reasoning for why the explanation state should change. For non-referenced explanations that were explained last, update their states to “understood”. If the agent used a ScaffoldingStrategy as the last explanation, ignore this explanation in the user model as it is not a part of the explanation plan.
2.	If the user demonstrates wrong assumptions or misunderstandings on previously understood explanations, mark them with an according state.
3.	Provide only the changes to explanation states, omitting states that remain unchanged. Do not suggest which new explanations should be shown to the user.
"""
