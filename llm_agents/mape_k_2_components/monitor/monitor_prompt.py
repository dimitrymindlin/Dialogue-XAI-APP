from pydantic import BaseModel, Field


class MonitorResultModel(BaseModel):
    reasoning: str = Field(...,
                           description="The reasoning for the selection of the understanding_displays, mode of engagement and model_changes.")
    understanding_displays: list[str] = Field(...,
                                              description="A list of understanding displays that the user message exhibits.")
    mode_of_engagement: str = Field(..., description="The cognitive mode of engagement that the user message exhibits.")
    model_changes: list = Field(...,
                                description="List of changes to the user model with dicts containing keys `(explanation_name, step, state)`. Default should be empty list.")
    explanation_questions: list[str] = Field(...,
                                             description="A list of explanation questions that the user message exhibits. Default should be empty list.")


def get_monitor_prompt_template():
    return """
You are an analyst and AI assistant that interprets user messages and manages explanation states via Explainable AI (XAI) methods.

<<Context>>:
- **Domain Description:** {domain_description}
- **Model Features:** {feature_names}
- **Current Local Instance of Interest:** {instance}
- **Predicted Class by AI Model:** {predicted_class_name}

<<Possible Understanding Display Labels>>:
{understanding_displays}

<<Possible Cognitive Modes of Engagement>>:
{modes_of_engagement}

<<Possible Explanation Questions>>:
{explanation_questions}

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
This is the user model, indicating which explanations were understood, not yet explained, or currently being shown: {user_model}.

<<Task>>:
The user sent the following message: "{user_message}".


1. Analyze the user's latest message in the context of the conversation history.
2. If an explanation was provided and the user replies, classify the message into one or more of the **Understanding Display Labels**.
3. If the user initiates the conversation, identify the **Cognitive Mode of Engagement** that best describes the user's engagement and classify his question into the explanation questions.
Interpret nuances since a simple 'yes' or 'no' might indicate agreement or partial understanding rather than complete comprehension.
4. Suggest updates to the states of the last shown explanations (provided as {last_shown_explanations}) based on the user’s message, cognitive engagement, and identified understanding displays.
   - Consider indirect confirmations like 'yes' or 'okay' as potential indicators of understanding.
   - Evaluate each explanation individually to determine if the user understood it.
   - For non-referenced explanations that were explained last, update their states to “understood.”
   - If the last explanation employed a ScaffoldingStrategy, ignore it in the user model.
5. If the user demonstrates misconceptions or misunderstandings regarding previously understood explanations, update their states accordingly.
6. Provide only the changes to explanation states, omitting those that remain unchanged.
"""
