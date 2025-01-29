from pydantic import BaseModel, Field


class MonitorResultModel(BaseModel):
    reasoning: str = Field(..., description="The reasoning behind the classification of the user message.")
    understanding_displays: list[str] = Field(..., description="A list of understanding displays that the user message exhibits.")
    mode_of_engagement: str = Field(..., description="The cognitive mode of engagement that the user message exhibits.")


def get_monitor_prompt_template():
    return """
You are an analyst that interprets user messages to identify users understanding and cognitive engagement based on the provided chat and his recent message. The user is curious about an AI models prediction and is presented with explanations vie explainable AI tools.

**Possible Understanding Display Labels:**
{understanding_displays}

**Possible Cognitive Modes of Engagement:**
{modes_of_engagement}

**Task:**

Analyze the user's latest message in the context of the conversation history. 1. If an explanation was provided and the user replies, classify it into one or more of the **Understanding Display Labels** listed above. The user may express multiple displays simultaneously. If the user leads and initiates the conversation,   2. Identify the **Cognitive Mode of Engagement** that best describe the user's engagement. Interpret the user message in the context of the conversation history to capture nuances since a 'yes' or 'no' might refer to understanding something or agreeing to a suggestion. The latter is a sign of engagement but not necessarily complete understanding.

**Conversation History:**
{chat_history}

**User Message:**
{user_message}
"""
