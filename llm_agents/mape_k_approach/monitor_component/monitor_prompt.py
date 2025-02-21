from pydantic import BaseModel, Field


class MonitorResultModel(BaseModel):
    reasoning: str = Field(description="Short reasoning for the classification of the user message.", default="")
    explicit_understanding_displays: list[str] = Field(
        description="A list of explicitly stated understanding displays by the user",
        default_factory=list)
    mode_of_engagement: str = Field(description="The cognitive mode of engagement that the user message exhibits",
                                    default="")


def get_monitor_prompt_template():
    return """
You are an analyst that interprets user messages to identify users understanding and cognitive engagement based on the provided chat and his recent message. The user is curious about an AI models prediction and is presented with explanations vie explainable AI tools.

**Possible Understanding Display Labels:**
{understanding_displays}

**Possible Cognitive Modes of Engagement:**
{modes_of_engagement}

**Task:**

Analyze the user's latest message in the context of the conversation history. 
1. If an explanation was provided and the user shows **explicit** signs of understanding as described in the **Understanding Display Labels** listed above, classify his explicit understanding. The user may express multiple understanding displays or just ask a question without explicitely signalling understanding. If it is not explicitely stated, return an empty list []. 
2. Identify the **Cognitive Mode of Engagement** that best describe the user's engagement. Interpret the user message in the context of the conversation history to disambiguate nuances since a 'yes' or 'no' might refer to understanding something or agreeing to a suggestion. This should always be defined by the given user question and history.

**Conversation History:**
{chat_history}

**User Message:**
{user_message}

Think step by step which understanding displays are present in the user message and what cognitive mode of engagement is exhibited.
"""
