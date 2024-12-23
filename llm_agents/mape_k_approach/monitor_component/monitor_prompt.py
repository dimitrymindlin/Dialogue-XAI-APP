from pydantic import BaseModel, Field


class MonitorResultModel(BaseModel):
    reasoning: str = Field(..., description="The reasoning behind the classification of the user message.")
    understanding_displays: list[str] = Field(..., description="A list of understanding displays that the user message exhibits.")
    mode_of_engagement: str = Field(..., description="The cognitive mode of engagement that the user message exhibits.")


def get_monitor_prompt_template():
    return """
You are an AI monitoring component that analyzes user text messages to identify specific types of understanding and engagement based on the explanations provided. The user interacts with an AI model that presents explanations for predictions made on a dataset instance.

**Possible Understanding Display Labels:**
{understanding_displays}

**Possible Cognitive Modes of Engagement:**
{modes_of_engagement}

**Task:**

Analyze the user's latest message in the context of the conversation history. 1. Classify it into one or more of the
 **Understanding Display Labels** listed above. The user may express multiple displays simultaneously. 2. Identify 
 the **Cognitive Mode of Engagement** that best describe the user's engagement.

**Conversation History:**
{chat_history}

**User Message:**
{user_message}
"""
