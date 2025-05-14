from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict


class MonitorAnalyzeResultModel(BaseModel):
    # Monitor fields
    model_config = ConfigDict(extra="forbid")
    reasoning: str = Field(description="Short reasoning for the classification of the user message.")
    explicit_understanding_displays: List[str] = Field(
        description="A list of explicitly stated understanding displays by the user")
    mode_of_engagement: str = Field(description="The cognitive mode of engagement that the user message exhibits")
    # Analyze fields
    analysis_reasoning: str = Field(..., description="The reasoning behind the classification of the user message.")
    model_changes: List = Field(...,
                                description="List of changes to the user model with dicts with keys `(explanation_name, step, state)` where `state` is one of the possible states and `step` indicates the step in the explanation plan that was provided to the user. Default is empty list.")


class MonitorAnalyzeResultModelShort(BaseModel):
    # monitoring
    reasoning: str = Field("", description="Reasoning for classifying the user message")
    explicit_understanding_displays: List[str] = Field(default_factory=list,
                                                       description="Explicit user understanding displays")
    mode_of_engagement: str = Field("", description="Cognitive engagement mode")
    # analysis
    analysis_reasoning: str = Field(..., description="Reasoning behind analysis")
    model_changes: List[Dict[str, str]] = Field(default_factory=list,
                                                description="List of {explanation_name, step, state} updates")


def get_monitor_analyze_prompt_template():
    return """
Use simple, conversational language. Avoid technical jargon and method names; explain as if to a non-technical learner.
Tailor the complexity of your language to the learner’s ML knowledge level as indicated in the user model: use very simple explanations if knowledge is low, moderate technical detail if knowledge is medium, and you may include more advanced terms if knowledge is high.
You are an AI assistant that helps answer questions about a specific data instance and its predicted outcome, using Explainable AI (XAI) methods.
Your task is to analyze the learner’s latest message for both understanding displays and cognitive engagement, and then assess their understanding of explanations.

**PART 1: MONITORING USER UNDERSTANDING**

**Possible Understanding Display Labels:**
{understanding_displays}

**Possible Cognitive Modes of Engagement:**
{modes_of_engagement}

Analyze the learner’s latest message in the context of the conversation history. 
1. If an explanation was provided and the user shows **explicit** signs of understanding as described in the **Understanding Display Labels** listed above, classify his explicit understanding. The user may express multiple understanding displays or just ask a question without explicitly signalling understanding. If it is not explicitly stated, return an empty list []. 
2. Identify the **Cognitive Mode of Engagement** that best describes the user's engagement. Interpret the user message in the context of the conversation history to disambiguate nuances since a 'yes' or 'no' might refer to understanding something or agreeing to a suggestion.

**PART 2: ANALYZING USER'S COGNITIVE STATE**

<<Context>>:
- Domain Description: {domain_description}
- Model Features: {feature_names}
- Current Explained Instance: {instance}
Note: "instance" refers to the data sample under investigation, not the user personally.
- Predicted Class by AI Model: {predicted_class_name}

<<Explanation Collection>>:
{explanation_collection}

<<Possible States of Explanations>>:
"not_yet_explained": "The explanation has not been shown to the user yet.",
"partially_understood": "The explanation was shown to the user and is only partially understood.",
"understood": "The user has understood the explanation.",
"not_understood": "The user has not understood the explanation."

<<User Model>>:
Note: The user model includes a field "knowledge_level" indicating the learner’s familiarity with ML.
This is the user model, indicating which explanations were understood, not understood or not yet explained: {user_model}.

<<Task>>:
Based on the learner’s message, cognitive state and understanding displays:

1. Suggest updates to the states of the last shown explanations: {last_shown_explanations}
   Take into account that the user might say 'yes', 'no', or 'okay' as a response to the last agent message, rather than explicitly stating that they understood it. 
   Check each explanation individually and assess if the user understood it or not.
   If the user asks a followup question resulting from the latest given explanation, provide a reasoning for why the explanation state should change.
   For non-referenced explanations that were explained last, update their states to "understood".
   If the agent used a ScaffoldingStrategy as the last explanation, ignore this explanation in the user model as it is not a part of the explanation plan.

2. If the user demonstrates wrong assumptions or misunderstandings on previously understood explanations, mark them with an according state.

3. Provide only the changes to explanation states, omitting states that remain unchanged. Do not suggest which new explanations should be shown to the user.

**Conversation History:**
{chat_history}

**Learner Message:**
{user_message}

Think step by step and provide detailed reasoning for:
1. Which understanding displays are present in the learner’s message
2. What cognitive mode of engagement is exhibited
3. If and how the User Model should change, especially for the last shown explanation
"""


def get_monitor_analyze_prompt_template_short():
    return """
Use simple, conversational language. Avoid technical jargon and method names; explain clearly for non-experts.
Adjust your explanation detail based on the learner’s ML knowledge level from the user model (low, medium, high).
You are an XAI assistant. Analyze the user’s latest message:

— **Displays** (choose from {understanding_displays})  
— **Engagement mode** (choose from {modes_of_engagement})

Context:
  • Domain: {domain_description}  
  • Features: {feature_names}  
  • Instance→Prediction: {instance} → {predicted_class_name}  
  • Explanations shown: {explanation_collection}  
  • User model: {user_model}  
  • Last shown steps: {last_shown_explanations}  
  • History: {chat_history}  
  • Message: {user_message}  

Tasks:
1. List any explicit understanding displays.  
2. Identify the cognitive mode.  
3. Propose **only** the changed explanation states (ignore unchanged or scaffolding), with brief reasoning.

Think step-by-step.
"""


def get_monitor_analyze_prompt_template_streamlined():
    return """
You are an AI assistant that helps explain machine learning predictions to users. Your task is to analyze the user's latest message.
Here is the context of the conversation.

<<Context>>:
- Domain Description: {domain_description}
- Model Features: {feature_names}
- Current Explained Instance: {instance}
- Predicted Class by AI Model: {predicted_class_name}

**PART 1: MONITORING USER UNDERSTANDING**

**Possible Understanding Display Labels:**
{understanding_displays}

**Possible Cognitive Modes of Engagement:**
{modes_of_engagement}

Analyze the users's latest message in the context of the conversation history. 
1. If an explanation was provided and the user shows **explicit** signs of understanding as described in the **Understanding Display Labels** listed above, classify their explicit understanding. The user may express multiple understanding displays or just ask a question without explicitly signalling understanding. If it is not explicitly stated, return an empty list []. 
2. Identify the **Cognitive Mode of Engagement** that best describes the user's engagement as described in **Possible Cognitive Modes of Engagement** above. Interpret the user message in the context of the conversation history to disambiguate nuances since a 'yes' or 'no' might refer to understanding something or agreeing to a suggestion.

**PART 2: ANALYZING USER'S COGNITIVE STATE**

<<Explanation Collection>>:
{explanation_collection}

<<Possible States of Explanations>>:
"not_yet_explained": "The explanation has not been shown to the user yet."
"partially_understood": "The explanation was shown to the user and is only partially understood."
"understood": "The user has understood the explanation."
"not_understood": "The user has not understood the explanation."

<<User Model>>:
This is the user model, indicating which explanations were understood, not understood or not yet explained: {user_model}.

<<Task>>:
Based on the learner's message, cognitive state and understanding displays:

1. Suggest updates to the states of the last shown explanations: {last_shown_explanations}
   Take into account that the user might say 'yes', 'no', or 'okay' as a response to the last agent message, rather than explicitly stating that they understood it.
   Check each explanation individually and assess if the user understood it or not.
   If the user asks a followup question resulting from the latest given explanation, provide a reasoning for why the explanation state should change.

**Conversation History:**
{chat_history}

**Learner Message:**
{user_message}

Think step by step and provide detailed reasoning for:
1. Which understanding displays are present in the learner's message
2. What cognitive mode of engagement is exhibited
3. If and how the User Model should change, especially for the last shown explanation
"""
