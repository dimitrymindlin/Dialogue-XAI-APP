from typing import List

from pydantic import BaseModel, Field


class CommunicationGoal(BaseModel):
    """
    Data model for an explanation step for the current ExplanationTarget.
    """
    goal: str = Field(...,
                      description="Each goal should focus on a specific aspect, which can be a determining if the user already understands a concept, or an explanation that answers the user’s question step by step. The step should be expressed as a concise heading, such as “Elicit if the user knows Machine Learning” or “Explain the importance of feature importance in XAI.”")


class ExplanationTarget(BaseModel):
    """
    Data model for the explanandum (the current explanation target).
    """
    reasoning: str = Field(..., description="The reasoning behind the choice of the current explanandum.")
    explanation_name: str = Field(..., description="The name of the current explanation concept.")
    step_name: str = Field(..., description="Step name that is the current explanandum.")
    communication_goals: List[CommunicationGoal] = Field(...,
                                                         description="List of atomic goals while communicating the complete explanation to the user, breaking down each ExplanationTarget into multiple CommunicationGoal,"
                                                                     "including eliciting information and providing"
                                                                     "information. Each atomic goal should be short to communicate to not overwhelm the user."
                                                                     "Begins with assessing the user’s familiarity if the user question was super general like 'why'. Do not elicit user's familiarity if the user actively asks for a certain explanation type."
                                                                     "with key concepts if not already understood as indicated by the user model or if it is not evident by the question that the user knows the concept, like he is directly asking for an explanation type, the most important features, either directly through questions or "
                                                                     "implicitly by observing their query, chat history and user model")

    def __str__(self):
        communication_goals_str = "\n".join(
            f"- {comm_goal.goal}" for comm_goal in self.communication_goals
        )
        return (
            f"Explanation Name: {self.explanation_name}\n"
            f"Step Name: {self.step_name}\n"
            f"Communication Goals:\n{communication_goals_str}"
        )


class ExplanationStepModel(BaseModel):
    """
    Data model for an explanation step in the explanation plan.
    """
    step_name: str = Field(..., description="The name of the explanation step.")
    description: str = Field(..., description="Description of the explanation step.")
    dependencies: list = Field(..., description="List of dependencies for the explanation step.")
    is_optional: bool = Field(..., description="Whether the explanation step is optional or not.")


class NewExplanationModel(BaseModel):
    """
    Data model for a new explanation concept to be added to the explanation plan.
    """
    explanation_name: str = Field(..., description="The name of the new explanation concept.")
    description: str = Field(..., description="Description of the new explanation concept.")
    explanation_steps: List[ExplanationStepModel] = Field(...,
                                                          description="List of steps for the new explanation concept. Each step is a dict with "
                                                                      "a 'step_name', 'description', 'dependencies' and 'is_optional' keys.")


class ChosenExplanationModel(BaseModel):
    """
    Data model for a chosen explanation concept to be added to the explanation plan.
    """
    explanation_name: str = Field(..., description="The name of the explanation concept.")
    step: str = Field(..., description="The name or label of the step of the explanation.")


class PlanResultModel(BaseModel):
    """
    Data model for the result of the explanation plan generation.
    """
    reasoning: str = Field(...,
                           description="The reasoning behind the decision for new explanations and which explanations to include in the next steps.")
    new_explanations: List[NewExplanationModel] = Field(...,
                                                        description="List of new explanations to be added to the explanation plan. Each new explanation is a dict with an explanation_name, a description, and a list of steps called 'explanations'. Each step is a dict with a 'step_name', 'description' and 'dependencies' and 'is_optional' keys.")
    explanation_plan: List[ChosenExplanationModel] = Field(...,
                                                           description="Mandatory List of explanations or scaffolding with dicts with keys `(explanation_name, step)`, indicating long term steps to explain to the user. Cannot be empty list, at least contains the next explanations.")
    next_response: List[ExplanationTarget] = Field(...,
                                                   description="A list of explanations and steps to include in the next response to answer the user's question. Only include multiple if the user requests the full reasoning of the model or asks for multiple explanations.")


def get_plan_prompt_template():
    return """
You are an explainer specialist who plans the explanation strategy in a dialogue with a user. The user is curious about an AI models prediction and is presented with explanations vie explainable AI tools. Your primary responsibilities include:

1. **Define New Explanations**
2. **Maintain the General Explanation Plan**
3. **Generate the Next Explanation to show the user**
    
<<Context>>:
- Domain Description: {domain_description}
- Model Features: {feature_names}
- Current Explained Instance:: {instance}
- Predicted Class by AI Model: {predicted_class_name}\n

<<User Model>>:
{user_model}\n

<<Explanation Collection>>:
{explanation_collection}
In the explanation collection above, the scaffolding strategies can be used to better grasp the user's understanding. When planning to use ScaffoldingStrategy, this should be the only step in the next explanation step.

<<Chat History>>:
{chat_history}\n

<<User Message>>:
"{user_message}".\n

<<Explanation Plan>>:
{previous_plan}\n
Note: The Explanation Plan serves as a high-level roadmap and should only be updated when shifts in user understanding occur, for example the user cannot understand an explanation because he lacks some more general concept knowledge, or the user explicitely wants to explore other concepts or explanation directions.

<<Last Explanation>>:
{last_explanation}\n
    
<<Task>>:
You have three primary tasks:
1. **Defining New Explanations**:
   - Determine if the latest input requires introducing a new concept by checking each possible explanation. If the user does not explicitly request one, apply scaffolding to address understanding gaps.
   - Define any new concept and integrate it into the explanation_plan if needed.

2. **Maintaining the Explanation Plan**:
    - consider that the user might only ask one or maximally three questions in a row:
    - If no explanation_plan exists, generate one based on the latest input.
    - Continuously assess the plan’s relevance. Update it only when significant gaps, shifts in user understanding, or requests for new overarching concepts occur.
    - Continuously check if the user's question that can be mapped to another explanation. If the user asks a question that fits any of the explanation methods, avoid explaining the concept of that explanations and assume the user knows it.
    - Address minor misunderstandings through communication_goals without altering the plan. Remove concepts that the user fully understands, as justified by the UserModel and recent input.

3. **Generating the Next Explanations**:
   - Based on the latest input, previous explanations, the user's cognitive state and ML knowledge, create the next_explanations along with tailored communication_goals. If the last explanation was not understood, consider scaffolding strategies and put them back into the next communication goal.
   - Ensure these goals are concise, engaging, and matched to the user’s current state. If the user’s ML knowledge is low or unclear, first assess and elicit their familiarity with key concepts.
   - For ambiguous requests, use scaffolding to clarify intent before providing details.
   - Adapt content dynamically—delving deeper, simplifying, or redirecting based on the user’s responses.
   - Avoid repetition unless the user explicitly asks for clarification, and prioritize reacting to user queries over strictly following the plan.
   - If the user asks question unrelated to understanding the current explanation, provide a short answer that you are not able to respond to that and can only talk about the model prediction and the instance shown.\n

Think step by step and provide a reasoning for each decision based on the users model indicating the UNDERSTOOD explanations, the users's latest message, the conversation history, and the current explanation plan.
"""
