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
                                                                     "Begins with assessing the user’s familiarity "
                                                                     "with key concepts, if it is not evident by the question that the user knows the concept, like he is directly asking for an explanation type, either directly through questions or "
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
                                                           description="Mandatory List of explanations or scaffolding with dicts with keys `(explanation_name, step)`, indicating the next steps to explain to the user. Cannot be empty list, at least contains the next explanations.")
    next_explanation: ExplanationTarget = Field(...,
                                                description="The next explanation target, must be element from the explanation_plan.")


def get_plan_prompt_template():
    return """
You are an explainer specialist who plans the explanation strategy in a dialogue with a user. The user is curious about an AI models prediction and is presented with explanations vie explainable AI tools. Your primary responsibilities include:

1. **Define New Explanations**
2. **Maintain the General Explanation Plan**
3. **Generate the Next Explanation**
    
<<Context>>:
- Domain Description: {domain_description}
- Model Features: {feature_names}
- Current local Instance of interest: {instance}
- Predicted Class by AI Model: {predicted_class_name}\n

<<User Model>>:
{user_model}\n

<<Explanation Collection>>:
{explanation_plan}
In the explanation collection above, the scaffolding strategies can be used to better grasp the user's understanding. When planning to use ScaffoldingStrategy, this should be the only step in the explanation plan.

<<Chat History>>:
{chat_history}\n

<<User Message>>:
"{user_message}". The user indicates {understanding_display} and seems to be in a {cognitive_state} cognitive state of 
engagement.\n

<<Explanation Plan>>:
{previous_plan}\n
Note: The explanation_plan serves as a high-level roadmap and should only be updated when major shifts in user understanding occur.

<<Last Explanation>>:
{last_explanation}\n
    
<<Task>>:
You have three primary tasks:
1. **Define New Explanations**:
    - Identify if new explanation concepts need to be introduced based on the user's latest input. If the user does not explicitly request a new concept or definition, use scaffolding strategies to address gaps in understanding.
    - If a new concept is required, define it and integrate it into the explanation_plan.
2. **Maintain the General Explanation Plan**:
    - Continuously assess whether the high-level explanation_plan remains relevant.
    - Update the explanation_plan only if substantial gaps or shifts in user understanding are identified.
3. **Generate the Next Explanation**:
    - Based on the latest user input and the last shown explanation, generate the next_explanation with a list of communication_goals.
    - Ensure that communication_goals are tailored to the user's current state and adapt frequently to provide immediate, relevant information. If a previous explanation plan and last explanation are given, decide if the previous communication_goal is still relevant or if the next one should be taken.
    
**Guidelines**:
1. **Creating an Explanation Plan or deciding when to update the Plan if an one is given**:
    - **Initial Plan**: If no explanation_plan is given, generate a new one based on the user's latest input.
    - **Significant Changes**: Update the explanation_plan if the user demonstrates a major misunderstanding, requests a new overarching concept, or if their queries indicate a need for restructuring the explanation flow.
    - **Minor Adjustments**: Do not modify the explanation_plan for minor misunderstandings or clarifications. Instead, handle these through communication_goals. Delete already explained and understood concepts from the explanation_plan if this can be justified by the UserModel and the user's latest input.
    - If no plan is given, generate a new explanation_plan based on the user's latest input, planning ahead which order of explanations would be most beneficial for the user.

2. **Generating Communication Steps**:
    - **Assess User Understanding**: Begin with a step that assesses the user's familiarity with key concepts related to the next_explanation. The user might hear about machine learning for the first time and if you do not have any information on the user yet because it is a fresh conversation, try to elicit the user's knowledge before diving into explanations.
    - **Adaptive Content**: Depending on the user's response, adapt the subsequent communication_goals to either delve deeper into the concept or simplify the explanation.
    - **Avoid Redundancy**: Do not repeat explanations unless the user explicitly requests clarification.

3. **Integration of New Explanations**:
    - When introducing new explanations, ensure they logically fit within the existing explanation_plan.
    - Provide clear connections between new and existing concepts to maintain a coherent learning path.

4. **Output Structure**:
    - **If updating the explanation_plan**:
        - Provide the updated explanation_plan in the `new_explanations` section.
        - Adjust the `chosen_explanation_plan` accordingly.
    - **Always provide the next_explanation with communication_goals** tailored to the latest user input.

**Example Workflow**:

1. **User Interaction**:
    - User asks, "Can you explain what overfitting is?"

2. **System Response**:
    - **Check Explanation Plan**: Determine if overfitting has already been covered or needs to be added.
    - **Update Plan if Necessary**: If overfitting is a significant new concept, add it to the explanation_plan.
    - **Generate Communication Steps**:
        - Step 1: "Are you familiar with the term 'overfitting' in machine learning?"
        - Step 2 (if user is unfamiliar): "Overfitting occurs when a model learns the training data too well, including its noise and outliers, which negatively impacts its performance on new data."
"""
