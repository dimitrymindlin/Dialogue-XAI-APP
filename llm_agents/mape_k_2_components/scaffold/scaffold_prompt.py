from typing import List
from pydantic import BaseModel, Field

class CommunicationGoal(BaseModel):
    goal: str = Field(
        ...,
        description="Each goal should focus on a specific aspect, such as eliciting if the user understands a concept or providing a step-by-step explanation."
    )

class ExplanationTarget(BaseModel):
    reasoning: str = Field(..., description="The reasoning behind the choice of the current explanandum.")
    explanation_name: str = Field(..., description="The name of the current explanation concept.")
    step_name: str = Field(..., description="Step name that is the current explanandum.")
    communication_goals: List[CommunicationGoal] = Field(
        ...,
        description="List of atomic communication goals for the complete explanation, including eliciting and providing information."
    )

    def __str__(self):
        communication_goals_str = "\n".join(f"- {cg.goal}" for cg in self.communication_goals)
        return (f"Explanation Name: {self.explanation_name}\n"
                f"Step Name: {self.step_name}\n"
                f"Communication Goals:\n{communication_goals_str}")

class ExplanationStepModel(BaseModel):
    step_name: str = Field(..., description="The name of the explanation step.")
    description: str = Field(..., description="Description of the explanation step.")
    dependencies: list = Field(..., description="List of dependencies for the explanation step.")
    is_optional: bool = Field(..., description="Whether the explanation step is optional or not.")

class NewExplanationModel(BaseModel):
    explanation_name: str = Field(..., description="The name of the new explanation concept.")
    description: str = Field(..., description="Description of the new explanation concept.")
    explanation_steps: List[ExplanationStepModel] = Field(
        ...,
        description="List of steps for the new explanation concept."
    )

class ChosenExplanationModel(BaseModel):
    explanation_name: str = Field(..., description="The name of the explanation concept.")
    step: str = Field(..., description="The name or label of the step of the explanation.")

class ScaffoldingResultModel(BaseModel):
    plan_reasoning: str = Field(
        ...,
        description="The reasoning behind the updated explanation plan and decisions regarding new explanations."
    )
    new_explanations: List[NewExplanationModel] = Field(
        ...,
        description="List of new explanations to be added to the explanation plan."
    )
    explanation_plan: List[ChosenExplanationModel] = Field(
        ...,
        description="The updated explanation plan with the next steps to explain to the user."
    )
    next_explanation: ExplanationTarget = Field(
        ...,
        description="The next explanation target with communication goals tailored to the user's state."
    )
    response: str = Field(
        ...,
        description="The response to the user's question styled with appropriate HTML elements (e.g., <b>, <ul>, <li>, <p>)."
    )
    summary_sentence: str = Field(
        ...,
        description="A concise summary of the response highlighting key delivered facts and whether the explanation ended with a question."
    )

# --- Merged Prompt Template ---
def get_scaffolding_prompt_template():
    return """
You are an expert explainer specialist and tutor responsible for both planning an explanation strategy and executing a tailored response to a user's query about an AI model's prediction.

<<Plan Section>>:
Your tasks in planning are:
1. **Define New Explanations**:
    - Evaluate the user's latest input to determine if new explanation concepts should be introduced.
    - If needed, define the new explanation and integrate it into the explanation plan.
2. **Maintain the General Explanation Plan**:
    - Assess whether the existing explanation plan remains relevant.
    - Update the plan if major shifts in user understanding occur.
3. **Generate the Next Explanation**:
    - Based on the user's input and the last provided explanation, generate the next explanation target with specific communication goals.
    - Ensure communication goals are tailored to the user's current state.

<<Context for Planning>>:
- Domain Description: {domain_description}
- Model Features: {feature_names}
- Current Instance of Interest: {instance}
- Predicted Class by AI Model: {predicted_class_name}
- User Model: {user_model}
- Explanation Collection: {explanation_collection}
- Chat History: {chat_history}
- User Message: "{user_message}"
- Previous Explanation Plan: {previous_plan}
- Last Explanation: {last_explanation}

<<Plan Task>>:
Analyze the above context to:
- Determine if new explanations are needed.
- Update the explanation plan if necessary.
- Generate the next explanation target with tailored communication goals.

Using the updated explanation plan and the next explanation target:
- Generate a response to the user's query about the model's prediction.
- Tailor the response to the user's language proficiency and cognitive state.
- Use HTML formatting (e.g., <b>, <ul>, <li>, <p>) to structure the response.
- Conclude with a question or prompt to encourage further interaction if appropriate.
"""