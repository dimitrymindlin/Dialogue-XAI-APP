from typing import List

from pydantic import BaseModel, Field


class MonitorResultModel(BaseModel):
    reasoning: str = Field(description="Short reasoning for the classification of the user message.", default="")
    explicit_understanding_displays: list[str] = Field(
        description="A list of explicitly stated understanding displays by the user",
        default_factory=list)
    mode_of_engagement: str = Field(description="The cognitive mode of engagement that the user message exhibits",
                                    default="")


class AnalyzeResult(BaseModel):
    reasoning: str = Field(..., description="Short reasoning behind the suggested changes of the user model.")
    model_changes: list = Field(...,
                                description="List of changes to the user model with dicts with keys `(explanation_name, step, state)` where `state` is one of the possible states and `step` indicates the step in the explanation plan that was provided to the user. Default is empty list. Return only changes that are not already in the user model and that can be seen by the user's question.")


class MonitorAnalyzeResultModel(MonitorResultModel, AnalyzeResult):
    """
    Data model for the result of the monitor and analyze components combined.
    """
    pass


class ExplanationTarget(BaseModel):
    """
    Data model for the explanandum (the current explanation target).
    """
    explanation_name: str = Field(..., description="The name of the current explanation concept.")
    step_name: str = Field(..., description="Step name that is the current explanandum.")
    communication_goals: List[str] = Field(
        ...,
        description="List of atomic communication goals (strings) for this explanation target, each breaking down the explanation into a concise user-facing step."
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
    step_name: str = Field(..., description="The name or label of the step of the explanation.")


class PlanResultModel(BaseModel):
    """
    Data model for the result of the explanation plan generation.
    """
    reasoning: str = Field(...,
                           description="The reasoning behind the decision for new explanations and which explanations to include in the next steps.")
    new_explanations: List[NewExplanationModel] = Field(...,
                                                        description="List of new explanations to be added to the explanation plan, if not already in the plan. Each new explanation is a dict with an explanation_name, a description, and a list of steps called 'explanations'. Each step is a dict with a 'step_name', 'description' and 'dependencies' and 'is_optional' keys.")
    explanation_plan: List[ChosenExplanationModel] = Field(...,
                                                           description="Mandatory list of explanations or scaffolding with dicts (explanation_name, step_name), indicating long-term steps to explain to the user. Cannot be an empty list; must contain at least the next explanations.")
    next_response: List[ExplanationTarget] = Field(...,
                                                   description="A list of explanations and steps to include in the next response to answer the user's question. Only include multiple if the user requests the full reasoning of the model or asks for multiple explanations.")


class ExecuteResult(BaseModel):
    reasoning: str = Field(..., description="The reasoning behind the response.")
    response: str = Field(...,
                          description="The response to the user's question about the shown instance and prediction only using information from the chat history and explanation plan styled with appropriate html elements such as <b> for bold text or bullet points.")
    summary_sentence: str = Field(...,
                                  description="Single sentence summary of the response, highlighting the key delivered facts to keep track of what the user should have understood. Specific bits of knowledge that the user should keep after reading the explanation, also indicating whether the explanation ended with a question.")


class PlanExecuteResultModel(PlanResultModel, ExecuteResult):
    """
    Data model for the result of the plan and execute components combined.
    """
    pass


class SinglePromptResultModel(MonitorAnalyzeResultModel, PlanExecuteResultModel):
    """
    Data model combining all the components of the MAPE-K approach in a single prompt.
    """
    pass
