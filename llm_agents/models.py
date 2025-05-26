from typing import List

from pydantic import BaseModel, Field

from llm_agents.explanation_state import ExplanationState


class MonitorResultModel(BaseModel):
    reasoning: str = Field(
        description="Short reasoning for the choice of explicit_understanding_displays and mode_of_engagement.",
        default="")
    explicit_understanding_displays: list[str] = Field(
        description="A list of explicitly stated understanding displays by the user",
        default_factory=list)
    mode_of_engagement: str = Field(description="The cognitive mode of engagement that the user message exhibits",
                                    default="")


class ModelChange(BaseModel):
    explanation_name: str
    step_name: str
    state: ExplanationState


class AnalyzeResult(BaseModel):
    reasoning: str = Field(..., description="Short reasoning for suggested changes to the user model.")
    model_changes: list[ModelChange] = Field(...,
                                             description="List of changes to the user model where `state` is one of the possible states and `step_name` indicates the step in the explanation plan that was provided to the user. Default is empty list. Return only changes that are not already in the user model and that can be seen by the user's question.")


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


class ExplanationStepModel(BaseModel):
    """
    Data model for an explanation step in the explanation plan.
    """
    step_name: str = Field(..., description="The name of the explanation step.")
    description: str = Field(..., description="Description of the explanation step.")


class NewExplanationModel(BaseModel):
    """
    Data model for a new explanation concept to be added to the explanation plan.
    """
    explanation_name: str = Field(..., description="The name of the new explanation concept.")
    description: str = Field(..., description="Description of the new explanation concept.")
    explanation_steps: List[ExplanationStepModel] = Field(...,
                                                          description="List of steps for the new explanation concept.")


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
                           description="Short reasoning for the decision of new explanations and which explanations to include in the next steps.")
    new_explanations: List[NewExplanationModel] = Field(default_factory=list,
                                                        description="List of completely new explanations to be added to the explanation collection.")
    explanation_plan: List[ChosenExplanationModel] = Field(default_factory=list,
                                                           description="List of explanations or scaffolding techniques, indicating a long-term plan to explain the whole model prediction to the user.")
    next_response: ExplanationTarget = Field(default_factory=list,
                                             description="An explanation target for the next agent response and steps to include in the next response.")


class ExecuteResult(BaseModel):
    reasoning: str = Field(..., description="Short reasoning behind how to craft the response.")
    response: str = Field(...,
                          description="The response to the user's question about the shown instance and prediction only using information from the chat history and explanation plan styled with appropriate html elements such as <b> for bold text or bullet points.")

class PlanExecuteResultModel(PlanResultModel, ExecuteResult):
    """
    Data model for the result of the plan and execute components combined.
    """
    # override reasoning to use ExecuteResult's definition only
    reasoning: str = Field(
        ...,
        description="Short reasoning for the decision of new explanations and which explanations to include in the next steps."
    )


class SinglePromptResultModel(MonitorAnalyzeResultModel, PlanExecuteResultModel):
    """
    Data model combining all the components of the MAPE-K approach in a single prompt.
    """
    pass
