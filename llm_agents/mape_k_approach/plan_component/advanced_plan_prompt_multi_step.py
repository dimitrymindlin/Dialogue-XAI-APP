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
                           description="The reasoning behind the decision for new explanations and which explanations"
                                       "to include in the next steps.")
    new_explanations: List[NewExplanationModel] = Field(...,
                                                        description="List of new explanations to be added to the explanation plan."
                                                                    "Each new explanation is a dict with an explanation_name, a description, and a list of "
                                                                    "steps called 'explanations'. Each step is a dict with a 'step_name', 'description' and 'dependencies' and "
                                                                    "'is_optional' keys.")
    next_explanations: List[ChosenExplanationModel] = Field(...,
                                                            description="List of explanations or scaffolding with dicts with keys "
                                                                        "`(explanation_name, step)`, indicating the next steps "
                                                                        "to explain to the user.")


def get_plan_prompt_template():
    return """
You are an AI planning assistant that helps to plan the next steps in a conversation with a user. You make decisions
whether new explanation concepts need to be defined and added to the plan and which explanations to include in the next 
steps.

<<Context>>:
- Domain Description: {domain_description}
- Model Features: {feature_names}
- Current local Instance of interest: {instance}
- Predicted Class by AI Model: {predicted_class_name}

<<User Model>>:
{user_model}

<<Explanation Plan Collection>>:
{explanation_plan} \n
In the explanation plan above, the scaffolding strategies can be used to better grasp the user's understanding. When
planning to use ScaffoldingStrategy, this should be the only step in the explanation plan.

<<Chat History>>:
{chat_history}\n

<<User Message>>:
"{user_message}". The user indicates {understanding_display} and seems to be in a {cognitive_state} cognitive state of 
engagement.\n

<<Task>>:
If a previous plan is given, decide if it should be continued or updated based on the user's model, message and cognitive 
state. Your goal is to create a logical and progressive explanation plan to lead the user to a complete understanding of 
the AI model's decision.

** First, decide whether new explanation concepts need to be defined and added to the plan**:
When the user requests a new concept or definition that is not part of the current explanation plan:
1.	**Analyze the Concept**:
    - Identify the concept and its relationship to the explanation plan. Determine whether it relies on any sub-concepts or 
    prerequisites for understanding.
2.	**Define a Progressive Plan**:
    - Create a new explanation definition for the requested concept, including steps.
    - If sub-concepts are already explained in the current plan, reuse them; otherwise, add them to the plan.
3.	**Avoid Redundancy**:
    - Ensure the new explanation steps are not redundant with existing steps in the explanation plan.
    - Integrate the new concept with the current explanation plan where logical dependencies exist.
4.	**Example**:
    - If the user asks for a definition of “machine learning,” structure a new explanation plan:
        - MachineLearningDefinition (main concept).
        - Sub-concepts like “model training,” “features,” or “learning algorithms” (if not already in the plan).
        - Seamlessly connect the new explanation plan with existing steps.
5.	**Output New Explanation Steps**:
    - Return the revised explanation plan in next_explanations, ensuring:
    - The new plan logically integrates with the existing plan.
    - Sub-concepts are progressively explained and linked to the requested concept.

** Second, consider the following when generating the next steps in the plan**:

1. **Prioritize Explanation Progression**:
    - Do not suggest returning to more basic explanations if the user has already demonstrated understanding of those 
    concepts, unless the user explicitly requests clarification or exhibits signs of misunderstanding.
    - If the user is engaged with advanced explanations and signals understanding (e.g., causal reasoning), continue 
    exploring deeper levels of those explanations such as multiple features instead of one.

2. **Balance Progression and Scaffolding**:
    - If the user shows signs of misunderstanding, suggest scaffolding strategies to address the gap, such as 
    reformulating explanations, using examples, or simplifying concepts.
    - Avoid introducing unrelated or less complex concepts unless they are essential for addressing user confusion.

3. **Leverage the User Model**:
    - Refer to the user model to identify what the user has already understood and focus on building upon that foundation.
    - Avoid redundant explanations unless the user requests clarification.

4. **Plan Explanation Dependencies**:
    - When the user has signaled understanding of a key concept (e.g., feature importance), introduce dependent 
    explanations (e.g., counterfactuals or causal relationships).
    - Do not repeat unrelated or previously covered explanations unless necessary to scaffold a gap in understanding.

5. **Consider Explanation Priority**:
    - Concepts like feature importance and causal reasoning (e.g., counterfactuals, Ceteris Paribus explanations) 
    should be emphasized as critical milestones in the explanation journey.
    
6. **Do not repeat previous explanations**:
    - If the user expresses partial_understanding or non_understanding, do not repeat the same explanation in the plan.
    Rather, use a scaffolding technique or provide a slightly new explanation that builds upon the previous one.
    
<<Response Instructions>>:
As a general rule, do not provide more than two new explanations in a single turn. The user should have the opportunity
to ask questions and engage with the explanations provided and when using scaffolding strategies, this should be the only
step in the explanation plan.

"""
