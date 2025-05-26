from pydantic import BaseModel, Field


class PlanResult(BaseModel):
    reasoning: str = Field(..., description="The reasoning behind the classification of the user message.")
    next_explanations: list = Field(..., description="List of explanations or scaffolding with dicts with keys "
                                                     "`(explanation_name, step)`, indicating the next step "
                                                     "in the explanation plan.")


def get_plan_prompt_template():
    return """
You are an AI planning assistant that helps to plan the next steps in a conversation with a user.

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
Your goal is to create a logical and progressive explanation plan to lead the user to a **complete understanding** of 
the AI model's decision. Consider the following when generating the next steps:

1. **Prioritize Explanation Progression**:
    - Do not suggest returning to more basic explanations if the user has already demonstrated understanding of those 
    concepts, unless the user explicitly requests clarification or exhibits signs of misunderstanding.
    - If the user is engaged with advanced explanations (e.g., causal reasoning), continue exploring deeper levels of 
    those explanations.

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

**Special Case if the user asks for a concept definition or information beyond the explanation plan**:
- If the user requests a concept definition or additional information beyond the explanation plan, make a plan on how
to explain this concept with steps that need to be understood before explaining the requested concept. **Reason** about how
to best explain this concept and which sub-concepts or steps need to be understood before explaining the requested concept.
If the sub-concepts are already explained in the explanation plan, you can reuse them. If not, define them in the plan.
Then, return next_explanations with your newly defined explanation plan. The new explanation should be called
similar as the requested concept by the user. If the user asks for a definition of "machine learning", you can return
next_explanations with the explanation plan for MachineLearningDefinition. Once you have the main concept defined,
recursingly define the sub-concepts that are part of the main concept that need to be understood before the main concept.
Make sure the newly explanations are necessary and are not redundant with explanations in the existing explanation plan.
You can also mix some new concept defition that is dependant or a prerequisite for an existing explanation in the plan.
Each newly defined step should also contain the keys 'description' and 'dependencies'.

<<Response Instructions>>:
As a general rule, do not provide more than two new explanations in a single turn. The user should have the opportunity
to ask questions and engage with the explanations provided and when using scaffolding strategies, this should be the only
step in the explanation plan.

"""
