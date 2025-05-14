from pydantic import BaseModel, Field
from typing import List, Optional


class GoalModel(BaseModel):
    goal: str = Field(..., description="Each goal should focus on a specific aspect")


class CommunicationGoalModel(BaseModel):
    reasoning: Optional[str] = Field(None, description="The reasoning behind the choice of the current explanandum.")
    explanation_name: str = Field(..., description="The name of the current explanation concept.")
    step_name: str = Field(..., description="Step name that is the current explanandum.")
    communication_goals: List[GoalModel] = Field(...,
                                                 description="List of atomic goals while communicating the complete explanation to the user")


class ExplanationStepModel(BaseModel):
    step_name: str = Field(..., description="The name of the explanation step.")
    description: str = Field(..., description="Description of the explanation step.")
    dependencies: List[str] = Field(description="List of dependencies for the explanation step.")
    is_optional: bool = Field(description="Whether the explanation step is optional or not.")


class NewExplanationModel(BaseModel):
    explanation_name: str = Field(..., description="The name of the new explanation concept.")
    description: str = Field(..., description="Description of the new explanation concept.")
    explanation_steps: List[ExplanationStepModel] = Field(...,
                                                          description="List of steps for the new explanation concept.")


class ChosenExplanationModel(BaseModel):
    explanation_name: str = Field(..., description="The name of the explanation concept.")
    step: str = Field(..., description="The name or label of the step of the explanation.")


class PlanExecuteResultModel(BaseModel):
    """Model for the combined plan and execute output structure."""
    # Planning section
    planning_reasoning: str = Field(...,
                                    description="The reasoning behind the decision for new explanations and which explanations to include in the next steps.")
    new_explanations: List[NewExplanationModel] = Field(description="List of new explanations to be added to the explanation plan.")
    explanation_plan: List[ChosenExplanationModel] = Field(description="List of explanations or scaffolding indicating long term steps to explain to the user.")
    next_response: List[CommunicationGoalModel] = Field(...,
                                                        description="A list of explanations and steps to include in the next response to answer the user's question.")

    # Execution section
    execution_reasoning: str = Field(..., description="The reasoning behind the response generation.")
    response: str = Field(...,
                          description="The final response to the user's question about the shown instance and prediction.")


def get_plan_execute_prompt_template():
    """Template for the combined plan and execute prompt."""
    return """
Use a friendly, conversational tone as if speaking to a friend.
Avoid technical jargon and method names; explain in plain language.
Keep your answer under 70 words or three sentences.
If listing reasons, use at most three bullet points.
End with a concise invitation for follow-up.
You are a specialized XAI (Explainable AI) agent using a MAPE-K approach to communicate with a user about AI decisions. 
You need to both PLAN and EXECUTE a response to the user's question.

# USER INFORMATION AND CONTEXT
- Domain: {domain_description}
- Features: {feature_names}
- Instance to explain: {instance}
- Predicted class: {predicted_class_name}
- Current chat history: {chat_history}
- Current User Model: {user_model}
- User's message: {user_message}

# EXPLANATION STATE
Current explanation collection:
{explanation_collection}

# TASK 1: PLANNING
First, you need to plan your explanation strategy.

## Step 1: Consider the explanation strategy
- What explanation concepts should be introduced, explained, or reinforced?
- How should new explanation concepts be broken down into steps?
- What sequence of explanations will effectively address the user's question?
- What concepts remain unexplained or need refinement?

## Step 2: Generate new explanation concepts if needed
- Identify gaps in the user's understanding
- Create new explanation concepts with clear steps

## Step 3: Create an explanation plan
- Determine the step-by-step approach to building user understanding

## Step 4: Select specific explanations for the next response
- Choose which explanation concepts and steps to include in your immediate response
- Consider communication goals for each explanation
- IMPORTANT: Always use explanation names that already exist in the explanation collection - do not invent new names. If you need a fallback response, use "ModelPredictionConfidence" as the explanation name and "Concept" as the step name.

# TASK 2: EXECUTION
Now, execute your plan by generating a response that implements your strategy.

## Step 1: Craft a coherent explanation based on your plan
- Use the planned explanation concepts and steps as your guide
- Sequence the information logically
- Tailor the language to the user's level of understanding

## Step 2: Structure the response effectively
- Start by acknowledging the question without repeating it or information from the chat history
- Present the explanations in a logical order
- Include appropriate examples and analogies
- End with a concise summary that reinforces key points if appropriate
- If appropriate, invite further questions and suggest new explanations to explore in the users language

## Step 3: Ensure the response is accurate and helpful
- Verify that your explanation addresses the user's specific question
- Check that the explanation is consistent with the instance features and prediction
- Make sure the content is appropriate for the user's current understanding level

## Step 4: Polish the final response
- Use clear, concise language
- Break up dense information into readable chunks
- Use html formatting to enhance readability

Remember to focus on addressing the specific user question while gradually building a more comprehensive understanding.
"""


def get_plan_execute_prompt_template_short():
    """
    A compact PLAN+EXECUTE template for an XAI agent.
    """
    return """
Use a friendly, conversational tone.
Avoid technical jargon; keep language simple.
Answer in under 70 words or three sentences.
Use up to three bullet points for clarity.
End with a short follow-up prompt.
You are an XAI agent using a MAPE-K loop.  Integrate both planning and response-generation.

CONTEXT:
- Domain: {domain_description}
- Features: {feature_names}
- Instance & prediction: {instance} → {predicted_class_name}
- Chat history & user model: {chat_history}, {user_model}
- Latest user message: {user_message}

EXISTING EXPLANATIONS:
{explanation_collection}

=== PLANNING ===
• Identify gaps in understanding & any new concepts (with steps)  
• Build a stepwise explanation plan  
• Select which concepts/steps + communication goals for your next reply  
• IMPORTANT: Only use explanation names that already exist in the explanation collection. For fallback responses, use "ModelPredictionConfidence" as the explanation name and "Concept" as the step name.

=== EXECUTION ===
• Write a clear intro, deliver chosen explanations in order, use examples/analogies  
• Summarize key points & invite follow-up  
• Ensure accuracy & readability  

Remember: keep it tailored, coherent and concise."""
