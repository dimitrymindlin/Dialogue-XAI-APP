import asyncio
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from agents import Agent, Runner, function_tool
from llm_agents.base_agent import XAIBaseAgent
from llm_agents.explanation_state import ExplanationState
from create_experiment_data.instance_datapoint import InstanceDatapoint

# Load environment variables
load_dotenv()

# Function tools for the agent
@function_tool
def analyze_user_message(message: str) -> Dict:
    """
    Analyze the user message to identify understanding displays and cognitive state.
    
    Args:
        message: The user's message
        
    Returns:
        A dictionary containing analysis results
    """
    # This function only serves as a schema definition for the agent
    return {
        "understanding_displays": ["asking_questions"],
        "cognitive_state": "active"
    }

@function_tool
def update_explanation_states(explanation_name: str, new_state: str) -> str:
    """
    Update the state of an explanation.
    
    Args:
        explanation_name: The name of the explanation to update
        new_state: The new state (understood, not_understood, etc.)
        
    Returns:
        Confirmation message
    """
    return f"Updated {explanation_name} to {new_state}"

@function_tool
def generate_explanation(explanation_name: str) -> str:
    """
    Generate an explanation for the user.
    
    Args:
        explanation_name: The name of the explanation to generate
        
    Returns:
        HTML-formatted explanation
    """
    return f"<p>Here's an explanation about <b>{explanation_name}</b>...</p>"


class SimpleOpenAIMapeKAgent(XAIBaseAgent):
    """
    A simplified MAPE-K agent implementation using OpenAI Agents SDK.
    """
    def __init__(
            self,
            feature_names="",
            domain_description="",
            user_ml_knowledge="",
            experiment_id="",
            **kwargs
    ):
        self.feature_names = feature_names
        self.domain_description = domain_description
        self.user_ml_knowledge = user_ml_knowledge
        self.experiment_id = experiment_id
        self.instance = None
        self.predicted_class_name = None
        self.opposite_class_name = None
        self.visual_explanations_dict = {}
        self.datapoint_count = 0
        
        # Create OpenAI agent
        self.agent = self._create_agent()
        
    def _create_agent(self):
        """Create an OpenAI Agent with appropriate instructions and tools."""
        # Create an agent with appropriate instructions
        instructions = f"""
        You are an advanced Explainable AI (XAI) assistant that manages interactions with a user about a machine learning model's predictions.
        You will perform the Monitor-Analyze-Plan-Execute (MAPE-K) workflow to help users understand model predictions.
        
        DOMAIN CONTEXT:
        - Domain Description: {self.domain_description}
        - Model Features: {self.feature_names}
        - Current Explained Instance: {self.instance}
        - Predicted Class by AI Model: {self.predicted_class_name if self.predicted_class_name else "Not set yet"}
        
        Your goal is to provide explanations that help the user understand the model's predictions.
        Always consider the user's knowledge level and previous interactions.
        """
        
        mape_k_agent = Agent(
            name="MAPE-K XAI Assistant",
            instructions=instructions,
            tools=[
                analyze_user_message,
                update_explanation_states,
                generate_explanation
            ]
        )
        
        return mape_k_agent

    def initialize_new_datapoint(self,
                                 instance: InstanceDatapoint,
                                 xai_explanations,
                                 xai_visual_explanations,
                                 predicted_class_name,
                                 opposite_class_name,
                                 datapoint_count):
        """Initialize a new datapoint for analysis.
        
        Args:
            instance: The instance datapoint
            xai_explanations: The XAI explanations
            xai_visual_explanations: The visual XAI explanations
            predicted_class_name: The predicted class name
            opposite_class_name: The opposite class name
            datapoint_count: The datapoint count
        """
        self.instance = instance.displayable_features
        self.predicted_class_name = predicted_class_name
        self.opposite_class_name = opposite_class_name
        self.datapoint_count = datapoint_count + 1
        self.visual_explanations_dict = xai_visual_explanations
        
        # For SimpleOpenAIMapeKAgent, we only need to update the domain description
        # based on the current instance to guide the agent
        model_description = f"""
        Domain: {self.domain_description}
        
        Current Instance Features:
        {json.dumps(self.instance, indent=2)}
        
        Predicted Class: {predicted_class_name}
        (Opposite Class: {opposite_class_name})
        
        Available Feature Names: {self.feature_names}
        """
        
        # Update the domain description with instance-specific information
        self.domain_description = model_description
        
        # Recreate the agent with updated context
        self.agent = self._create_agent()
        
        print(f"SimpleOpenAIMapeKAgent initialized with datapoint {self.datapoint_count}")

    async def answer_user_question(self, user_question):
        """
        Public method to answer a user question using the simplified MAPE-K workflow.
        Maintains compatibility with the existing XAIBaseAgent interface.
        """
        try:
            # Create input message with all necessary context
            input_message = f"""
            Please analyze and respond to this user message: "{user_question}"
            
            Follow these steps:
            1. MONITOR: Analyze the user message to identify understanding displays and cognitive state
            2. ANALYZE: Determine which explanations need to be updated based on the user's question
            3. PLAN: Choose which explanations to present to the user
            4. EXECUTE: Generate an HTML response with the explanations
            
            Domain: {self.domain_description}
            Features: {self.feature_names}
            Instance: {json.dumps(self.instance) if self.instance else "Not available"}
            Prediction: {self.predicted_class_name if self.predicted_class_name else "Not available"}
            
            The response should be in HTML format for the frontend.
            """
            
            # Run the agent
            result = await Runner.run(self.agent, input=input_message)
            
            # Return a tuple of (reasoning, response) to maintain compatibility
            reasoning = "Processed user question using simplified MAPE-K workflow"
            response = result.final_output
            
            # Check if response is HTML formatted
            if not response.strip().startswith("<"):
                response = f"<p>{response}</p>"
                
            return reasoning, response
            
        except Exception as e:
            print(f"Error in simplified MAPE-K agent: {e}")
            return "Error processing request", "<p>I encountered an error processing your request. Please try again.</p>"


async def test_agent():
    """Test the simplified OpenAI MAPE-K agent"""
    agent = SimpleOpenAIMapeKAgent(
        feature_names="age, income, education",
        domain_description="An income prediction model that determines if a person earns above or below $50K per year",
        user_ml_knowledge="beginner"
    )
    
    # Test with some questions
    questions = [
        "How does this model work?",
        "Why was the prediction made for this person?",
        "What's the most important feature?",
        "Can you explain how age affects the prediction?"
    ]
    
    for question in questions:
        print(f"\n\nQuestion: {question}")
        reasoning, response = await agent.answer_user_question(question)
        print(f"Response: {response}")
        print(f"Reasoning: {reasoning}")
        await asyncio.sleep(1)  # Wait a bit between questions


if __name__ == "__main__":
    asyncio.run(test_agent()) 