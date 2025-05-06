"""
Test script for SimpleOpenAIMapeKAgent integration with ExplainBot.

This script demonstrates how to use the SimpleOpenAIMapeKAgent with ExplainBot
by simulating the initialization and usage flow in the actual application.
"""

import asyncio
import os
from dotenv import load_dotenv
import json

# Load environment variables for OpenAI API key
load_dotenv()

# Ensure you have set your OpenAI API key in .env file or environment variable
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable is not set!")
    print("Please set it before running this script.")

# Import the SimpleOpenAIMapeKAgent
from llm_agents.openai_agents_mape_k import SimpleOpenAIMapeKAgent

# Create a mock InstanceDatapoint class for testing
class MockInstanceDatapoint:
    def __init__(self, features):
        self.displayable_features = features
        self.instance_id = "mock_instance_1"


async def test_openai_agents_mape_k():
    print("Testing SimpleOpenAIMapeKAgent...")
    
    # Create the agent with example values
    agent = SimpleOpenAIMapeKAgent(
        feature_names="age, education, occupation, hours_per_week, income",
        domain_description="Income prediction model that predicts whether a person earns above or below $50K per year",
        user_ml_knowledge="beginner",
        experiment_id="test_integration"
    )
    
    # Create a mock instance with example features
    mock_instance = MockInstanceDatapoint({
        "age": 45,
        "education": "Bachelors",
        "occupation": "Executive-Managerial", 
        "hours_per_week": 60,
        "income": ">50K"
    })
    
    # Mock explanations
    mock_explanations = {
        "feature_importance": {
            "age": 0.3,
            "education": 0.25,
            "occupation": 0.2,
            "hours_per_week": 0.15
        }
    }
    
    # Mock visual explanations
    mock_visual_explanations = {
        "shap_values": {
            "image_data": "base64_encoded_image_data_would_go_here"
        }
    }
    
    # Initialize the agent with the mock datapoint
    agent.initialize_new_datapoint(
        instance=mock_instance,
        xai_explanations=mock_explanations,
        xai_visual_explanations=mock_visual_explanations,
        predicted_class_name=">50K",
        opposite_class_name="<=50K",
        datapoint_count=0
    )
    
    # Test a series of user questions
    test_questions = [
        "What is the most important feature for this prediction?",
        "How does education affect the prediction?",
        "Why does the model predict this person earns above $50K?",
        "What would need to change for the prediction to be different?"
    ]
    
    for question in test_questions:
        print(f"\n--- Testing question: '{question}' ---")
        try:
            reasoning, response = await agent.answer_user_question(question)
            print(f"Reasoning: {reasoning}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_openai_agents_mape_k()) 