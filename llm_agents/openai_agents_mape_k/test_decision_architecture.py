"""
Test script for the new decision-based agent architecture.

This script demonstrates how the decision agent routes queries to either
the MAPE-K agent or the RAG system based on the query type.
"""

import asyncio
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set! Please set it before running this script.")

# Create data directories if they don't exist
os.makedirs("data/documents", exist_ok=True)

# Import our agents
from llm_agents.openai_agents_mape_k.decision_agent import DecisionAgent
from llm_agents.openai_agents_mape_k.mape_k_agent import MapeKAgent
from llm_agents.openai_agents_mape_k.rag_system import RAGSystem

# Create a mock InstanceDatapoint class for testing
class MockInstanceDatapoint:
    def __init__(self, features):
        self.displayable_features = features
        self.instance_id = "mock_instance_1"

async def create_test_dataset():
    """Create a test dataset CSV file for the RAG system."""
    import pandas as pd
    
    # Create a test directory
    os.makedirs("data/documents", exist_ok=True)
    
    # Create a test CSV file
    test_data = {
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "education": ["High School", "Bachelors", "Masters", "PhD", "Bachelors", "Masters", "High School", "PhD"],
        "occupation": ["Clerk", "Manager", "Director", "Executive", "Engineer", "Technician", "Sales", "Professional"],
        "hours_per_week": [40, 45, 50, 60, 35, 40, 30, 55],
        "income": ["<=50K", ">50K", ">50K", ">50K", ">50K", ">50K", "<=50K", ">50K"]
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv("data/documents/income_dataset.csv", index=False)
    print("Created test dataset at data/documents/income_dataset.csv")

async def setup_test_environment():
    """Setup the test environment with necessary data."""
    await create_test_dataset()
    
    # Initialize the RAG system to build vector store
    rag_system = RAGSystem(domain_description="Income prediction model using census data")
    await rag_system.ensure_indexed()
    
    print("Test environment setup completed")

async def test_architecture():
    """Test the decision-based agent architecture."""
    # Setup environment first
    await setup_test_environment()
    
    # Create the decision agent
    decision_agent = DecisionAgent(
        feature_names="age, education, occupation, hours_per_week",
        domain_description="An income prediction model that determines if a person earns above or below $50K per year based on census data attributes",
        user_ml_knowledge="beginner",
        experiment_id="test_architecture"
    )
    
    # Initialize a mock instance
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
        "FeatureInfluencesPlot": "<img src='plot_data' />"
    }
    
    # First get the MAPE-K agent and initialize with datapoint
    mape_k_agent = await decision_agent._get_mape_k_agent()
    mape_k_agent.initialize_new_datapoint(
        instance=mock_instance,
        xai_explanations=mock_explanations,
        xai_visual_explanations=mock_visual_explanations,
        predicted_class_name=">50K",
        opposite_class_name="<=50K",
        datapoint_count=0
    )
    
    # Test queries
    model_explanation_queries = [
        "Why was this person predicted to earn more than $50K?",
        "What are the most important features for this prediction?",
        "How does education affect the model's decision?",
        "What would need to change for the prediction to be different?"
    ]
    
    dataset_information_queries = [
        "What is the average age in the dataset?",
        "How many people have a Bachelor's degree in the dataset?",
        "What is the distribution of income in the dataset?",
        "What columns are available in the dataset?"
    ]
    
    print("\n==== Testing Model Explanation Queries (should route to MAPE-K agent) ====\n")
    for query in model_explanation_queries:
        print(f"\nQuery: {query}")
        reasoning, response = await decision_agent.answer_user_question(query)
        print(f"Routing: {reasoning.split('.')[0]}")  # Just print the first part of reasoning
        print(f"Response: {response}")
    
    print("\n==== Testing Dataset Information Queries (should route to RAG system) ====\n")
    for query in dataset_information_queries:
        print(f"\nQuery: {query}")
        reasoning, response = await decision_agent.answer_user_question(query)
        print(f"Routing: {reasoning.split('.')[0]}")  # Just print the first part of reasoning
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(test_architecture()) 