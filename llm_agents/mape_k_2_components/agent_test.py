import asyncio
import os
import sys
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from llm_agents.mape_k_2_components.unified_mape_k_agent import UnifiedMapeKAgent, MAPE_K_ResultModel

load_dotenv()


# Sample data for testing
class InstanceDatapoint:
    """Mock implementation of InstanceDatapoint for testing"""
    def __init__(self, instance_dict, target, index):
        self.instance_dict = instance_dict
        self.target = target
        self.index = index
        self.displayable_features = instance_dict


class ExplanationModel(BaseModel):
    explanation_name: str
    explanation_steps: List[Dict[str, str]]
    concept_type: str
    explanation_type: str


# Generate dummy visual explanations for testing
def generate_dummy_explanations():
    return {
        "FeatureInfluencesPlot": "<div class='visualization'>Feature Influences Plot Placeholder</div>",
        "CounterfactualPlot": "<div class='visualization'>Counterfactual Plot Placeholder</div>",
        "GlobalFeatureImportancePlot": "<div class='visualization'>Global Feature Importance Plot Placeholder</div>",
        "DecisionTreePlot": "<div class='visualization'>Decision Tree Plot Placeholder</div>"
    }


# Create test instance datapoint
def create_test_datapoint():
    instance = InstanceDatapoint(
        instance_dict={
            "age": 35,
            "workclass": "Private",
            "education": "Bachelors",
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 10000,
            "capital_loss": 0,
            "hours_per_week": 45,
            "native_country": "United-States"
        },
        target="income",
        index=0
    )
    return instance


# Create test explanations
def create_test_explanations():
    return {
        "global_explanation": ExplanationModel(
            explanation_name="global_explanation",
            explanation_steps=[
                {"step_name": "model_purpose", "content": "This model predicts if a person earns >$50K annually."},
                {"step_name": "features_overview", "content": "Key features include age, education, workclass, and hours worked."}
            ],
            concept_type="general",
            explanation_type="global"
        ),
        "feature_influences": ExplanationModel(
            explanation_name="feature_influences",
            explanation_steps=[
                {"step_name": "overall_influences", "content": "Your education, occupation, and hours worked positively influence the prediction."},
                {"step_name": "specific_features", "content": "Having a Bachelors degree increases likelihood of >$50K income by 20%."}
            ],
            concept_type="specific",
            explanation_type="local"
        ),
        "counterfactual_explanation": ExplanationModel(
            explanation_name="counterfactual_explanation",
            explanation_steps=[
                {"step_name": "basic_counterfactual", "content": "If your education was 'Some-college' instead of 'Bachelors', the prediction would likely change."},
                {"step_name": "actionable_changes", "content": "Increasing weekly hours from 45 to 55 would significantly increase likelihood of >$50K income."}
            ],
            concept_type="specific",
            explanation_type="counterfactual"
        )
    }


async def test_agent():
    print("Testing UnifiedMapeKAgent with structured MAPE_K_ResultModel output")
    
    # Create agent instance
    agent = UnifiedMapeKAgent(
        feature_names="age, workclass, education, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country",
        domain_description="This dataset contains census data and the model predicts whether an individual earns more than $50K per year.",
        user_ml_knowledge="low",
        experiment_id="test_experiment"
    )
    
    # Create test data
    instance = create_test_datapoint()
    xai_explanations = create_test_explanations()
    xai_visual_explanations = generate_dummy_explanations()
    
    # Initialize datapoint
    agent.initialize_new_datapoint(
        instance=instance,
        xai_explanations=xai_explanations,
        xai_visual_explanations=xai_visual_explanations,
        predicted_class_name=">50K",
        opposite_class_name="<=50K",
        datapoint_count=0
    )
    
    # Test user questions
    test_questions = [
        "Why does the model predict that I earn more than $50K?",
        "How much does my education influence the prediction?",
        "What would I need to change to earn less than $50K?",
        "Can you explain how the model works in general?"
    ]
    
    for i, question in enumerate(test_questions):
        print(f"\n\nTesting question {i+1}: {question}\n")
        try:
            reasoning, response = await agent.answer_user_question(question)
            print(f"Reasoning: {reasoning}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error processing question: {e}")
    
    print("\nTest completed")


if __name__ == "__main__":
    asyncio.run(test_agent()) 