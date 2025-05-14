import asyncio
import os
import json
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

from llm_agents.mape_k_2_components.unified_mape_k_agent import UnifiedMapeKAgent
from create_experiment_data.instance_datapoint import InstanceDatapoint

# Load environment variables for API keys
load_dotenv()

# Sample test data
INSTANCE_DATA = {
    "age": 38,
    "workclass": "Private",
    "education": "Bachelors",
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

FEATURE_NAMES = ["age", "workclass", "education", "marital_status", "occupation", 
                "relationship", "race", "sex", "capital_gain", "capital_loss", 
                "hours_per_week", "native_country"]

DOMAIN_DESCRIPTION = """
The task is income prediction based on demographic and employment information.
The model predicts whether a person earns over or under $50,000 per year.
"""

XAI_EXPLANATIONS = {
    "FeatureImportances": {
        "Concept": "Feature importance shows how much each feature contributes to the model's prediction. Features with higher importance have more influence on the prediction outcome.",
        "FeaturesInFavourOfOver50k": "The most important features contributing to the prediction of income OVER $50K are: marital_status, education, occupation, and hours_per_week.",
        "FeaturesInFavourOfUnder50k": "The most important features contributing to the prediction of income UNDER $50K are: capital_gain, relationship, and age."
    },
    "LocalExplanation": {
        "Concept": "Local explanations show how specific feature values for this individual case impact the prediction.",
        "PositiveLocalFeatures": "Features that increase likelihood of OVER $50K: Being married, Having a Bachelors degree, Working in Prof-specialty, and working 40 hours per week.",
        "NegativeLocalFeatures": "Features that decrease likelihood of OVER $50K: Having zero capital gain, Being in a Husband relationship, and Being 38 years old."
    }
}

XAI_VISUAL_EXPLANATIONS = {
    "FeatureInfluencesPlot": """
    <div style="text-align: center;">
        <img src="http://example.com/placeholder-feature-plot.png" alt="Feature Influences Plot" style="max-width: 90%; height: auto;">
        <p style="font-style: italic; font-size: 0.9em;">Feature influences showing the most important factors in this prediction.</p>
    </div>
    """
}


async def run_unified_mape_k_test():
    """Test the unified MAPE-K workflow with sample data"""
    
    # Initialize LLM with specified model
    llm = OpenAI(model=os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo'))
    
    # Create instance datapoint with correct parameters
    instance = InstanceDatapoint(
        instance_id=1,
        instance_as_dict=INSTANCE_DATA,
        class_probabilities=[0.8, 0.2],  # Probability for each class
        model_predicted_label_string="<=50K",
        model_predicted_label=0,  # 0 for the first class (<=50K)
        instance_type="test"
    )
    
    # Set displayable features
    instance.displayable_features = INSTANCE_DATA
    
    # Initialize the unified agent
    agent = UnifiedMapeKAgent(
        llm=llm,
        feature_names=FEATURE_NAMES,
        domain_description=DOMAIN_DESCRIPTION,
        user_ml_knowledge="Beginner",
        experiment_id="unified_test"
    )
    
    # Initialize a new datapoint
    agent.initialize_new_datapoint(
        instance=instance,
        xai_explanations=XAI_EXPLANATIONS,
        xai_visual_explanations=XAI_VISUAL_EXPLANATIONS,
        predicted_class_name="<=50K",
        opposite_class_name=">50K",
        datapoint_count=0
    )
    
    # Test with a simple first question
    print("=== Testing with first user question ===")
    analysis, response = await agent.answer_user_question("Why was this person predicted to earn under $50K?")
    print(f"\nUser model after first question:\n{json.dumps(agent.user_model.get_state_summary(as_dict=True), indent=2)}")
    print(f"\nAnalysis: {analysis}")
    print(f"\nResponse:\n{response}")
    
    # Test with a follow-up question
    print("\n\n=== Testing with follow-up question ===")
    analysis, response = await agent.answer_user_question("What would need to change for this person to earn over $50K?")
    print(f"\nUser model after follow-up:\n{json.dumps(agent.user_model.get_state_summary(as_dict=True), indent=2)}")
    print(f"\nAnalysis: {analysis}")
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    # Run the test
    asyncio.run(run_unified_mape_k_test()) 