import asyncio
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

from llm_agents.mape_k_2_components.unified_mape_k_agent import UnifiedMapeKAgent
from create_experiment_data.instance_datapoint import InstanceDatapoint

# Load API keys
load_dotenv()

# Sample data
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

# List of features
FEATURE_NAMES = ["age", "workclass", "education", "marital_status", "occupation", 
                "relationship", "race", "sex", "capital_gain", "capital_loss", 
                "hours_per_week", "native_country"]

# Domain description
DOMAIN_DESCRIPTION = "The task is income prediction based on demographic and employment information. The model predicts whether a person earns over or under $50,000 per year."

# Simple XAI explanations
XAI_EXPLANATIONS = {
    "FeatureImportances": {
        "Concept": "Feature importance shows how much each feature contributes to the model's prediction. Features with higher importance have more influence on the prediction outcome.",
        "FeaturesInFavourOfOver50k": "The most important features contributing to the prediction of income OVER $50K are: marital_status, education, occupation, and hours_per_week.",
        "FeaturesInFavourOfUnder50k": "The most important features contributing to the prediction of income UNDER $50K are: capital_gain, relationship, and age."
    }
}

# Visual explanations
XAI_VISUAL_EXPLANATIONS = {
    "FeatureInfluencesPlot": """
    <div style="text-align: center;">
        <img src="http://example.com/placeholder-feature-plot.png" alt="Feature Influences Plot" style="max-width: 90%; height: auto;">
        <p style="font-style: italic; font-size: 0.9em;">Feature influences showing the most important factors in this prediction.</p>
    </div>
    """
}

async def main():
    """Run a simple test"""
    print("Starting Unified MAPE-K Agent Test...")
    
    # Initialize LLM model
    llm = OpenAI(model=os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo'))
    
    # Create instance datapoint
    instance = InstanceDatapoint(
        instance_id=1,
        instance_as_dict=INSTANCE_DATA,
        class_probabilities=[0.8, 0.2],
        model_predicted_label_string="<=50K",
        model_predicted_label=0,
        instance_type="test"
    )
    
    # Set displayable features
    instance.displayable_features = INSTANCE_DATA
    
    # Initialize unified agent
    agent = UnifiedMapeKAgent(
        llm=llm,
        feature_names=FEATURE_NAMES,
        domain_description=DOMAIN_DESCRIPTION,
        user_ml_knowledge="Beginner",
        experiment_id="simple_test"
    )
    
    # Initialize datapoint
    agent.initialize_new_datapoint(
        instance=instance,
        xai_explanations=XAI_EXPLANATIONS,
        xai_visual_explanations=XAI_VISUAL_EXPLANATIONS,
        predicted_class_name="<=50K",
        opposite_class_name=">50K",
        datapoint_count=0
    )
    
    # Simple question test
    print("\nFirst question test:")
    analysis, response = await agent.answer_user_question("Can you give me some example about the project?")
    
    print(f"\nAnalysis: {analysis}")
    print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    # Run main function
    asyncio.run(main()) 