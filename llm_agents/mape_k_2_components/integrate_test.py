import asyncio
import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

from llm_agents.mape_k_2_components.mape_k_workflow_agent import MapeK2Component
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


async def run_comparison_test(agent_class, use_unified=True, agent_name="Unknown"):
    """Test an agent with sample data and measure performance"""
    print(f"\n=== Testing {agent_name} ===")
    
    # Initialize LLM with specified model
    llm = OpenAI(model=os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo'))
    
    # Create instance datapoint with correct parameters
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
    
    # Initialize the agent
    if agent_class == MapeK2Component:
        agent = agent_class(
            llm=llm,
            feature_names=FEATURE_NAMES,
            domain_description=DOMAIN_DESCRIPTION,
            user_ml_knowledge="Beginner",
            experiment_id=f"{agent_name}_test",
            use_unified=use_unified
        )
    else:
        agent = agent_class(
            llm=llm,
            feature_names=FEATURE_NAMES,
            domain_description=DOMAIN_DESCRIPTION,
            user_ml_knowledge="Beginner",
            experiment_id=f"{agent_name}_test"
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
    
    # Test questions and measure performance
    questions = [
        "Why was this person predicted to earn under $50K?",
        "What would need to change for this person to earn over $50K?",
        "Can you explain more about how education impacts the prediction?"
    ]
    
    results = []
    
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        start_time = time.time()
        analysis, response = await agent.answer_user_question(question)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        results.append({
            "agent": agent_name,
            "question": question,
            "time": elapsed_time,
            "response_length": len(response)
        })
        
        print(f"Response time: {elapsed_time:.2f} seconds")
        print(f"Response length: {len(response)} characters")
        # Uncomment to see full responses:
        # print(f"Response: {response}")
    
    return results


async def main():
    """Run comparison tests between different agent configurations"""
    print("Starting MAPE-K Agent Integration Test")
    
    all_results = []
    
    # Test traditional MAPE-K with multiple steps
    results = await run_comparison_test(
        MapeK2Component, 
        use_unified=False, 
        agent_name="Traditional MAPE-K"
    )
    all_results.extend(results)
    
    # Test unified MAPE-K with a single step
    results = await run_comparison_test(
        MapeK2Component, 
        use_unified=True, 
        agent_name="Unified MAPE-K (MapeK2Component)"
    )
    all_results.extend(results)
    
    # Test dedicated Unified agent
    results = await run_comparison_test(
        UnifiedMapeKAgent, 
        agent_name="Unified MAPE-K (Dedicated Agent)"
    )
    all_results.extend(results)
    
    # Convert results to DataFrame and calculate statistics
    df = pd.DataFrame(all_results)
    print("\n=== Performance Comparison ===")
    
    avg_times = df.groupby("agent")["time"].mean()
    print("\nAverage response times (seconds):")
    for agent, avg_time in avg_times.items():
        print(f"{agent}: {avg_time:.2f}s")
    
    # Calculate relative speed improvements
    baseline = avg_times["Traditional MAPE-K"]
    for agent, avg_time in avg_times.items():
        if agent != "Traditional MAPE-K":
            improvement = (baseline - avg_time) / baseline * 100
            print(f"{agent} improvement: {improvement:.1f}% faster than traditional approach")

    # Generate detailed performance table
    print("\nDetailed performance by question:")
    pivot_table = df.pivot_table(
        index="question", 
        columns="agent", 
        values="time",
        aggfunc="mean"
    )
    print(pivot_table)


if __name__ == "__main__":
    # Run the comparison tests
    asyncio.run(main()) 