import asyncio
import os
from dotenv import load_dotenv

# Import the OpenAI Unified MAPE-K Agent
from llm_agents.openai_agents_mape_k import SimpleOpenAIMapeKAgent

# Load environment variables
load_dotenv()

async def main():
    """
    Example usage of the OpenAI Simplified MAPE-K Agent
    """
    print("Initializing OpenAI Simple MAPE-K Agent...")
    
    # Create the agent with sample parameters
    agent = SimpleOpenAIMapeKAgent(
        feature_names="age, income, education_level, employment_status, credit_score",
        domain_description="This is a loan approval prediction system that determines if a loan application should be approved or rejected.",
        user_ml_knowledge="beginner",
        experiment_id="test_openai_agents"
    )
    
    # Sample questions to test the agent
    questions = [
        "What does the model predict for this customer?",
        "Why was this person rejected for a loan?",
        "I don't understand how credit score affects the decision.",
        "How important is age in the decision process?",
        "Could you please explain everything again?"
    ]
    
    # Process each question sequentially
    for i, question in enumerate(questions):
        print(f"\n\n--- Question {i+1}: '{question}' ---\n")
        reasoning, response = await agent.answer_user_question(question)
        print(f"Response:\n{response}\n")
        print(f"Reasoning: {reasoning}")
        
        # Wait a moment between questions
        await asyncio.sleep(1)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main()) 