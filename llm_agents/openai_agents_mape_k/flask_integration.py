"""
Flask Integration Example Code

This file demonstrates how to integrate the MAPE-K agent created with OpenAI Agents SDK
into a Flask API.
"""

from llm_agents.openai_agents_mape_k import SimpleOpenAIMapeKAgent

def initialize_openai_agent(user_ml_knowledge, experiment_id):
    """
    Creates a new OpenAI MAPE-K agent.
    
    Args:
        user_ml_knowledge (str): User's ML knowledge level
        experiment_id (str): Experiment ID
        
    Returns:
        SimpleOpenAIMapeKAgent: The created agent
    """
    return SimpleOpenAIMapeKAgent(
        feature_names="",  # These will be set in initialize_new_datapoint
        domain_description="",  # These will be set in initialize_new_datapoint
        user_ml_knowledge=user_ml_knowledge,
        experiment_id=experiment_id
    )

# Example code to be added to Flask app.py
"""
from llm_agents.openai_agents_mape_k.flask_integration import initialize_openai_agent

# Inside the ExplainBot class
def __init__(self, study_group, ml_knowledge="beginner", user_id="TEST"):
    self.user_id = user_id
    self.study_group = study_group
    self.use_llm_agent = study_group == "chat" or study_group == "interactive"
    self.use_active_dialogue_manager = False

    # Other code...
    
    # To use OpenAI MAPE-K Agent
    if self.use_llm_agent:
        self.agent = initialize_openai_agent(ml_knowledge, user_id)
    
    # Other code...

# In update_state_from_nl method
async def update_state_from_nl(self, user_input):
    if self.use_llm_agent:
        try:
            # Generate response using OpenAI Agent
            reasoning, response = await self.agent.answer_user_question(user_input)
            
            # response is already in HTML format
            return response, None, None, reasoning
        except Exception as e:
            print(f"Error using OpenAI Agent: {e}")
            # Fallback to original method
            # Call original method...
    
    # Original code...
"""

# Test code
async def test_integration():
    """
    Tests the integration example
    """
    import asyncio
    
    # Create the agent
    agent = initialize_openai_agent("beginner", "test_integration")
    
    # Set domain and features (normally set in initialize_new_datapoint)
    agent.domain_description = "Loan approval prediction system"
    agent.feature_names = "age, income, education, credit_score"
    
    # Test question
    response = await agent.answer_user_question("How does credit score affect the prediction?")
    
    print("Integration test successful!")
    print(f"Response: {response[1]}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_integration()) 