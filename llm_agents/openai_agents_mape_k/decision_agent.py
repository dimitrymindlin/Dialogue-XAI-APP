import os
from dotenv import load_dotenv
from typing import Tuple, Dict, Any
import asyncio

from agents import Agent, Runner, function_tool

# Load environment variables
load_dotenv()

@function_tool
def determine_query_type(user_message: str) -> Dict[str, Any]:
    """
    Determine the type of user query to decide which agent should handle it.
    
    Args:
        user_message: The user's message/question
        
    Returns:
        A dictionary containing the query type and routing decision
    """
    return {
        "query_type": "model_explanation",  # or "dataset_information", "general_question", etc.
        "should_use_mape_k": True,  # Whether to use the MAPE-K agent or RAG system
        "reasoning": "Explanation of why this routing decision was made"
    }

class DecisionAgent:
    """
    Decision agent that routes queries to either the MAPE-K agent or RAG system.
    """
    def __init__(self, feature_names="", domain_description="", user_ml_knowledge="", experiment_id=""):
        """
        Initialize the decision agent.
        
        Args:
            feature_names: Names of features in the model
            domain_description: Description of the domain
            user_ml_knowledge: User's ML knowledge level
            experiment_id: Experiment ID
        """
        self.feature_names = feature_names
        self.domain_description = domain_description
        self.user_ml_knowledge = user_ml_knowledge
        self.experiment_id = experiment_id
        
        # Initialize the agent
        self.agent = self._create_agent()
        
        # We'll initialize the MAPE-K agent and RAG system on demand
        self.mape_k_agent = None
        self.rag_system = None
    
    def _create_agent(self):
        """Create the decision agent with OpenAI Agents SDK."""
        instructions = f"""
        You are a decision agent that determines how to route user queries about an AI model and its predictions.
        
        Your task is to analyze the user's query and decide whether it should be handled by:
        1. The MAPE-K agent (for questions about model predictions, explanations, features, etc.)
        2. The RAG system (for general questions about datasets, concepts, summaries, etc.)
        
        DOMAIN CONTEXT:
        - Domain Description: {self.domain_description}
        - Model Features: {self.feature_names}
        
        Consider the user's query carefully and decide which system is best suited to answer it.
        """
        
        decision_agent = Agent(
            name="XAI Query Router",
            instructions=instructions,
            tools=[determine_query_type]
        )
        
        return decision_agent
    
    async def _get_mape_k_agent(self):
        """Lazily initialize the MAPE-K agent if not already initialized."""
        if self.mape_k_agent is None:
            from llm_agents.openai_agents_mape_k.mape_k_agent import MapeKAgent
            self.mape_k_agent = MapeKAgent(
                feature_names=self.feature_names,
                domain_description=self.domain_description,
                user_ml_knowledge=self.user_ml_knowledge,
                experiment_id=self.experiment_id
            )
        return self.mape_k_agent
    
    async def _get_rag_system(self):
        """Lazily initialize the RAG system if not already initialized."""
        if self.rag_system is None:
            from llm_agents.openai_agents_mape_k.rag_system import RAGSystem
            self.rag_system = RAGSystem(
                domain_description=self.domain_description
            )
        return self.rag_system
    
    async def process_query(self, user_query: str) -> Tuple[str, str]:
        """
        Process the user query by routing it to the appropriate agent.
        
        Args:
            user_query: The user's query/question
            
        Returns:
            Tuple of (reasoning, response)
        """
        # Run the decision agent to determine query type
        result = await Runner.run(self.agent, input=f"Analyze this user query: {user_query}")
        
        # Extract the decision
        decision = result.outputs.get("determine_query_type", {})
        should_use_mape_k = decision.get("should_use_mape_k", True)
        query_type = decision.get("query_type", "model_explanation")
        reasoning = decision.get("reasoning", "No reasoning provided")
        
        if should_use_mape_k:
            # Use the MAPE-K agent for model explanation queries
            mape_k_agent = await self._get_mape_k_agent()
            mape_k_reasoning, mape_k_response = await mape_k_agent.answer_user_question(user_query)
            reasoning = f"Decision: Route to MAPE-K agent. {reasoning}. MAPE-K reasoning: {mape_k_reasoning}"
            return reasoning, mape_k_response
        else:
            # Use the RAG system for dataset information or general queries
            rag_system = await self._get_rag_system()
            rag_reasoning, rag_response = await rag_system.answer_query(user_query)
            reasoning = f"Decision: Route to RAG system. {reasoning}. RAG reasoning: {rag_reasoning}"
            return reasoning, rag_response
    
    # Maintain compatibility with the XAIBaseAgent interface
    async def answer_user_question(self, user_question: str) -> Tuple[str, str]:
        """
        Answer a user question by routing to the appropriate system.
        
        Args:
            user_question: The user's question
            
        Returns:
            Tuple of (reasoning, response)
        """
        return await self.process_query(user_question)

# Test function
async def test_decision_agent():
    agent = DecisionAgent(
        feature_names="age, income, education",
        domain_description="An income prediction model that determines if a person earns above or below $50K per year",
        user_ml_knowledge="beginner",
        experiment_id="test"
    )
    
    questions = [
        "How does this model predict income?",
        "Can you explain what features affect the prediction?",
        "What dataset was used to train this model?",
        "How many samples are in the training data?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        reasoning, response = await agent.answer_user_question(question)
        print(f"Routing: {reasoning}")
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(test_decision_agent()) 