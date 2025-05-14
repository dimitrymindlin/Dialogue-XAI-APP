"""
Advanced MAPE-K agent wrapper module.

This module provides a non-async wrapper around the enhanced MAPE-K agent implementation
to make it easier to use in synchronous code contexts.
"""

import asyncio
import os
import logging
from typing import Dict, Any, List, Tuple, Optional

# Import the enhanced MAPE-K agent
from llm_agents.mape_k_2_components.mape_k_agent_openai_2_enhanced import (
    MapeKXAIWorkflowAgentEnhanced
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mape-k-advanced-wrapper")

# Ensure the logs directory exists
os.makedirs("mape-k-logs", exist_ok=True)
file_handler = logging.FileHandler("mape-k-logs/advanced-mape-k-wrapper.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


class MapeKAgentAdvanced:
    """
    Synchronous wrapper for the enhanced MAPE-K agent.
    
    This class provides a synchronous interface for the asynchronous
    MapeKXAIWorkflowAgentEnhanced class, making it easier to use in
    synchronous code contexts.
    """
    
    def __init__(
        self,
        feature_names: str,
        domain_description: str,
        user_ml_knowledge: str = "beginner",
        experiment_id: str = None,
        model: str = "gpt-4-turbo",
        include_reasoning: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the MapeKAgentAdvanced wrapper.
        
        Args:
            feature_names: Names of features in the dataset
            domain_description: Description of the application domain
            user_ml_knowledge: User's level of ML knowledge
            experiment_id: Identifier for the experiment
            model: OpenAI model to use
            include_reasoning: Whether to include reasoning in the response
            verbose: Whether to print verbose output
        """
        self.async_agent = MapeKXAIWorkflowAgentEnhanced(
            feature_names=feature_names,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge,
            experiment_id=experiment_id,
            model=model,
            include_monitor_reasoning=include_reasoning,
            include_analyze_reasoning=include_reasoning,
            include_plan_reasoning=include_reasoning,
            include_execute_reasoning=include_reasoning,
            verbose=verbose
        )
        self.include_reasoning = include_reasoning
        self.verbose = verbose

    def initialize_new_datapoint(
        self, 
        datapoint, 
        xai_report: str, 
        visual_explanation_dict: Dict[str, str], 
        current_prediction: str,
        opposite_class_name: str,
        datapoint_count: int
    ):
        """
        Initialize the agent with a new datapoint.
        
        Args:
            datapoint: The new datapoint to analyze
            xai_report: Textual XAI report for the datapoint
            visual_explanation_dict: Dictionary of visual explanations
            current_prediction: Current model prediction
            opposite_class_name: Name of the opposite class
            datapoint_count: Datapoint count in the dataset
        """
        self.async_agent.initialize_new_datapoint(
            datapoint=datapoint,
            xai_report=xai_report,
            visual_explanation_dict=visual_explanation_dict,
            current_prediction=current_prediction,
            opposite_class_name=opposite_class_name,
            datapoint_count=datapoint_count
        )

    def answer_user_question(self, user_question: str) -> Tuple[str, str]:
        """
        Answer a user question using the MAPE-K agent.
        
        Args:
            user_question: The user's question
            
        Returns:
            Tuple of (analysis, response)
        """
        try:
            # Run the async method in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            reasoning, response = loop.run_until_complete(
                self.async_agent.answer_user_question(user_question)
            )
            loop.close()
            
            if self.verbose:
                logger.info(f"User question: {user_question}")
                logger.info(f"Reasoning: {reasoning}")
                logger.info(f"Response: {response}")
            
            return reasoning, response
        except Exception as e:
            logger.error(f"Error in answer_user_question: {str(e)}")
            return "Error analyzing question", f"I'm having trouble processing your question. Could you rephrase it?"

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the agent.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.async_agent.get_performance_metrics()


def run_example_advanced_mape_k_agent(
    user_message: str,
    feature_names: str = "Placeholder feature names",
    domain_description: str = "Placeholder domain description",
    user_ml_knowledge: str = "beginner",
    experiment_id: str = "advanced-mape-k-test"
) -> str:
    """
    Example function to demonstrate using the advanced MAPE-K agent
    
    Args:
        user_message: Initial user message to process
        feature_names: Names of features in the dataset
        domain_description: Description of the application domain
        user_ml_knowledge: User's ML knowledge level
        experiment_id: ID for logging purposes
        
    Returns:
        The agent's response
    """
    try:
        # Create an instance of the agent
        agent = MapeKAgentAdvanced(
            feature_names=feature_names,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge,
            experiment_id=experiment_id,
            include_reasoning=True,
            verbose=True
        )
        
        # Sample XAI report for testing
        sample_xai_report = """
        Feature Importance:
        - Age: High importance (0.32)
        - Income: Medium importance (0.21)
        - Education: Low importance (0.09)
        
        Model Prediction: Approved
        Confidence: 87%
        """
        
        # Initialize with test data
        agent.initialize_new_datapoint(
            datapoint={"Age": 35, "Income": 75000, "Education": "Bachelor's"},
            xai_report=sample_xai_report,
            visual_explanation_dict={"feature_importance": "path/to/image.png"},
            current_prediction="Approved",
            opposite_class_name="Rejected",
            datapoint_count=1
        )
        
        # Process the user's question
        analysis, response = agent.answer_user_question(user_message)
        
        print(f"Analysis: {analysis}")
        print(f"Response: {response}")
        
        return response
    except Exception as e:
        logger.error(f"Error running advanced MAPE-K agent example: {str(e)}")
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Example usage
    user_question = "Why was my loan application approved?"
    response = run_example_advanced_mape_k_agent(user_question)
    print(f"Final response: {response}")