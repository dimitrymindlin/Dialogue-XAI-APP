"""
LlamaIndex-specific implementation of the XAI agent.

This module contains the LlamaIndex-specific implementation of the XAI agent,
inheriting common functionality from the BaseAgent class.
"""
from abc import abstractmethod
from typing import Any, Dict, List
import logging

from llama_index.core.workflow.workflow import WorkflowMeta
from llm_agents.agent_utils import configure_logger
from llm_agents.base_agent_class import BaseAgent


# Configure a logger for this module
logger = configure_logger(__name__, "llama_index_agent.log")


class XAIBaseAgentMeta(WorkflowMeta, type(BaseAgent)):
    """
    Metaclass that combines LlamaIndex's WorkflowMeta with ABC.
    """
    pass


class LlamaIndexBaseAgent(BaseAgent, metaclass=XAIBaseAgentMeta):
    """
    LlamaIndex-specific implementation of the XAI agent.
    
    This class extends the BaseAgent with LlamaIndex-specific functionality
    for workflow management and execution.
    """
    
    def __init__(
            self,
            experiment_id: str,
            feature_names: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
            **kwargs
    ):
        super().__init__(
            experiment_id=experiment_id,
            feature_names=feature_names,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge,
            **kwargs
        )
        # Add any LlamaIndex-specific initialization here
        
        logger.info(f"LlamaIndexAgent initialized for experiment: {experiment_id}")

    # Implementation-specific methods can be added here

    async def answer_user_question(self, user_question: str) -> Any:
        """
        Process a user question through the MAPE-K workflow using LlamaIndex.
        
        Args:
            user_question: The user's question text
            
        Returns:
            A tuple: (analysis, response, recommend_visualization)
        """
        # This method should be implemented by concrete subclasses or
        # can contain LlamaIndex-specific implementation here
        raise NotImplementedError("Concrete LlamaIndex agent must implement this method")
