"""
LlamaIndex-specific implementation of the XAI agent.

This module contains the LlamaIndex-specific implementation of the XAI agent,
inheriting common functionality from the BaseAgent class.
"""
from abc import abstractmethod
from typing import Any, Dict, List, Optional
import logging
import json

from llama_index.core import PromptTemplate
from llama_index.core.workflow.workflow import WorkflowMeta
from llm_agents.agent_utils import configure_logger
from llm_agents.base_agent_class import BaseAgent
from llm_agents.prompt_mixins import DemographicsPrompt
from llm_agents.models import UserDemographics, DemographicsResponse


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

    def set_user_demographics(self, demographics: UserDemographics):
        """Set the user demographics."""
        self.user_demographics = demographics

    async def analyze_demographics(self, user_message: str):
        """Analyzes user demographics based on their message."""
        demographics_pm = DemographicsPrompt()
        template = demographics_pm.get_prompts()["default"].get_template()
        prompt_str = template.format(user_message=user_message)
        demographics_prompt = PromptTemplate(prompt_str)
        demographics_result = await self.llm.astructured_predict(DemographicsResponse, demographics_prompt)
        self.user_demographics = demographics_result.demographics
        
        # Log the demographics result to the console for debugging
        logger.info("--- DEMOGRAPHIC ANALYSIS RESULT ---")
        try:
            # Pydantic v2
            logger.info(json.dumps(demographics_result.model_dump(), indent=2))
        except AttributeError:
            # Pydantic v1
            logger.info(json.dumps(demographics_result.dict(), indent=2))
        logger.info("---------------------------------")
        
        return demographics_result
