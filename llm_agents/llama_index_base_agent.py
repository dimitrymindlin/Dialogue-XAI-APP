"""
LlamaIndex-specific implementation of the XAI agent.

This module contains the LlamaIndex-specific implementation of the XAI agent,
inheriting common functionality from the BaseAgent class.
"""
from typing import Any

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
            feature_names: str = "",
            feature_units: str = "",
            feature_tooltips: str = "",
            domain_description: str = "",
            user_ml_knowledge: str = "",
            **kwargs
    ):
        super().__init__(
            feature_names=feature_names,
            feature_units=feature_units,
            feature_tooltips=feature_tooltips,
            domain_description=domain_description,
            user_ml_knowledge=user_ml_knowledge,
            **kwargs
        )

    @staticmethod
    def _get_default_llm(model_name: str, **kwargs):
        """Get a default LLM instance."""
        from llama_index.llms.openai import OpenAI
        return OpenAI(model=model_name, **kwargs)

    async def answer_user_question(self, user_question: str) -> Any:
        """
        Process a user question through the MAPE-K workflow using LlamaIndex.
        
        Args:
            user_question: The user's question text
            
        Returns:
            A tuple: (analysis, response)
        """
        # Begin run: initialize CSV buffer
        self._csv_items.clear()
        self.log_prompt("UserQuestion", user_question)

        if not hasattr(self, 'run'):
            raise NotImplementedError("Agent must have a 'run' method for the workflow.")

        # Execute the workflow
        result = await self.run(input=user_question)

        # Extract reasoning and response with fallbacks
        analysis = getattr(result, "reasoning", "No reasoning available")
        response = getattr(result, "response", "Sorry, please try again")

        # Buffer analysis and response for CSV
        self.log_prompt("Analysis", analysis)
        self.log_prompt("Response", response)

        # End run: finalize CSV log row
        try:
            self.finalize_log()
        except Exception as e:
            logger.error(f"Error writing CSV row: {e}", exc_info=True)
        
        return analysis, response
