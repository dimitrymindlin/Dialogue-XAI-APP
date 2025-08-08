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

        # Log state before execution
        self.log_state_snapshot("pre_execution")
        
        try:
            analysis, response = await self._execute_workflow(user_question)
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            analysis = "No reasoning available"
            response = f"Agent execution failed: {str(e)}"
        
        # Log state after execution and finalize
        self.log_state_snapshot("post_execution")
        self.log_component_input_output("Turn", user_question, response)
        
        try:
            self.finalize_log()
        except Exception as e:
            logger.error(f"Error finalizing CSV log: {e}", exc_info=True)
        
        return analysis, response

    async def answer_user_question_stream(self, user_question: str, stream_callback=None):
        """
        Stream-enabled version of answer_user_question with unified logging.
        
        Args:
            user_question: The user's question text
            stream_callback: Optional callback for streaming chunks
            
        Yields:
            Dict with streaming data: {"type": "partial"|"final"|"error", "content": str, "is_complete": bool}
        """
        # Begin run: initialize CSV buffer
        self._csv_items.clear()
        
        # Set up streaming callback if provided
        if stream_callback:
            self.set_stream_callback(stream_callback)
        
        # Log state before execution
        self.log_state_snapshot("pre_execution")
        
        final_analysis = "No reasoning available"
        final_response = "Sorry, please try again"
        
        try:
            async for chunk in self._execute_workflow_streaming(user_question):
                if chunk["type"] == "final":
                    final_analysis = chunk.get("reasoning", "No reasoning available")
                    final_response = chunk.get("content", "Sorry, please try again")
                    yield chunk
                    break
                elif chunk["type"] == "error":
                    final_response = f"Agent execution failed: {chunk['content']}"
                    yield chunk
                    break
                else:
                    yield chunk
        except Exception as e:
            logger.error(f"Streaming workflow execution failed: {e}", exc_info=True)
            final_response = f"Agent execution failed: {str(e)}"
            yield {"type": "error", "content": str(e), "is_complete": True}
        
        # Log state after execution and finalize
        self.log_state_snapshot("post_execution")
        self.log_component_input_output("Turn", user_question, final_response)
        
        try:
            self.finalize_log()
        except Exception as e:
            logger.error(f"Error finalizing CSV log: {e}", exc_info=True)
