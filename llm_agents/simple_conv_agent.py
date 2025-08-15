from llama_index.core.llms.llm import LLM
import logging

# Configure logger
logger = logging.getLogger(__name__)

from llama_index.core.workflow import Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy

from llm_agents.helper_mixins import ConversationHelperMixin

from llama_index.core import PromptTemplate
from llm_agents.llama_index_base_agent import LlamaIndexBaseAgent
from llm_agents.utils.postprocess_message import replace_plot_placeholders

# Reduce verbosity for OpenAI and HTTP libraries
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Import mixins from mape_k_component_mixins
from llm_agents.mape_k_component_mixins import BaseAgentInitMixin, StreamingMixin


class ConversationalMixin(ConversationHelperMixin):
    """
    Simplified mixin that handles conversation in a single unified step.
    No user modeling or plan tracking - just conversation history and direct response generation.
    """

    @step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=0))
    async def respond(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """
        Single unified step that processes user question and generates response.
        """
        user_message = ev.input
        await ctx.set("user_message", user_message)

        # Create simplified unified prompt
        from llm_agents.prompt_mixins import ConversationalPrompt
        conv_pm = ConversationalPrompt()
        template = conv_pm.get_prompts()["default"].get_template()

        # Get available explanations as formatted string for the prompt
        available_explanations = self._get_available_explanations_text()

        prompt_str = template.format(
            domain_description=self.domain_description,
            feature_context=self.get_formatted_feature_context(),
            instance=self.instance,
            predicted_class_name=self.predicted_class_name,
            chat_history=self.get_chat_history_as_xml(),
            user_message=user_message,
            available_explanations=available_explanations,
        )

        conv_prompt = PromptTemplate(prompt_str)

        # Import the model here to avoid circular imports
        from llm_agents.models import ConversationalResultModel
        result = await self._predict(ConversationalResultModel, conv_prompt, "ConversationalResult")

        # Get explanations that were used in the response
        used_explanations = self._extract_used_explanations(result)

        # Update conversation history
        self.update_conversation_history(user_message, result.response)

        # Process any visual explanations in the response
        result.response = replace_plot_placeholders(result.response, self.visual_explanations_dict)

        return StopEvent(result=result)

    def _get_available_explanations_text(self) -> str:
        """
        Get a formatted text representation of all available XAI explanations.
        Uses the same method as MAPE-K agents to access explanations through the user model.
        """
        if not hasattr(self, 'user_model') or not self.user_model:
            return "No explanations available"

        try:
            # Use the same method as MAPE-K agents to get explanation collection
            explanation_collection = self.user_model.get_complete_explanation_collection(as_dict=False)
            
            if not explanation_collection or explanation_collection.strip() == "":
                return "No explanations available"
                
            return explanation_collection
        except Exception as e:
            logger.warning(f"Error getting explanation collection: {e}")
            return "No explanations available"

    def _extract_used_explanations(self, result) -> list[str]:
        """
        Extract which explanations were actually used in the response.
        This is a simplified version that doesn't need complex plan tracking.
        """
        # For now, return empty list - can be enhanced later if needed
        return []


class ConversationalStreamAgent(Workflow, LlamaIndexBaseAgent, ConversationalMixin,
                                StreamingMixin, BaseAgentInitMixin):
    """
    Simplified conversational agent that maintains conversation history but does not
    track user models or explanation plans. Uses a single unified workflow step.
    """

    def __init__(self, llm: LLM = None, structured_output: bool = True, timeout: float = 100.0, **kwargs):
        # Initialize LlamaIndexBaseAgent with all the base parameters
        LlamaIndexBaseAgent.__init__(self, **kwargs)

        # Initialize agent-specific components
        self._init_agent_components(
            llm=llm,
            structured_output=structured_output,
            timeout=timeout,
            **kwargs
        )

    def log_state_snapshot(self, snapshot_type: str):
        """
        Log minimal state for ConversationalStreamAgent.
        
        Unlike MAPE-K agents, this agent doesn't track user models or explanation plans,
        so we log minimal state that accurately reflects its stateless conversational nature.
        
        Args:
            snapshot_type: Type of snapshot ('pre_execution' or 'post_execution')
        """
        import json
        import datetime
        
        # Log minimal conversational state - just conversation status
        conversation_status = "active" if hasattr(self, 'chat_history') and self.chat_history else "empty"
        
        # Create minimal state data that reflects what this agent actually uses
        state_data = {
            "user_model_state": {},  # ConversationalStreamAgent doesn't track user understanding
            "explanation_plan_state": ""  # ConversationalStreamAgent doesn't use explanation plans
        }

        # Log the state snapshot with timestamp for timing analysis
        timestamp = datetime.datetime.now().isoformat()
        self._log_state_with_timing(
            f"{snapshot_type}_state",
            json.dumps(state_data),
            timestamp
        )