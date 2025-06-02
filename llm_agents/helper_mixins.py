"""
Helper mixins for the MAPE-K agent classes to reduce code duplication.
These functions handle common operations related to user model updates, logging, and explanation state management.
"""

import datetime
import logging
from typing import Dict, Any, Union, Optional

from llm_agents.agent_utils import append_new_log_row, update_last_log_row
from llm_agents.explanation_state import ExplanationState
from llm_agents.models import MonitorResultModel, AnalyzeResult, PlanResultModel, ExecuteResult
from llm_agents.models import ChosenExplanationModel, SinglePromptResultModel, PlanApprovalModel
from llm_agents.utils.postprocess_message import replace_plot_placeholders

# Configure logger
logger = logging.getLogger(__name__)


class UserModelHelperMixin:
    """Helper methods for managing user model updates."""

    def update_user_model_from_monitor(self, monitor_result: Union[MonitorResultModel, Any]) -> None:
        """
        Update the user model based on monitor component results.
        
        Args:
            monitor_result: The result from the monitor component
        """
        if hasattr(monitor_result, 'mode_of_engagement') and monitor_result.mode_of_engagement:
            self.user_model.cognitive_state = self.modes_of_engagement.get_differentiating_description(
                monitor_result.mode_of_engagement)

        if hasattr(monitor_result,
                   'explicit_understanding_displays') and monitor_result.explicit_understanding_displays:
            self.user_model.set_explicit_understanding_signals(monitor_result.explicit_understanding_displays)

    def update_user_model_from_analyze(self, analyze_result: Union[AnalyzeResult, Any]) -> None:
        """
        Update the user model based on analyze component results.
        
        Args:
            analyze_result: The result from the analyze component
        """
        if not hasattr(analyze_result, 'model_changes') or not analyze_result.model_changes:
            return

        for change in analyze_result.model_changes:
            if isinstance(change, dict):
                exp = change.get("explanation_name")
                step = change.get("step")
                state = change.get("state")
                # If any of the required fields is missing, skip this change
                if not all([exp, step, state]):
                    continue
            else:
                # Assuming structure as (explanation_name, state, step)
                try:
                    exp, state, step = change
                except (ValueError, TypeError):
                    logger.warning(f"Invalid model change format: {change}")
                    continue

            self.user_model.update_explanation_step_state(exp, step, state)

    def update_user_model_from_plan(self, plan_result: Union[PlanResultModel, Any]) -> None:
        """
        Update the user model and explanation plan based on plan component results.
        
        Args:
            plan_result: The result from the plan component
        """
        # Update explanation plan if provided
        if hasattr(plan_result, 'explanation_plan') and plan_result.explanation_plan:
            self.explanation_plan = plan_result.explanation_plan

        # Add new explanations to user model if provided
        if hasattr(plan_result, 'new_explanations') and plan_result.new_explanations:
            self.user_model.add_explanations_from_plan_result(plan_result.new_explanations)

    def update_user_model_from_execute(self, execute_result: Union[ExecuteResult, Any],
                                       next_response: ChosenExplanationModel = None) -> None:
        """
        Update the user model after execute component and prepare the response.
        
        Args:
            execute_result: The result from the execute component
            next_response: The next response to be shown (from plan result)
        """
        if next_response:
            for nxt in next_response:
                if hasattr(nxt, 'explanation_name') and hasattr(nxt, 'step_name'):
                    self.user_model.update_explanation_step_state(
                        nxt.explanation_name, nxt.step_name, ExplanationState.UNDERSTOOD.value)

        # Process any visual explanations in the response
        if hasattr(execute_result, 'response') and hasattr(self, 'visual_explanations_dict'):
            execute_result.response = replace_plot_placeholders(
                execute_result.response, self.visual_explanations_dict)


class LoggingHelperMixin:
    """Helper methods for managing logging."""

    def initialize_log_row(self, user_message: str) -> Dict[str, Any]:
        """
        Initialize a new log row with default values.
        
        Args:
            user_message: The user's message
            
        Returns:
            Dict: A new log row dictionary
        """
        self.current_log_row = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": self.experiment_id,
            "datapoint_count": self.datapoint_count,
            "user_message": user_message,
            "monitor": "",
            "analyze": "",
            "plan": "",
            "execute": "",
            "user_model": ""
        }
        append_new_log_row(self.current_log_row, self.log_file)
        return self.current_log_row

    def update_log(self, component: str, result: Any) -> None:
        """
        Update the current log row with results from a MAPE-K component.
        
        Args:
            component: The name of the component ('monitor', 'analyze', 'plan', 'execute')
            result: The result from the component
        """
        if component not in self.current_log_row:
            logger.warning(f"Unknown component: {component}")
            return

        self.current_log_row[component] = result
        update_last_log_row(self.current_log_row, self.log_file)

    def finalize_log_row(self) -> None:
        """
        Finalize the current log row with the updated user model.
        """
        self.current_log_row["user_model"] = self.user_model.get_state_summary(as_dict=True)
        update_last_log_row(self.current_log_row, self.log_file)

    def format_predefined_plan_for_prompt(self) -> str:
        """
        Format the current predefined explanation plan as a numbered list for LLM prompts.
        
        Returns:
            str: A formatted string with numbered plan items, or a fallback message if no plan exists.
        """
        if hasattr(self, 'explanation_plan') and self.explanation_plan:
            predefined_plan_items = []
            for number, plan_item in enumerate(self.explanation_plan):
                predefined_plan_items.append(f"{number + 1}. {plan_item.explanation_name}: {plan_item.step_name}")
            return "\n".join(predefined_plan_items)
        else:
            return "No predefined plan available."


class ConversationHelperMixin:
    """Helper methods for managing conversation history."""

    def update_conversation_history(self, user_message: str, agent_response: str) -> None:
        """
        Update the conversation history with user and agent messages.
        
        Args:
            user_message: The user's message
            agent_response: The agent's response
        """
        self.append_to_history("user", user_message)
        self.append_to_history("agent", agent_response)


class UnifiedHelperMixin(UserModelHelperMixin, LoggingHelperMixin, ConversationHelperMixin):
    """
    Combined helper mixin that provides all helper methods in one class.
    """

    def process_unified_result(self, user_message: str, result: SinglePromptResultModel) -> None:
        """
        Process a unified MAPE-K result (from single-prompt approach).
        
        Args:
            user_message: The user's message
            result: The unified MAPE-K result
        """
        # Monitor & Analyze phase
        self.update_user_model_from_monitor(result)

        # Log monitor/analyze
        self.update_log("monitor", {
            "mode_of_engagement": result.mode_of_engagement,
            "explicit_understanding_displays": result.explicit_understanding_displays
        })
        self.update_log("analyze", result.model_changes)

        # Plan phase
        self.update_user_model_from_plan(result)

        # Log plan
        self.update_log("plan", {
            "new_explanations": result.new_explanations,
            "explanation_plan": result.explanation_plan,
        })

        # Execute phase
        self.update_conversation_history(user_message, result.response)
        response_with_plots = replace_plot_placeholders(result.response, self.visual_explanations_dict)

        next_response = result.explanation_plan[0]

        for target in next_response:
            self.user_model.update_explanation_step_state(
                target.explanation_name, target.step_name, ExplanationState.UNDERSTOOD.value)
            self.last_shown_explanations.append(target)

        # Log execute & user model
        self.update_log("execute", result.response)
        self.finalize_log_row()

        # Update datapoint
        self.user_model.new_datapoint()

        # Return modified result
        return ExecuteResult(
            reasoning=result.reasoning,
            response=response_with_plots,
            summary_sentence=result.summary_sentence,
        )
