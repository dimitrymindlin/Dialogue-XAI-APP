"""
Helper mixins for the MAPE-K agent classes to reduce code duplication.
These functions handle common operations related to user model updates, logging, and explanation state management.
"""

import datetime
import logging
from typing import Dict, Any, Union, List, Tuple

from llm_agents.agent_utils import append_new_log_row, update_last_log_row
from llm_agents.explanation_state import ExplanationState
from llm_agents.models import MonitorResultModel, AnalyzeResult, PlanResultModel, ExecuteResult, \
    PlanApprovalExecuteResultModel
from llm_agents.models import ChosenExplanationModel, SinglePromptResultModel, PlanApprovalModel
from llm_agents.utils.postprocess_message import replace_plot_placeholders, remove_html_plots_and_restore_placeholders

# Configure logger
logger = logging.getLogger(__name__)


class UserModelHelperMixin:
    """Helper methods for managing user model updates."""

    def __init__(self):
        # Initialize last_shown_explanations if not already present
        if not hasattr(self, 'last_shown_explanations'):
            self.last_shown_explanations = []

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
                step = change.get("step") or change.get("step_name")  # Handle both possible field names
                state = change.get("state")
                # If any of the required fields is missing, skip this change
                if not all([exp, step, state]):
                    continue
            elif hasattr(change, 'explanation_name'):
                # Handle ModelChange objects
                exp = change.explanation_name
                step = change.step_name
                state = change.state
            else:
                # Assuming structure as (explanation_name, state, step) - legacy format
                try:
                    exp, state, step = change
                except (ValueError, TypeError):
                    logger.warning(f"Invalid model change format: {change}")
                    continue

            self.user_model.update_explanation_step_state(exp, step, state)

    def update_explanation_plan(self, plan_result: Union[PlanResultModel, Any]) -> None:
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

    def remove_explanation_from_plan(self, explanation: ChosenExplanationModel) -> None:
        """
        Remove a specific explanation from the predefined plan to avoid duplication.
        
        This focused helper method handles the removal of explanations from self.explanation_plan
        by comparing explanation_name and step fields. It handles field name variations between
        'step_name' and 'step' used in different parts of the codebase.
        
        Args:
            explanation: The ChosenExplanationModel to remove from the predefined plan
        """
        if not hasattr(self, 'explanation_plan') or not self.explanation_plan:
            return

        # Get the step field name from the explanation (handles step_name vs step variations)
        explanation_step = getattr(explanation, 'step_name', None) or getattr(explanation, 'step', None)

        if not explanation_step:
            logger.warning(f"Cannot remove explanation: no step field found in {explanation}")
            return

        # Remove matching explanations from the predefined plan
        self.explanation_plan = [
            exp for exp in self.explanation_plan
            if not (exp.explanation_name == explanation.explanation_name and
                    (getattr(exp, 'step_name', None) == explanation_step or
                     getattr(exp, 'step', None) == explanation_step))
        ]

    def move_explanation_to_front_of_plan(self, explanation: ChosenExplanationModel) -> None:
        """
        Move a specific explanation to the front of the predefined plan to prioritize it.
        
        This helper method finds a matching explanation in self.explanation_plan and moves it to the
        front of the list so it will be considered next. It handles field name variations
        between 'step_name' and 'step' used in different parts of the codebase.
        
        Args:
            explanation: The ChosenExplanationModel to move to the front of the predefined plan
        """
        if not hasattr(self, 'explanation_plan') or not self.explanation_plan:
            return

        # Get the step field name from the explanation (handles step_name vs step variations)
        explanation_step = getattr(explanation, 'step_name', None) or getattr(explanation, 'step', None)

        if not explanation_step:
            logger.warning(f"Cannot move explanation: no step field found in {explanation}")
            return

        # Find the matching explanation in the plan
        target_index = -1
        for i, exp in enumerate(self.explanation_plan):
            if (exp.explanation_name == explanation.explanation_name and
                    (getattr(exp, 'step_name', None) == explanation_step or
                     getattr(exp, 'step', None) == explanation_step)):
                target_index = i
                break

        # If found, move it to the front
        if target_index >= 0:
            target_explanation = self.explanation_plan.pop(target_index)
            self.explanation_plan.insert(0, target_explanation)

    def _determine_explanations_from_approval(self, result: Union[PlanApprovalModel, PlanApprovalExecuteResultModel]) -> Tuple[List[ChosenExplanationModel], int]:
        """
        Core logic to determine which explanations should be used based on approval results.
        This is a pure function that doesn't modify state.
        
        Args:
            result: The plan approval result containing approval status and potential alternative explanations
            
        Returns:
            tuple: (explanations_to_use, explanations_consumed_from_plan)
        """
        explanations_count = getattr(result, 'explanations_count', 1)
        explanations_count = max(1, explanations_count)  # At least 1 explanation should be used
        
        if result.approved and hasattr(self, 'explanation_plan') and self.explanation_plan:
            # Use predefined plan - get the explanations based on count
            end_index = min(explanations_count, len(self.explanation_plan))
            explanations_to_use = self.explanation_plan[:end_index]
            explanations_consumed = end_index
            return explanations_to_use, explanations_consumed
            
        elif not result.approved and result.next_response:
            # Use alternative explanation as the first one
            explanations_to_use = [result.next_response]
            
            # If explanations_count > 1, add additional explanations from the predefined plan
            if explanations_count > 1 and hasattr(self, 'explanation_plan') and self.explanation_plan:
                # Get additional explanations from the plan (explanations_count - 1 more)
                additional_count = explanations_count - 1
                end_index = min(additional_count, len(self.explanation_plan))
                explanations_to_use.extend(self.explanation_plan[:end_index])
                explanations_consumed = end_index  # Only plan explanations are consumed
            else:
                explanations_consumed = 0  # No plan explanations consumed
                
            return explanations_to_use, explanations_consumed
        
        # No explanation available
        return [], 0

    def update_explanation_plan_after_approval(self, result: Union[PlanApprovalModel, PlanApprovalExecuteResultModel]) -> PlanResultModel:
        """
        Update the explanation plan based on plan approval results and return a new plan result.
        
        This method handles the plan updating logic based on approval status:
        - If approved: removes the used explanations from the plan based on explanations_count
        - If not approved: moves the next_response explanation to the front of the plan
        
        Args:
            result: The plan approval result containing approval status and potential alternative explanations
            
        Returns:
            PlanResultModel: A new plan result with the updated explanation plan
        """
        _, explanations_consumed = self._determine_explanations_from_approval(result)
        
        if result.approved and hasattr(self, 'explanation_plan') and self.explanation_plan:
            # Remove the used explanations from the beginning of the plan
            updated_plan = self.explanation_plan[explanations_consumed:] if len(self.explanation_plan) > explanations_consumed else []
            self.explanation_plan = updated_plan
        elif not result.approved and result.next_response:
            # Use alternative explanation - move it to the front of the plan
            self.move_explanation_to_front_of_plan(result.next_response)

        # Return a new PlanResultModel with the current state of the explanation plan
        return PlanResultModel(
            reasoning="Plan updated based on approval status",
            new_explanations=result.new_explanations if hasattr(result, 'new_explanations') else [],
            explanation_plan=self.explanation_plan if hasattr(self, 'explanation_plan') else []
        )

    def get_target_explanations_from_approval(self, result: Union[PlanApprovalModel, PlanApprovalExecuteResultModel]) -> List[ChosenExplanationModel]:
        """
        Get the list of target explanations to be executed based on plan approval results and explanations_count.
        
        This method determines which explanations should be executed:
        - If approved: returns explanations from the predefined plan based on explanations_count
        - If not approved: returns the next_response explanation plus additional explanations from the plan
          to reach the total specified by explanations_count
        
        Args:
            result: The plan approval result containing approval status and potential alternative explanations
            
        Returns:
            List[ChosenExplanationModel]: The target explanations to be used
        """
        explanations_to_use, _ = self._determine_explanations_from_approval(result)
        return explanations_to_use

    def update_explanation_plan_after_execute(self, 
                                            target_explanations: List[ChosenExplanationModel]) -> None:
        """
        Update the explanation plan after execution to remove used explanations.
        
        This method handles post-execution plan updates by modifying the explanation plan in-place:
        - Removes the used target explanations from the current plan
        
        Args:
            target_explanations: The explanations that were actually used/shown during execution
        """
        # Remove used explanations from the plan (in-place modification)
        if hasattr(self, 'explanation_plan') and self.explanation_plan and target_explanations:
            for used_explanation in target_explanations:
                self.remove_explanation_from_plan(used_explanation)

    def update_explanation_plan_from_scaffolding(self, scaffolding_result: Union[Any]) -> PlanResultModel:
        """
        Update the explanation plan based on scaffolding/PlanExecute results.
        
        This method handles plan updates for combined Plan+Execute operations:
        - Updates the explanation plan with new explanations and ordering from scaffolding
        - Adds new explanations to the user model
        - Removes the used explanation from the plan
        
        Args:
            scaffolding_result: The result from the scaffolding/PlanExecute component
            
        Returns:
            PlanResultModel: A new plan result with the updated explanation plan
        """
        new_explanations = []
        
        # Add new explanations to user model if provided
        if hasattr(scaffolding_result, 'new_explanations') and scaffolding_result.new_explanations:
            new_explanations = scaffolding_result.new_explanations
            self.user_model.add_explanations_from_plan_result(new_explanations)
        
        # Update explanation plan if provided
        if hasattr(scaffolding_result, 'explanation_plan') and scaffolding_result.explanation_plan:
            self.explanation_plan = scaffolding_result.explanation_plan
        
        # Remove the first explanation from the plan if it was used
        if hasattr(self, 'explanation_plan') and self.explanation_plan:
            target = scaffolding_result.explanation_plan[0] if hasattr(scaffolding_result, 'explanation_plan') and scaffolding_result.explanation_plan else None
            if target:
                self.remove_explanation_from_plan(target)
        
        # Return a new PlanResultModel with the updated explanation plan
        return PlanResultModel(
            reasoning="Plan updated from scaffolding - updated with new explanations and ordering",
            new_explanations=new_explanations,
            explanation_plan=self.explanation_plan if hasattr(self, 'explanation_plan') else []
        )

    def mark_explanations_as_shown(self, target_explanations: List[ChosenExplanationModel]) -> None:
        """
        Mark multiple explanations as SHOWN in the user model.
        
        This helper method handles the common pattern of iterating through target explanations
        and marking each one as SHOWN in the user model. It eliminates code duplication across
        different mixins that need to perform this operation.
        
        Args:
            target_explanations: List of ChosenExplanationModel objects to mark as shown
        """
        for target_explanation in target_explanations:
            if target_explanation:
                self.user_model.update_explanation_step_state(
                    target_explanation.explanation_name,
                    target_explanation.step_name,
                    ExplanationState.SHOWN.value
                )

    def update_last_shown_explanations(self, explanations: Union[List[ChosenExplanationModel], ChosenExplanationModel]) -> None:
        """
        Update the last_shown_explanations list with new explanations.
        
        This helper method provides a consistent way to update the tracking of recently
        shown explanations across all execute mixins. It handles both single explanations
        and lists of explanations, and ensures ONLY the most recent explanations are kept
        by replacing the entire list with the current explanations.
        
        Args:
            explanations: Single ChosenExplanationModel or list of ChosenExplanationModel objects
                         that were shown to the user in the current execute operation
        """
        if not explanations:
            # Clear the list if no explanations were provided
            self.last_shown_explanations = []
            return
            
        # Always reset to only contain the current explanations (not accumulate)
        if isinstance(explanations, list):
            # Filter out None values and replace the entire list
            valid_explanations = [exp for exp in explanations if exp is not None]
            self.last_shown_explanations = valid_explanations
        else:
            # Single explanation - replace the entire list with just this one
            if explanations is not None:
                self.last_shown_explanations = [explanations]
            else:
                self.last_shown_explanations = []


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
            "plan_approval": "",
            "plan_approval_execute": "",
            "execute": "",
            "user_model": ""
        }
        append_new_log_row(self.current_log_row, self.log_file)
        return self.current_log_row

    def update_log(self, component: str, result: Any) -> None:
        """
        Update the current log row with results from a MAPE-K component.
        
        Args:
            component: The name of the component ('monitor', 'analyze', 'plan', 'plan_approval', 'plan_approval_execute', 'execute')
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
        Format the current predefined explanation plan as XML structure for LLM prompts.
        
        Returns:
            str: An XML-formatted string with plan items, or a fallback message if no plan exists.
        """
        if hasattr(self, 'explanation_plan') and self.explanation_plan:
            plan_items = []
            for plan_item in self.explanation_plan:
                # Create XML structure for each plan item
                plan_xml = f'        <plan_item>'
                plan_xml += f' <explanation_name>{plan_item.explanation_name}</explanation_name>'
                plan_xml += f' <step_name>{plan_item.step_name}</step_name>'
                plan_xml += ' </plan_item>'
                plan_items.append(plan_xml)
            
            # Return as a single-line XML structure to avoid YAML multiline issues
            return "<explanation_plan> " + " ".join(plan_items) + " </explanation_plan>"
        else:
            return "<explanation_plan>No predefined plan available.</explanation_plan>"


class ConversationHelperMixin:
    """Helper methods for managing conversation history."""

    def update_conversation_history(self, user_message: str, agent_response: str) -> None:
        """
        Update the conversation history with user and agent messages.
        
        Args:
            user_message: The user's message
            agent_response: The agent's response
        """
        # Remove HTML plots and restore placeholders before storing in conversation history
        if hasattr(self, 'visual_explanations_dict') and self.visual_explanations_dict:
            cleaned_response = remove_html_plots_and_restore_placeholders(
                agent_response, self.visual_explanations_dict
            )
        else:
            cleaned_response = agent_response
            
        self.append_to_history("user", user_message)
        self.append_to_history("agent", cleaned_response)

    def get_chat_history_as_xml(self) -> str:
        """
        Convert the chat history to XML format.
        
        Returns:
            str: The chat history formatted as XML
        """
        import re
        
        if not hasattr(self, 'chat_history') or not self.chat_history:
            return "<chat_history></chat_history>"
        
        # Handle the case where chat_history is the initial placeholder
        if self.chat_history.startswith("No history available"):
            return "<chat_history></chat_history>"
        
        xml_lines = ["<chat_history>"]
        
        # Split the chat history into lines and parse each message
        lines = self.chat_history.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse user messages
            user_match = re.match(r'^User:\s*(.*)', line)
            if user_match:
                message_content = user_match.group(1)
                # Only escape characters that could break XML: & and "
                escaped_content = message_content.replace('&', '&amp;').replace('"', '&quot;')
                xml_lines.append(f"  <message role=\"user\">{escaped_content}</message>")
                continue
            
            # Parse agent messages
            agent_match = re.match(r'^Agent:\s*(.*)', line)
            if agent_match:
                message_content = agent_match.group(1)
                # Only escape characters that could break XML: & and "
                # Preserve HTML tags and placeholder patterns
                escaped_content = message_content.replace('&', '&amp;').replace('"', '&quot;')
                xml_lines.append(f"  <message role=\"agent\">{escaped_content}</message>")
                continue
        
        xml_lines.append("</chat_history>")
        
        return '\n'.join(xml_lines)


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
        self.update_explanation_plan(result)

        # Log plan
        self.update_log("plan", {
            "new_explanations": result.new_explanations,
            "explanation_plan": result.explanation_plan,
        })

        # Execute phase
        self.update_conversation_history(user_message, result.response)
        response_with_plots = replace_plot_placeholders(result.response, self.visual_explanations_dict)

        # Process all explanations in the plan
        for target in result.explanation_plan:
            # Handle both step_name and step attributes for compatibility
            step_name = getattr(target, 'step_name', None) or getattr(target, 'step', None)
            if step_name:
                self.user_model.update_explanation_step_state(
                    target.explanation_name, step_name, ExplanationState.UNDERSTOOD.value)
                self.last_shown_explanations.append(target)

        # Log execute & user model
        self.update_log("execute", result.response)
        self.finalize_log_row()

        # Update datapoint
        self.user_model.reset_understanding_displays()

        # Return modified result
        return ExecuteResult(
            reasoning=result.reasoning,
            explanations_count=result.explanations_count,
            response=response_with_plots,
        )
