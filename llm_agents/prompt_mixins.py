from typing import Optional, List
from collections import OrderedDict
from llama_index.core.prompts.mixin import PromptMixin, PromptDictType, PromptMixinType
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate


# --- Helper to enumerate and combine task prompts ---
def enumerate_and_combine_tasks(tasks) -> PromptTemplate:
    parts = []
    for idx, t in enumerate(tasks, start=1):
        template = t.get_prompts()["default"].get_template().strip()
        parts.append(f"{idx}. {template}")
    return PromptTemplate("\n\n".join(parts))


# --- SimplePromptMixin ---
class SimplePromptMixin(PromptMixin):
    """Mixin for prompts that only need a single default template."""

    def __init__(self, tpl_str: str):
        self._tpl = PromptTemplate(tpl_str)

    def _get_prompts(self) -> PromptDictType:
        return {"default": self._tpl}

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        if "default" in prompts_dict:
            self._tpl = prompts_dict["default"]


class CompositePromptMixin(PromptMixin):
    """Mixin that auto-composes prompts from a fixed set of sub-modules."""

    def __init__(self, modules: PromptMixinType, exclude_task: bool = False):
        # modules: dict of name -> PromptMixin
        self._modules = modules
        self._exclude_task = exclude_task
        # expose sub-modules as attributes for backward compatibility
        for name, mod in modules.items():
            setattr(self, name, mod)
        self._override: Optional[BasePromptTemplate] = None

    def _get_prompts(self) -> PromptDictType:
        if self._override is not None:
            return {"default": self._override}
        # recursively collect non-task leaf modules to preserve order without duplication
        merged: OrderedDict[str, PromptMixin] = OrderedDict()

        def collect(mods: PromptMixinType):
            for name, m in mods.items():
                if name == "task" or name.endswith("_task"):
                    continue
                if isinstance(m, CompositePromptMixin):
                    collect(m._modules)
                else:
                    if name not in merged:
                        merged[name] = m

        collect(self._modules)
        parts = [
            m.get_prompts()["default"].get_template()
            for m in merged.values()
        ]
        # append any task modules at the end
        if not self._exclude_task:
            for name, mod in self._modules.items():
                if name == "task" or name.endswith("_task"):
                    parts.append(mod.get_prompts()["default"].get_template())
        # remove duplicate blocks while preserving order
        unique_parts = []
        seen = set()
        for part in parts:
            text = part.strip()
            if text not in seen:
                seen.add(text)
                unique_parts.append(part)
        return {"default": PromptTemplate("\n\n".join(unique_parts))}

    def _get_prompt_modules(self) -> PromptMixinType:
        return self._modules

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        for key, tpl in prompts_dict.items():
            if ":" in key:
                mod_name, sub_key = key.split(":", 1)
                if mod_name in self._modules:
                    self._modules[mod_name].update_prompts({sub_key: tpl})
            elif key == "default":
                self._override = prompts_dict["default"]


# --- Prompt building blocks for MonitorPrompt ---

class ContextPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Context>>:
- Domain Description: {domain_description}
- Model Features: {feature_names}
- Current Local Instance of Interest: {instance}
- Predicted Class by AI Model: {predicted_class_name}
"""
        )


class UnderstandingPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Explicit Understanding Display Labels>>:
{understanding_displays}

<<Possible Cognitive Modes of Engagement>>:
{modes_of_engagement}
"""
        )


class HistoryPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Conversation History>>:
{chat_history}
"""
        )


class UserMessagePrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Current User Message>>:
{user_message}
"""
        )


class MonitorTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Task (Monitor)>>:
Analyze the user's latest message in the context of the conversation history. 
1. If an explanation was provided and the user shows **explicit** signs of understanding as described in the **Understanding Display Labels** listed above, classify his explicit understanding. The user may express multiple understanding displays or just ask a question without explicitely signalling understanding.
2. Identify the **Cognitive Mode of Engagement** that best describe the user's engagement. Interpret the user message in the context of the conversation history to disambiguate nuances since a 'yes' or 'no' might refer to understanding something or agreeing to a suggestion. This should always be defined by the given user question and history.
"""
        )


class MonitorAgentSystemPrompt(CompositePromptMixin):
    def __init__(self):
        agent_instructions = """
        You are an analyst that interprets user messages to identify users understanding and cognitive engagement based on the provided chat and his recent message. The user is curious about an AI models prediction and is presented with explanations via explainable AI tools. Your task is to analyze the user's latest message in the context of the conversation history and define the class of cognitive engagement and understanding displays as defined below.
        """
        modules = {
            "instructions": SimplePromptMixin(agent_instructions),
            "context": ContextPrompt(),
            "understanding": UnderstandingPrompt(),
            "monitor_task": MonitorTaskPrompt(),
        }
        super().__init__(modules, exclude_task=True)


# Modular MonitorPrompt using the above building blocks
class MonitorPrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "context": ContextPrompt(),
            "understanding": UnderstandingPrompt(),
            "history": HistoryPrompt(),
            "user_message": UserMessagePrompt(),
            "task": MonitorTaskPrompt(),  # Always in the end
        }
        super().__init__(modules, exclude_task=exclude_task)


# --- Building blocks for Analyze ---

class ExplanationCollectionPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Collection of all explanations as background knowledge to make plans and update the user model>>:
{explanation_collection}
"""
        )


class UserModelPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<User Model>>:
This is the user model, indicating which explanations were understood, not understood or not yet explained: /n
{user_model}.
"""
        )


class LastShownExpPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Explanations that were shown to the user in the last agent message>>:
{last_shown_explanations}
"""
        )


class AnalyzeTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Task (Analyze)>>:
1.	Suggest updates the states of the last shown explanations based on the user’s response and explicit understanding displays. If the user replies with “yes,” “no,” or “okay,” treat it as a possible reaction to the last message, even if they don’t say they understood. For each explanation mentioned in the chat history, assess whether it was understood. If a follow-up question relates to the latest explanation, justify any state change. Mark last shown but unreferenced explanations as “understood.” Ignore ScaffoldingStrategy explanations, as they are not part of the explanation plan.
2.	If the user demonstrates wrong assumptions or misunderstandings on previously understood explanations, mark them with an according state.
3.	Provide only the changes to explanation states, omitting states that remain unchanged. Do not suggest which new explanations should be shown to the user.
"""
        )


class AnalyzePrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "history": HistoryPrompt(),
            "last_shown": LastShownExpPrompt(),
            "user_message": UserMessagePrompt(),
            "task": AnalyzeTaskPrompt(),  # Always in the end
        }
        super().__init__(modules, exclude_task=exclude_task)


class AnalyzeAgentSystemPrompt(CompositePromptMixin):
    def __init__(self):
        agent_instructions = """
        You are an analyst that interprets user messages to identify users understanding based on the provided chat and his recent message. The user is curious about an AI models prediction and is presented with explanations via explainable AI tools. Your task is to analyze the user's latest message in the context of the conversation history and suggest updates to the user model.
        """
        modules = {
            "instructions": SimplePromptMixin(agent_instructions),
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "task": AnalyzeTaskPrompt(),  # Always in the end
        }
        super().__init__(modules, exclude_task=True)


class MonitorAnalyzeTaskPrompt(PromptMixin):
    def __init__(self):
        # combine monitor and analyze tasks using unnumbered templates
        tasks = [MonitorTaskPrompt(), AnalyzeTaskPrompt()]
        combined = "\n\n".join(
            t.get_prompts()["default"].get_template().strip()
            for t in tasks
        )
        self._tpl = PromptTemplate(combined)

    def _get_prompts(self) -> PromptDictType:
        return {"default": self._tpl}

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        if "default" in prompts_dict:
            self._tpl = prompts_dict["default"]


# --- Combined task prompt classes ---
class PlanExecuteTaskPrompt(PromptMixin):
    def __init__(self):
        tasks = [PlanTaskPrompt(), ExecuteTaskPrompt()]
        combined = "\n\n".join(
            t.get_prompts()["default"].get_template().strip()
            for t in tasks
        )
        self._tpl = PromptTemplate(combined)

    def _get_prompts(self) -> PromptDictType:
        return {"default": self._tpl}

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        if "default" in prompts_dict:
            self._tpl = prompts_dict["default"]


class UnifiedTaskPrompt(PromptMixin):
    def __init__(self):
        tasks = [
            MonitorTaskPrompt(),
            AnalyzeTaskPrompt(),
            PlanTaskPrompt(),
            ExecuteTaskPrompt(),
        ]
        self._tpl = enumerate_and_combine_tasks(tasks)

    def _get_prompts(self) -> PromptDictType:
        return {"default": self._tpl}

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        if "default" in prompts_dict:
            self._tpl = prompts_dict["default"]


class MonitorAnalyzePrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "monitor": MonitorPrompt(exclude_task=True),
            "analyze": AnalyzePrompt(exclude_task=True),
            "task": MonitorAnalyzeTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


# --- Building blocks for Plan & Execute ---


class PreviousPlanPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<This is the previous explanation plan that was established for the user before the user's current message>>:
{explanation_plan}
"""
        )


class PlanTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Task (Plan)>>:
You have three steps:
1. **Defining New Explanations if needed**:
   - First, evaluate whether the user’s message expresses a clear information need. If it does, check whether this need can be satisfied using any existing explanations. If no suitable explanation is available in the collection above, define a new one tailored to the identified need and it will be added to the explanation collection.   
   - If the user’s need is unclear, do not create a new explanation yet. Instead, use scaffolding techniques from the explantion collection.

2. **Construct and maintaining an Explanation Plan**:
    - Assume the user may ask only a few questions. The explanation plan should prioritize diverse, informative explanations that highlight relevant and unexpected aspects of the current data instance to clarify the model’s decision. If no explanation plan exists, create one based on the latest user message, prioritizing explanations that address the identified need. The first item will guide the next response.
    - Revise the plan only when there are major gaps, shifts in user understanding, or new high-level concepts are introduced.
    - Map new user questions to existing explanations when possible; if matched, assume familiarity with the explanation’s concept.

3. **Generating the Next ExplanationTarget**:
   - Use the latest input, prior explanations, and the user’s cognitive state and ML knowledge to generate a tailored ExplanationTarget based on the next item in the plan. If the last explanation was unclear, apply scaffolding and integrate it into the next communication goal.
   - Ensure communication goals are concise, engaging, and aligned with the user’s current understanding. If ML knowledge is low or unclear, assess familiarity through conversation context or follow-up questions.
   - For ambiguous inputs, use scaffolding to clarify intent before proceeding.
   - Adapt content dynamically, starting with an overview of key facts, suggesting to delving deeper, simplifying, or redirecting based on the user’s responses.
   - Avoid repetition unless requested, and prioritize addressing user queries over rigidly following the plan.
   - If the user asks an unrelated question, briefly explain that you can only discuss the model’s prediction and the current instance, suggesting new explanations to explore without explicitely mentioning the names but rather what they reveal.
"""
        )


class ExecuteTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Task (Execute)>>

Generate a concise response (max 3 sentences) based on the current User Model, conversation history, and explanation plan. Use only information from the chat history or what can be reasonably inferred from the user’s prior behavior and questions. If the user has agreed to revisit or elaborate on an explanation, continue with that before introducing new concepts. Do not repeat information in the same way, check the conversation history to know what was already communicated and don't repeat it.

Craft the Response:
- Content Alignment: Use the explanation plan and chat history. If eliciting knowledge, prompt briefly rather than explaining fully.
- Tone and Language: Match the user’s cognitive state and ML expertise. Use plain language for lay users; avoid technical terms and XAI method names unless the user is ML-proficient.
- Clarity and Relevance: Be concise and avoid jargon. Focus on explanation over naming techniques. Maintain the flow of conversation. Before generating a sentence, check whether the same explanation or wording was used earlier in the conversation. If yes, do not repeat unless the user explicitly asks for it.
- Stay Focused: If the user goes off-topic, respond that you can only discuss the model’s prediction and the current instance.
- Formatting: Use HTML tags:
  <b> or <strong> for bold,
  <ul> and <li> for bullet lists,
  <p> for paragraphs.
- Visuals: Insert placeholders like ##FeatureInfluencesPlot##. Present the plot first, explain briefly, then ask for understanding. Do not repeat plots, since the user can see the conversation history.
- Engagement: End with a prompt or question. Use scaffolding for ambiguous or low-knowledge input. Don’t repeat previous content unless asked.
"""
        )


# --- PlanPrompt, ExecutePrompt, PlanExecutePrompt ---

class PlanPrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "explanation_plan": PreviousPlanPrompt(),
            "user_model": UserModelPrompt(),
            "last_shown_explanations": LastShownExpPrompt(),
            "history": HistoryPrompt(),
            "user_message": UserMessagePrompt(),
            "task": PlanTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


class ExecutePrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "context": ContextPrompt(),
            "explanation_plan": PreviousPlanPrompt(),
            "history": HistoryPrompt(),
            "user_message": UserMessagePrompt(),
            "task": ExecuteTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


class PlanExecutePrompt(CompositePromptMixin):
    """Composite prompt for Plan + Execute phases with a combined task."""

    def __init__(self, exclude_task: bool = False):
        # Get Execute without next_exp_content
        exec_prompt = ExecutePrompt(exclude_task=True)
        exec_prompt._modules.pop("explanation_plan")
        modules = {
            "plan": PlanPrompt(exclude_task=True),
            "execute": exec_prompt,
            "task": PlanExecuteTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


# --- Plan Approval System Prompts ---

class PlanApprovalSystemPrompt(CompositePromptMixin):
    def __init__(self):
        agent_instructions = """
        You are a plan approval specialist that evaluates predefined explanation plans and determines whether they should be followed or modified based on the user's current needs. The user is curious about an AI model's prediction and has been provided with a predefined explanation plan. Your task is to analyze the user's latest message in the context of the conversation history and decide whether to approve the predefined plan or modify it by selecting a more appropriate explanation.
        """
        modules = {
            "instructions": SimplePromptMixin(agent_instructions),
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "task": PlanApprovalTaskPrompt(),
        }
        super().__init__(modules, exclude_task=True)


class PlanApprovalExecuteSystemPrompt(CompositePromptMixin):
    def __init__(self):
        agent_instructions = """
        You are a planning specialist and communicator that evaluates predefined explanation plans and generates explanations about AI model predictions for users. The user is curious about an AI model's prediction and has been provided with a predefined explanation plan. Your task is to analyze the user's latest message in the context of the conversation history, decide whether to approve the predefined plan or modify it by selecting a more appropriate explanation, and then craft a natural, engaging response based on your decision.
        """
        modules = {
            "instructions": SimplePromptMixin(agent_instructions),
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "task": PlanApprovalExecuteTaskPrompt(),
        }
        super().__init__(modules, exclude_task=True)


# --- SinglePromptPrompt ---


class UnifiedPrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "monitor_analyze": MonitorAnalyzePrompt(exclude_task=True),
            "plan_execute": PlanExecutePrompt(exclude_task=True),
            "task": UnifiedTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


# --- MonitorAnalyzeSystemPrompt ---
class MonitorAnalyzeSystemPrompt(CompositePromptMixin):
    def __init__(self):
        agent_instructions = """
        You are an analyst that interprets user messages to identify users' understanding and cognitive engagement based on the provided chat and recent message. The user is curious about an AI model's prediction and is presented with explanations via explainable AI tools. Your task is to analyze the user's latest message in the context of the conversation history, define the class of cognitive engagement and understanding displays, and suggest updates to the user model as appropriate.
        """
        modules = {
            "instructions": SimplePromptMixin(agent_instructions),
            "context": ContextPrompt(),
            "understanding": UnderstandingPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "task": MonitorAnalyzeTaskPrompt(),
        }
        super().__init__(modules, exclude_task=True)


# --- PlanExecuteSystemPrompt ---
class PlanExecuteSystemPrompt(CompositePromptMixin):
    def __init__(self):
        agent_instructions = """
        You are a planner and communicator that designs explanation plans and generates explanations about AI model predictions for users. The user is curious about an AI model's prediction and is presented with explanations via explainable AI tools. Your task is to analyze the user's latest message in the context of the conversation history, user model, and previous explanations to create a logical explanation plan tailored to the user's needs and understanding level, and then craft a natural, engaging response based on the next explanation content that has been planned.
        """
        modules = {
            "instructions": SimplePromptMixin(agent_instructions),
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "task": PlanExecuteTaskPrompt(),
        }
        super().__init__(modules, exclude_task=True)


# --- Building blocks for Plan Approval ---


class PlanApprovalTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Task (Plan Approval)>>:
You are presented with a previous explanation plan that was created based on the user's initial needs and context. Your task is to evaluate whether this plan should be approved as-is or modified based on the user's latest message and current understanding state.

**Decision Process:**
1. **Analyze User's Current State**: Consider the user's message, understanding displays, cognitive engagement, and any changes in their information needs since the predefined plan was created.

2. **Evaluate Plan Relevance**: Determine if the next step in the plan above still addresses the user's current question or if their needs have shifted. If the user accepts an explanation or elaboration, the plan should adapt by either deepening the explanation or selecting a related, yet unexplored, explanation that better meets the user’s current need.

3. **Make Approval Decision**:
   - **APPROVE (approved=True)**: If the predefined plan's next step is still relevant and appropriate for the user's current need.
   - **MODIFY (approved=False)**: If the user's needs have changed, they're asking about something different, or the predefined plan no longer fits their current understanding level.

4. **Provide Alternative **: When not approving, select a select a more suitable next step that directly addresses the user's current message and state that should be prepanded to the ordered plan.

**Guidelines:**
- Always prioritize the user’s current question or request over the predefined plan.
- Adjust the plan to match the user’s demonstrated level of understanding.
- If the user shows confusion, consider scaffolded or simpler explanations.
- If the user shifts to a new topic, replace the next step with a relevant explanation from the collection.
"""
        )


class PlanApprovalPrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "user_model": UserModelPrompt(),
            "explanation_plan": PreviousPlanPrompt(),
            "history": HistoryPrompt(),
            "last_shown": LastShownExpPrompt(),
            "user_message": UserMessagePrompt(),
            "task": PlanApprovalTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


# --- Combined task prompt classes ---
class PlanApprovalExecuteTaskPrompt(PromptMixin):
    def __init__(self):
        # Combine plan approval and execute tasks
        tasks = [PlanApprovalTaskPrompt(), ExecuteTaskPrompt()]
        combined = "\n\n".join(
            t.get_prompts()["default"].get_template().strip()
            for t in tasks
        )
        self._tpl = PromptTemplate(combined)

    def _get_prompts(self) -> PromptDictType:
        return {"default": self._tpl}

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        if "default" in prompts_dict:
            self._tpl = prompts_dict["default"]


class PlanApprovalExecutePrompt(CompositePromptMixin):
    """Composite prompt for Plan Approval + Execute phases with a combined task."""

    def __init__(self, exclude_task: bool = False):
        # Get Execute prompt for combined use with plan approval
        exec_prompt = ExecutePrompt(exclude_task=True)
        modules = {
            "approval": PlanApprovalPrompt(exclude_task=True),
            "execute": exec_prompt,
            "task": PlanApprovalExecuteTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


# --- Demographics Prompt ---
class DemographicsPrompt(SimplePromptMixin):
    """Prompts for the demographics component."""

    def __init__(self):
        super().__init__(
            """You are an expert in analyzing text to predict user demographics.
Based on the user's input, predict their age, gender, socio-economic status, and education level.
Also, assess their machine learning knowledge level from their message.

Here is the user's message:
---
{user_message}
---

Provide your answer in a JSON format. For each demographic attribute (Age, Gender, Socio-economic status, Education), provide the most likely value and a confidence score between 0 and 100.
If you have alternative predictions, include them as well.
For ML knowledge, provide a string value.

Example for a single attribute:
"age": {{
    "main_prediction": {{"value": "Adolescent", "confidence": 87}},
    "alternative_predictions": [
        {{"value": "Young Adult", "confidence": 10}}
    ]
}}

Provide a brief reasoning for your predictions.
"""
        )


# === TEST HARNESS ===
if __name__ == "__main__":
    """
    Test harness that instantiates each PromptMixin and prints all prompt templates.
    """


    def dump_prompts(pm, title=None):
        print(f"\n=== {title or pm.__class__.__name__} ===")
        tpl = pm.get_prompts().get("default")
        if tpl:
            print(f"\n{tpl.get_template().strip()}\n")


    # List of all prompt classes to test
    prompt_instances = [
        MonitorPrompt(),
        AnalyzePrompt(),
        MonitorAnalyzePrompt(),
        PlanPrompt(),
        ExecutePrompt(),
        PlanExecutePrompt(),
        UnifiedPrompt(),
    ]

    # Dump each prompt
    for pm in prompt_instances:
        dump_prompts(pm)
