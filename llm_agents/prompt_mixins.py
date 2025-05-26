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
<<This is the entire collection of all available explanations>>:
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
<<This are the explanations that were shown to the user in the last agent message>>:
{last_shown_explanations}
"""
        )


class AnalyzeTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Task (Analyze)>>:
1.	Suggest updates the states of the last shown explanations based  on the user’s message, cognitive state and understanding displays. Take into account that the user might say 'yes', 'no', or 'okay' as a response to the last agent message, rather than explicitly stating that they understood it. Check  each explanation individually and assess if the user understood it or not. If the user asks a followup question resulting from the latest given explanation, provide a reasoning for why the explanation state should change. For non-referenced explanations that were explained last, update their states to “understood”. If the agent used a ScaffoldingStrategy as the last explanation, ignore this explanation in the user model as it is not a part of the explanation plan.
2.	If the user demonstrates wrong assumptions or misunderstandings on previously understood explanations, mark them with an according state.
3.	Provide only the changes to explanation states, omitting states that remain unchanged. Do not suggest which new explanations should be shown to the user.

Reason step by step about the updates to the user model and why an explantion should be marked as understood or not understood.
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


class NextExplanationPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<The plan suggests that the next explanation should be>>:
{next_exp_content}
"""
        )


class PlanTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Task (Plan)>>:
You have three steps:
1. **Defining New Explanations if needed**:
   - First, evaluate whether the user’s message expresses a clear information need. If it does, check whether this need can be satisfied using any existing explanations. If no suitable explanation is available, define a new one tailored to the identified need and add it to the explanation plan.   
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
   
Think step by step about each step and provide a reasoning for each decision based on the users model indicating the UNDERSTOOD explanations, the users's latest message, the conversation history, and the current explanation plan. Especially when deciding to create a new explanation, you should provide a reasoning for why the explanation is needed and how it relates to the user's message and why it cannot be answered with existing explanations from the collection.
"""
        )


class ExecuteTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Task (Execute)>>

Using the current User Model, generate a concise response (max 3 sentences per Explanation Goal) that fits the user’s ML knowledge, understanding level, and chat history. Respond directly to the user’s query using only the information from the conversation and clear assumptions.

Craft the Response:
- Content Alignment: Use the explanation plan and chat history. If eliciting knowledge, prompt briefly rather than explaining fully.
- Tone and Language: Match the user’s cognitive state and ML expertise. Use plain language for lay users; avoid technical terms unless the user is ML-proficient.
- Clarity and Relevance: Be concise and avoid jargon. Focus on explanation over naming techniques. Maintain the flow of conversation.
- Stay Focused: If the user goes off-topic, respond that you can only discuss the model’s prediction and the current instance.
- User Reasoning Context: If the user’s guess was correct (see first agent message), ask them to reflect on their reasoning. Acknowledge or correct their view while continuing with the plan.
- Formatting: Use HTML tags:
  <b> or <strong> for bold,
  <ul> and <li> for bullet lists,
  <p> for paragraphs.
- Visuals: Insert placeholders like ##FeatureInfluencesPlot##. Present the plot first, explain briefly, then ask for understanding.
- Engagement: End with a prompt or question. Use scaffolding for ambiguous or low-knowledge input. Don’t repeat previous content unless asked.
"""
        )


# --- PlanPrompt, ExecutePrompt, PlanExecutePrompt ---

class PlanPrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "user_message": UserMessagePrompt(),
            "explanation_plan": PreviousPlanPrompt(),
            "user_model": UserModelPrompt(),
            "last_shown_explanations": LastShownExpPrompt(),
            "task": PlanTaskPrompt(),
            "history": HistoryPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


class ExecutePrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "context": ContextPrompt(),
            "history": HistoryPrompt(),
            "user_message": UserMessagePrompt(),
            "next_exp_content": NextExplanationPrompt(),
            "task": ExecuteTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


class PlanExecutePrompt(CompositePromptMixin):
    """Composite prompt for Plan + Execute phases with a combined task."""

    def __init__(self, exclude_task: bool = False):
        # Get Execute without next_exp_content
        exec_prompt = ExecutePrompt(exclude_task=True)
        exec_prompt._modules.pop("next_exp_content")
        modules = {
            "plan": PlanPrompt(exclude_task=True),
            "execute": exec_prompt,
            "task": PlanExecuteTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


# --- Plan and Execute System Prompts ---

class PlanAgentSystemPrompt(CompositePromptMixin):
    def __init__(self):
        agent_instructions = """
        You are a planner that designs explanation plans for users interacting with AI model predictions. The user is curious about an AI model's prediction and is presented with explanations via explainable AI tools. Your task is to analyze the user's latest message in the context of the conversation history, user model, and previous explanations to create a logical explanation plan tailored to the user's needs and understanding level.
        """
        modules = {
            "instructions": SimplePromptMixin(agent_instructions),
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "task": PlanTaskPrompt(),
        }
        super().__init__(modules, exclude_task=True)


class ExecuteAgentSystemPrompt(CompositePromptMixin):
    def __init__(self):
        agent_instructions = """
        You are a communicator that generates explanations about AI model predictions for users. The user is curious about an AI model's prediction and needs clear, concise explanations tailored to their understanding level. Your task is to craft a natural, engaging response based on the next explanation content that has been planned, taking into account the user's message, conversation history, and cognitive state.
        """
        modules = {
            "instructions": SimplePromptMixin(agent_instructions),
            "context": ContextPrompt(),
            "task": ExecuteTaskPrompt(),
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
