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


# Modular MonitorPrompt using the above building blocks
class MonitorPrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "context": ContextPrompt(),
            "understanding": UnderstandingPrompt(),
            "history": HistoryPrompt(),
            "user_message": UserMessagePrompt(),
            "monitor_task": MonitorTaskPrompt(),  # Always in the end
        }
        super().__init__(modules, exclude_task=exclude_task)


# --- Building blocks for Analyze ---

class ExplanationCollectionPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Explanation Collection>>:
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
<<Last Shown Explanations>>:
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
<<Previous Explanation Plan>>:
{explanation_plan}
"""
        )


class LastExplanationPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Last Shown Explanations>>:
{last_shown_explanations}
"""
        )


class NextExplanationPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Planning decided that the next explanation should be>>:
{next_exp_content}
"""
        )


class PlanTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Task (Plan)>>:
You have three steps:
1. **Defining New Explanations**:
   - Determine if the latest input requires introducing a new concept by checking each possible explanation. If the user does not explicitly request one, apply scaffolding to address understanding gaps.
   - Define any new concept and integrate it into the explanation_plan if needed.

2. **Maintaining the Explanation Plan**:
    - consider that the user might only ask one or maximally three questions in a row:
    - If no explanation_plan exists, generate one based on the latest input.
    - Continuously assess the plan’s relevance. Update it only when significant gaps, shifts in user understanding, or requests for new overarching concepts occur.
    - Continuously check if the user's question that can be mapped to another explanation. If the user asks a question that fits any of the explanation methods, avoid explaining the concept of that explanations and assume the user knows it.
    - Address minor misunderstandings through communication_goals without altering the plan. Remove concepts that the user fully understands, as justified by the UserModel and recent input.

3. **Generating the Next Explanations**:
   - Based on the latest input, previous explanations, the user's cognitive state and ML knowledge, create the next_explanations along with tailored communication_goals. If the last explanation was not understood, consider scaffolding strategies and put them back into the next communication goal.
   - Ensure these goals are concise, engaging, and matched to the user’s current state. If the user’s ML knowledge is low or unclear, first assess and elicit their familiarity with key concepts.
   - For ambiguous requests, use scaffolding to clarify intent before providing details.
   - Adapt content dynamically—delving deeper, simplifying, or redirecting based on the user’s responses.
   - Avoid repetition unless the user explicitly asks for clarification, and prioritize reacting to user queries over strictly following the plan.
   - If the user asks question unrelated to understanding the current explanation, provide a short answer that you are not able to respond to that and can only talk about the model prediction and the instance shown.\n

Think step by step and provide a reasoning for each decision based on the users model indicating the UNDERSTOOD explanations, the users's latest message, the conversation history, and the current explanation plan.
"""
        )


class ExecuteTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<<Task (Execute)>>:
Using the current user model, generate a response that aligns with the user's understanding level, ML knowledge, and conversation history. Your answer should be concise (no more than 3 sentences per Explanation Goal) and directly address the user's query to not upset the user with irrelevant information.
Rely solely on the information in the chat history and any clear, deducible assumptions.

**Craft the Response**:
- **Content Alignment**: Use only the information from the chat history and explanation plan to fulfill the goal of the next explanation. If the objective is to elicit knowledge from the user, do so with a concise prompt rather than a full explanation.
- **Language and Tone**: Match the user’s proficiency and cognitive state. Maintain a natural, teacher-like tone, ensuring clarity without unnecessary repetition. For lay users, use everyday language while preserving accuracy—highlighting key and less important points. Avoid technical terms like features, plot, unless the user is knowledgeable in ML. For lay users, try to express clear explanations with wording like: Most important, least important, and do not simplify the content too much as to not lose the meaning and accuracy.
- **Clarity and Conciseness**: Present information in a clear and accessible manner, minimizing technical jargon and excessive details, keeping the conversation flow as seen by the chat history. It is less about mentioning the specific XAI techniques that are used and more about using them to explain the model's prediction and answer the user's understanding needs.
- **Stay Focused**: If the user asks a question unrelated to understanding the current explanation, provide a short answer that you are not able to respond to that and can only talk about the model prediction and the instance shown.
- **Contextualize User's furst question**: If the user's guess was correct, indicating by the first agent message in the chat history, the user is prompted to check if his reasoning alignes with the model reasoning. Therefore, the user might indicate why he thinks the model predicts a certain class. In this case, consider the explanation plan and next explanation but react to the users's reasoning by varifying his decision making or correcting it. 
- **Formatting**: Use HTML elements for structure and emphasis:
    - `<b>` or `<strong>` for bold text,
    - `<ul>` and `<li>` for bullet points,
    - `<p>` for paragraphs.
- **Visual Placeholders**: If visuals (e.g., plots) are required, insert placeholders in the format `##plot_name##` (e.g., `##FeatureInfluencesPlot##`) but keep the text as if the placeholder is substituted already. When a plot is shown, display it first, provide a brief explanation, then ask if the user understood.
- **Engagement and Adaptive Strategy**: Conclude with a question or prompt that invites further interaction without overwhelming the user. If the user's ML knowledge is low or if the request is ambiguous, assess their familiarity with key concepts using scaffolding before expanding. Avoid repeating previously explained content unless explicitly requested.

Think step by step to craft a natural response that clearly connects the user's question with your answer and consider the User Model to see alrady UNDERSTOOD explanations to not repeat them, and consider the chat history as well as if the user's guess about the ML model prediction was correct.
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
            "last_shown_explanations": LastExplanationPrompt(),
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


# --- SinglePromptPrompt ---


class UnifiedPrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "monitor_analyze": MonitorAnalyzePrompt(exclude_task=True),
            "plan_execute": PlanExecutePrompt(exclude_task=True),
            "task": UnifiedTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


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
