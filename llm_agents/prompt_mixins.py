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
<context>
  <domain>{domain_description}</domain>
  <features>
    {feature_context}
  </features>
  <current_instance>{instance}</current_instance>
  <model_prediction>{predicted_class_name}</model_prediction>
</context>
"""
        )


class UnderstandingPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<user_signals>
    <displays>{understanding_displays}</displays>
    <engagement_modes>{modes_of_engagement}</engagement_modes>
</user_signals>
"""
        )


class HistoryPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<conversation_history>
    {chat_history}
</conversation_history>
"""
        )


class UserMessagePrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<user_message>
    {user_message}
</user_message>
"""
        )


class MonitorTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<task>
  <objective>Monitor and analyze user engagement</objective>
  <instructions>
    Analyze the user's latest message in the context of the conversation history.
    <steps>
      <step>If an explanation was provided and the user shows **explicit** signs of understanding as described in the **Understanding Display Labels** listed above, classify his explicit understanding. The user may express multiple understanding displays or just ask a question without explicitely signalling understanding.</step>
      <step>Identify the **Cognitive Mode of Engagement** that best describe the user's engagement. Interpret the user message in the context of the conversation history to disambiguate nuances since a 'yes' or 'no' might refer to understanding something or agreeing to a suggestion. This should always be defined by the given user question and history.</step>
    </steps>
  </instructions>
</task>
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
<explanation_collection>
    {explanation_collection}
</explanation_collection>
"""
        )


class UserModelPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<user_model>
    <description>This is the user model, indicating the user's cognitive state and machine learning models as well as which explanations were understood, not yet explained, or currently being shown. Consider it to be the best explainer possible and adapt to the user</description>
    {user_model}
</user_model>
"""
        )


class LastShownExpPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<last_shown_explanations>
    {last_shown_explanations}
</last_shown_explanations>
"""
        )


class AnalyzeTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<task>
  <objective>Analyze</objective>
  <steps>
    <step number="1">
      <instructions>Suggest updates to the states of the last shown explanations based on the user’s response and explicit understanding displays. If the user replies with “yes,” “no,” or “okay,” treat it as a possible reaction to the last message, even if they don’t say they understood. For each explanation mentioned in the chat history, assess whether it was understood. If a follow-up question relates to the latest explanation, justify any state change. Mark last shown but unreferenced explanations as “understood.” Ignore ScaffoldingStrategy explanations, as they are not part of the explanation plan.</instructions>
    </step>
    <step number="2">
      <instructions>If the user demonstrates wrong assumptions or misunderstandings on previously understood explanations, mark them with an according state.</instructions>
    </step>
    <step number="3">
      <instructions>Provide only the changes to explanation states, omitting states that remain unchanged. Do not suggest which new explanations should be shown to the user. Differentiate between a user understanding or acknowledging the understanding and a user conforming to see a suggested explanation.</instructions>
    </step>
  </steps>
</task>
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

class MinitorAnalyzePersona(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
You are a Cognitive Engagement Analyst & Understanding Tracker: an expert in monitoring how users engage with AI explanations and how their understanding evolves. You interpret subtle communication signals to assess both real-time cognitive engagement and deeper comprehension, distinguishing surface acknowledgment from genuine understanding. Your focus includes classifying engagement modes and tracking knowledge progression over time. Guided by principles of evidence-based assessment, contextual interpretation, and dynamic model updating, you ensure that user models reflect both current engagement and long-term learning trajectories.
            """
        )
class MonitorAnalyzePrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "persona": MinitorAnalyzePersona(),
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
<previous_plan>
    <description>This is the previous explanation plan ordered from top to bottom that was established for the user before the user's current message</description>
    <content>{explanation_plan}</content>
</previous_plan>
"""
        )


class PlanTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<task>
  <objective>Plan</objective>
  <description>You have three steps to create and manage explanation plans.</description>
  <definitions>
    <AGREEMENT>Short affirmations such as “yes”, “ok”, “sure”, “show me”, “go ahead”, 👍, or lack of objection after an offer.</AGREEMENT>
    <UNCLEAR>Replies that are off-topic, vague, or express confusion (e.g., “I don’t get it”).</UNCLEAR>
  </definitions>
  <steps>
    <step number="1">
      <title>Defining New Explanations if needed</title>
      <instructions>
        - First, evaluate whether the user's message expresses a clear information need. If it does, check whether this need can be satisfied using any existing explanations. If no suitable explanation is available in the collection above, define a new one tailored to the identified need and it will be added to the explanation collection.   
        - If the user explicitly or implicitly AGREES (e.g., “yes”, “ok”, “show me”, “go ahead”), immediately mark that explanation as APPROVED and set it as the next ExplanationTarget. Do NOT ask again or scaffold—proceed to execution.
        - If the user's need is unclear, do not create a new explanation yet. Instead, scaffold briefly using techniques from the explanation collection.
      </instructions>
    </step>
    <step number="2">
      <title>Construct and maintaining an Explanation Plan</title>
      <instructions>
        - Assume the user may ask only a few questions. Prioritize diverse, informative explanations that clarify the model's decision and surface unexpected aspects. If no explanation plan exists, create one based on the latest user message. The first item MUST be executed in the very next response once AGREEMENT is detected; do not re-ask.
        - Revise the plan only when there are major gaps, shifts in user understanding, or new high-level concepts are introduced.
        - Map new user questions to existing explanations when possible; if matched, assume familiarity with the explanation's concept. If the user asks a direct question like "what if", skip introducing the concept and proceed directly with the explanation content.
      </instructions>
    </step>
    <step number="3">
      <title>Generating the Next ExplanationTarget</title>
      <instructions>
        - Use the latest input, prior explanations, and the user's cognitive state and ML knowledge to generate a tailored ExplanationTarget based on the next item in the plan. If the last explanation was unclear, apply scaffolding and integrate it into the next communication goal.
        - Ensure communication goals are concise, engaging, and aligned with the user's current understanding. If ML knowledge is low or unclear, assess familiarity through conversation context or brief follow-up questions.
        - For ambiguous inputs, use scaffolding to clarify intent before proceeding.
        - Adapt content dynamically: start with key facts, then (optionally) offer to delve deeper, simplify, or redirect based on the user's responses.
        - Avoid repetition unless requested, and prioritize addressing user queries over rigidly following the plan.
        - If the user asks an unrelated question, briefly state that you can only discuss the model's prediction and the current instance, and offer a relevant next insight (describe what it reveals, not its method name).
      </instructions>
    </step>
  </steps>
</task>
"""
        )


class ExecuteTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<task>
  <objective>Execute</objective>
  <description>
    Generate a concise response (3–4 sentences) based on the current user model, conversation history, and explanation plan. Do not meta-comment on your process; if the user AGREED, start directly with the promised content. Introduce a concept only if required, in ≤1 sentence, then continue. You have exactly one message per user question.
  </description>
  <guidelines>
    <ul>
      <li><b>Quantity:</b> Include exactly the information needed—no more, no less—by checking prior turns and omitting redundant content.</li>
      <li><b>Quality:</b> State only verifiable facts drawn from chat history or logical inference; avoid unsupported or speculative wording.</li>
      <li><b>Relation:</b> Ensure every sentence directly advances the current objective—executing a concise response—without branching into unrelated methods.</li>
      <li><b>Manner:</b> Write clearly, briefly, and in logical order; avoid ambiguity, obscurity, and unnecessary prolixity.</li>
    </ul>
  </guidelines>
  <response_crafting>
    <content_alignment>
      Use the explanation plan and chat history to guide responses. When eliciting knowledge, prompt briefly instead of fully explaining. If the user’s question aligns with an explanation method, do not introduce that concept first—proceed directly with the explanation. When the user agrees to see a suggestion, show it immediately without narration.
    </content_alignment>
    <tone_and_language>
      Match the user’s cognitive state and ML expertise. Use plain language for lay users; avoid technical method names unless the user is clearly ML-proficient. Present model behavior with calibrated certainty—include numbers or ranges when available; avoid unsupported certainty. Use modal verbs when evidence is indirect.
    </tone_and_language>
    <clarity_and_relevance>
      Be concise and avoid jargon. Focus on explanation results rather than naming techniques or repeating what the user has already seen. Before generating each sentence, verify it hasn’t been used earlier—do not repeat unless explicitly requested. When responding to AGREEMENT, lead with the promised artifact/result (e.g., numbers/plot), then (optionally) one short orienting sentence.
    </clarity_and_relevance>
    <focus>
      If the user goes off-topic, respond that you can only discuss the model’s prediction and the current instance.
    </focus>
    <formatting>
      Use HTML tags:
      <ul>
        <li><b>&lt;b&gt;</b> for bold</li>
        <li><b>&lt;ul&gt;</b> and <b>&lt;li&gt;</b> for bullet lists</li>
        <li><b>&lt;p&gt;</b> for paragraphs</li>
      </ul>
    </formatting>
    <visuals>
      Insert placeholders like ##FeatureInfluencesPlot##. Present the visual first, then give a one-sentence reading. Ask for understanding only if AGREEMENT was not given or clarification is needed. Do not repeat visuals already shown.
    </visuals>
    <engagement>
      • On AGREEMENT: deliver the explanation/artifact and stop—no follow-up question.  
      • If a concrete choice is required and no preference is stated, ask exactly one short question (e.g., “Compare to global results?”).  
      • To guide proactively, append a single optional next-step suggestion phrased as an offer (“Next, I can show how X interacts with Y.”). Describe what it reveals; skip method names unless the user shows high ML literacy.
    </engagement>
  </response_crafting>
</task>
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


class ExecutePersona(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
You are an Adaptive XAI Communicator: an expert in delivering user-tailored explanations based on the user's cognitive state and ML knowledge. You craft concise, engaging responses that align with the user's understanding and the current explanation plan. Your focus is on clarity, relevance, and maintaining conversational flow while adapting dynamically to user feedback and questions.
"""
        )


class ExecutePrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "persona": ExecutePersona(),
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
        exec_prompt._modules.pop("persona")
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
<task>
  <objective>Plan Approval</objective>
  <description>You are presented with a previous explanation plan that was created based on the user's initial needs and context. Your task is to evaluate whether this plan should be approved as-is or modified based on the user's latest message and current understanding state.</description>
  <decision_process>
    <step number="1">
      <title>Analyze User's Current State</title>
      <description>Consider the user's message, understanding displays, cognitive engagement, and any changes in their information needs since the predefined plan was created.</description>
    </step>
    <step number="2">
      <title>Evaluate Plan Relevance</title>
      <description>Determine if the next step in the plan above still addresses the user's current question or if their needs have shifted. If the user accepts an explanation or elaboration, the plan should adapt by either deepening the explanation or selecting a related, yet unexplored, explanation that better meets the user's current need.</description>
    </step>
    <step number="3">
      <title>Make Approval Decision</title>
      <options>
        <approve>APPROVE (approved=True): If the predefined plan's next step is still relevant and appropriate for the user's current need.</approve>
        <modify>MODIFY (approved=False): If the user's needs have changed, they're asking about something different, or the predefined plan no longer fits their current understanding level.</modify>
      </options>
    </step>
    <step number="4">
      <title>Provide Alternative</title>
      <description>When not approving, select a more suitable next step that directly addresses the user's current message and state that should be prepended to the ordered plan.</description>
    </step>
  </decision_process>
  <guidelines>
    - Always prioritize the user's current question or request over the predefined plan.
    - Adjust the plan to match the user's demonstrated level of understanding.
    - If the user shows confusion, consider scaffolded or simpler explanations.
    - If the user shifts to a new topic, replace the next step with a relevant explanation from the collection.
  </guidelines>
</task>
"""
        )


class PlanApprovalPersona(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
You are the best Co-Constructive XAI Learning Facilitator—an adaptive explainer who identifies user understanding gaps and tailors explanation plans accordingly, ensuring foundational concepts are addressed before introducing complex ideas. You specialize in scaffolded, collaborative learning where explanations evolve dynamically based on user comprehension, avoiding overload while guiding users through AI model predictions in digestible, logically sequenced steps.
""")


class PlanApprovalPrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "persona": PlanApprovalPersona(),
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


class PlanApprovalExecutePersona(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
You are an Adaptive XAI Communication Specialist who plans and delivers explanations that adjust to user understanding. You identify knowledge gaps, reorganize content to meet user needs, and use techniques like progressive disclosure and language-level adjustment. By managing cognitive load, you ensure each interaction builds comprehension and supports incremental learning.
""")


class PlanApprovalExecutePrompt(CompositePromptMixin):
    """Composite prompt for Plan Approval + Execute phases with a combined task."""

    def __init__(self, exclude_task: bool = False):
        # Get Execute prompt for combined use with plan approval
        exec_prompt = ExecutePrompt(exclude_task=True)
        exec_prompt._modules.pop("persona")
        # Get PlanApprovalPrompt without persona and task
        plan_approval_prompt = PlanApprovalPrompt(exclude_task=True)
        plan_approval_prompt._modules.pop("persona")
        modules = {
            "persona": PlanApprovalExecutePersona(),
            "approval": plan_approval_prompt,
            "execute": exec_prompt,
            "task": PlanApprovalExecuteTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)
