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
  <attributes>
    {feature_context}
  </attributes>
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


class PlanSpecificTaskPrompt(SimplePromptMixin):
    """Plan-specific task instructions without redundant content"""

    def __init__(self):
        super().__init__(
            """
<task>
  <objective>Plan</objective>
  <description>You have two steps to create and manage explanation plans.</description>

  <steps>
    <step number="1">
      <title>Define New Explanations only if needed</title>
      <instructions>
        - First, evaluate whether the user's message expresses a clear information need
        - Check whether this need can be satisfied using any existing explanations
        - If no suitable explanation is available, define a new one tailored to the identified need
        - If the user's need is unclear, scaffold briefly using techniques from the explanation collection
      </instructions>
    </step>

    <step number="2">
      <title>Construct and Maintain an Explanation Plan</title>
      <instructions>
        - Assume the user may ask only a few questions and adhere to human explanation design principles stated above when picking explanations
        - Prioritize diverse, informative explanations that clarify the model's decision. Do not plan confidence unless user specifically asked for it, as stated by human explanation design principles.
        - Create plan covering at least three main explanations if none exists
        - The first item MUST be executed in the very next response once AGREEMENT is detected
        - Revise the plan only when there are major gaps or shifts in understanding
        - Emit the *full* ordered list of upcoming step_names for each explanation
      </instructions>
    </step>
  </steps>
</task>
"""
        )


class PlanTaskPrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "principles": ExplanationPrinciplesPrompt(),
            "scheme": CorrelationCausalSchemePrompt(),
            "engagement": UserEngagementGuidelinesPrompt(),
            "task": PlanSpecificTaskPrompt(),  # Task-specific content below
        }
        super().__init__(modules, exclude_task=exclude_task)


class ExecuteTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<task>
  <objective>Execute</objective>
  <description>
    Generate a concise response (3–4 sentences) based on the current user model, conversation history, and explanation plan. Do not meta-comment on your process; if the user AGREED, start directly with the promised content. Introduce a concept only if required, in ≤1 sentence, then continue. You have exactly one message per user question so make it self-contained.
    Your responses should accurately convey XAI method outputs without adding speculation about real-world causations beyond what the data tells us. Stay faithful to what the methods actually reveal, unless the user explicitly asks for a broader interpretation.
  </description>
  <fidelity_principles>
    Always report only what the XAI output shows, what influences the prediction, without inventing causal stories about why those patterns exist.
    <faithful_translation>
      Translate technical concepts accurately without adding interpretation:
      - FeatureImportance: "how much each attribute influences the prediction"
      - Counterfactual: "what would need to change for a different prediction"
      - Anchor: "conditions that guarantee this prediction"
      - Confidence: "how certain the model is"
    </faithful_translation>
    If the user explicitly asks for a causal interpretation, you can provide it, but making it clear that this is not what can be interpreted from the model, but rather a causal interpretation that might be wrong.
    Do not endorse causal/stigmatizing claims if not supported by explanation methods; clarify that attributions reflect correlations in data, not nexessarly ability or causation in real world. If encountered, add a one-sentence caution, then proceed with model-scope facts.
  </fidelity_principles>
  <human_explanations_principles>
      <contrastive>
        Explanations answer "Why P rather than Q?"—focus on the implied counterfactual foil, not exhaustive detail.
      </contrastive>
      <selectivity>
        Good explanations pick a few relevant causes (not every possible one), guided by what the user finds most helpful.
      </selectivity>
      <social>
        Explanations occur in conversation—tailor them to the user's current knowledge and engage in a two-way dialogue.
      </social>
  </human_explanations_principles>
  <guidelines>
    <ul>
      <li><b>Quantity:</b> Include exactly the information needed to answer the user's information need, no more and no less, by checking prior turns and omitting redundant content. Prioritise not repeating over following the plan. </li>
      <li><b>Quality:</b> State only verifiable facts drawn from chat history or logical inference; avoid unsupported or speculative wording or interpretations that are not supported by the explanations.</li>
      <li><b>Relation:</b> Ensure every sentence directly advances the current objective—executing a concise response—without branching into unrelated methods.</li>
      <li><b>Manner:</b> Write clearly, briefly, and in logical order; avoid ambiguity, obscurity, and unnecessary prolixity.</li>
    </ul>
  </guidelines>
  <response_crafting>
    <content_alignment>
      - Use the explanation plan and chat history to guide responses, but prioritize not repeating previously shown explanations if the user did not show confusion, even if they are in the plan.
      - When eliciting knowledge, prompt briefly instead of fully explaining. 
      - If the user’s question aligns with an explanation method, do not introduce that concept first—proceed directly with the explanation. 
      - When the user agrees to see a suggestion, show it immediately without narration. 
      - Always state scope (individual/global) and target class/polarity in the first sentence naturally without like "For this individual,..." or "In general, ..." and if ambiguous, ask one short clarifying question before answering.
      - When listing top features, show top-3 only, then say ‘others smaller’ to stay within 3–4 sentences.
    </content_alignment>
    <tone_and_language>
        Match the user’s cognitive state, ML expertise, and conversational style, mirror their formality and phrasing and metaphors or established concepts.
        Use plain language for lay users; avoid technical method names and terms like "anchoring" unless the user is clearly ML-proficient.
    </tone_and_language>
    <clarity_and_relevance>
      Be concise and avoid jargon. Focus on explanation results rather than naming techniques or repeating what the user has already seen. Before generating each sentence, verify it hasn’t been used earlier—do not repeat unless explicitly requested. When responding to AGREEMENT, lead with the promised artifact/result (e.g., numbers/plot), then (optionally) one short orienting sentence.
    </clarity_and_relevance>
    <focus>
      If the user goes off-topic, respond that you can only discuss the model’s prediction and the current instance. Counterfactuals: present the closest single-change flip first; if none, the smallest multi-change set;
    </focus>
    <visuals>
      Insert placeholders like ##FeatureInfluencesPlot## last in your response. First, give one insight sentence. Do not repeat visuals already shown.
    </visuals>
    <engagement>
      - On AGREEMENT: deliver the explanation/artifact right away, without reintroducing or motivating the explanation, (e.g., features supporting prediction A). Do not switch polarity or scope. Ask for understanding only if AGREEMENT was not given or clarification is needed.
      - Focus on WHAT the model learned, not WHY society works that way
      - Be honest about the limits of what XAI methods reveal
      - If a concrete choice is required and no preference is stated, ask exactly one short question (e.g., “Compare to global results?”).  
      - To guide proactively, append a single optional next-step suggestion (that is in the explanation plan) phrased as an offer (“Next, we can explore which attributes change the model prediction.”). Describe what it reveals and skip method names unless the user shows high ML literacy.
    </engagement>
    <formatting>
      ALWAYS Use HTML tags to make the response more readable:
      <ul>
        <li><b>&lt;b&gt;</b> for bold</li>
        <li><b>&lt;ul&gt;</b> and <b>&lt;li&gt;</b> for bullet lists</li>
        <li><b>&lt;p&gt;</b> for paragraphs</li>
      </ul>
    </formatting>
  </response_crafting>
  <rendered_step_names>
    include exactly which explanations and steps were rendered in the response to track and update the plan and avoind repetition.
  </rendered_step_names>
</task>
"""
        )


# --- PlanPrompt, ExecutePrompt, PlanExecutePrompt ---


# Reusable Modules to Extract

class ExplanationPrinciplesPrompt(SimplePromptMixin):
    """Human-explanation design principles used by both Plan and Approval tasks"""

    def __init__(self):
        super().__init__(
            """
<!-- Human-explanation design principles -->
<principles>
  <contrastive>
    Explanations answer "Why P rather than Q?"—focus on the implied counterfactual foil, not exhaustive detail.
  </contrastive>
  <selectivity>
    Good explanations pick a few relevant causes (not every possible one), guided by what the user finds most helpful.
  </selectivity>
  <social>
    Explanations occur in conversation—tailor them to the user's current knowledge and engage in a two-way dialogue.
  </social>
</principles>
"""
        )


class CorrelationCausalSchemePrompt(SimplePromptMixin):
    """Correlation to causal reasoning scheme"""

    def __init__(self):
        super().__init__(
            """
<!-- Correlation→Causal scheme -->
<scheme>
  1. Look at high-level correlations like feature importances to identify candidate attributes to investigate further.<br/>
  2. Check if these attributes appear in counterfactuals or other explanations to infer causal relationships.
  Example: Attribute A is most important and switching only it would lead to a different prediction. Attribute B is second important, it has a rare value and is often among the attributes to change for a different model prediction.
</scheme>
"""
        )


class UserEngagementGuidelinesPrompt(SimplePromptMixin):
    """Shared guidelines for user interaction and adaptation"""

    def __init__(self):
        super().__init__(
            """
<engagement_guidelines>
  <agreement_detection>
    - Detect explicit or implicit AGREEMENT (e.g., "yes", "ok", "show me", "go ahead")
    - When agreement is detected, proceed immediately without re-asking or additional scaffolding
  </agreement_detection>

  <adaptation_rules>
    - Always prioritize the user's current question or request over predefined plans
    - Adjust to match the user's demonstrated level of understanding and ML knowledge
    - If user shows confusion, consider scaffolded or simpler explanations
    - When user asks direct questions like "what if", skip conceptual introduction and proceed directly
    - Avoid repetition unless explicitly requested
    - Maintain correlation→causal flow when possible, but prioritize user needs over rigid sequencing
  </adaptation_rules>

  <conversation_flow>
    - If user shifts to a new topic, adapt immediately with relevant explanation
    - Map new user questions to existing explanations when possible
    - If matched, assume familiarity with the explanation's concept
    - Briefly redirect if user asks unrelated questions outside model prediction scope
  </conversation_flow>
</engagement_guidelines>
"""
        )


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
            "persona": PlanExecutePersona(),
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

class ApprovalSpecificTaskPrompt(SimplePromptMixin):
    """Approval-specific task instructions without redundant content"""

    def __init__(self):
        super().__init__(
            """
<task>
  <objective>Plan Approval</objective>
  <description>Evaluate whether the existing plan should be approved as-is or modified based on the user's latest message and current understanding state.</description>
    
  <invariants>
    <rule>Never plan or repeat steps already delivered in this thread (derive from history).</rule>
    <rule>If the previous reply said a version of “Next, we can …”, the very next action must deliver that artifact (no concept-only explanation).</rule>
    <rule>Treat acknowledgements (“ok/okay/yes/right”) as approval to execute the last suggested artifact</rule>
    <rule>Artifacts = plots, counterfactual lists, top-k attributes, numeric outputs. Concepts = ≤1 sentence inline context only.</rule>
  </invariants>
  
  <decision_process>
      <step number="1">
      <title>Lock Scope, Polarity, and Artifact</title>
      <description>
        - Determine scope: for this individual vs in general without explicitly using these words. Determine target/polarity (e.g., "supporting class A" vs "supporting class B") consistent with the user's last request/AGREEMENT.
        - Choose the next explanation; prefer an artifact for “what-if/compare/show” intents. Do not repeat visuals already shown.
      </description>
    </step>
    
    <step number="2">
      <title>Evaluate Plan Relevance & Social Context</title>
      <description>
        - If the user asks about a single attribute/value, prefer a local single-feature explanation with a signed magnitude and brief foil comparison (micro-contrast), potentially adding if it is included in counterfactuals and it's ceteris paribus explanation.
        - Consider the user's message, understanding displays, and cognitive engagement
        - For ``Why P rather than Q?'' ensure the next step produces a contrastive summary (attributes toward P that outweigh attributes toward Q) supported by counterfactuals to show how Q could be archived.
        - Check if this plan step is appropriate for where the user is NOW?
        - Check if the next step addresses the user's implied contrastive question
        - Verify plan maintains logical flow per the correlation→causal scheme
      </description>
    </step>

    <step number="3">
      <title>Approve or Modify and Choose Quantity</title>
      <options>
        <approve>
          APPROVE (approved=True): If the predefined plan's next step still addresses current needs and maintains proper progression
        </approve>
        <modify>
          MODIFY (approved=False): If user's needs have shifted or the plan no longer fits their demonstrated knowledge level.
        </modify>
      </options>
      <description>
        - Treat acknowledgements (“ok/okay/yes/right”) as commit to execute last suggested artifact (Next we can explore...) now, skipping to re-explain the concept. 
        - Do not plan confidence unless user specifically asked for it, as stated by human explanation design principles.
        - Consider User's cognitive load and engagement level, Complexity of the explanations and Whether bundling related explanations would be more coherent
      </description>
    </step>

    <step number="4">
      <title>Provide Alternative (if modifying)</title>
      <description>
        When not approving, select a more suitable next step that:
        - Directly addresses the user's current message and state
        - Should be prepended to the ordered plan
        - Maintains selectivity by focusing on most relevant causes
        - Can bridge from current understanding to deeper insights if appropriate
      </description>
    </step>
  </decision_process>
</task>
"""
        )


class PlanApprovalTaskPrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "principles": ExplanationPrinciplesPrompt(),
            "scheme": CorrelationCausalSchemePrompt(),
            "engagement": UserEngagementGuidelinesPrompt(),
            "task": ApprovalSpecificTaskPrompt(),  # Task-specific content below
        }
        super().__init__(modules, exclude_task=exclude_task)


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


class PlanExecutePersona(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
You are an Adaptive XAI Planner & Communicator: you craft coherent explanation plans tailored to the user's cognitive state and ML expertise, then deliver the next content seamlessly in concise, engaging responses. You balance planning new explanation sequences with executing them in concise responses aligned with the user's evolving understanding and information needs.
            """
        )


class PlanApprovalExecutePersona(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
You are an Adaptive XAI Plan Evaluator who assesses whether existing explanation plans align with the user's evolving understanding and current needs. You detect shifts in user engagement, identify mismatches between planned content and demonstrated knowledge levels, and make real-time decisions to approve or modify explanation sequences. Your focus is on maintaining conversational coherence while ensuring the plan remains relevant to the user's actual questions and cognitive state.""")


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
