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
                if name == "task" or name.endswith("_task") or name == "user_message":
                    continue  # Skip tasks AND user_message
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
        
        # append user_message modules LAST (after tasks)
        for name, mod in self._modules.items():
            if name == "user_message":
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
  <domain>
    {domain_description}
  </domain>
  <attributes>
    {feature_context}
  </attributes>
  <current_instance>
    {instance}
  </current_instance>
  <model_prediction>
    {predicted_class_name}
  </model_prediction>
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
{chat_history}
"""
        )


class UserMessagePrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<current_user_message>
    {user_message}
</current_user_message>
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
        <system_role>
            You are an analyst that interprets user messages to identify users understanding and cognitive engagement based on the provided chat and his recent message. The user is curious about an AI models prediction and is presented with explanations via explainable AI tools. Your task is to analyze the user's latest message in the context of the conversation history and define the class of cognitive engagement and understanding displays as defined below.
        </system_role>"""
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
            "last_shown": LastShownExpPrompt(),
            "history": HistoryPrompt(),
            "user_message": UserMessagePrompt(),
            "task": AnalyzeTaskPrompt(),  # Always in the end
        }
        super().__init__(modules, exclude_task=exclude_task)


class AnalyzeAgentSystemPrompt(CompositePromptMixin):
    def __init__(self):
        agent_instructions = """
        <system_role>
            You are an analyst that interprets user messages to identify users understanding based on the provided chat and his recent message. The user is curious about an AI models prediction and is presented with explanations via explainable AI tools. Your task is to analyze the user's latest message in the context of the conversation history and suggest updates to the user model.
        </system_role>
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
<system_role>
    You are a Cognitive Engagement Analyst & Understanding Tracker: an expert in monitoring how users engage with AI explanations and how their understanding evolves. You interpret subtle communication signals to assess both real-time cognitive engagement and deeper comprehension, distinguishing surface acknowledgment from genuine understanding. Your focus includes classifying engagement modes and tracking knowledge progression over time. Guided by principles of evidence-based assessment, contextual interpretation, and dynamic model updating, you ensure that user models reflect both current engagement and long-term learning trajectories.
</system_role>"""
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
        - Prioritize diverse, informative explanations that clarify the model's decision.
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
    Do not endorse causal/stigmatizing claims if not supported by explanation methods; clarify that attributions reflect correlations in data, not necessarily ability or causation in real world. If encountered, add a one-sentence caution, then proceed with model-scope facts.
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
  
  Follow response_generation_guidelines and emit rendered_step_names accordingly.
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


class ResponseGenerationGuidelinesPrompt(SimplePromptMixin):
    """Shared response generation guidelines used by both Plan Execute and Plan Approval tasks"""
    
    def __init__(self):
        super().__init__(
            """
<response_generation_guidelines>
  <response_generation>
    - Generate a concise, helpful response (3-4 sentences) using selected explanations
    - Do not meta-comment on your process; if user AGREED, start directly with promised content
    - Make response self-contained as you have exactly one message per user question
  </response_generation>
  
  <fidelity_principles>
    - Report only what XAI outputs show without inventing causal stories about why patterns exist
    - Translate technical concepts faithfully: FeatureImportance="how much each attribute influences", Counterfactual="what would need to change", Anchor="conditions that guarantee this prediction"
    - If user asks for causal interpretation, clarify this is interpretation beyond model scope that might be wrong
  </fidelity_principles>
  
  <content_alignment>
    - Always state scope explicitly. Preferred openers: 'For this person, …' or 'Overall, …'
    - When listing top features, show top-3 only, then say 'others smaller' to stay within 3-4 sentences
    - Do not repeat explanations already shown in conversation history
    - When user agrees to see suggestion, show it immediately without narration
  </content_alignment>
  
  <tone_and_formatting>
    - Match user's ML expertise level: use plain language not mentioning the method names for lay users but rather what they explain, avoid technical jargon unless user shows ML proficiency and uses the terms himself
    - ALWAYS use HTML formatting: <b> for bold, <ul>/<li> for lists, <p> for paragraphs
    - Include visual placeholders like ##FeatureInfluencesPlot## at end after giving one insight sentence.
  </tone_and_formatting>
  
  <engagement>
    - Focus on WHAT the model learned, not WHY society works that way and be honest about limits of what XAI methods reveal
    - Guide users naturally to important next insights: Always include at least one next-step suggestion from the plan (or even two options, letting the user decide) to help users discover key aspects they may not know to ask about
    - Keep the conversation fluent based on the chat history and current user message. Connect to the most recent user message and introduce the next explanation step naturally, showing why it's valuable for understanding the prediction
  </engagement>
</response_generation_guidelines>
"""
        )


class RenderedStepNamesPrompt(SimplePromptMixin):
    """Shared rendered step names instructions for tracking plan updates"""
    
    def __init__(self):
        super().__init__(
            """
<rendered_step_names>
  ALWAYS include the rendered_step_names field in your JSON response, specifying exactly which explanations and steps were rendered. 
  - If you showed specific XAI explanations (features, counterfactuals, etc.), list them
  - If you provided clarification, asked questions, or went off-topic, use empty list []
  - If you gave conceptual/scaffolding explanation, specify which explanation concept was introduced
  This field is required for proper tracking and plan updates.
</rendered_step_names>
"""
        )


class PlanExecuteInvariantsPrompt(SimplePromptMixin):
    """Shared invariants for plan execute and approval tasks"""
    
    def __init__(self):
        super().__init__(
            """
<invariants>
  <rule>Never plan or repeat steps already delivered in this thread (derive from history).</rule>
  <rule>If the previous reply said "Next, we can …", the very next action must deliver that artifact (no concept-only explanation).</rule>
  <rule>Treat acknowledgements ("ok/okay/yes/right") as approval to execute the last suggested artifact</rule>
  <rule>Artifacts = plots, counterfactual lists, top-k attributes, numeric outputs. Concepts = ≤1 sentence inline context only.</rule>
</invariants>
"""
        )


class NewExplanationCreationPrompt(SimplePromptMixin):
    """Shared instructions for creating new explanations when needed"""
    
    def __init__(self):
        super().__init__(
            """
<new_explanation_creation>
  <evaluation>
    - First, evaluate whether the user's message expresses a clear information need
    - Check whether this need can be satisfied using any existing explanations
    - If no suitable explanation is available, define a new one tailored to the identified need
    - If the user's need is unclear, scaffold briefly using techniques from the explanation collection
  </evaluation>
  
  <creation_guidelines>
    - Only create new explanations when existing ones cannot address the user's specific need
    - Ensure new explanations complement rather than duplicate existing explanation methods
    - Design new explanations to be reusable for similar future questions
    - Focus on the user's demonstrated level of understanding and information gaps
  </creation_guidelines>
</new_explanation_creation>
"""
        )


class ExplanationPlanningGuidelinesPrompt(SimplePromptMixin):
    """Shared guidelines for explanation planning and sequencing"""
    
    def __init__(self):
        super().__init__(
            """
<explanation_planning_guidelines>
  <diversity_guidelines>
    - **Start with foundation**: Begin with FeatureImportances to show which attributes influence the prediction and how
    - **Add contrastive perspective**: Include Counterfactuals to answer "what would need to change for different prediction"
    - **Provide decision boundaries**: Use AnchorExplanation to show minimal conditions that guarantee this prediction
    - **Include local vs global context**: Combine CeterisParibus (this instance) with TextualPartialDependence (general patterns) when relevant
    - **Add contextual understanding**: Include FeatureStatistics when instance values are unusual or need dataset context
    - **End with engagement**: Consider PossibleClarifications for common questions or deeper exploration opportunities
    - **Avoid confidence** unless user specifically requested it
  </diversity_guidelines>
  
  <human_explanation_principles>
    - **Contrastive focus**: Ensure plan answers "Why this prediction rather than the alternative?" through complementary explanation types
    - **Selective relevance**: Choose explanations most informative for this specific instance (e.g., if features have extreme values, include statistics)
    - **Social adaptation**: Order explanations for logical progression from basic understanding to deeper insights
    - **Conversational flow**: Plan should build understanding step-by-step, with each explanation preparing ground for the next
  </human_explanation_principles>
  
  <logical_progression>
    - Follow correlation→causation flow: Start with what influences prediction, then show how changes would affect it
    - Build from individual to general: Local explanations first, then global context when needed
    - Progress from concrete to abstract: Feature influences → counterfactual scenarios → decision rules → broader patterns
    - Maintain coherent narrative: Each step should connect naturally to previous understanding
  </logical_progression>
  
  <instance_specific_selection>
    - Prioritize explanations that are most informative for the current instance's feature values and prediction
    - If instance has unusual feature values, include FeatureStatistics for context
    - If prediction is close to decision boundary, emphasize Counterfactuals and Anchors
    - If features show strong patterns, connect CeterisParibus with TextualPartialDependence
    - Select step_names within explanations that directly address the instance's characteristics
  </instance_specific_selection>
</explanation_planning_guidelines>
"""
        )


class ExecuteStepPrompt(SimplePromptMixin):
    """Shared execute response step for both plan creation and approval tasks"""
    
    def __init__(self):
        super().__init__(
            """
<execute_step>
  <title>Execute Response</title>
  <instructions>Generate response using selected explanations from the plan, applying response_generation_guidelines and including rendered_step_names.</instructions>
</execute_step>
"""
        )


class PlanPrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "explanation_plan": PreviousPlanPrompt(),
            "last_shown_explanations": LastShownExpPrompt(),
            "user_model": UserModelPrompt(),
            "history": HistoryPrompt(),
            "user_message": UserMessagePrompt(),
            "task": PlanTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


class ExecutePersona(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<system_role>
    You are an Adaptive XAI Communicator: an expert in delivering user-tailored explanations based on the user's cognitive state and ML knowledge. You craft concise, engaging responses that align with the user's understanding and the current explanation plan. Your focus is on clarity, relevance, and maintaining conversational flow while adapting dynamically to user feedback and questions.
</system_role>
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
    """Restructured prompt with clear hierarchy for initial Plan + Execute phases."""

    def __init__(self, exclude_task: bool = False):
        modules = {
            "persona": PlanExecutePersona(),
            "behavioral_guidelines": BehavioralGuidelinesPrompt(),
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "conversation_state": ConversationStatePrompt(),
            "task": SimplifiedPlanExecuteTaskPrompt(),
            "user_message": UserMessagePrompt(),
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
        <system_role>
            You are an analyst that interprets user messages to identify users' understanding and cognitive engagement based on the provided chat and recent message. The user is curious about an AI model's prediction and is presented with explanations via explainable AI tools. Your task is to analyze the user's latest message in the context of the conversation history, define the class of cognitive engagement and understanding displays, and suggest updates to the user model as appropriate.
        </system_role>
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
        <system_role>
            You are a planner and communicator that designs explanation plans and generates explanations about AI model predictions for users. The user is curious about an AI model's prediction and is presented with explanations via explainable AI tools. Your task is to analyze the user's latest message in the context of the conversation history, user model, and previous explanations to create a logical explanation plan tailored to the user's needs and understanding level, and then craft a natural, engaging response based on the next explanation content that has been planned.
        </system_role>
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
        - Determine scope: for this individual vs in general. Determine target/polarity (e.g., "supporting class A" vs "supporting class B") consistent with the user's last request/AGREEMENT.
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
        - Treat acknowledgements ("ok/okay/yes/right") as commit to execute last suggested artifact (Next we can explore...) now, skipping to re-explain the concept. 
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
<system_role>
    You are the best Co-Constructive XAI Learning Facilitator—an adaptive explainer who identifies user understanding gaps and tailors explanation plans accordingly, ensuring foundational concepts are addressed before introducing complex ideas. You specialize in scaffolded, collaborative learning where explanations evolve dynamically based on user comprehension, avoiding overload while guiding users through AI model predictions in digestible, logically sequenced steps.
</system_role>
""")


class PlanApprovalPrompt(CompositePromptMixin):
    def __init__(self, exclude_task: bool = False):
        modules = {
            "persona": PlanApprovalPersona(),
            "principles": ExplanationPrinciplesPrompt(),
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "user_model": UserModelPrompt(),
            "history": HistoryPrompt(),
            "explanation_plan": PreviousPlanPrompt(),
            "last_shown": LastShownExpPrompt(),
            "user_message": UserMessagePrompt(),
            "task": PlanApprovalTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


# --- New Structured Prompt Classes ---

class BehavioralGuidelinesPrompt(SimplePromptMixin):
    """Groups all behavioral guidelines together"""
    def __init__(self):
        super().__init__(
            """
<behavioral_guidelines>
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

  <scheme>
    1. Look at high-level correlations like feature importances to identify candidate attributes to investigate further.
    2. Check if these attributes appear in counterfactuals or other explanations to infer causal relationships.
    Example: Attribute A is most important and switching only it would lead to a different prediction. Attribute B is second important, it has a rare value and is often among the attributes to change for a different model prediction.
  </scheme>

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
</behavioral_guidelines>
"""
        )


class ConversationStatePrompt(SimplePromptMixin):
    """Groups all conversation state information together"""
    def __init__(self):
        super().__init__(
            """
<conversation_state>
  {chat_history}
  
  <user_model>
    <description>This is the user model, indicating the user's cognitive state and machine learning models as well as which explanations were understood, not yet explained, or currently being shown. Consider it to be the best explainer possible and adapt to the user</description>
    {user_model}
  </user_model>
  
  <previous_plan>
    <description>This is the previous explanation plan ordered from top to bottom that was established for the user before the user's current message</description>
    <content>{explanation_plan}</content>
  </previous_plan>
  
  <last_shown_explanations>
    {last_shown_explanations}
  </last_shown_explanations>
</conversation_state>
"""
        )


class SimplifiedPlanApprovalTaskPrompt(CompositePromptMixin):
    """Plan approval task focused only on approval logic, using shared execution components"""
    def __init__(self):
        modules = {
            "invariants": PlanExecuteInvariantsPrompt(),
            "new_explanation_creation": NewExplanationCreationPrompt(),
            "planning_guidelines": ExplanationPlanningGuidelinesPrompt(),
            "execute_step": ExecuteStepPrompt(),
            "task_specific": SimplePromptMixin(
            """
<task>
  <objective>Approve or Modify, then Execute</objective>
  <description>Evaluate the existing predefined plan against the user's current message and needs. Either approve the plan's next step as-is, or modify by selecting a different step or creating individual new explanations. Then execute the response.</description>
  
  <decision_process>
    <step number="1">
      <title>Evaluate Plan Relevance and Approve or Modify</title>
      <description>
        - Lock scope and polarity: Determine scope (individual vs general) and target/polarity consistent with the user's last request/AGREEMENT
        - Tie-breaker rule: If prior turn offered multiple artifacts ("anchor or counterfactuals") and user says "Okay", prefer the first mentioned option
        - Choose the next explanation; prefer an artifact for "what-if/compare/show" intents. Do not repeat visuals already shown
        - If the user asks about a single attribute/value, prefer a local single-feature explanation with a signed magnitude and brief foil comparison
        - For "Why P rather than Q?" ensure the next step produces a contrastive summary (attributes toward P that outweigh attributes toward Q) supported by counterfactuals
        - Confidence gating: Only include confidence/certainty explanations when user explicitly asks about model confidence or certainty
        - Check if the predefined plan's next step is appropriate for where the user is NOW and addresses their implied contrastive question
        - Consider user's cognitive load and engagement level, complexity of explanations and whether bundling related explanations would be more coherent
        - When modifying plans, can create individual new explanations but do not create comprehensive new plans (that's the plan creation workflow's role)
      </description>
      <options>
        <approve>APPROVE (approved=True): If the predefined plan's next step still addresses current needs and maintains proper progression</approve>
        <modify>MODIFY (approved=False): If user's needs have shifted or the plan no longer fits their demonstrated knowledge level. Use new_explanations if needed and set next_response accordingly.</modify>
      </options>
    </step>

    <step number="2">
      Apply the execute_step instructions.
    </step>
  </decision_process>
</task>
"""),
            "response_guidelines": ResponseGenerationGuidelinesPrompt(),
            "rendered_steps": RenderedStepNamesPrompt(),
        }
        super().__init__(modules)


class SimplifiedPlanExecuteTaskPrompt(CompositePromptMixin):
    """Plan creation task focused only on plan creation logic, using shared execution components"""
    def __init__(self):
        modules = {
            "invariants": PlanExecuteInvariantsPrompt(),
            "new_explanation_creation": NewExplanationCreationPrompt(),
            "planning_guidelines": ExplanationPlanningGuidelinesPrompt(),
            "execute_step": ExecuteStepPrompt(),
            "task_specific": SimplePromptMixin(
            """
<task>
  <objective>Plan Creation and Execute</objective>
  <description>Create a new explanation plan and execute the first response based on the user's information needs.</description>

  <planning_steps>
    <step number="1">
      <title>Construct Explanation Plan</title>
      <instructions>
        <plan_requirements>
          - Create a comprehensive 6-7 step plan that provides diverse perspectives on the model's decision for this specific instance
          - Assume the user may ask only a few questions, so prioritize the most informative and complementary explanations
          - The first item MUST be executed in the very next response once AGREEMENT is detected
          - Emit the *full* ordered list of upcoming step_names for each explanation
        </plan_requirements>
        
        Apply the explanation_planning_guidelines for diversity, human principles, logical progression, and instance-specific selection.
      </instructions>
    </step>
    
    <step number="2">
      Apply the execute_step instructions.
    </step>
  </planning_steps>
</task>
"""),
            "response_guidelines": ResponseGenerationGuidelinesPrompt(),
            "rendered_steps": RenderedStepNamesPrompt(),
        }
        super().__init__(modules)


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
<system_role>
    You are an Adaptive XAI Planner and Communicator: you craft coherent explanation plans tailored to the user's cognitive state and ML expertise, then deliver the next content seamlessly in concise, engaging responses. You balance planning new explanation sequences with executing them in concise responses aligned with the user's evolving understanding and information needs.
</system_role>
            """
        )


class PlanApprovalExecutePersona(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<system_role>
    You are an Adaptive XAI Evaluator & Communicator: you assess and adapt existing explanation plans based on the user's evolving cognitive state and ML expertise, then deliver the next content seamlessly in concise, engaging responses. You balance evaluating plan relevance with modifying explanation sequences to match the user's current understanding and information needs.
</system_role>""")


class PlanApprovalExecutePrompt(CompositePromptMixin):
    """Restructured prompt with clear hierarchy for Plan Approval + Execute phases."""

    def __init__(self, exclude_task: bool = False):
        modules = {
            "persona": PlanApprovalExecutePersona(),
            "behavioral_guidelines": BehavioralGuidelinesPrompt(),
            "context": ContextPrompt(),
            "collection": ExplanationCollectionPrompt(),
            "conversation_state": ConversationStatePrompt(),
            "task": SimplifiedPlanApprovalTaskPrompt(),
            "user_message": UserMessagePrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)


# --- Conversational Prompt for simplified agent ---

class ConversationalPersona(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<system_role>
    You are a Conversational XAI Assistant: an expert in explaining machine learning model predictions through natural dialogue. You maintain conversation context while providing clear, relevant explanations tailored to user questions. You select appropriate explanations from available XAI methods and present them in an engaging, conversational manner using HTML formatting for clarity.
</system_role>"""
        )


class AvailableExplanationsPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<available_explanations>
    {available_explanations}
</available_explanations>
"""
        )


class ConversationalTaskPrompt(SimplePromptMixin):
    def __init__(self):
        super().__init__(
            """
<task>
  <objective>Generate a concise response (3–4 sentences) based on the conversation history and current user question. You have exactly one message per user question so make it self-contained.</objective>
  Your responses should accurately convey XAI method outputs without adding speculation about real-world causations beyond what the data tells us. Stay faithful to what the methods actually reveal, unless the user explicitly asks for a broader interpretation.
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
    Do not endorse causal/stigmatizing claims if not supported by explanation methods; clarify that attributions reflect correlations in data, not necessarily ability or causation in real world. If encountered, add a one-sentence caution, then proceed with model-scope facts.
  </fidelity_principles>
  <instructions>
    <step>Analyze the user's question in the context of the conversation history</step>
    <step>Select the most appropriate explanations from the available XAI methods to address their question</step>
    <step>Provide a clear, conversational response using the selected explanations and only use the names of explanations if user uses them too. Mirror the language and keep it rather simple unless the user seems proficient in data science or machine leanring.</step>
    <step>Use HTML formatting (like <b>, <i>, <ul>, <li>) to enhance readability</step>
    <step>Maintain conversational flow while being informative and accurate</step>
    <step>Be proactive and suggest follow up explanations to explore</step>
  </instructions>
</task>
"""
        )


class ConversationalPrompt(CompositePromptMixin):
    """Simplified prompt for conversational agent with single unified step."""
    
    def __init__(self, exclude_task: bool = False):
        modules = {
            "persona": ConversationalPersona(),
            "context": ContextPrompt(),
            "available_explanations": AvailableExplanationsPrompt(), 
            "history": HistoryPrompt(),
            "user_message": UserMessagePrompt(),
            "task": ConversationalTaskPrompt(),
        }
        super().__init__(modules, exclude_task=exclude_task)
