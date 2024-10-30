from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from parsing.llm_intent_recognition.llm_pipeline_setup.openai_pipeline.openai_pipeline import LLMBase


class PlanLLM(LLMBase):
    def __init__(self):
        super().__init__()
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt()),
                HumanMessagePromptTemplate.from_template("")
            ]
        )

    def system_prompt(self):
        return """
You are an AI assistant responsible for planning the next actions in a dialogue system.
"""

    def plan_next_action(self, method, feature, understood_concepts, misconceptions, xai_knowledge, last_explanation):
        prompt = f"""
The user has asked for the method: "{method}" and feature: "{feature}".
They currently understand: {understood_concepts}.
They may have misconceptions about: {misconceptions}.
Their xAI knowledge level is: {xai_knowledge}.
The last explanation provided was: "{last_explanation}".

Based on this information, decide whether to:
- Provide the requested explanation.
- Clarify any misconceptions.
- Ask a question to gauge understanding.

Provide your decision and reasoning in JSON format:
{{
    "action": "...",
    "reasoning": "...",
    "method": "...",
    "feature": "..."
}}
"""
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt}
        ]
        return self.llm_call(messages)
