from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from parsing.llm_intent_recognition.llm_pipeline_setup.openai_pipeline.openai_pipeline import LLMBase


def get_analyze_prompt(feature_names=""):
    return f"""
<<Context>>:

The user was presented with a data point from a machine learning dataset with various features. The model predicted a class.

The possible feature names that the user might ask about are: {feature_names}

<<Methods>>:

[Include the methods checklist as in your previous prompt, listing all possible methods and how to map user inputs to them.]

<<Task>>:

1. Based on the user's input, decide which method fits best by reasoning over every possible method.
2. Summarize the user's intent.

<<Previous User Inputs and Assistant Responses>>:
{{chat_history}}

<<Knowledge base of what the user knows>>:
{{knowledge_base}}

<<Current User Input>>:
"{{user_input}}"

<<Format Instructions>>:

Return a single JSON object with the following keys:

- "classification": "...",  // The user's intent or classification from the methods above.
- "method": "...",          // The method name corresponding to the user's intent.
- "feature": "...",         // The relevant feature, if any.
- "reasoning": "..."        // Your reasoning for choosing this method.
"""


class AnalyzeLLM(LLMBase):
    def __init__(self, feature_names):
        super().__init__()
        self.feature_names = feature_names
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(get_analyze_prompt(feature_names)),
                HumanMessagePromptTemplate.from_template("")
            ]
        )

    def analyze_user_input(self, user_input, chat_history, knowledge_base_as_text):
        # Format the chat history
        formatted_chat_history = self.format_chat_history(chat_history)
        formatted_messages = self.prompt_template.format_messages(
            user_input=user_input,
            chat_history=formatted_chat_history,
            knowledge_base=knowledge_base_as_text
        )
        messages = [
            {"role": "system", "content": formatted_messages[0].content},
            {"role": "user", "content": ""}
        ]
        return self.llm_call(messages)

    def format_chat_history(self, chat_history):
        # Same as in MonitorLLM
        history_str = ""
        for entry in chat_history:
            user_input = entry.get('user_input', '')
            assistant_response = entry.get('assistant_response', '')
            history_str += f"User: {user_input}\nAssistant: {assistant_response}\n"
        return history_str
