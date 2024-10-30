from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from parsing.llm_intent_recognition.llm_pipeline_setup.openai_pipeline.openai_pipeline import LLMBase


def get_monitor_prompt(feature_names):
    return f"""
<<Context>>:
The user was presented with datapoint from a machine learning dataset with various features. The model predicted a class. Based on the 
user's question about the prediction, estimate the user's expertise level and understanding of machine learning. The possible feature names that the user might ask about are: {feature_names}\n
Given the users words, infer their understanding, misunderstandings, and expertise level. Are there obvious misconceptions or concepts that the user understands?

<<Task>>:

1. Think step by step and identify any concepts the user understands or misunderstands, as well as misconceptions.
2. Determine if the user seems like an expert or a novice:
   - Experts know how to ask for specific explanations and use the right terms.
   - Novices may not know how to phrase their questions correctly and may ask for general explanations.

<<Previous User Inputs and Assistant Responses>>:\n
{{chat_history}}

<<Current User Input>>:\n
"{{user_input}}"

<<Format Instructions>>:\n
Return a single JSON object with the following keys:

"thoughts": "...",          // Step-by-step thoughts about the user's understanding.\n
"xAIknowledge": "...",      // "Expert" or "Novice", based on the user's input.\n
"concepts_learned": ["..."],// List of concepts the user understands.\n
"misconceptions": ["..."]   // List of concepts the user misunderstands.\n
"""


class MonitorLLM(LLMBase):
    def __init__(self, feature_names):
        super().__init__()
        self.feature_names = feature_names
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(get_monitor_prompt(self.feature_names)),
                HumanMessagePromptTemplate.from_template("")
            ]
        )

    def monitor_user_input(self, user_input, chat_history):
        # Format the chat history
        formatted_chat_history = self.format_chat_history(chat_history)
        formatted_messages = self.prompt_template.format_messages(
            user_input=user_input,
            chat_history=formatted_chat_history
        )
        messages = [
            {"role": "system", "content": formatted_messages[0].content},
            {"role": "user", "content": ""}
        ]
        return self.llm_call(messages)

    def format_chat_history(self, chat_history):
        # Convert chat history to a formatted string
        history_str = ""
        for entry in chat_history:
            user_input = entry.get('user_input', '')
            assistant_response = entry.get('assistant_response', '')
            history_str += f"User: {user_input}\nAssistant: {assistant_response}\n"
        return history_str
