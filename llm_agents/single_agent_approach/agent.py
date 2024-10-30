import json
import os

from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_community.adapters import openai as lc_openai
from langchain_core.prompts import ChatPromptTemplate

from llm_agents.base_agent import XAIBaseAgent
from llm_agents.output_parsers import get_response_output_parser
from llm_agents.xai_utils import process_xai_explanations, extract_instance_information
from parsing.llm_intent_recognition.prompts.explanations_prompt_clean import openai_user_prompt

LLM_MODEL = os.getenv('OPENAI_MODEL_NAME')
response_output_parser = get_response_output_parser()
response_format_instructions = response_output_parser.get_format_instructions()


def get_system_message_prompt_template(feature_names=""):
    return f"""
    <<Context>>:\n
    You are an AI assistant providing explanations about a machine learning model's decision in a dialogue.
    The user may ask about previous predictions, but you can only refer to the current datapoint. Focus on
    xAI explanations to respond, as appropriate.\n 
    The application domain is {{domain_description}}.\n
    The machine learning model uses the following features: {feature_names}.\n
    The user sees the following instance: {{instance}}.\n
    And the model predicted {{predicted_class_name}}.\n
    
    <<Explainable AI Explanations about the current prediction>>:\n
    You have access to multiple xAI explanations (e.g., feature importances, counterfactuals, anchors, feature statistics, ceteris paribus). Choose
    the most relevant explanation(s) to address the user’s question:\n
    {{xai_explanations}}\n
    
    Based on the user's question, decide which explanation to provide and how to respond. Reason over what the user
    is asking and how to best answer their question.\n
    
    <<Chat History>>:\n
    Consider the chat history for the current question, if relevant:\n
    {{chat_history}}\n
    
    <<Response Instructions>>:\n
    {{format_instructions}}
    """


class XAIAgent(XAIBaseAgent):

    def __init__(self,
                 feature_names,
                 domain_description):
        """
        Initialize the agent with xAI explanations.

        """
        self.xai_explanations = None
        self.predicted_class_name = None
        self.instance = None
        self.feature_names = feature_names
        self.domain_description = domain_description

        # Initialize the LLM predictor (you can choose the model you prefer)
        self.llm_predictor = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

        self.memory = ConversationBufferMemory(memory_key="chat_history")

        self.dialogue_prompt_chat_tamplate = ChatPromptTemplate.from_messages(
            [
                ("system", get_system_message_prompt_template(feature_names)),
                openai_user_prompt(),
            ]
        )

    def initialize_new_datapoint(self, instance_information, xai_explanations, predicted_class_name):
        self.xai_explanations = process_xai_explanations(xai_explanations)
        self.instance = extract_instance_information(instance_information)
        self.memory = ConversationBufferMemory(memory_key="chat_history")  # Reset chat history
        self.predicted_class_name = predicted_class_name

    def call_user_question_function(self, question):
        # Retrieve chat history
        chat_history = self.memory.load_memory_variables({})["chat_history"]

        # Directly pass question and chat_history as inputs
        formatted_messages = self.dialogue_prompt_chat_tamplate.format_messages(
            input=question,
            domain_description=self.domain_description,
            predicted_class_name=self.predicted_class_name,
            instance=self.instance,
            xai_explanations=self.xai_explanations,
            chat_history=chat_history,
            format_instructions=response_format_instructions)

        messages = [
            {"role": "system", "content": formatted_messages[0].content},
            {"role": "user", "content": formatted_messages[1].content},
        ]
        # response_format = ResponseFormat(type="json_object")

        # Wrapper to use Langsmith client
        response = lc_openai.chat.completions.create(
            messages=messages, model=LLM_MODEL, temperature=0)

        # Print reasoning
        print(formatted_messages[0].content)
        print(response.choices[0].message['content'])

        """response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            messages=messages,
            response_format=response_format,
        )"""

        try:
            response = response.choices[0].message['content']
        # Exception if not a dict
        except KeyError:
            response = response.choices[0].message.content

        response = json.loads(response)

        self.memory.save_context({"question": question}, {"response": response['response'].__str__()})

        return response

    def answer_user_question(self, user_question):
        """
        Answer user question given the xAI explanations.
        """
        response = self.call_user_question_function(user_question)
        try:
            reasoning = response[0]
            response = response[1]
        except KeyError:
            reasoning = response["reasoning"]
            response = response["response"]
        return reasoning, response
