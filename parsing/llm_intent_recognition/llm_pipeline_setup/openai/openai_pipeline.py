import json
import re
import sys

import pandas as pd
from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.messages import AIMessage
from langchain.chains import ConversationChain
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.output_parsers import JsonOutputParser
from openai import OpenAI
from openai.types.chat.completion_create_params import ResponseFormat

from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from difflib import SequenceMatcher
import tqdm
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

from parsing.llm_intent_recognition.prompts import get_template_with_full_descriptions, \
    get_xai_template_with_descriptions, \
    possible_categories, question_to_id_mapping, \
    get_template_with_full_descr_step_by_step, get_template_wich_checklist, \
    get_tempalte_wich_checklist_and_memory, get_template_with_checklist_condensed, get_system_template_with_checklist, \
    simple_user_question_prompt, get_system_prompt_condensed, openai_system_prompt, openai_user_prompt
from parsing.llm_intent_recognition.router_chain import generate_destination_chains, generate_router_chain
from langchain_community.adapters import openai as lc_openai

load_dotenv()

### Langsmith
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

### OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_ORGANIZATION"] = os.getenv('OPENAI_ORGANIZATION_ID')

LLM_MODEL = "gpt-3.5-turbo-0125"
client = OpenAI()
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)

response_schemas = [
    ResponseSchema(name="method_name", description="name of the method to answer the user question."),
    ResponseSchema(name="feature", description="Feature that the user mentioned, can be None."),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


def extract_json_from_response(response) -> dict:
    """
    Extracts JSON object from a response string that contains both justification and JSON formatted within triple backticks.

    Parameters:
        response (str): The response string from which to extract the JSON.

    Returns:
        dict: The extracted JSON object.
    """
    # Use regular expression to find the JSON object within triple backticks
    try:
        json_obj = json.loads(response)
        return json_obj
    except json.JSONDecodeError:
        json_match = re.search(r'```(.*?)```', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                json_obj = json.loads(json_str)
                return json_obj
            except json.JSONDecodeError:
                raise ValueError("Found JSON block, but it is not valid JSON.")

    # If no JSON object is found, raise an error
    raise ValueError("No JSON block found in the response.")


def extract_tuple_from_response(response) -> tuple:
    """
    Extracts a Python tuple from a response string that contains both justification and tuple formatted within triple backticks.

    Parameters:
        response (str): The response string from which to extract the tuple.

    Returns:
        tuple: The extracted Python tuple.
    """
    # Use regular expression to find the tuple within triple backticks
    if isinstance(response, dict):
        answer = response['text']
    elif isinstance(response, AIMessage):
        answer = response.content
    else:  # str
        answer = response

    tuple_match = re.search(r'\(([^)]+)\)', answer)

    if tuple_match:
        tuple_str = tuple_match.group(1).strip()
        try:
            # Evaluate the string as a Python tuple
            tuple_obj = eval(tuple_str)
            if isinstance(tuple_obj, tuple):
                return tuple_obj
            else:
                print(answer)
                raise ValueError("Found block, but it is not a valid tuple.")
        except (SyntaxError, NameError):
            print(answer)
            raise ValueError("Found block, but it is not a valid tuple.")

    # If no tuple is found, raise an error
    raise ValueError("No tuple block found in the response.")


"""chain = (
        PromptTemplate.from_template(
            template=get_tempalte_wich_checklist(),
            partial_variables={"input": "{question}"}) | Ollama(model=LLM_MODEL,
                                                                    temperature=0) | extract_tuple_from_response
)"""


class LLMSinglePromptSystemMessage:
    def __init__(self):
        self.system_prompt = get_system_prompt_condensed()
        self.chain = (
                PromptTemplate.from_template(
                    template=simple_user_question_prompt(),
                    partial_variables={"input": "{question}"}) | llm | extract_json_from_response
        )

    def predict(self, question):
        message = self.chain.invoke({"input": f"{question}"})
        if isinstance(message, dict):
            question = message["method"]
            feature = message["feature"]
        else:
            question = message[0]
            feature = message[1]
        # get ids
        question_id = question_to_id_mapping[question]
        return question_id, feature


class LLMSinglePromptWithMemoryAndSystemMessage:
    def __init__(self, feature_names):
        self.memory = ConversationBufferMemory(memory_key="chat_history")

        self.chat_template = ChatPromptTemplate.from_messages(
            [
                openai_system_prompt(feature_names),
                openai_user_prompt(),
            ]
        )

    def process_question(self, question):
        # Retrieve chat history
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        # Directly pass question and chat_history as inputs
        formatted_messages = self.chat_template.format_messages(
            input=question, chat_history=chat_history, format_instructions=format_instructions)

        messages = [
            {"role": "system", "content": formatted_messages[0].content},
            {"role": "user", "content": formatted_messages[1].content},
        ]
        response_format = ResponseFormat(type="json_object")

        # Wrapper to use Langsmith client
        response = lc_openai.chat.completions.create(
            messages=messages, model=LLM_MODEL, temperature=0, response_format=response_format)

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

        self.memory.save_context({"question": question}, {"response": response.__str__()})

        return response

    def predict(self, user_question):
        response = self.process_question(user_question)
        try:
            mapped_question = response[0]
            feature = response[1]
        except KeyError:
            try:
                mapped_question = response["method_name"]
            except KeyError:
                mapped_question = response["method"]
            feature = response["feature"]
        question_id = question_to_id_mapping.get(mapped_question, -1)
        return question_id, feature


class LLMMultiPromptChainModel:
    def __init__(self):
        self.chain = multi_prompt_chain

    def predict(self, question):
        response = self.chain.invoke({"input": question})
        response["text"] = response["text"].replace("\n", "")
        response["text"] = response["text"].strip()
        response = json.loads(response["text"])
        method_name = response.get("method_name")
        try:
            feature = response.get("feature")
        except KeyError:
            feature = None
        question_id = question_to_id_mapping.get(method_name)
        return question_id, feature


class LLMMultiChain:
    def __init__(self):
        self.chain = None
        self.build_chain()

    def build_chain(self):
        prompt_infos, destination_chains, default_chain = generate_destination_chains()
        self.chain = generate_router_chain(prompt_infos, destination_chains, default_chain)

    def predict(self, question):
        message = self.chain.invoke({"input": f"{question}"})
        if not isinstance(message["text"], dict):
            answer_dict = json.loads(message["text"])
        else:
            answer_dict = message["text"]
        question = answer_dict["method_name"]
        feature = answer_dict["feature"]
        # get ids
        question_id = question_to_id_mapping[question]
        return question_id, feature


# Define prompt names and descriptions
prompt_names = [
    "xai_method", "whyExplanation", "notXaiMethod", "followUp", "greeting"
]

prompt_descriptions = [
    "Prompt to select the most appropriate explainability method for the user's question about the model prediction. Useful when the user seeks explanations that can be answered by XAI methods like anchors (sufficient conditions), SHAP values (feature importances), counterfactuals (alternative scenarios), or feature statistics (data analysis).",
    "Prompt to address general 'why did this happen' questions related to model predictions. Suitable for inquiries that require a straightforward explanation without delving into specific alternative scenarios or feature changes.",
    "Prompt to respond to user questions unrelated to model predictions or feature specifics, which cannot be addressed using standard XAI methods.",
    "Prompt to handle follow-up questions that reference previous interactions, especially when the user shifts focus to a different feature or aspect. Useful for maintaining context continuity.",
    "Prompt to manage greetings and general inquiries not directly tied to the model predictions or specific features, ensuring a friendly and informative interaction."
]

# Define the templates for each prompt
xai_method_template = get_xai_template_with_descriptions()

why_template = """Respond with the following json: {"method_name": "whyExplanation"}"""

greeting_template = """Respond with the following json: {"method_name": "greeting"}"""

notXaiMethod_template = """Respond with the following json: {"method_name": "notXaiMethod"}"""

followUp_template = """Respond with the following json: {"method_name": "followUp"}"""

# Collect all templates in a list
prompt_templates = [
    xai_method_template, why_template, notXaiMethod_template, followUp_template, greeting_template
]

# Create prompt_infos dict with "name" and "description" and "prompt_template" keys
prompt_infos = [
    {"name": name, "description": description, "prompt_template": template}
    for name, description, template in zip(prompt_names, prompt_descriptions, prompt_templates)
]

# Initialize the MultiPromptChain with the LLM and the prompts
multi_prompt_chain = MultiPromptChain.from_prompts(
    llm,
    prompt_infos=prompt_infos,
)


def parse_categorical(ai_message: AIMessage) -> str:
    """
    Parse the AI output to identify one of the possible categories. This includes checking for exact matches
    and very close matches that account for potential misspellings or minor variations in the text.
    """
    message_content = ai_message.content.lower()
    question = None

    for category in possible_categories:
        if category in message_content:
            question = category
        # Check for very close matches using a similarity threshold
        elif SequenceMatcher(None, category, message_content).ratio() > 0.9:
            question = category

    # get question id
    question_id = question_to_id_mapping[question]

    return question_id


def current_approach_performance(question_to_id_mapping, load_previous_results=False):
    # reverse question_to_id_mapping
    id_to_question_mapping = {v: k for k, v in question_to_id_mapping.items()}
    # load test questions csv
    if not load_previous_results:
        test_questions_df = pd.read_csv("../../llm_intent_test_set.csv", delimiter=";")

        # Keep only half of every xai method (10 questions per method)
        # test_questions_df = test_questions_df.groupby("xai method").head(10)

        # only keep certain methods
        test_questions_df = test_questions_df[test_questions_df["xai method"].isin(["followUp"])]

        # load llm model
        llm_model = LLMSinglePromptWithMemoryAndSystemMessage(
            feature_names=["feature1", "feature2", "feature3", "feature4"])

        # predict for each question
        correct = 0
        correct_features = 0
        wrong_predictions = {}
        total_predictions = {q_id: 0 for q_id in question_to_id_mapping.values()}
        wrong_features = {}
        parsing_errors = {}
        for index, row in tqdm.tqdm(test_questions_df.iterrows(), total=len(test_questions_df), desc="Testing"):
            correct_q_id = question_to_id_mapping[row["xai method"]]
            correct_feature = row["feature"]
            question = row["question"]
            total_predictions[correct_q_id] += 1

            try:
                predicted_q_id, feature = llm_model.predict(question)
            except Exception as e:
                print(f"Error for question {question}: {e}")
                try:
                    parsing_errors[correct_q_id] += 1
                except KeyError:
                    parsing_errors[correct_q_id] = 1
                continue
                # Check if method is correct
            if predicted_q_id == correct_q_id:
                correct += 1
            else:
                if correct_q_id not in wrong_predictions:
                    wrong_predictions[correct_q_id] = []
                wrong_predictions[correct_q_id].append(predicted_q_id)
                print(f"Question: {question}", f"Correct: {id_to_question_mapping[correct_q_id]}",
                      f"Predicted: {id_to_question_mapping[predicted_q_id]}")
                # Check if feature is correct if applicable
            if correct_feature is not None:
                if feature == correct_feature:
                    correct_features += 1
                else:
                    wrong_features[correct_feature] = feature

        # Print performance summary.
        print(f"Correct predictions: {correct}/{len(test_questions_df)}, Total questions: {len(test_questions_df)}")
        print(f"Wrong predictions: {wrong_predictions}")
        print(f"Wrong features: {wrong_features} of total {len(wrong_features) + correct_features}")

        # Save predictions to a file for further analysis.
        def save_results(filepath, correct_predictions, incorrect_predictions, total_predictions,
                         question_to_id_mapping,
                         id_to_question_mapping):
            results = {
                "correct_predictions": correct_predictions,
                "incorrect_predictions": incorrect_predictions,
                "total_predictions": total_predictions,
                "question_to_id_mapping": question_to_id_mapping,
                "id_to_question_mapping": id_to_question_mapping,
                "parsing_errors": parsing_errors
            }

            with open(filepath, 'w') as f:
                json.dump(results, f)

        save_results("llm_classification_results.json", wrong_predictions, wrong_features, total_predictions,
                     question_to_id_mapping, id_to_question_mapping)
    else:
        with open("llm_classification_results.json", 'r') as f:
            results = json.load(f)

        wrong_predictions = results["correct_predictions"]
        wrong_features = results["incorrect_predictions"]
        total_predictions = results["total_predictions"]
        question_to_id_mapping = results["question_to_id_mapping"]
        parsing_errors = results["parsing_errors"]

    # Prepare data for plotting
    question_names = [id_to_question_mapping[q_id] for q_id in question_to_id_mapping.values()]
    correct_predictions = []
    incorrect_predictions = []
    parsing_errors_list = []

    for q_id in id_to_question_mapping.keys():
        q_id = str(q_id)
        correct_count = total_predictions[q_id]
        incorrect_count = 0
        parsing_error_count = 0

        try:
            incorrect_count = len(wrong_predictions[q_id])
        except (KeyError, TypeError):
            pass

        try:
            parsing_error_count = parsing_errors[q_id]
        except (KeyError, TypeError):
            pass

        incorrect_predictions.append(incorrect_count)
        parsing_errors_list.append(parsing_error_count)
        correct_predictions.append(correct_count - incorrect_count - parsing_error_count)

    # Plotting the stacked bar plot
    fig, ax = plt.subplots()
    bar_width = 0.5

    ax.bar(question_names, correct_predictions, bar_width, label='Correct predictions', color='g')
    ax.bar(question_names, incorrect_predictions, bar_width, bottom=correct_predictions,
           label='Incorrect predictions', color='r')
    ax.bar(question_names, parsing_errors_list, bar_width,
           bottom=[i + j for i, j in zip(correct_predictions, incorrect_predictions)],
           label='Parsing errors', color='b')

    ax.set_xlabel('Questions')
    ax.set_ylabel('Proportion')
    ax.set_title('Proportion of Correct, Incorrect Predictions, and Parsing Errors per Question')
    ax.legend()

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    current_approach_performance(question_to_id_mapping, load_previous_results=False)
    """feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5"]
    llm_model = LLMSinglePromptWithMemory(feature_names)

    while True:
        question = input("Please enter your question (or type 'exit' to quit): ")

        if question.lower() == 'exit':
            break

        question_text, question_id, feature = llm_model.predict(question)
        print(f"Question ID: {question_text}, Feature: {feature}")

    sys.exit()"""
