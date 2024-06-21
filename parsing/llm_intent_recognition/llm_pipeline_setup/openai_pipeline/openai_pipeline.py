import json

import pandas as pd

from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.adapters import openai as lc_openai

from openai import OpenAI
from openai.types.chat.completion_create_params import ResponseFormat

import tqdm
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

from parsing.llm_intent_recognition.prompts import question_to_id_mapping, openai_system_prompt, openai_user_prompt

load_dotenv()

### Langsmith
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

### OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_ORGANIZATION"] = os.getenv('OPENAI_ORGANIZATION_ID')

LLM_MODEL = os.getenv('OPENAI_MODEL_NAME')
client = OpenAI()
# llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)

response_schemas = [
    ResponseSchema(name="method_name", description="name of the method to answer the user question."),
    ResponseSchema(name="feature", description="Feature that the user mentioned, can be None."),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


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


def current_approach_performance(question_to_id_mapping, load_previous_results=False):
    # reverse question_to_id_mapping
    id_to_question_mapping = {v: k for k, v in question_to_id_mapping.items()}
    # load test questions csv
    if not load_previous_results:
        test_questions_df = pd.read_csv("../../llm_intent_test_set.csv", delimiter=";")

        # Keep only half of every xai method (10 questions per method)
        # test_questions_df = test_questions_df.groupby("xai method").head(10)

        # only keep certain methods
        # test_questions_df = test_questions_df[test_questions_df["xai method"].isin(["followUp"])]

        # load llm model
        # llm_model = LLMSinglePromptWithMemoryAndSystemMessage(
        # feature_names=["feature1", "feature2", "feature3", "feature4"])

        # predict for each question
        correct = 0
        correct_features = 0
        wrong_predictions = {}
        total_predictions = {q_id: 0 for q_id in question_to_id_mapping.values()}
        wrong_features = {}
        parsing_errors = {}
        for index, row in tqdm.tqdm(test_questions_df.iterrows(), total=len(test_questions_df), desc="Testing"):
            # Create a new model for each question to avoid memory issues
            llm_model = LLMSinglePromptWithMemoryAndSystemMessage(
                feature_names=["feature1", "feature2", "feature3", "feature4"])
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

        save_results("openai_classification_results.json", wrong_predictions, wrong_features, total_predictions,
                     question_to_id_mapping, id_to_question_mapping)
    else:
        with open("openai_classification_results.json", 'r') as f:
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
    current_approach_performance(question_to_id_mapping, load_previous_results=True)
    """feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5"]
    llm_model = LLMSinglePromptWithMemory(feature_names)

    while True:
        question = input("Please enter your question (or type 'exit' to quit): ")

        if question.lower() == 'exit':
            break

        question_text, question_id, feature = llm_model.predict(question)
        print(f"Question ID: {question_text}, Feature: {feature}")

    sys.exit()"""