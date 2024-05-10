import json

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.messages import AIMessage

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from difflib import SequenceMatcher
from langchain.memory import ConversationBufferMemory
import os

import os
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

LLM_MODEL = "llama3"

general_questions = [
    "anchor",
    "shap_all_features",
    "shap_top_3_features",
    "shap_least_3_features",
    "counterfactual_any_change"]

feature_specific_q = [
    "ceteris_paribus",
    "feature_statistics",
]

extra_dialogue_intents = [
    "follow_up",
    "why_explanation",
    "not_xai_method",
    "greeting"
]

possible_categories = general_questions + feature_specific_q + extra_dialogue_intents
possible_features = [
    'Age',
    'EducationLevel',
    'MaritalStatus',
    'Occupation',
    'WeeklyWorkingHours',
    'WorkLifeBalance',
    'InvestmentOutcome']

question_to_id_mapping = {
    "shap_top_3_features": 23,
    "anchor": 11,
    "shap_all_features": 24,
    "shap_least_3_features": 27,
    "ceteris_paribus": 25,
    "feature_statistics": 13,
    "counterfactual_any_change": 7,
    "follow_up": 0,
    "why_explanation": 1,
    "not_xai_method": 100,
    "greeting": 99
}

response_schemas = [
    ResponseSchema(name="method_name", description="name of the method to answer the user question."),
    ResponseSchema(name="feature", description="Feature that the user mentioned, can be none."),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


class LLMClassificationModel:
    def __init__(self):
        self.chain = (
                PromptTemplate.from_template(
                    template=get_template_with_descriptions(),
                    partial_variables={"input": "{question}",
                                       "format_instructions": format_instructions}) | ChatOllama(model=LLM_MODEL,
                                                                                                 temperature=0) | output_parser
        )

    def predict(self, question):
        message = self.chain.invoke({"question": f"{question}"})
        print(message)
        question = message["method_name"]
        feature = message["feature"]
        # get ids
        question_id = question_to_id_mapping[question]
        return question_id, feature


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


def get_template_with_descriptions():
    return f"""The user was presented an instance from the adult dataset with the features:
        "age", "workclass", "education", "marital_status", "occupation", "relationship", "work life balance"
        The machine learning model predicted whether the individual's income was above or below 50k.
        Given the user question about the model prediction, decide which method fits best to answer it. There are
        standalone methods that work without requiring a feature specification: {general_questions}
         and some that are specific to a feature: {feature_specific_q}. If you choose a feature specific method,
         reply with the method and the feature the user referenced from the following list of features {possible_features}""" + """
    Here are definitions for the available methods:
    - anchor: Anchor explanations identify the minimal set of conditions and feature values that, when held constant, 
    ensure a specific prediction result does not change, irrespective of alterations to other features. 
    Best for questions like "What factors guarantee this prediction remains the same?"
    Answer example: "If 'age' and 'occupation' remain constant, the model's prediction of the income being above 
    50k will remain unchanged. These features are the anchors in this scenario."
    - shap_all_features: uses SHAP values to measure and visualize all feature's individual contribution to the 
    overall prediction, considering the marginal effect of each feature across all possible combinations. 
    This comprehensive approach provides a complete overview by showing the impact of all features of the instance.
    Best for questions like: "What is the strength of each feature?"
    Answer example: "Here is a visualization, showing the contribution of each feature..."
    - shap_top_3_features: This method focuses on identifying and visualizing the contributions of the top three 
    features that have the highest impact on the model's prediction, according to SHAP values. It simplifies the 
    explanation by concentrating on the most influential variables.
    Best for questions like: "Which features had the greatest impact on this prediction?"
    Answer example: "The top influences on the prediction were 'education level', 'occupation', and 'hours per week'."
    - shap_least_3_features: This method concentrates on the three features with the least impact on the model’s
    prediction, according to SHAP values. It provides insights into the features that have minimal influence on
    the outcome, which can be critical for understanding the robustness of the model or for identifying potential
    areas of model simplification.
    Best for questions like: "Which features had the least impact on this prediction?"
    Answer example: "The least influential features were 'marital status', 'race', and 'sex', each contributing minimally to the overall prediction according to their SHAP values."
    - ceteris_paribus: This method focuses on looking for the impact of changes in a feature requested by the 
    user. It is useful to investigate what would happen if some feature was different. i.e higher or lower.
     Best for questions like: "What would happen if marital status was different?"
     Answer example: "Changing 'marital status' from 'single' to 'married' would result in the income prediction 
     changing from below 50k to above 50k."
    - counterfactual_any_change: Provide possible feature alterations 
    to understand scenarios under which the model's prediction is changed. This method is suited for 
    exploring changes in the features of the current instance that would lead to a different prediction, 
    thereby clarifying why the current instance didn't classify as another category.
    Best for questions like: "Why is it not class [other class]?", or "In which case would be other class?"
    Answer example: "The prediction would switch from class A to class B if 'hours per week' increased by more than 10 hours,
     or 'age' was above 45 or ..."
    - feature_statistics: Provides a statistical summary or visual representation of the features in a dataset.
    It calculates the mean and standard deviation for numerical features, offering a clear quantitative 
    overview of data spread and central tendency, giving intuition about whether a certain current value is 
    particularly high, low or average. For categorical features, it generates a frequency plot, 
    visually representing the distribution of categories within the data. 
    Best for questions like: "What are the typical values and distributions of 'age' in my dataset?"
    Answer example: "The mean age is 40 years with a standard deviation of 12.5.
    - follow_up: Indicates that the user did not ask a standalone question and mentioned a feature or some change. The
    user assumes to use the previous explanation method again and therefore does not repeat it explicitly.
    Best for questions like: "And what about age?"
    Answer example: "Here is the distribution for the feature age"
    - why_explanation: Explains which explanations are possible when the user asks a general why question.
    Best for questions like: "Why this prediction"?
    Answer example: "To understand the prediction, I can tell you about the most important attributes, or 
    which changes would have led to a different prediction."
    - greeting: Greets the user when he does not ask a specific question. If the user just says hi or how are you,
    tell him what questions you can answer.
    Best for questions like: "Hey, how are you?"
    Answer example: "Hello, I am an assistant to help you understand the prediction of the Machine Learning model.
    You can ask about the most or least important attributes, how certain changes in attributes influence the prediction
    or what alternative attributes would lead to a different prediction."
    - not_xai_method: Used to answer questions that are not possible to answer with the previous methods. Could
    be that the user asks for a clarification on an explanation or questions related to the domain.
    Best for questions like: "What does it mean?"
    Answer example: "[Description of the questioned thing]".

Decide which methods fits best and if its a feature specific one, also reply with the feature.
Immediately answer with the method and feature if applicable, without justification.
\n{format_instructions}\n{question}
"""
