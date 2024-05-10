# https://python.langchain.com/docs/expression_language/how_to/routing/

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

chain = (
    PromptTemplate.from_template(
        """The user was presented an instance from the adult dataset with the features:
        "age", "workclass", "education", "marital_status", "occupation", "relationship...
        The machine learning model predicted whether the individual's income was above or below 50k.
        Given the user question about the model prediction below, classify it as either being about 'feature_attributions', 'counterfactuals' or 'general'.
        
        Here are definitions for the three classes:
        - feature_attributions: quantify the importance of input features in model predictions, providing essential 
        insights into which features significantly influence outcomes and why certain decisions are made.
        - counterfactuals: provide alternative scenarios that would change a model's decision, illustrating how slight
        modifications to input features could result in different outcomes and clarifying the conditions under which a decision would be altered.
        Anwers questions like "Why not predicted other class?" or "How should the input be changed to get a different prediction?"
        - general: any other questions that do not fall into the above categories.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | ChatOllama(model="llama3")
    | StrOutputParser()
)

print(chain.invoke({"question": "Is he predicted less than 50k because he is young?"}))