from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain

import os
from dotenv import load_dotenv

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

from parsing.llm_intent_recognition.prompts.prompt_A import get_template_with_full_descriptions

LLM_MODEL = "llama3"

llm = ChatOllama(model=LLM_MODEL, temperature=0)

PROMPT = get_template_with_full_descriptions()

conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)
