import os

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"

from parsing.llm_intent_recognition.descriptive_prompt_classification import get_template_with_descriptions

LLM_MODEL = "llama3"

llm = ChatOllama(model=LLM_MODEL, temperature=0)

PROMPT = get_template_with_descriptions()

conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)
