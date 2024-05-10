# Import things that are needed generically
import os

from langchain_community.embeddings import OllamaEmbeddings

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"

from langchain.utils.math import cosine_similarity
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

feature_importances_template = """Feature attribution methods in XAI quantify feature importance in model predictions, 
offering insights into which features significantly impact outcomes. These methods are essential for answering questions
about why models make certain decisions by computing which features are most influential."""

counterfactuals_template = """Counterfactual explanations in XAI provide alternative scenarios that would change a 
model's decision, helping to understand model behavior by illustrating how slight modifications to input features could 
result in different outcomes. These explanations are crucial for answering questions about how specific changes in input 
data could lead to different predictions, thus clarifying the conditions under which a decision would be altered. They
are also used to describe how someone would need to be different to receive a different prediction."""

embeddings = OllamaEmbeddings(model="llama3")
prompt_templates = [feature_importances_template, counterfactuals_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)


# Route question to prompt
def prompt_router(input):
    # Embed question
    query_embedding = embeddings.embed_query(input["query"])
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # Chosen prompt
    if most_similar == feature_importances_template:
        print("Using FI")
    if most_similar == counterfactuals_template:
        print("Using CF")
    else:
        print("Dmi")


chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(prompt_router)
)

print(chain.invoke("What's the most important feature?"))

print("DIMI")
