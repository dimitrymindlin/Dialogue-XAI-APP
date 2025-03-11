import os
from pathlib import Path
import re

from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import IndexNode
from llama_index.readers.file import FlatReader
from llama_index.embeddings.openai import OpenAIEmbedding
import psycopg2
from sqlalchemy import make_url

from llm_agents.pg_vec_store import get_pg_vector_store

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_ORGANIZATION"] = os.getenv('OPENAI_ORGANIZATION_ID')
LLM_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')

markdown_files_path = "../knowledge_base"
db_name = "llm-agents-db"

embed_model = OpenAIEmbedding(model="text-embedding-3-large")
embedding_dim = 1536

connection_string = os.getenv('PG_VEC_CONNECTION_STRING')
conn = psycopg2.connect(connection_string)
conn.autocommit = True
url = make_url(connection_string)

node_parser = MarkdownNodeParser()


def sanitize_text(text):
    # Remove NUL characters from the text
    return re.sub(r'\x00', '', text)


def add_chunks_to_pg_vec_index(table_name, chunks):
    # Init Vector Store
    vector_store = get_pg_vector_store(url, db_name, table_name, embedding_dim)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    for chunk in chunks:
        chunk.text = sanitize_text(chunk.text)
    # Turn chunks to index nodes
    index_nodes = [IndexNode.from_text_node(chunk, index_id=chunk.id_) for chunk in chunks]
    # Insert nodes to index
    index.insert_nodes(index_nodes)  # TODO: (Dimi) How to make sure to update only the new chunks?
    return True


def create_indexes():
    for root, dirs, files in os.walk(markdown_files_path):
        markdown_docs = [os.path.join(root, file) for file in files if file.endswith(".md")]
        for file_path in markdown_docs:
            file_name = Path(file_path).name.replace(".md", "")
            # Create vector store with table name same as file name
            # Load and preprocess markdown documents
            md_doc = FlatReader().load_data(Path(file_path))
            nodes = node_parser.get_nodes_from_documents(md_doc)
            # Create llamaindex document from nodes
            add_chunks_to_pg_vec_index(file_name, nodes)
            print(f"Added nodes from {file_name} to vector store")


if __name__ == "__main__":
    create_indexes()
