import json

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from sqlalchemy import inspect, make_url

from llm_agents import db
from llm_agents.knowledge_base.create_index import get_pg_vector_store
import os

OPENAI_EMBEDDING_SIZE = int(os.getenv('OPENAI_EMBEDDING_SIZE'))
RETRIEVER_NODE_AMOUNT = int(os.getenv('RETRIEVER_NODE_AMOUNT', "5"))

# Load summaries from json
summaries_path = "llm_agents/knowledge_base/summaries.json"
SUMMARIES = json.load(open(summaries_path, "r"))['tools']


def get_citation_query_engine(index, similarity_top_k=8, hybrid=False):
    if not hybrid:
        vector_store_query_mode = VectorStoreQueryMode.DEFAULT
    else:
        vector_store_query_mode = VectorStoreQueryMode.HYBRID

    return CitationQueryEngine(retriever=VectorIndexRetriever(index=index,
                                                              similarity_top_k=similarity_top_k,
                                                              vector_store_query_mode=vector_store_query_mode))


def create_query_engine_tool(query_engine, name, summary):
    """
    Method to create a query engine tool from a query engine.
    :param query_engine: The query engine to create the tool from.
    :param name: The name of the tool.
    :param summary: The summary of the tool, a description for the llm to know what the tool does.
    """
    tool_description = f"Provides information from the paper titled '{name}'. Summary: {summary}"
    metadata = ToolMetadata(
        name=name,
        description=tool_description)

    return QueryEngineTool(
        query_engine=query_engine,
        metadata=metadata
    )


def build_query_engine_tools() -> list:
    """
    Builds query engine tools for the agent using paper summaries as tool descriptions. The summaries are retrieved
    from the paper_overview table in the database. The tools are created for each document table in the database
    (excluding the paper_overview table).
    :param session: SQLAlchemy session
    :param embedding_size: The size of the embeddings in the vector store
    :param retriever_node_amount: The number of nodes to retrieve by the retriever
    :param llm: The language model to use for the query engine
    :return: List of query engine tools to be used by the agent
    """

    load_dotenv()
    connection_string = os.getenv('PG_VEC_CONNECTION_STRING')
    DB_NAME = os.getenv('DB_NAME')

    # Convert summaries to a dictionary
    paper_summaries = {paper['title']: paper['summary'] for paper in SUMMARIES}

    # Use SQLAlchemy Inspector to retrieve table names
    with db.engine.connect() as connection:
        inspector = inspect(connection)
        all_table_names = inspector.get_table_names()

    query_engine_tools = []
    for table_name in all_table_names:
        if table_name == "items":
            continue
        # Process the table name (remove any unwanted prefixes)
        table_name = table_name.replace("data_", "")

        # Get the PG Vector store for the document table
        vec_store = get_pg_vector_store(
            url=make_url(connection_string),
            db_name=DB_NAME,
            table_name=table_name,
            embedding_size=OPENAI_EMBEDDING_SIZE
        )

        # Get the vector index from the store
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vec_store, show_progress=True)

        # Retrieve summary for the paper
        summary = paper_summaries.get(table_name)

        # Use the hybrid query engine with the vector index
        query_engine = get_citation_query_engine(
            vector_index,
            similarity_top_k=RETRIEVER_NODE_AMOUNT,
            hybrid=True
        )

        # Create a query engine tool and add it to the list
        query_engine_tool = create_query_engine_tool(query_engine, table_name, summary)
        query_engine_tools.append(query_engine_tool)

        # Limit the number of tools to prevent overwhelming the agent
        if len(query_engine_tools) >= 123:
            break

    return query_engine_tools
