from llama_index.vector_stores.postgres import PGVectorStore


def get_pg_vector_store(url, db_name, table_name, embedding_size):
    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name=table_name,
        embed_dim=embedding_size,
        hybrid_search=True,
    )
    return vector_store
