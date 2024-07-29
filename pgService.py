from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os

from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvecto_rs import PGVecto_rs
from langchain_core.documents import Document



# # importin the embedding model from hugging face
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

# # creating an object for embedding model
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},  # use cuda, if gpu is available
    encode_kwargs=encode_kwargs
    )

def getContext(databaseNmae,query):
    db = "class"+databaseNmae
    PORT = os.getenv("DB_PORT", 5432)
    HOST = os.getenv("DB_HOST", "localhost")
    USER = os.getenv("DB_USER", "postgres")
    PASS = os.getenv("DB_PASS", "mysecretpassword")
    DB_NAME = os.getenv("DB_NAME",db)

    # Run tests with shell:
    URL = "postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}".format(
        port=PORT,
        host=HOST,
        username=USER,
        password=PASS,
        db_name=DB_NAME,
    )

    db1 = PGVecto_rs.from_collection_name(
        embedding=embeddings,
        db_url=URL,
        collection_name=db,
    )

    docs: List[Document] = db1.similarity_search(query, k=2)
    return docs
