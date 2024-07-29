from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvecto_rs import PGVecto_rs
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# science 
loader = TextLoader("class/class 8/class 8 english.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = text_splitter.split_documents(documents)
print("CHUNCKS ARE READY!!--------------------------------- ")

# Model for text embedding
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True}

# Initialize embeddings model
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},  # use 'cuda' if GPU is available
    encode_kwargs=encode_kwargs
)


## PGVecto.rs needs the connection string to the database.
## We will load it from the environment variables.
import os

PORT = os.getenv("DB_PORT", 5432)
HOST = os.getenv("DB_HOST", "localhost")
USER = os.getenv("DB_USER", "postgres")
PASS = os.getenv("DB_PASS", "mysecretpassword")
DB_NAME = os.getenv("DB_NAME", "class8")

# Run tests with shell:
URL = "postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}".format(
    port=PORT,
    host=HOST,
    username=USER,
    password=PASS,
    db_name=DB_NAME,
)
print("READY    TO      INSERT !!--------------------------------- ")

db1 = PGVecto_rs.from_documents(
    documents=docs,
    embedding=embeddings,
    db_url=URL,
    # The table name is f"collection_{collection_name}", so that it should be unique.
    collection_name="class8",
)
print("EMBEDDINGS ARE STORED INTO THE DATABASE!!--------------------------------- ")
