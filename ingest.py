import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_PORT = 80
COLLECTION_NAME = "lanchain-chunking"

# Files to ingest
FILES = [
    "Business_Strategy.docx",
    "Financial_Planning_Essentials_22_Pages.docx"
]

def ingest_data():
    documents = []
    print("Loading documents...")
    for file_name in FILES:
        if os.path.exists(file_name):
            loader = Docx2txtLoader(file_name)
            docs = loader.load()
            print(f"Loaded {file_name}: {len(docs)} documents")
            documents.extend(docs)
        else:
            print(f"File not found: {file_name}")

    if not documents:
        print("No documents to process.")
        return

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks.")

    print("Initializing Embeddings...")
 
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_base="https://litellm.confer.today"
    )

    print("Initializing Qdrant Client...")
    client = QdrantClient(
        url=QDRANT_URL,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
    )

    print(f"Upserting to collection '{COLLECTION_NAME}'...")
    try:
        QdrantVectorStore.from_documents(
            splits,
            embeddings,
            url=QDRANT_URL,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
            force_recreate=True 
        )
        print("Ingestion complete!")
    except Exception as e:
        print(f"Error during upsert: {e}")

if __name__ == "__main__":
    ingest_data()
