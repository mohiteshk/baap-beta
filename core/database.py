import chromadb
from core.config import config

def get_chroma_collection():
    print("Connecting to Knowledge Base...")
    db_folder = config.get("db_folder", "./chroma_db")
    collection_name = config.get("collection_name", "drone_footage")
    
    chroma_client = chromadb.PersistentClient(path=db_folder)
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )