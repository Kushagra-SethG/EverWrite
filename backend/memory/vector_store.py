import chromadb
from sentence_transformers import SentenceTransformer
from config import TOP_K

client = chromadb.Client()

collection = client.get_or_create_collection(name="game_memory")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def add_memory(text):
    embedding = embedder.encode(text).tolist()
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(hash(text))]
    )

def get_memory(query):
    embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K
    )

    return results["documents"][0] if results["documents"] else []