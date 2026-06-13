"""
embed.py — Embedding and Vector Store Builder

What this script does:
1. Calls ingest.py to get all 153 chunks
2. Loads the all-MiniLM-L6-v2 embedding model (runs locally, no API key)
3. Converts each chunk's text into a vector (list of 384 numbers)
4. Stores all vectors + text + source metadata in ChromaDB
5. Saves the ChromaDB collection to disk in the chroma_db/ folder

Run this ONCE to build the vector store.
After that, query.py reads from it without rebuilding.
"""

from sentence_transformers import SentenceTransformer
import chromadb
from ingest import ingest


def build_vector_store(chunks, persist_dir="chroma_db"):
    """
    Embed all chunks and store them in ChromaDB.

    Args:
        chunks: list of dicts from ingest.py, each with 'text', 'source', 'chunk_id'
        persist_dir: folder where ChromaDB saves its data to disk
    """

    # Step 1: Load the embedding model
    # all-MiniLM-L6-v2 converts text into 384-dimensional vectors
    # It runs entirely on your machine — no API key, no internet needed
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 2: Set up ChromaDB with a persistent local storage
    # This creates a chroma_db/ folder in your project directory
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if rebuilding from scratch
    try:
        client.delete_collection("professor_reviews")
        print("Deleted existing collection.")
    except Exception:
        pass

    # Create a fresh collection
    collection = client.create_collection(
        name="professor_reviews",
        metadata={"hnsw:space": "cosine"}  # use cosine similarity for search
    )

    # Step 3: Embed all chunks
    print(f"Embedding {len(chunks)} chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    # embeddings is now a numpy array of shape (153, 384)
    # each row is the vector for one chunk

    # Step 4: Store everything in ChromaDB
    print("Storing in ChromaDB...")
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],           # unique ID per chunk
        documents=[chunk["text"] for chunk in chunks],          # raw text
        embeddings=embeddings.tolist(),                         # vectors
        metadatas=[{"source": chunk["source"]} for chunk in chunks]  # source filename
    )

    print(f"Done. {collection.count()} chunks stored in '{persist_dir}/'.")
    return collection


if __name__ == "__main__":
    # Run the full pipeline: ingest → embed → store
    chunks = ingest()
    build_vector_store(chunks)
