"""
query.py — Retrieval and Generation

What this script does:
1. retrieve(): takes a question, finds the top-k most relevant chunks from ChromaDB
2. ask(): takes a question, retrieves chunks, sends them to Groq LLM, returns a grounded answer

The LLM is instructed to answer ONLY from the retrieved chunks.
If the chunks don't contain enough information, it must say so.
Source attribution is included in every response.
"""

import os
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY from your .env file

# Load the same embedding model used in embed.py
# Must be the same model — queries and chunks need to live in the same vector space
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to the ChromaDB collection saved to disk by embed.py
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("professor_reviews")

# Connect to Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def retrieve(query, k=5):
    """
    Find the top-k chunks most semantically similar to the query.

    How it works:
    - Embed the query into a 384-dimensional vector (same space as the chunks)
    - ChromaDB computes cosine similarity between the query vector and all stored vectors
    - Returns the k closest chunks

    Returns a list of dicts: {text, source, distance}
    """
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": text,
            "source": metadata["source"],
            "distance": round(distance, 4)
        })

    return chunks


def ask(question):
    """
    Full RAG pipeline: retrieve relevant chunks → generate grounded answer.

    Grounding mechanism: the system prompt explicitly forbids the LLM from
    using any knowledge outside the provided documents. If the context is
    insufficient, the model must say so rather than guess.

    Returns a dict: {answer, sources}
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(question, k=5)

    # Step 2: Format chunks as context for the LLM
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    context = "\n\n".join(context_blocks)

    # Step 3: Build the prompt
    system_prompt = """You are a helpful assistant for students at Alfred University.
You answer questions about professors using ONLY the student reviews provided below.
Do NOT use any knowledge from your training data.
Do NOT make up information that is not in the provided reviews.
If the provided reviews do not contain enough information to answer the question,
say exactly: "I don't have enough information on that in the available reviews."
Always cite which source document(s) your answer draws from."""

    user_prompt = f"""Student reviews (use ONLY these to answer):

{context}

Question: {question}

Answer based only on the reviews above, and cite your sources."""

    # Step 4: Call Groq LLM
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,  # low temperature = more factual, less creative
        max_tokens=512
    )

    answer = response.choices[0].message.content

    # Step 5: Collect unique sources from retrieved chunks
    sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    # Test retrieval with 3 evaluation questions
    test_queries = [
        "Does Lynn Petrillo give good feedback on writing assignments?",
        "Is Juliana Gray a harsh grader?",
        "Is Joseph Petrillo good for students who struggle with math?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        chunks = retrieve(query, k=5)
        for chunk in chunks:
            print(f"  Source: {chunk['source']} | Distance: {chunk['distance']}")
            print(f"  Text: {chunk['text'][:150]}...")
            print()
