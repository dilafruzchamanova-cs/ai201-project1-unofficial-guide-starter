"""
ingest.py — Document Ingestion and Chunking Pipeline

What this script does:
1. Loads every .txt file from the documents/ folder
2. Strips the metadata header (professor name, dept, rating) above the first ---
3. Splits the remaining review text into overlapping chunks
4. Returns a list of dicts: {text, source, chunk_id}

Each chunk is ~300 characters with 50-character overlap so that
a key opinion sitting at a chunk boundary appears in at least one
complete chunk.
"""

import os
import glob


def load_documents(folder="documents"):
    """
    Read all .txt files from the documents folder.
    Returns a list of dicts with keys: 'text' and 'source'.
    """
    docs = []
    filepaths = glob.glob(os.path.join(folder, "*.txt"))

    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        source = os.path.basename(filepath)  # e.g. "prof_lynn_petrillo.txt"
        docs.append({"text": raw_text, "source": source})

    print(f"Loaded {len(docs)} documents.")
    return docs


def clean_document(text):
    """
    Remove the metadata header at the top of each file.

    Each file starts with professor name, department, school, and rating,
    followed by a line of '---'. Everything above the first '---' is header.
    We strip it and keep only the review content below.

    Also normalizes whitespace: collapses multiple blank lines into one.
    """
    # Split on the first '---' separator
    parts = text.split("---", 1)

    if len(parts) == 2:
        # parts[0] is the header, parts[1] is the review content
        content = parts[1]
    else:
        # No separator found — use the full text as-is
        content = text

    # Normalize whitespace: strip leading/trailing space, collapse blank lines
    lines = content.splitlines()
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True
        else:
            cleaned_lines.append(stripped)
            prev_blank = False

    return "\n".join(cleaned_lines).strip()


def chunk_text(text, source, chunk_size=300, overlap=50):
    """
    Split text into overlapping chunks of ~chunk_size characters.

    chunk_size=300: large enough to capture a complete student opinion,
                    small enough to stay focused on one topic.
    overlap=50:     if a key sentence sits at a boundary, the next chunk
                    will still include its tail end for context.

    Returns a list of dicts: {text, source, chunk_id}
    """
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        # Skip empty or very short chunks (less than 20 chars = not useful)
        if len(chunk) >= 20:
            chunks.append({
                "text": chunk,
                "source": source,
                "chunk_id": f"{source}_chunk_{chunk_id}"
            })
            chunk_id += 1

        # Move forward by (chunk_size - overlap) so chunks overlap
        start += chunk_size - overlap

    return chunks


def ingest(folder="documents", chunk_size=300, overlap=50):
    """
    Full pipeline: load → clean → chunk.
    Returns a flat list of all chunks across all documents.
    """
    docs = load_documents(folder)
    all_chunks = []

    for doc in docs:
        cleaned = clean_document(doc["text"])
        chunks = chunk_text(cleaned, doc["source"], chunk_size, overlap)
        all_chunks.extend(chunks)

    print(f"Total chunks produced: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    # Run this file directly to inspect chunks
    chunks = ingest()

    print("\n--- 5 SAMPLE CHUNKS ---\n")
    import random
    samples = random.sample(chunks, min(5, len(chunks)))
    for i, chunk in enumerate(samples):
        print(f"[Chunk {i+1}] Source: {chunk['source']}")
        print(f"Text: {chunk['text']}")
        print()
