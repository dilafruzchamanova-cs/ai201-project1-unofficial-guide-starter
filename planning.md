# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

This guide covers student-generated reviews of professors at Alfred University. Official sources like course catalogs and department websites describe what a course covers, but they tell you nothing about whether a professor gives useful feedback, curves exams, cancels class frequently, or is approachable outside of class. Students share this knowledge informally — on Rate My Professors, in group chats, between friends — but it's scattered and unsearchable. This system makes it queryable: a student can ask a plain-language question and get a grounded answer drawn directly from real peer reviews.

---

## Documents

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Rate My Professors | Reviews of Lynn Petrillo (English) | https://www.ratemyprofessors.com/professor/1131643 |
| 2 | Rate My Professors | Reviews of Juliana Gray (English) | https://www.ratemyprofessors.com/professor/917815 |
| 3 | Rate My Professors | Reviews of Sarah Cote (English) | https://www.ratemyprofessors.com/professor/2289745 |
| 4 | Rate My Professors | Reviews of Joseph Petrillo (Mathematics) | https://www.ratemyprofessors.com/professor/740992 |
| 5 | Rate My Professors | Reviews of Robert Myers (Anthropology) | https://www.ratemyprofessors.com/professor/139130 |
| 6 | Rate My Professors | Reviews of Hakan Karaaytu (Communication) | https://www.ratemyprofessors.com/professor/2943652 |
| 7 | Rate My Professors | Reviews of Allen Grove (English) | https://www.ratemyprofessors.com/professor/133214 |
| 8 | Rate My Professors | Reviews of Melissa Ryan (English) | https://www.ratemyprofessors.com/professor/553002 |
| 9 | Rate My Professors | Reviews of Pam Schultz (Communication) | https://www.ratemyprofessors.com/professor/305438 |
| 10 | Rate My Professors | Reviews of Elizabeth Matson (Mathematics) | https://www.ratemyprofessors.com/professor/2457783 |

---

## Chunking Strategy

**Chunk size:** 300 characters

**Overlap:** 50 characters

**Reasoning:** The documents are made up of short, opinion-based student reviews — typically 1 to 4 sentences per review. A chunk size of 300 characters is large enough to capture a complete thought (e.g., "She gives great feedback and is always available outside class") without merging unrelated opinions from different reviewers into the same chunk. Larger chunks (e.g., 800+ characters) would blend multiple student opinions into one embedding, making it harder to match a specific question like "does she give good feedback?" to the right piece of text. An overlap of 50 characters ensures that if a key opinion sits at the boundary between two chunks, at least one chunk captures it in full context.

---

## Retrieval Approach

**Embedding model:** all-MiniLM-L6-v2 via sentence-transformers

**Top-k:** 5

**Production tradeoff reflection:** all-MiniLM-L6-v2 was chosen because it runs entirely locally with no API key, no cost, and no rate limits — ideal for a course project. For a real production deployment, the tradeoffs worth weighing are: (1) context length — MiniLM has a 256-token limit, which is fine for short reviews but would truncate long documents; a model like text-embedding-3-small handles up to 8191 tokens; (2) multilingual support — MiniLM is English-only, so a multilingual student body would need a model like paraphrase-multilingual-MiniLM-L12-v2; (3) domain accuracy — general-purpose embeddings may underperform on highly informal, slang-heavy review text compared to a fine-tuned model; (4) latency and cost — API-hosted models like OpenAI's embeddings add per-token cost and network latency, while local models trade speed for zero cost.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Does Lynn Petrillo give good feedback on writing assignments? | Yes — multiple reviews specifically mention clear feedback, helpful comments, and that she reviews your work before you submit it. |
| 2 | Is Juliana Gray a harsh grader? | Yes — the majority of reviews describe her as a tough grader who is strict on attendance and does not tolerate inconvenience. Several reviews warn students to stay away. |
| 3 | Is Joseph Petrillo a good professor for students who struggle with math? | Yes — reviews consistently say he is very accessible, explains material clearly, gives real-world examples, and is willing to re-explain until students understand. |
| 4 | What do students say about Pam Schultz's lectures in Communication? | Mixed — some students say she is kind and encouraging; others say her lectures are disconnected from assignments, that she rambles about current events instead of teaching, and that grading feels biased. |
| 5 | Is Sarah Cote a good choice for Writing 1 or Writing 2 at Alfred? | Yes — reviews consistently describe her as kind, clear on expectations, flexible with attendance, and one of the best writing professors on campus. |

---

## Anticipated Challenges

1. **Name collision between Lynn Petrillo and Joseph Petrillo:** Both professors share the last name Petrillo and teach at Alfred. A query like "Is Petrillo good?" is ambiguous — the retrieval system may return chunks from both professors' files, and the LLM may produce a confused or blended answer. This is a retrieval failure risk caused by shared surface-level token overlap between documents.

2. **Short chunks losing context:** Because reviews are short and informal, chunks at 300 characters may contain very little semantic signal — a chunk like "She's amazing!" carries almost no information about what specifically makes her good. This could cause retrieval to match on tone rather than topic, returning enthusiastic-sounding chunks that don't actually answer the question.

---

## Architecture

```
┌─────────────────────┐
│  Raw .txt files      │
│  (documents/ folder) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Document Ingestion  │  ← Python (ingest.py)
│  Load + clean text   │    strips metadata headers,
│  from each file      │    normalizes whitespace
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Chunking            │  ← LangChain CharacterTextSplitter
│  chunk_size=300      │    or custom sliding window
│  overlap=50          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Embedding + Vector Store            │  ← sentence-transformers
│  all-MiniLM-L6-v2                   │    all-MiniLM-L6-v2
│  Stored in ChromaDB (local)          │  ← ChromaDB
│  Metadata: source filename, chunk ID │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Retrieval           │  ← ChromaDB semantic search
│  top-k = 5 chunks    │    cosine similarity
│  + source metadata   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Generation          │  ← Groq API
│  llama-3.3-70b       │    grounded prompt: answer
│  -versatile          │    only from retrieved chunks
│  + source citation   │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Query Interface     │  ← Gradio (app.py)
│  Text input + output │    localhost:7860
└─────────────────────┘
```

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**
I will give Claude my Documents section (10 .txt files, one professor per file, structured with headers and review blocks separated by `---`) and my Chunking Strategy section (300-char chunks, 50-char overlap). I will ask Claude to implement `ingest.py` — a script that loads all files from the `documents/` folder, strips the metadata header at the top of each file, splits the remaining text into chunks using the specified size and overlap, and returns a list of dicts with keys `text`, `source`, and `chunk_id`. I will verify the output by printing 5 random chunks and checking they are readable, self-contained, and free of header artifacts.

**Milestone 4 — Embedding and retrieval:**
I will give Claude my Architecture diagram (showing all-MiniLM-L6-v2 → ChromaDB) and my Retrieval Approach section. I will ask Claude to implement `embed.py` — a script that takes the chunk list from `ingest.py`, embeds each chunk using SentenceTransformer("all-MiniLM-L6-v2"), stores them in a ChromaDB collection with source metadata, and saves the collection to disk. I will also ask Claude to implement a `retrieve(query, k=5)` function in `query.py` that returns the top-k chunks and their source filenames. I will verify by running 3 of my evaluation queries and checking that returned chunks are visibly relevant and have distance scores below 0.5.

**Milestone 5 — Generation and interface:**
I will give Claude my grounding requirement (LLM must answer only from retrieved chunks, must cite sources, must refuse if context is insufficient) and the Gradio skeleton from the project instructions. I will ask Claude to implement the `ask(question)` function in `query.py` that calls `retrieve()`, formats the chunks into a prompt, calls the Groq API, and returns a dict with keys `answer` and `sources`. I will also ask Claude to wire this into `app.py` using Gradio. I will verify grounding by checking that responses directly reference content from the retrieved chunks and that asking an out-of-scope question produces a refusal.
