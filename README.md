# The Unofficial Guide — Project 1

---

## Domain

This system covers student-generated reviews of professors at Alfred University. Official sources like course catalogs and department websites describe what a course covers but tell you nothing about whether a professor curves exams, gives useful feedback, cancels class frequently, or is approachable outside of class. Students share this knowledge informally — on Rate My Professors, in group chats, between friends — but it is scattered and unsearchable. This RAG system makes it queryable: a student types a plain-language question and gets a grounded answer drawn directly from real peer reviews, with the source documents cited.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Rate My Professors — Lynn Petrillo | Student reviews (English) | https://www.ratemyprofessors.com/professor/1131643 |
| 2 | Rate My Professors — Juliana Gray | Student reviews (English) | https://www.ratemyprofessors.com/professor/917815 |
| 3 | Rate My Professors — Sarah Cote | Student reviews (English) | https://www.ratemyprofessors.com/professor/2289745 |
| 4 | Rate My Professors — Joseph Petrillo | Student reviews (Mathematics) | https://www.ratemyprofessors.com/professor/740992 |
| 5 | Rate My Professors — Robert Myers | Student reviews (Anthropology) | https://www.ratemyprofessors.com/professor/139130 |
| 6 | Rate My Professors — Hakan Karaaytu | Student reviews (Communication) | https://www.ratemyprofessors.com/professor/2943652 |
| 7 | Rate My Professors — Allen Grove | Student reviews (English) | https://www.ratemyprofessors.com/professor/133214 |
| 8 | Rate My Professors — Melissa Ryan | Student reviews (English) | https://www.ratemyprofessors.com/professor/553002 |
| 9 | Rate My Professors — Pam Schultz | Student reviews (Communication) | https://www.ratemyprofessors.com/professor/305438 |
| 10 | Rate My Professors — Elizabeth Matson | Student reviews (Mathematics) | https://www.ratemyprofessors.com/professor/2457783 |

Documents were collected manually by copying review text from Rate My Professors and saving each professor's reviews as a plain .txt file in the `documents/` folder. Each file includes the professor's name, department, overall rating, and individual reviews with course, date, quality score, difficulty score, grade, and review text.

---

## Chunking Strategy

**Chunk size:** 300 characters

**Overlap:** 50 characters

**Why these choices fit your documents:** The documents consist of short, opinion-based student reviews — typically 1 to 4 sentences per review. A chunk size of 300 characters captures one complete student opinion without merging unrelated opinions from different reviewers into a single embedding. Larger chunks (e.g., 800+ characters) would blend multiple opinions — exam difficulty, teaching style, office hours accessibility — into one vector, making it harder to match a specific query like "does she give good feedback?" to the right piece of text. An overlap of 50 characters ensures that a key sentence sitting at a chunk boundary appears in full in at least one of the two adjacent chunks. Before chunking, the metadata header at the top of each file (professor name, department, school, rating) is stripped — the `clean_document()` function finds the first `---` separator and discards everything above it.

**Final chunk count:** 153 chunks across 10 documents.

**Sample chunks:**

```
[Source: prof_lynn_petrillo.txt]
clear on her expectations and does not give a heavy workload. I definitely recommend her to everyone.
---
Course: WRIT101
Date: Nov 8, 2012
Quality:

[Source: prof_juliana_gray.txt]
ew: Dr. Gray is a very complicated person. She will actually take pleasure in being malicious towards her students and deliberately make them feel bad...

[Source: prof_joseph_petrillo.txt]
uality: 4.5 | Difficulty: 2.0
Grade: A+ | Would Take Again: Yes
Review: Petrillo tries his best to make Math151 as fun as possible while making sure y...

[Source: prof_robert_myers.txt]
nd is very interested in what a student has to say. Tests are a little difficult due to lots of vocabulary. Other than that, a really great teacher.

[Source: prof_sarah_cote.txt]
et her halfway and you'll succeed. Highly recommend.
---
Course: ENGL102
Date: Apr 21, 2021
Quality: 4.0 | Difficulty: 3.0
```

One visible issue in the sample chunks: some chunks begin mid-word (e.g., "culty: 2.0" instead of "Difficulty: 2.0") because the character-based splitter cuts at a fixed position regardless of word boundaries. This is a known limitation of fixed-size character chunking.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via the `sentence-transformers` library. This model runs entirely locally — no API key, no internet connection, no rate limits. It converts each chunk of text into a 384-dimensional vector. Chunks with similar meaning produce vectors that are close together in this 384-dimensional space, which is how semantic search works: the query vector is compared to all chunk vectors using cosine similarity, and the closest ones are returned.

**Production tradeoff reflection:** For a real deployment, the tradeoffs worth weighing are: (1) context length — `all-MiniLM-L6-v2` has a 256-token limit, which is fine for short reviews but would truncate longer documents; `text-embedding-3-small` from OpenAI handles up to 8191 tokens; (2) multilingual support — MiniLM is English-only, which would be a problem for a university with international students; `paraphrase-multilingual-MiniLM-L12-v2` handles 50+ languages; (3) domain accuracy — general-purpose embeddings may underperform on informal, slang-heavy review text compared to a fine-tuned model; (4) cost and latency — API-hosted models add per-token cost and network round-trip latency, while local models like MiniLM are free but slower on CPU.

---

## Grounded Generation

**System prompt grounding instruction:** The LLM (Groq's `llama-3.3-70b-versatile`) receives the following system prompt on every request:

> "You are a helpful assistant for students at Alfred University. You answer questions about professors using ONLY the student reviews provided below. Do NOT use any knowledge from your training data. Do NOT make up information that is not in the provided reviews. If the provided reviews do not contain enough information to answer the question, say exactly: 'I don't have enough information on that in the available reviews.' Always cite which source document(s) your answer draws from."

**How source attribution is surfaced in the response:** Source attribution is enforced in two ways. First, the system prompt instructs the model to cite sources inline within its response. Second, the `ask()` function in `query.py` programmatically collects the source filenames from the retrieved chunks and appends them to every response in a separate "Retrieved from" field — this means source attribution appears even if the model fails to cite inline.

---

## Retrieval Test Results

**Query 1: Does Lynn Petrillo give good feedback on writing assignments?**

Top returned chunks:
- `prof_lynn_petrillo.txt` — "clear on her expectations and does not give a heavy workload. I definitely recommend her to everyone." (distance: 0.391)
- `prof_lynn_petrillo.txt` — "always happy, and tries to keep the class in a positive mood. Class is interesting, never felt like skipping..." (distance: 0.399)
- `prof_lynn_petrillo.txt` — "Wonderful Professor! I had her for English 101 in Fall 2013 and English 102 in Spring 2014..." (distance: 0.3999)
- `prof_pam_schultz.txt` — "good at making you feel comfortable about giving them [speeches]..." (distance: 0.4123)
- `prof_lynn_petrillo.txt` — "She's honestly amazing! 100% take her for Writing 102..." (distance: 0.42)

Why the top chunks are relevant: The top 3 results correctly come from the Lynn Petrillo document. They describe her general teaching quality and expectations, which are semantically related to the question about feedback. The Pam Schultz chunk (rank 4) is off-target — it discusses speech feedback in a communications class, not writing feedback.

**Query 2: Is Juliana Gray a harsh grader?**

Top returned chunks:
- `prof_sarah_cote.txt` — "She is very soft — anytime she says something controversial she has to follow up with an excuse..." (distance: 0.4224)
- `prof_juliana_gray.txt` — "Dr. Gray is a very complicated person. She will actually take pleasure in being malicious..." (distance: 0.453)
- `prof_juliana_gray.txt` — "Toughest grader I have seen. So many papers to write." (distance: 0.4634)
- `prof_juliana_gray.txt` — "Dr. Gray has a great sense of humor, though she can be very sarcastic..." (distance: 0.4712)
- `prof_juliana_gray.txt` — "Extremely tough grader, lots of homework, attendance mandatory." (distance: 0.4975)

Why the top chunks are relevant: Chunks 2–5 are directly relevant — they all describe Gray's grading harshness. The top result (Sarah Cote) is off-target and represents a retrieval failure discussed in the Failure Case section below.

**Query 3: Is Joseph Petrillo good for students who struggle with math?**

Top returned chunks:
- `prof_joseph_petrillo.txt` — "are no surprise — you will have a general idea of what is on them ahead of time. I would definitely take another class of his." (distance: 0.3354)
- `prof_joseph_petrillo.txt` — "He loves what he does. Dr. Petrillo is very passionate about his field..." (distance: 0.3649)
- `prof_joseph_petrillo.txt` — "Petrillo tries his best to make Math151 as fun as possible while making sure you learn the content..." (distance: 0.3812)
- `prof_joseph_petrillo.txt` — "helps you learn. His homework is very manageable..." (distance: 0.3812)
- `prof_joseph_petrillo.txt` — "Dr. Petrillo is a great professor. He is willing to help and actually takes the time to make sure you truly understand..." (distance: 0.4014)

Why the top chunks are relevant: All 5 returned chunks are from the correct document and directly address accessibility, patience, and willingness to help — exactly what a struggling student would care about. Distances are all below 0.41, indicating strong retrieval.

---

## Grounded Generation Examples

**Example 1 — Accurate, grounded response:**

Query: "Is Joseph Petrillo good for students who struggle with math?"

Response: "Dr. Petrillo seems to be a great professor for students who struggle with math. According to the reviews, he 'genuinely cares about his students' (prof_joseph_petrillo.txt), is 'willing to help and actually takes the time to make sure you truly understand what you are learning' (prof_joseph_petrillo.txt), and his homework is 'very manageable' (prof_joseph_petrillo.txt, two sources). Additionally, one review states that 'His test prep tends to be difficult, but the tests are usually quite easy if you understand the material' (prof_joseph_petrillo.txt), which suggests that he helps students prepare well for tests. Another review mentions that he 'tries his best to make Math151 as fun as possible while making sure you learn the content' (prof_joseph_petrillo.txt). Overall, the reviews suggest that Dr. Petrillo is a supportive and effective professor for students who struggle with math (prof_joseph_petrillo.txt, multiple sources)."

Sources: prof_joseph_petrillo.txt

**Example 2 — Accurate, grounded response:**

Query: "Is Sarah Cote a good choice for Writing 1 or Writing 2?"

Response: "According to the reviews, Sarah Cote is a good choice for Writing 1 and Writing 2. One review states, 'Compared to some of my friends I would say Sarah is the best choice for Writing 1 and probably Writing 2 on this whole campus' (Source: prof_sarah_cote.txt)."

Sources: prof_sarah_cote.txt, prof_lynn_petrillo.txt

**Out-of-scope query — refusal response:**

Query: "What is the best restaurant near Alfred University?"

The system returns: "I don't have enough information on that in the available reviews." — correctly refusing to answer a question outside its document domain.

---

## Query Interface

The interface is built with Gradio and runs locally at `http://localhost:7860`. Start it with:

```bash
python app.py
```

**Input field:** A text box labeled "Your question" where the student types a plain-language question about Alfred University professors.

**Output fields:** Two boxes — "Answer" displays the LLM-generated response with inline source citations; "Retrieved from" lists the source document filenames the answer was drawn from.

**Sample interaction transcript:**

Input: "Is Sarah Cote a good choice for Writing 1 or Writing 2?"

Answer: "According to the reviews, Sarah Cote is a good choice for Writing 1 and Writing 2. One review states, 'Compared to some of my friends I would say Sarah is the best choice for Writing 1 and probably Writing 2 on this whole campus' (Source: prof_sarah_cote.txt)."

Retrieved from:
- prof_sarah_cote.txt
- prof_lynn_petrillo.txt

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Does Lynn Petrillo give good feedback on writing assignments? | Yes — multiple reviews mention clear feedback and that she reviews work before submission | Concluded "I don't have enough information" — retrieved general praise chunks but not the specific feedback chunks | Partially relevant | Inaccurate |
| 2 | Is Juliana Gray a harsh grader? | Yes — majority of reviews describe her as tough grader, strict attendance, harsh comments | Correctly answered yes with direct quotes from prof_juliana_gray.txt | Partially relevant (Sarah Cote appeared as top result) | Accurate |
| 3 | Is Joseph Petrillo good for students who struggle with math? | Yes — very accessible, explains clearly, willing to re-explain | Correctly answered yes with multiple specific citations | Relevant | Accurate |
| 4 | What do students say about Pam Schultz's lectures? | Mixed — some say she rambles, others find her kind and encouraging | Correctly described mixed opinions, cited rambling and positive reviews | Partially relevant (3 of 4 sources were wrong professors) | Partially accurate |
| 5 | Is Sarah Cote a good choice for Writing 1 or Writing 2? | Yes — consistently described as best writing professor on campus | Correctly answered yes with direct quote | Relevant | Accurate |

---
 
## Failure Case Analysis

**Question that failed:** "Does Lynn Petrillo give good feedback on writing assignments?"

**What the system returned:** The system concluded "I don't have enough information on that in the available reviews" — a refusal. This is incorrect because the `prof_lynn_petrillo.txt` document contains at least two reviews that directly mention feedback quality: "she also gives very helpful feedback" and "She gives good feedback and would even review your writing before you submitted."

**Root cause (tied to a specific pipeline stage):** This is a retrieval failure at the chunking stage. The phrase "gives very helpful feedback" is 31 characters long. Because of the 300-character chunk size and 50-character overlap, that phrase ended up embedded in a chunk that was dominated by other content — specifically general praise about her personality ("Lynn is probably the nicest person I've ever met"). When the query "does she give good feedback" was embedded and compared to stored chunks, the cosine similarity matched on general positivity and classroom atmosphere rather than on the specific concept of written feedback. The feedback-specific phrase was diluted by surrounding text and never surfaced in the top-5 results.

**What you would change to fix it:** Reducing chunk size to 150 characters would isolate individual sentences, making it more likely that "she gives very helpful feedback" lands in its own chunk with a strong semantic signal. Alternatively, sentence-level chunking (splitting on periods rather than fixed character counts) would keep each distinct opinion in its own embedding.

---

## Spec Reflection

**One way the spec helped you during implementation:** The chunking strategy section of `planning.md` forced a specific decision before writing any code — 300-character chunks with 50-character overlap. Having this written down meant that when implementing `chunk_text()`, there was no ambiguity about parameters. It also made it easy to recognize during evaluation that a chunk size issue was the root cause of the Q1 failure, because the spec had documented the reasoning behind the choice.

**One way your implementation diverged from the spec, and why:** The spec listed LangChain's `CharacterTextSplitter` as the chunking tool in the architecture diagram. During implementation, a custom sliding window loop was used instead — plain Python string slicing with `text[start:end]`. This happened because importing LangChain added unnecessary complexity and dependencies for what is fundamentally a simple operation. The custom implementation produces identical output and is easier to read and debug.

---

## AI Usage

**Instance 1**

- *What I gave the AI:* The domain description, the list of 10 document sources (professor names, file structure), and the chunking strategy section from planning.md specifying 300-character chunks and 50-character overlap.
- *What it produced:* `ingest.py` with four functions: `load_documents()`, `clean_document()`, `chunk_text()`, and `ingest()`. The cleaning approach used `.split("---", 1)` to strip the header above the first separator.
- *What I changed or overrode:* The initial version did not filter out chunks shorter than 20 characters. After running `python ingest.py` and seeing empty and near-empty chunks in the output, I added a `len(chunk) >= 20` filter inside `chunk_text()` to skip fragments that carry no semantic signal.

**Instance 2**

- *What I gave the AI:* The architecture diagram from planning.md (showing all-MiniLM-L6-v2 → ChromaDB → Groq), the retrieval approach section specifying top-k=5, and the grounding requirement (LLM must answer only from retrieved chunks, must cite sources, must refuse if context is insufficient).
- *What it produced:* `embed.py` and `query.py` with `build_vector_store()`, `retrieve()`, and `ask()` functions. The system prompt in `ask()` included the explicit grounding instruction.
- *What I changed or overrode:* The initial `ask()` function set `temperature=0.7`, which produced verbose and occasionally speculative responses. I reduced this to `temperature=0.2` to make the model more factual and less likely to elaborate beyond what the retrieved chunks actually say.
