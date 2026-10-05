# DevAssist — grounded troubleshooting with hybrid RAG

DevAssist helps developers solve technical problems by retrieving relevant Stack Overflow question-and-answer passages, reranking them, generating a grounded response, and displaying verifiable source links. It is designed for the DevAssist hackathon instruction manual and implements every mandatory pipeline stage plus hybrid-search and CI bonuses.

> **Submission status:** code, small licensed sample, 25-query evaluation set, architecture diagram, tests, Dockerfile and UI are included. Before submitting, add your LLM settings, build the index, run evaluation, paste your real metrics into this README, record a backup demo, and replace the team placeholders.

## 1. Problem statement

Developers lose time opening several Stack Overflow threads, separating relevant answers from weak answers and checking whether a suggested fix is trustworthy. A normal chatbot can hallucinate or hide its evidence. DevAssist instead answers from an indexed Stack Overflow subset, cites the retrieved posts and declines when the evidence is insufficient.

**Target users:** Python, data and container developers.  
**Quality goals:** correct, grounded, referenced and honest answers.

## 2. Architecture

![DevAssist architecture](docs/architecture.svg)

There are two pipelines:

1. **Offline ingestion:** Stack Overflow questions/answers → HTML cleaning and join → token-aware chunks → normalized embeddings → persistent Qdrant collection. A BM25 index is built over the identical chunks.
2. **Online query:** clean query → vector and BM25 retrieval → Reciprocal Rank Fusion (RRF) → cross-encoder reranking → top-five context → grounded LLM prompt → answer plus metadata-built sources.

Source IDs and URLs never come from the LLM. They are copied from the retrieved chunk metadata in `generation/citations.py`.

## 3. Dataset

The repository contains a same-day-friendly subset downloaded through the Stack Exchange API. The API provides a recent Stack Overflow dataset while preserving the required question IDs, answer IDs, tags, scores, HTML bodies, authors and links.

| Item | Value |
|---|---:|
| Questions with positive-score answers | 196 |
| Selected answers | 392 |
| Tags requested | Python, Pandas, NumPy, Docker, TensorFlow |
| Selection | top 40 by votes/tag; at most 2 positive-score answers/question |
| Raw format | separate `questions.jsonl` and `answers.jsonl` |
| Join key | `answers.question_id = questions.question_id` |

The fetch date and settings are recorded in `data/raw/manifest.json`. Stack Overflow content is CC BY-SA as applicable; source IDs, post links and author names are retained for attribution. The code is MIT-licensed, but the Q&A content is not relicensed.

## 4. Data preprocessing and chunking

`ingestion/preprocess.py`:

- joins each answer to its question through `question_id`;
- keeps only positive-score answers and prioritizes accepted/high-score answers;
- converts HTML entities and removes markup;
- preserves `<pre>` and `<code>` content inside readable code fences;
- writes self-contained question-answer documents with IDs, title, tags, scores, authors and source URLs.

`ingestion/chunking.py` uses the selected embedding model's tokenizer. With MiniLM, chunks are **220 tokens with 30-token overlap**, below the model's 256-token input limit. Every chunk starts with the question title, carries parent metadata and receives stable ID `answer_id * 1000 + chunk_index`. Paragraphs/code blocks are preferred boundaries; only an over-limit unit is forcibly split.

These values are a latency/quality compromise for the included CPU-friendly baseline. Compare at least one second configuration before the final report if time permits.

## 5. Embedding model

Default: `sentence-transformers/all-MiniLM-L6-v2`.

- 384-dimensional vectors keep the local index small.
- It runs acceptably on CPU and is reliable for a live hackathon demo.
- Its 256-token limit directly determines the 220-token chunk setting.
- Both documents and queries use the identical model with normalized embeddings.

A stronger alternative is `BAAI/bge-small-en-v1.5` with a 512-token limit, but changing the model requires rebuilding the full index and retesting chunk size.

## 6. Vector database

DevAssist uses **Qdrant local mode**. Each point stores a normalized vector plus the exact chunk text and citation metadata. Qdrant was chosen because it provides persistent local storage, payloads, stable IDs, cosine search and a path to server/cloud deployment without changing the data model.

The application only reads an existing collection. Index construction remains in `ingestion/build_index.py`, so Streamlit does not rebuild data on every start.

## 7. Retrieval strategy

The first stage retrieves 20 candidates:

- **Semantic search:** normalized query embedding against Qdrant cosine similarity.
- **BM25:** exact technical-token matching over the same chunks. Its tokenizer preserves symbols in strings such as `torch.cuda`, `--no-cache`, `HTTP-422` and `numpy.ndarray`.
- **Fusion:** RRF combines ranks rather than adding incompatible vector/BM25 score scales.
- **Diversity:** no question contributes more than two chunks to the final candidate list.

For evaluation, the code can disable hybrid mode and compare vector-only against hybrid retrieval.

## 8. Reranking

Default: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

The cross-encoder jointly reads `(query, chunk)` pairs, assigns relevance logits to the 20 candidates and passes the best 5 to generation. Loading happens once through Streamlit resource caching. The top reranker score is also checked against a configurable insufficient-evidence threshold. **Calibrate `RERANK_MIN_SCORE` on answerable and unanswerable validation queries; do not present the default as a universal probability.**

## 9. LLM, grounding and failures

`generation/llm.py` supports any OpenAI-compatible chat endpoint configured through environment variables. The prompt:

- labels each context `[1]`, `[2]`, ... with real question and answer IDs;
- demands a likely cause, numbered fixes, verification and inline context citations;
- explicitly forbids outside knowledge and invented source IDs;
- permits an insufficient-information response;
- uses temperature `0.1` for consistency.

A code-level relevance gate declines weak queries before generation. Timeouts, empty output, missing configuration and missing indexes produce clear messages rather than tracebacks. Do not put real keys in the repository.

## 10. Source references

After generation, citation markers are validated against the supplied context range. `generation/citations.py` builds a de-duplicated source list only from chunk metadata, with links of the form:

- `https://stackoverflow.com/questions/<question_id>`
- `https://stackoverflow.com/a/<answer_id>`

This prevents the LLM from inventing attribution and satisfies both grounding and CC BY-SA attribution requirements.

## 11. Evaluation

`evaluation/test_set.jsonl` contains 25 realistic queries:

- 22 answerable queries across all five indexed technology areas;
- 3 intentionally unsupported queries (FastAPI, PyTorch and Kubernetes) for the refusal path;
- manually verified relevant question IDs as retrieval ground truth.

Retrieval metrics are counted at unique question-ID level: Recall@5, Precision@5 and MRR. Run:

```bash
python -m evaluation.run_retrieval
```

Then manually score generation with `evaluation/manual_generation_scores.csv`, where both team members independently assign 0–2 for faithfulness, answer relevance and context relevance.

### Real results — replace after running

Do **not** submit invented numbers. Paste the output from your machine here.

| Configuration | Recall@5 | Precision@5 | MRR | Faithfulness /2 | Answer relevance /2 |
|---|---:|---:|---:|---:|---:|
| Vector search | TODO | TODO | TODO | — | — |
| Vector + reranker | TODO | TODO | TODO | TODO | TODO |
| Hybrid search | TODO | TODO | TODO | — | — |
| Hybrid + reranker | TODO | TODO | TODO | TODO | TODO |

## 12. Run locally from a fresh clone

Use Python 3.11. Commands below are for macOS/Linux; on Windows use `.venv\Scripts\activate`.

```bash
git clone <YOUR_REPOSITORY_URL>
cd devassist
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set LLM_API_KEY, LLM_MODEL, and optionally LLM_BASE_URL.

# The small raw sample is already included. Build clean documents and index:
python -m ingestion.preprocess
python -m ingestion.build_index

# Tests and evaluation:
python -m pytest -q
python -m evaluation.run_retrieval

# Application:
streamlit run app/streamlit_app.py
```

To refresh the Stack Overflow sample first:

```bash
python -m ingestion.fetch_stackoverflow --tags python pandas numpy docker tensorflow --per-tag 40 --max-answers 2
```

First model downloads require internet. Prebuild the index and start the app before the presentation.

## 13. OpenAI-compatible provider examples

Use a model/account you are authorized to use. Confirm the model name in that provider's dashboard.

```dotenv
# Standard OpenAI-compatible account
LLM_API_KEY=...
LLM_BASE_URL=
LLM_MODEL=<provider-model-name>

# Local Ollama exposing its OpenAI-compatible endpoint
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=<installed-ollama-model>
```

For a remote provider, never commit `.env`. For Ollama, pull and test the model before going offline.

## 14. Docker

Build the Qdrant index locally first, or add an ingestion stage/volume appropriate to your deployment. For a demonstration image:

```bash
docker build -t devassist .
docker run --rm -p 8501:8501 --env-file .env -v "$PWD/qdrant_data:/app/qdrant_data" devassist
```

Model caches can also be mounted to avoid downloading on each container start.

## 15. Repository map

```text
devassist/
├── data/                  # small raw sample, manifest and processed files
├── ingestion/             # fetch, clean/join, token chunking, embedding, Qdrant
├── retrieval/             # vector + BM25 + RRF
├── reranking/             # cross-encoder
├── generation/            # grounded prompt, LLM client, citation validation
├── evaluation/            # 25-query test set, metrics, scoring sheet, results
├── app/                   # Streamlit UI
├── docs/architecture.svg
├── tests/
├── pipeline.py
├── config.py
├── requirements.txt
├── Dockerfile
└── .env.example
```

## 16. Team and work split

Replace names before submission.

| Team member | Ownership |
|---|---|
| Member 1 — Retrieval lead | data check, preprocessing, chunking, embedding, Qdrant, retrieval, reranker, retrieval metrics |
| Member 2 — Product lead | LLM/prompt, source validation, Streamlit, README, diagram, generation scoring, presentation/demo |
| Both | integration checkpoints, 3 demo queries, secret scan, final GitHub submission |

## 17. Limitations and future work

- The included dataset is deliberately small and biased toward high-vote posts, so it is not representative of all developer questions.
- MiniLM truncates above 256 tokens; longer-context embedding models should be compared later.
- The score threshold is model-specific and must be calibrated.
- Manual answer-quality scoring is small-scale; RAGAS or a carefully checked LLM judge could automate it.
- Future extensions: query rewriting, metadata filters, FastAPI, cached embeddings, distributed Qdrant, documentation search and agentic tool routing.

## 18. Demo script (8 minutes)

1. **Problem (45s):** search is slow; chatbots may hallucinate; DevAssist answers with evidence.
2. **Architecture (90s):** point at `docs/architecture.svg`; distinguish offline ingestion and online query.
3. **Implementation (90s):** 220/30 chunks, MiniLM, Qdrant, hybrid RRF, 20→5 cross-encoder, low-temperature grounded prompt.
4. **Live demo (2m):** one Pandas exact-error query, one Docker query, then one unsupported FastAPI/PyTorch query.
5. **Evaluation (60s):** show real vector/hybrid/reranker comparison.
6. **Trust and close (45s):** source IDs come from metadata; refusal path; license attribution; next steps.

## 19. Submission checklist

- [ ] Replace team names and repository URL.
- [ ] Set a working LLM provider locally; keep `.env` untracked.
- [ ] Run preprocessing and indexing successfully.
- [ ] Run all 25 queries; paste real evaluation metrics above.
- [ ] Complete both-member generation scoring.
- [ ] Verify three source links manually.
- [ ] Test an unsupported query and an empty query.
- [ ] Run `python -m pytest -q` from a clean environment.
- [ ] Run or honestly label Docker support.
- [ ] Record a backup demo video.
- [ ] Search Git history for secrets before making the repository public.

## 20. Models, APIs and libraries

- Dataset/API: Stack Exchange API 2.3; Stack Overflow user contributions, CC BY-SA as applicable.
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Vector database: Qdrant local mode.
- Keyword retrieval: `rank-bm25`.
- Generation: team-selected OpenAI-compatible LLM (record the exact provider and model before submitting).
- UI: Streamlit.
- See `requirements.txt` for pinned Python library versions.
