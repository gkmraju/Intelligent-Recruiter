# RecruiterTwin-AI — Implementation Documentation

Candidate-ranking system for the Redrob "India Runs on Data & AI" challenge. Given the Senior AI Engineer job description and a pool of 100,000 candidate profiles, the system produces a ranked shortlist of the top 100 candidates as a submission CSV — CPU-only, no network calls at rank time, finishing in about one minute (well inside the 5-minute / 16 GB budget).

---

## 1. What has been implemented

The pipeline is a **two-stage hybrid retrieval and ranking system**:

**Stage 1 — Recall (streaming evidence scoring).** A single streaming pass over `candidates.jsonl` scores every candidate with a cheap, rule-and-evidence-based scorer (must-have skill coverage, career trajectory, experience fit, location fit, behavioral availability signals). Honeypot profiles and zero-score candidates are dropped. A min-heap keeps only the top ~1,500 candidates in memory, so the full 100K pool never has to be loaded at once.

**Stage 2 — Precision (hybrid lexical + dense re-ranking).** The 1,500-candidate shortlist is re-ranked against the JD query text using three signals blended together:

1. **BM25 (Okapi)** — term-saturation and document-length-normalized lexical matching, so short and long profiles are compared fairly.
2. **TF-IDF cosine (1–2 grams)** — phrase-level lexical matching ("hybrid search", "learning to rank").
3. **Lightweight embeddings + FAISS** — each shortlist profile and the JD are embedded into a shared dense vector space; the vectors are indexed in a FAISS `IndexFlatIP` store and the JD vector is used as the query to get cosine similarity for every candidate. This catches semantically strong candidates whose wording doesn't lexically match the JD.

The dense similarity gets 50% of the blend weight, BM25 30%, TF-IDF 20%, and the blended lexical/semantic score is added as a bounded boost on top of the Stage-1 evidence score. The final top 100 get a human-readable reasoning string and are written as a spec-compliant CSV (non-increasing scores, unique ranks 1–100, candidate_id tie-breaking).

**Embedding backends.** The embedder picks automatically:

- *MiniLM* (`all-MiniLM-L6-v2`, 384-dim) if the model has been downloaded once via `scripts/download_model.py` — best quality.
- *LSA fallback* (always available, no downloads) — TF-IDF projected to a dense 256-dim space with TruncatedSVD and L2-normalized. Pure scikit-learn, deterministic (fixed seed), builds in ~1 s.

Both backends emit L2-normalized vectors, so FAISS inner-product search equals cosine similarity. If `faiss` itself is missing, the vector store transparently falls back to a NumPy matrix product with the identical API.

---

## 2. File and module roles

| File | Role |
|---|---|
| `rank.py` | Command-line entry point. The single reproduction command: reads candidates, runs the pipeline, writes the submission CSV. |
| `src/recruitertwin/ranking_engine/pipeline_v2.py` | Orchestrator. Streams candidates (Stage 1), runs BM25 + TF-IDF + embeddings/FAISS re-ranking (Stage 2), enforces submission-spec score rules, writes the CSV. Accepts both `.jsonl`(.gz) and JSON-array files. |
| `src/recruitertwin/ranking_engine/scorer_v2.py` | Stage-1 hybrid scorer: evidence-weighted JD fit × behavioral availability × penalty multipliers. Flags honeypots. |
| `src/recruitertwin/ranking_engine/features.py` | Feature extraction from a raw candidate record: must-have/nice-to-have skill evidence, career trajectory, location fit, experience fit, behavioral signals, honeypot flags, and the concatenated narrative text used for retrieval. |
| `src/recruitertwin/ranking_engine/embedder.py` | `LightweightEmbedder` (MiniLM or LSA backend) plus `semantic_similarities()`, which encodes the shortlist and JD and queries them through the FAISS store. |
| `src/recruitertwin/ranking_engine/vector_store.py` | `FaissVectorStore`: exact cosine search over normalized embeddings via `faiss.IndexFlatIP`, with `add()`, `search(query, k)` and `similarities(query)` methods, and a NumPy fallback. |
| `src/recruitertwin/ranking_engine/reasoning.py` | Builds the one-line, recruiter-readable reasoning string for each of the final 100 rows. |
| `src/recruitertwin/job_intelligence/jd_profile.py` | The parsed JD: must-have axes (embeddings/retrieval, vector-search infra, ranking systems, evaluation), nice-to-haves, disqualifiers, location preferences, and the `JD_QUERY_TEXT` used as the retrieval query. |
| `scripts/download_model.py` | One-time, internet-required step that saves MiniLM to `models/` so rank-time stays offline. Optional — LSA fallback works without it. |
| `tests/test_ranker.py` | Unit tests: scorer behavior, honeypot filtering, submission format, embedder backends, and FAISS store search correctness (11 tests). |
| `app/streamlit_app.py` | Optional Streamlit demo UI for exploring the ranked shortlist. |
| `submission/team_recruitertwin.csv` | The generated submission (100 rows, validated). |

---

## 3. Data files (challenge zip)

| File | Role |
|---|---|
| `candidates.jsonl` | **The real input.** 100,000 candidate records, one JSON object per line (~487 MB). Use this for every actual submission. |
| `sample_candidates.json` | A 50-candidate JSON array for smoke-testing only. After honeypot/zero-score filtering, roughly 43 candidates survive — so it can never produce a 100-row submission. Use it to check the code runs, never to generate a submission. |
| `job_description.docx` | The Senior AI Engineer JD the ranking targets (already encoded in `jd_profile.py`). |
| `candidate_schema.json` | Schema of a candidate record. |
| `sample_submission.csv` | Example of the required output format. |
| `validate_submission.py` | Official validator: exactly 100 data rows, ranks 1–100 unique, scores non-increasing, ID format `CAND_XXXXXXX`, tie-break rules. Run it on every submission before uploading. |

Each candidate record contains a `profile` block (headline, summary, years of experience, location, current role), `career_history` entries with free-text role descriptions, plus skills and behavioral fields — the narrative text from these is what BM25/TF-IDF/embeddings retrieve against.

---

## 4. How to run

**Setup (once):**

```bash
pip install -r requirements.txt
# optional, needs internet once — upgrades the dense layer from LSA to MiniLM:
python scripts/download_model.py
```

**Smoke test on the sample (sanity check only, ~2 s):**

```bash
python rank.py --candidates path/to/sample_candidates.json --out /tmp/sample_out.csv --top 20
```

**Full run on the 100K pool (~60 s, ~0.3 GB RAM):**

```bash
python rank.py --candidates path/to/candidates.jsonl --out submission/team_recruitertwin.csv
```

**Validate before submitting:**

```bash
python validate_submission.py submission/team_recruitertwin.csv
# expected output: "Submission is valid."
```

**Run the test suite:**

```bash
python -m pytest tests/ -q
# expected: 11 passed
```

Useful flags on `rank.py`: `--top N` (rows to emit, default 100), `--shortlist N` (Stage-1 shortlist size, default 1500), `--quiet`.

---

## 5. Verified results

- Full 100,000-candidate run: **63 seconds, 0.27 GB peak memory**, exit 0.
- Output: exactly 100 unique candidates, ranks 1–100, scores non-increasing — passes the official validator.
- Deterministic: same input always yields the same ranking (no randomness; fixed SVD seed; candidate_id tie-breaks).
- All 11 unit tests pass, including FAISS search correctness and embedder-backend tests.
- Top of the ranking is dominated by Senior/Staff ML Engineers with documented embeddings/retrieval, vector-search-infrastructure, and ranking-systems experience in or near the preferred Pune/Noida hubs — consistent with the JD's stated must-haves.
