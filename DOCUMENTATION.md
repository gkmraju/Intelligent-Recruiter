# Intelligent Recruiter Documentation

Intelligent Recruiter is a CPU-only candidate-ranking prototype for Senior AI Engineer hiring workflows. Given a role profile and a large candidate pool, it produces a deterministic top-100 CSV export with rank, score, and fact-grounded reasoning.

## Implemented Pipeline

The engine uses a two-stage hybrid retrieval and ranking design.

**Stage 1: Recall.** A streaming pass over `candidates.jsonl` scores every candidate with deterministic evidence features: required concept coverage, career trajectory, experience fit, location fit, behavioral availability, and profile plausibility. Zero-score and implausible profiles are removed. A min-heap keeps the top shortlist in memory.

**Stage 2: Precision.** The shortlist is re-ranked with:

1. **BM25 Okapi** for term saturation and document-length normalization.
2. **TF-IDF 1-2 grams** for phrase-level lexical matches.
3. **MiniLM or LSA embeddings** queried through FAISS, with a NumPy fallback.

The final rows are sorted deterministically and exported with non-increasing scores and unique ranks.

## Module Roles

| File | Role |
|---|---|
| `rank.py` | CLI entry point for ranked CSV export. |
| `src/intelligent_recruiter/ranking_engine/pipeline_v2.py` | Streaming orchestration, shortlist management, hybrid re-rank, CSV writer. |
| `src/intelligent_recruiter/ranking_engine/scorer_v2.py` | Stage-1 scoring and penalty multipliers. |
| `src/intelligent_recruiter/ranking_engine/features.py` | Evidence extraction, behavior signals, location fit, experience fit, plausibility checks. |
| `src/intelligent_recruiter/ranking_engine/embedder.py` | MiniLM or LSA semantic similarity layer. |
| `src/intelligent_recruiter/ranking_engine/vector_store.py` | FAISS exact cosine search with NumPy fallback. |
| `src/intelligent_recruiter/ranking_engine/reasoning.py` | Fact-grounded reasoning for exported rows. |
| `src/intelligent_recruiter/job_intelligence/jd_profile.py` | Role concept profile and retrieval query text. |
| `app/streamlit_app.py` | Interactive dashboard prototype. |
| `tests/test_ranker.py` | Unit tests for scoring, filters, reasoning, embeddings, and vector search. |

## Data Files

| File | Role |
|---|---|
| `candidates.jsonl` | Main 100,000-candidate input file. |
| `data/sample/ranking_sample_candidates.json` | 50-candidate smoke-test fixture. |
| `data/contracts/ranked_output_columns.csv` | Ranked-output schema contract. |
| `validate_submission.py` | CSV validation utility. |

## Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main export:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission/team_2892.csv
```

Run a sample smoke test:

```bash
python rank.py --candidates ./data/sample/ranking_sample_candidates.json --out ./submission/sample.csv --top 20 --embedding-backend none
```

Validate the main export:

```bash
python validate_submission.py ./submission/team_2892.csv
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Runtime Profile

- Main pool size: 100,000 candidates
- Target runtime: under 5 minutes on CPU
- Observed runtime: about 65-70 seconds on a laptop-class CPU
- Memory: under 2 GB in normal runs
- Network during ranking: none
- GPU during ranking: none
