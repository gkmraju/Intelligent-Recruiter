#!/usr/bin/env python3
"""Convenience wrapper: rank the bundled sample and write a demo CSV.

For the full export run:  python rank.py --candidates candidates.jsonl --out ranked.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_recruiter.ranking_engine.scorer_v2 import score_candidate  # noqa: E402
from intelligent_recruiter.ranking_engine.reasoning import build_reasoning  # noqa: E402
from intelligent_recruiter.ranking_engine.pipeline_v2 import write_submission  # noqa: E402

sample = ROOT / "data" / "sample" / "ranking_sample_candidates.json"
cands = json.loads(sample.read_text())
scored = [score_candidate(c) for c in cands if not score_candidate(c)["honeypot"]]
scored.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
rows = [{"candidate_id": r["candidate_id"], "rank": i, "score": r["final_score"],
         "reasoning": build_reasoning(r, i)} for i, r in enumerate(scored[:20], 1)]
out = ROOT / "submission" / "ranked_shortlist.csv"
write_submission(rows, out)
print(f"Wrote {len(rows)} rows to {out}")
