"""Tests for the v2 evidence-based ranking pipeline."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recruitertwin.ranking_engine import features as F
from recruitertwin.ranking_engine.scorer_v2 import score_candidate
from recruitertwin.ranking_engine.reasoning import build_reasoning


def _base_candidate(**over):
    c = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "Senior ML Engineer",
            "summary": "Built embeddings-based retrieval and ranking systems in production.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Senior Machine Learning Engineer",
            "current_company": "ProductCo",
            "current_company_size": "201-500",
            "current_industry": "SaaS",
        },
        "career_history": [{
            "company": "ProductCo", "title": "Senior ML Engineer",
            "start_date": "2021-01-01", "end_date": None, "duration_months": 65,
            "is_current": True, "industry": "SaaS", "company_size": "201-500",
            "description": ("Shipped a hybrid search and ranking system with FAISS and "
                            "Elasticsearch; built NDCG/MRR evaluation framework and A/B tests; "
                            "deployed sentence-transformer embeddings to production."),
        }],
        "education": [], "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 80},
        ],
        "redrob_signals": {
            "last_active_date": "2026-06-01", "recruiter_response_rate": 0.7,
            "open_to_work_flag": True, "interview_completion_rate": 0.9,
            "notice_period_days": 30, "verified_email": True, "verified_phone": True,
            "willing_to_relocate": True,
        },
    }
    c.update(over)
    return c


class TestScorer(unittest.TestCase):
    def test_strong_candidate_scores_high(self):
        r = score_candidate(_base_candidate())
        self.assertFalse(r["honeypot"])
        self.assertGreater(r["final_score"], 0.5)

    def test_keyword_stuffer_penalized(self):
        c = _base_candidate()
        c["profile"]["current_title"] = "Marketing Manager"
        c["profile"]["headline"] = "Marketing Manager"
        c["profile"]["summary"] = "Marketing campaigns and brand strategy."
        c["career_history"][0]["title"] = "Marketing Manager"
        c["career_history"][0]["description"] = "Ran email marketing campaigns."
        c["skills"] = [{"name": n, "proficiency": "advanced", "endorsements": 5,
                        "duration_months": 24}
                       for n in ["FAISS", "Pinecone", "RAG", "LLM", "Embedding", "Ranking"]]
        r = score_candidate(c)
        strong = score_candidate(_base_candidate())
        self.assertLess(r["final_score"], strong["final_score"] * 0.3)
        self.assertTrue(any("non-engineering" in p or "stuffer" in p for p in r["penalties"]))

    def test_honeypot_skill_duration_exceeds_career(self):
        c = _base_candidate()
        c["profile"]["years_of_experience"] = 3.0
        c["skills"] = [{"name": "Python", "proficiency": "expert",
                        "endorsements": 1, "duration_months": 120}]
        flags = F.honeypot_flags(c)
        self.assertIn("skill_duration_exceeds_career", flags)
        self.assertEqual(score_candidate(c)["final_score"], 0.0)

    def test_inactive_candidate_downweighted(self):
        c = _base_candidate()
        c["redrob_signals"]["last_active_date"] = "2025-09-01"
        c["redrob_signals"]["recruiter_response_rate"] = 0.05
        r_inactive = score_candidate(c)
        r_active = score_candidate(_base_candidate())
        self.assertLess(r_inactive["final_score"], r_active["final_score"])

    def test_consulting_only_penalized(self):
        c = _base_candidate()
        c["career_history"][0]["company"] = "Infosys"
        r = score_candidate(c)
        self.assertTrue(any("consulting" in p for p in r["penalties"]))

    def test_reasoning_is_specific_and_varied(self):
        r = score_candidate(_base_candidate())
        text = build_reasoning(r, 1)
        self.assertIn("7.0 yrs", text)
        self.assertIn("Senior Machine Learning Engineer", text)
        c2 = _base_candidate(candidate_id="CAND_0000002")
        text2 = build_reasoning(score_candidate(c2), 50)
        self.assertNotEqual(text, text2)


class TestSampleData(unittest.TestCase):
    def test_pipeline_on_bundled_sample(self):
        sample = ROOT / "data" / "sample" / "redrob_sample_candidates.json"
        cands = json.loads(sample.read_text())
        results = [score_candidate(c) for c in cands]
        self.assertEqual(len(results), len(cands))
        scores = sorted({r["final_score"] for r in results}, reverse=True)
        self.assertGreater(len(scores), 5, "model must differentiate candidates")


if __name__ == "__main__":
    unittest.main()


class TestEmbeddingFallback(unittest.TestCase):
    def test_dense_layer_always_available(self):
        """The lightweight embedder must work with no downloaded model
        (LSA backend) and return one similarity per input text."""
        from recruitertwin.ranking_engine.embedder import (
            LightweightEmbedder, semantic_similarities,
        )
        emb = LightweightEmbedder()
        self.assertIn(emb.backend, ("minilm", "lsa"))
        texts = ["built embeddings retrieval with faiss vector search",
                 "professional pastry chef and baker",
                 "ranking systems ndcg evaluation production"]
        sims = semantic_similarities(texts, "embeddings retrieval ranking faiss")
        self.assertEqual(len(sims), len(texts))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in sims))


class TestEmbedderFallback(unittest.TestCase):
    def test_relevant_text_scores_higher(self):
        from recruitertwin.ranking_engine import embedder
        sims = embedder.semantic_similarities(
            ["search ranking retrieval embeddings systems engineer",
             "cooking recipes and restaurant kitchen management"],
            "search ranking retrieval embeddings")
        self.assertEqual(len(sims), 2)
        self.assertGreaterEqual(sims[0], sims[1])


class TestFaissVectorStore(unittest.TestCase):
    def test_search_and_similarities(self):
        import numpy as np
        from recruitertwin.ranking_engine.vector_store import FaissVectorStore
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((20, 8)).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store = FaissVectorStore(dim=8)
        store.add(vecs)
        self.assertEqual(store.ntotal, 20)
        q = vecs[3]
        ids, scores = store.search(q, k=5)
        self.assertEqual(int(ids[0]), 3)           # exact match first
        self.assertAlmostEqual(float(scores[0]), 1.0, places=4)
        sims = store.similarities(q)
        self.assertEqual(len(sims), 20)
        self.assertAlmostEqual(float(sims[3]), 1.0, places=4)
