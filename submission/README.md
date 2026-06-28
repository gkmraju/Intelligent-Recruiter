# Ranked export artifacts

- `intelligent_recruiter.csv` is the top-100 ranking of the full 100K pool, produced by:

  ```bash
  python rank.py --candidates candidates.jsonl --out submission/intelligent_recruiter.csv
  ```

  It validates with:

  ```bash
  python validate_submission.py submission/intelligent_recruiter.csv
  ```

- `ranked_shortlist.csv` is a small demo output from:

  ```bash
  python scripts/generate_shortlist.py
  ```
