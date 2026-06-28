# Ranked export artifacts

- `team_2892.csv` is the top-100 ranking of the full 100K pool, produced by:

  ```bash
  python rank.py --candidates candidates.jsonl --out submission/team_2892.csv
  ```

  It validates with:

  ```bash
  python validate_submission.py submission/team_2892.csv
  ```

- `ranked_shortlist.csv` is a small demo output from:

  ```bash
  python scripts/generate_shortlist.py
  ```
