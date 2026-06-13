# RecruiterTwin-AI Work Summary

Date: 13 June 2026

## Dashboard Enhancements

- Added a reusable `render_processing_hud(...)` function in `app/streamlit_app.py`.
- Built a Matrix / Mission-Control style live processing HUD for candidate ranking.
- Added animated digital rain, scanline effects, glass-style stat cards, and active pipeline stage chips.
- HUD shows:
  - Scanned candidates
  - Shortlisted candidates
  - Honeypots filtered
  - Elapsed time
  - Current pipeline stage
- Pipeline stages shown:
  - Parser
  - Feature Builder
  - Filter Engine
  - Fast Ranker
  - Honeypot Detector
  - Reranker
  - Reasoning Writer
  - CSV Validator

## Ranking Flow Integration

- Integrated the HUD into `main_file_rank()` using a Streamlit placeholder.
- Updated the existing `on_progress(event)` callback to refresh the HUD in place.
- Added HUD support for sample/upload ranking during the scoring loop.
- Kept the original Streamlit progress bar and live stats panel.
- Cleared the HUD after ranking completes so the final results remain clean.

## UI And Theme Improvements

- Converted the dashboard toward a full dark cyber theme.
- Added Streamlit native dark theme settings in `.streamlit/config.toml`.
- Styled the app shell, hero section, metric cards, buttons, sliders, inputs, select boxes, alerts, progress area, and result panels.
- Fixed hero text formatting and casing.
- Added line breaks for the hero subtitle:
  - Dream Team Engineers
  - Secure | Speed | Scalable

## Ranked Shortlist Table Improvements

- Replaced the plain Streamlit shortlist dataframe with a custom styled HTML table.
- Added horizontal scrolling.
- Added sticky, readable table headers.
- Replaced raw CSV-style column names with cleaner labels:
  - Rank
  - Candidate ID
  - Score
  - Title
  - Company
  - YOE
  - Location Fit
  - Risk Flags
  - Reason
- Center-aligned Rank, Score, and YOE.
- Improved the Reason column with better width, wrapping, spacing, and dark-table styling.
- Escaped table values before rendering to keep HTML output safe.

## Speed Optimizations

### UI-Level Speedups

- Removed the artificial sample HUD delay by setting:

```python
MIN_PREVIEW_HUD_SECONDS = 0.0
```

- Throttled Streamlit UI updates during the main 100k ranking run.
- Throttled sample/upload HUD updates to reduce unnecessary re-rendering.
- This reduces dashboard overhead while keeping live progress visible.

### Ranking Engine Speedups

- Updated `src/recruitertwin/ranking_engine/pipeline_v2.py`.
- Added optional `orjson` support for faster JSON and JSONL parsing.
- Changed candidate file parsing to binary mode for faster line reads.
- Kept a safe fallback to Python's standard `json` module.
- Added `orjson>=3.10` to `requirements.txt`.
- Updated `src/recruitertwin/ranking_engine/scorer_v2.py` to carry `evidence_text` forward.
- Reused `evidence_text` in the pipeline instead of rebuilding candidate text for shortlisted records.
- Updated `src/recruitertwin/ranking_engine/features.py` to use `date.fromisoformat(...)` instead of slower `datetime.strptime(...)` for ISO dates.

## Measured Results

- Full main ranking run tested with:
  - `top_n=100`
  - `shortlist_size=5000`
  - Fast mode / no dense embeddings
  - Main `candidates.jsonl` file

Result:

```text
rows=100 elapsed=78.12s
top=CAND_0018499 score=0.984838
```

## Verification Completed

- Python syntax compilation passed for edited files.
- Ranking tests passed:

```text
10 passed
```

- Streamlit servers were restarted and verified:
  - `http://localhost:8501`
  - `http://localhost:8502`

## Practical Speed Notes

- Fast mode remains the best option for quick end-to-end ranking.
- The biggest runtime cost is still the full 100k candidate parse + score pass.
- Lowering Stage-1 Shortlist from `5000` closer to `1500-2500` can reduce reranking time.
- Larger shortlist sizes improve recall but make reranking heavier.
- Dashboard animations now have minimal impact compared with the parsing/ranking engine itself.
