"""RecruiterTwin-AI Streamlit dashboard."""

from __future__ import annotations

import io
import json
import sys
import time
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recruitertwin.ranking_engine.pipeline_v2 import rank_candidates  # noqa: E402
from recruitertwin.ranking_engine.reasoning import build_reasoning  # noqa: E402
from recruitertwin.ranking_engine.scorer_v2 import score_candidate  # noqa: E402

MAIN_CANDIDATES = ROOT / "candidates.jsonl"
SAMPLE_CANDIDATES = ROOT / "data" / "sample" / "redrob_sample_candidates.json"
ESTIMATED_MAIN_ROWS = 100_000
MAIN_UI_UPDATE_INTERVAL_SECONDS = 0.75
MIN_MAIN_UI_UPDATE_ROWS = 5_000
MIN_PREVIEW_HUD_SECONDS = 0.0
PREVIEW_UI_UPDATE_INTERVAL_SECONDS = 0.25
HUD_PIPELINE_STAGES = [
    "Parser",
    "Feature Builder",
    "Filter Engine",
    "Fast Ranker",
    "Honeypot Detector",
    "Reranker",
    "Reasoning Writer",
    "CSV Validator",
]
SHORTLIST_COLUMN_LABELS = {
    "rank": "Rank",
    "candidate_id": "Candidate ID",
    "score": "Score",
    "title": "Title",
    "company": "Company",
    "yoe": "YOE",
    "location_fit": "Location Fit",
    "risk_flags": "Risk Flags",
    "reasoning": "Reason",
}


def _hud_stage_index(stage: str) -> int:
    stage_key = (stage or "").strip().lower().replace("_", " ")
    aliases = {
        "parse": "Parser",
        "parser": "Parser",
        "scan": "Filter Engine",
        "scoring": "Fast Ranker",
        "feature builder": "Feature Builder",
        "filter": "Filter Engine",
        "filter engine": "Filter Engine",
        "fast ranker": "Fast Ranker",
        "honeypot": "Honeypot Detector",
        "honeypot detector": "Honeypot Detector",
        "shortlist ready": "Reranker",
        "rerank": "Reranker",
        "reranker": "Reranker",
        "complete": "Reasoning Writer",
        "reasoning": "Reasoning Writer",
        "reasoning writer": "Reasoning Writer",
        "output ready": "CSV Validator",
        "csv validator": "CSV Validator",
    }
    label = aliases.get(stage_key, stage)
    if label in HUD_PIPELINE_STAGES:
        return HUD_PIPELINE_STAGES.index(label)
    return 0


def render_processing_hud(
    stage: str,
    scanned: int = 0,
    shortlisted: int = 0,
    honeypots: int = 0,
    elapsed: float = 0.0,
) -> None:
    active_index = _hud_stage_index(stage)
    active_stage = HUD_PIPELINE_STAGES[active_index]
    rain_lines = [
        "01001100AI101101VECTOR",
        "PROFILE00101110EMBED",
        "MATCH11001001REASON",
        "NLP01101010EVIDENCE",
        "RANK100101FASTFIT",
        "FILTER0101HONEYPOT",
        "CSVVALIDATOR111000",
        "SIGNAL010011TRUST",
    ]
    rain_markup = "".join(f"<span>{escape(line)}</span>" for line in rain_lines)
    chips = "".join(
        f'<span class="hud-chip {"active" if index == active_index else "done" if index < active_index else ""}">'
        f"{escape(label)}</span>"
        for index, label in enumerate(HUD_PIPELINE_STAGES)
    )
    safe_stage = escape(active_stage)

    st.markdown(
        f"""
        <div class="processing-hud" role="status" aria-live="polite">
          <div class="hud-rain" aria-hidden="true">{rain_markup}</div>
          <div class="hud-scanline" aria-hidden="true"></div>
          <div class="hud-content">
            <div class="hud-header">
              <div>
                <div class="hud-kicker">Live Processing</div>
                <div class="hud-title">RecruiterTwin Neural Ranking Engine</div>
              </div>
              <div class="hud-stage">
                <span class="hud-pulse"></span>
                <span>{safe_stage}</span>
              </div>
            </div>
            <div class="hud-grid">
              <div class="hud-stat">
                <span>Scanned Candidates</span>
                <strong>{int(scanned):,}</strong>
              </div>
              <div class="hud-stat">
                <span>Shortlisted</span>
                <strong>{int(shortlisted):,}</strong>
              </div>
              <div class="hud-stat">
                <span>Honeypots Filtered</span>
                <strong>{int(honeypots):,}</strong>
              </div>
              <div class="hud-stat">
                <span>Elapsed Time</span>
                <strong>{float(elapsed):.1f}s</strong>
              </div>
            </div>
            <div class="hud-pipeline">{chips}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main_hud_stage(stage: str, scanned: int) -> str:
    if stage in {"shortlist_ready", "rerank"}:
        return "Reranker"
    if stage == "complete":
        return "Reasoning Writer"
    if stage == "output_ready":
        return "CSV Validator"
    scan_ratio = min(max(scanned / ESTIMATED_MAIN_ROWS, 0.0), 1.0)
    if scan_ratio < 0.08:
        return "Parser"
    if scan_ratio < 0.28:
        return "Feature Builder"
    if scan_ratio < 0.58:
        return "Filter Engine"
    if scan_ratio < 0.82:
        return "Fast Ranker"
    return "Honeypot Detector"


def sample_hud_stage(index: int, total: int, scoring_done: bool = False) -> str:
    if scoring_done:
        return "Reranker"
    ratio = min(max(index / max(total, 1), 0.0), 1.0)
    if ratio < 0.12:
        return "Parser"
    if ratio < 0.34:
        return "Feature Builder"
    if ratio < 0.58:
        return "Filter Engine"
    if ratio < 0.82:
        return "Fast Ranker"
    return "Honeypot Detector"


def set_source(source: str) -> None:
    st.session_state["candidate_source"] = source


def source_button(label: str, source: str, help_text: str) -> None:
    selected = st.session_state["candidate_source"] == source
    if st.button(
        label,
        key=f"source_{source}",
        type="primary" if selected else "secondary",
        width="stretch",
        help=help_text,
    ):
        set_source(source)
        st.rerun()


def render_metric(label: str, value: str, detail: str = "") -> None:
    safe_label = escape(str(label))
    safe_value = escape(str(value))
    safe_detail = escape(str(detail))
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{safe_label}</div>
          <div class="metric-value">{safe_value}</div>
          <div class="metric-detail">{safe_detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_timing_table(timings: list[dict[str, str]]) -> None:
    st.dataframe(pd.DataFrame(timings), width="stretch", hide_index=True)


def _format_shortlist_value(column: str, value: Any) -> str:
    if pd.isna(value):
        return "-"
    if column == "rank":
        return f"{int(value)}"
    if column == "score":
        return f"{float(value):.4f}"
    if column == "yoe":
        return f"{float(value):.1f}"
    text = " ".join(str(value).split())
    return text or "-"


def render_ranked_shortlist_table(df: pd.DataFrame, columns: list[str]) -> None:
    if df.empty:
        st.info("No ranked candidates to display.")
        return

    header_html = "".join(
        f'<th class="col-{escape(column.replace("_", "-"))}">{escape(SHORTLIST_COLUMN_LABELS.get(column, column.replace("_", " ").title()))}</th>'
        for column in columns
    )
    body_rows = []
    for _, row in df[columns].iterrows():
        cells = []
        for column in columns:
            formatted = _format_shortlist_value(column, row[column])
            safe_value = escape(formatted)
            class_name = f'col-{column.replace("_", "-")}'
            if column == "reasoning":
                cells.append(
                    f'<td class="{class_name}"><div class="reason-cell" title="{safe_value}">{safe_value}</div></td>'
                )
            else:
                cells.append(f'<td class="{class_name}">{safe_value}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    st.markdown(
        f"""
        <div class="shortlist-scroll">
          <table class="shortlist-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_download_payload(df: pd.DataFrame, output_choice: str) -> tuple[str, str, str]:
    if output_choice == "Detailed CSV":
        out_df = df.copy()
        return out_df.to_csv(index=False), "ranked_shortlist_detailed.csv", "text/csv"

    out_df = df[["candidate_id", "rank", "score", "reasoning"]]
    return out_df.to_csv(index=False), "ranked_shortlist.csv", "text/csv"


def parse_uploaded_candidates(raw: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw)
        candidates = payload if isinstance(payload, list) else [payload]
    except json.JSONDecodeError:
        candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return candidates[:100]


def sample_or_upload_rank(candidates: list[dict[str, Any]], top_n: int) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    scored: list[dict[str, Any]] = []
    progress = st.progress(0, text="Scoring candidate evidence")
    hud_placeholder = st.empty()
    total = max(len(candidates), 1)
    started = time.perf_counter()
    honeypots_seen = 0
    last_ui_update = started

    with hud_placeholder.container():
        render_processing_hud("Parser", elapsed=0.0)

    for index, candidate in enumerate(candidates, start=1):
        row = score_candidate(candidate)
        scored.append(row)
        if row["honeypot"]:
            honeypots_seen += 1
        kept_seen = index - honeypots_seen
        now = time.perf_counter()
        if index == total or now - last_ui_update >= PREVIEW_UI_UPDATE_INTERVAL_SECONDS:
            progress.progress(index / total, text=f"Scored {index:,} of {total:,} candidates")
            with hud_placeholder.container():
                render_processing_hud(
                    sample_hud_stage(index, total),
                    scanned=index,
                    shortlisted=kept_seen,
                    honeypots=honeypots_seen,
                    elapsed=now - started,
                )
            last_ui_update = now
    scored_at = time.perf_counter()

    kept = [row for row in scored if not row["honeypot"]]
    pots = [row for row in scored if row["honeypot"]]
    with hud_placeholder.container():
        render_processing_hud(
            sample_hud_stage(total, total, scoring_done=True),
            scanned=len(scored),
            shortlisted=len(kept),
            honeypots=len(pots),
            elapsed=time.perf_counter() - started,
        )
    kept.sort(key=lambda row: (-row["final_score"], row["candidate_id"]))
    top = kept[:top_n]
    filtered_at = time.perf_counter()

    rows = [
        {
            "rank": rank,
            "candidate_id": row["candidate_id"],
            "score": row["final_score"],
            "title": row["title"],
            "company": row["company"],
            "yoe": row["yoe"],
            "location_fit": row["location_label"],
            "risk_flags": "; ".join(row["penalties"]) or "-",
            "reasoning": build_reasoning(row, rank),
        }
        for rank, row in enumerate(top, start=1)
    ]
    df = pd.DataFrame(rows)
    reasoned_at = time.perf_counter()
    with hud_placeholder.container():
        render_processing_hud(
            "CSV Validator",
            scanned=len(candidates),
            shortlisted=len(kept),
            honeypots=len(pots),
            elapsed=reasoned_at - started,
        )
    progress.progress(1.0, text="Ranking complete")
    remaining_hud_time = MIN_PREVIEW_HUD_SECONDS - (time.perf_counter() - started)
    if remaining_hud_time > 0:
        time.sleep(remaining_hud_time)
    hud_placeholder.empty()

    stats = {
        "scanned": len(candidates),
        "shortlisted": len(kept),
        "skipped_honeypot": len(pots),
        "skipped_zero": 0,
        "emitted": len(df),
        "elapsed": reasoned_at - started,
        "timings": [
            {"Task": "Candidate scoring", "Time Taken": f"{scored_at - started:.2f}s"},
            {"Task": "Filtering and sorting", "Time Taken": f"{filtered_at - scored_at:.2f}s"},
            {"Task": "Reasoning generation", "Time Taken": f"{reasoned_at - filtered_at:.2f}s"},
        ],
    }
    pot_df = pd.DataFrame(
        [
            {
                "candidate_id": row["candidate_id"],
                "title": row["title"],
                "flags": "; ".join(row["penalties"]),
            }
            for row in pots
        ]
    )
    return df, stats, pot_df


def main_file_rank(
    top_n: int,
    shortlist_size: int,
    embedding_backend: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    progress = st.progress(0, text="Preparing ranking engine")
    hud_placeholder = st.empty()
    status = st.empty()
    stats_row = st.container()
    latest: dict[str, Any] = {
        "scanned": 0,
        "shortlisted": 0,
        "skipped_honeypot": 0,
        "skipped_zero": 0,
        "elapsed": 0.0,
    }
    stage_elapsed: dict[str, float] = {}
    last_ui_update: dict[str, Any] = {
        "elapsed": 0.0,
        "scanned": 0,
        "stage": "",
    }

    with hud_placeholder.container():
        render_processing_hud("Parser")

    def on_progress(event: dict[str, Any]) -> None:
        latest.update(event)
        stage = str(latest.get("stage", "scan"))
        stage_elapsed[stage] = float(latest.get("elapsed", 0.0))
        scanned = int(latest.get("scanned", 0))
        shortlisted = int(latest.get("shortlisted", 0))
        honeypots = int(latest.get("skipped_honeypot", 0))
        elapsed = float(latest.get("elapsed", 0.0))
        stage_changed = stage != last_ui_update["stage"]
        row_delta = scanned - int(last_ui_update["scanned"])
        time_delta = elapsed - float(last_ui_update["elapsed"])
        should_update_ui = (
            stage_changed
            or stage in {"shortlist_ready", "rerank", "complete", "output_ready"}
            or row_delta >= MIN_MAIN_UI_UPDATE_ROWS
            or time_delta >= MAIN_UI_UPDATE_INTERVAL_SECONDS
        )
        if not should_update_ui:
            return

        ratio = min(scanned / ESTIMATED_MAIN_ROWS, 1.0)
        if stage == "rerank":
            ratio = max(ratio, 0.92)
            label = "Hybrid re-ranking shortlist"
        elif stage in {"complete", "output_ready"}:
            ratio = 1.0
            label = "Ranking rows ready"
        elif stage == "shortlist_ready":
            ratio = max(ratio, 0.9)
            label = "Shortlist ready; preparing precision re-rank"
        else:
            label = f"Streaming candidate file: {scanned:,} scanned"

        progress.progress(ratio, text=label)
        with hud_placeholder.container():
            render_processing_hud(
                main_hud_stage(stage, scanned),
                scanned=scanned,
                shortlisted=shortlisted,
                honeypots=honeypots,
                elapsed=elapsed,
            )
        status.markdown(
            f"""
            <div class="run-panel">
              <span>Scanned <strong>{scanned:,}</strong></span>
              <span>Shortlist <strong>{shortlisted:,}</strong></span>
              <span>Honeypots <strong>{honeypots:,}</strong></span>
              <span>Elapsed <strong>{elapsed:.1f}s</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        last_ui_update.update({"elapsed": elapsed, "scanned": scanned, "stage": stage})

    with stats_row:
        ranked_rows = rank_candidates(
            MAIN_CANDIDATES,
            top_n=top_n,
            shortlist_size=shortlist_size,
            verbose=False,
            progress_callback=on_progress,
            embedding_backend=embedding_backend,
        )

    hud_placeholder.empty()
    df = pd.DataFrame(
        [
            {
                "rank": row["rank"],
                "candidate_id": row["candidate_id"],
                "score": row["score"],
                "reasoning": row["reasoning"],
            }
            for row in ranked_rows
        ]
    )
    latest["emitted"] = len(df)
    scan_done = stage_elapsed.get("shortlist_ready", stage_elapsed.get("scan", 0.0))
    rerank_done = stage_elapsed.get("complete", scan_done)
    output_done = stage_elapsed.get("output_ready", rerank_done)
    latest["timings"] = [
        {"Task": "Stage 1 scan and shortlist", "Time Taken": f"{scan_done:.2f}s"},
        {"Task": "Hybrid re-rank", "Time Taken": f"{max(0.0, rerank_done - scan_done):.2f}s"},
        {"Task": "Reasoning rows", "Time Taken": f"{max(0.0, output_done - rerank_done):.2f}s"},
    ]
    return df, latest


st.set_page_config(page_title="RecruiterTwin-AI", page_icon="RT", layout="wide")

st.markdown(
    """
    <style>
      :root {
        --rt-ink: #e8fff5;
        --rt-muted: #9fb7b0;
        --rt-panel: rgba(15, 23, 42, 0.78);
        --rt-line: rgba(142, 247, 191, 0.18);
        --rt-accent: #0f766e;
        --rt-accent-2: #35f59a;
        --rt-warm: #fbbf24;
        --rt-bg: #020617;
      }
      html, body, [data-testid="stAppViewContainer"], .main {
        background: #020617 !important;
        color: var(--rt-ink) !important;
      }
      .stApp {
        background:
          radial-gradient(circle at 16% 0%, rgba(20, 184, 166, 0.16), transparent 30%),
          radial-gradient(circle at 88% 12%, rgba(37, 99, 235, 0.18), transparent 28%),
          linear-gradient(180deg, #020617 0%, #071411 44%, #020617 100%) !important;
        color: var(--rt-ink);
      }
      [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
        background: transparent !important;
      }
      [data-testid="stSidebar"] {
        background: #020617 !important;
      }
      .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1240px;
      }
      .hero {
        border: 1px solid var(--rt-line);
        background:
          linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(5, 24, 20, 0.9) 48%, rgba(15, 23, 42, 0.88));
        border-radius: 8px;
        padding: 28px 30px;
        box-shadow:
          0 18px 45px rgba(0, 0, 0, 0.34),
          inset 0 1px 0 rgba(255, 255, 255, 0.06);
      }
      .hero h1 {
        color: var(--rt-ink);
        font-size: 2.45rem;
        line-height: 1.05;
        margin: 0 0 8px;
        letter-spacing: 0;
      }
      .hero .built {
        color: var(--rt-accent-2);
        font-size: 1rem;
        font-weight: 800;
        text-transform: none;
        letter-spacing: 0;
        margin-bottom: 14px;
        line-height: 1.45;
      }
      .hero p {
        color: var(--rt-muted);
        font-size: 1.02rem;
        margin: 0;
        max-width: 920px;
      }
      .section-title {
        color: var(--rt-ink);
        font-size: 1.15rem;
        font-weight: 800;
        margin: 22px 0 8px;
      }
      .stMarkdown, .stMarkdown p, .stCaptionContainer, label, [data-testid="stWidgetLabel"] {
        color: var(--rt-ink) !important;
      }
      div.stButton > button {
        background: rgba(15, 23, 42, 0.84) !important;
        color: #e8fff5 !important;
        border-radius: 8px;
        min-height: 58px;
        font-weight: 800;
        border: 1px solid rgba(142, 247, 191, 0.22) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
      }
      div.stButton > button:hover {
        border-color: rgba(53, 245, 154, 0.72) !important;
        color: #ffffff !important;
      }
      div.stButton > button[kind="primary"],
      [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #0f766e, #14b8a6) !important;
        border-color: rgba(53, 245, 154, 0.58) !important;
        color: #ffffff !important;
      }
      .metric-card {
        background: var(--rt-panel);
        border: 1px solid var(--rt-line);
        border-radius: 8px;
        padding: 18px 18px 16px;
        box-shadow:
          0 8px 24px rgba(0, 0, 0, 0.22),
          inset 0 1px 0 rgba(255, 255, 255, 0.05);
        min-height: 116px;
      }
      .metric-label {
        color: var(--rt-muted);
        font-size: 0.77rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0;
      }
      .metric-value {
        color: var(--rt-ink);
        font-size: 1.9rem;
        font-weight: 900;
        margin-top: 7px;
      }
      .metric-detail {
        color: var(--rt-muted);
        font-size: 0.86rem;
        margin-top: 4px;
      }
      [data-testid="stSlider"] {
        color: var(--rt-ink) !important;
      }
      [data-testid="stSlider"] [role="slider"] {
        background: #35f59a !important;
        border: 2px solid #d9fff0 !important;
        box-shadow: 0 0 14px rgba(53, 245, 154, 0.35);
      }
      [data-baseweb="select"] > div,
      [data-testid="stNumberInput"] input,
      [data-testid="stTextInput"] input,
      [data-testid="stFileUploader"] section,
      textarea {
        background: rgba(15, 23, 42, 0.86) !important;
        color: #e8fff5 !important;
        border-color: rgba(142, 247, 191, 0.22) !important;
      }
      [data-baseweb="select"] span,
      [data-testid="stNumberInput"] input,
      [data-testid="stTextInput"] input {
        color: #e8fff5 !important;
      }
      [data-testid="stNumberInput"] button {
        background: rgba(15, 23, 42, 0.9) !important;
        color: #e8fff5 !important;
        border-color: rgba(142, 247, 191, 0.2) !important;
      }
      [data-testid="stProgress"] > div {
        background: rgba(15, 23, 42, 0.74) !important;
      }
      [data-testid="stDataFrame"],
      [data-testid="stTable"] {
        border: 1px solid var(--rt-line);
        border-radius: 8px;
        overflow: hidden;
      }
      .shortlist-scroll {
        width: 100%;
        max-height: 520px;
        overflow: auto;
        border: 1px solid rgba(142, 247, 191, 0.18);
        border-radius: 8px;
        background: rgba(2, 6, 23, 0.72);
        box-shadow:
          0 18px 42px rgba(0, 0, 0, 0.24),
          inset 0 1px 0 rgba(255, 255, 255, 0.05);
      }
      .shortlist-scroll::-webkit-scrollbar {
        width: 10px;
        height: 10px;
      }
      .shortlist-scroll::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.84);
      }
      .shortlist-scroll::-webkit-scrollbar-thumb {
        background: rgba(53, 245, 154, 0.42);
        border-radius: 999px;
      }
      .shortlist-table {
        width: max-content;
        min-width: 1320px;
        border-collapse: collapse;
        table-layout: fixed;
        color: #e8fff5;
        font-size: 0.88rem;
      }
      .shortlist-table thead th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: #111827;
        color: #b8f7d1;
        border-bottom: 1px solid rgba(142, 247, 191, 0.2);
        border-right: 1px solid rgba(142, 247, 191, 0.12);
        padding: 11px 12px;
        text-align: left;
        font-weight: 900;
        letter-spacing: 0;
      }
      .shortlist-table tbody td {
        border-bottom: 1px solid rgba(148, 163, 184, 0.14);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
        padding: 10px 12px;
        vertical-align: top;
        color: #f8fffb;
        background: rgba(2, 6, 23, 0.58);
      }
      .shortlist-table tbody tr:nth-child(even) td {
        background: rgba(15, 23, 42, 0.38);
      }
      .shortlist-table tbody tr:hover td {
        background: rgba(20, 184, 166, 0.11);
      }
      .shortlist-table .col-rank,
      .shortlist-table .col-score,
      .shortlist-table .col-yoe {
        text-align: center;
        font-variant-numeric: tabular-nums;
      }
      .shortlist-table .col-rank { width: 80px; }
      .shortlist-table .col-candidate-id { width: 170px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
      .shortlist-table .col-score { width: 110px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
      .shortlist-table .col-title { width: 190px; }
      .shortlist-table .col-company { width: 170px; }
      .shortlist-table .col-yoe { width: 90px; }
      .shortlist-table .col-location-fit { width: 220px; }
      .shortlist-table .col-risk-flags { width: 170px; }
      .shortlist-table .col-reasoning { width: 520px; }
      .reason-cell {
        max-width: 500px;
        color: #d9fff0;
        line-height: 1.45;
        white-space: normal;
        overflow-wrap: anywhere;
      }
      [data-testid="stAlert"] {
        background: rgba(15, 23, 42, 0.84) !important;
        color: var(--rt-ink) !important;
        border-color: var(--rt-line) !important;
      }
      .run-panel {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        background: #0f172a;
        color: #e5edf7;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 10px 0 2px;
      }
      .run-panel span {
        padding-right: 14px;
        border-right: 1px solid rgba(255, 255, 255, 0.18);
      }
      .run-panel span:last-child {
        border-right: 0;
      }
      .processing-hud {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(37, 211, 102, 0.34);
        border-radius: 8px;
        margin: 8px 0 12px;
        background:
          radial-gradient(circle at 18% 10%, rgba(37, 211, 102, 0.18), transparent 30%),
          radial-gradient(circle at 86% 18%, rgba(37, 99, 235, 0.18), transparent 28%),
          linear-gradient(135deg, #020617 0%, #071411 48%, #0f172a 100%);
        box-shadow:
          0 18px 46px rgba(15, 23, 42, 0.24),
          inset 0 1px 0 rgba(255, 255, 255, 0.08);
        color: #e5fff2;
      }
      .processing-hud::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
          linear-gradient(rgba(37, 211, 102, 0.07) 1px, transparent 1px),
          linear-gradient(90deg, rgba(37, 211, 102, 0.05) 1px, transparent 1px);
        background-size: 26px 26px;
        mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.72), rgba(0, 0, 0, 0.12));
        pointer-events: none;
      }
      .processing-hud::after {
        content: "";
        position: absolute;
        top: -20%;
        bottom: -20%;
        left: -45%;
        width: 42%;
        background: linear-gradient(90deg, transparent, rgba(53, 245, 154, 0.16), transparent);
        transform: skewX(-14deg);
        animation: hudSweep 3.2s ease-in-out infinite;
        pointer-events: none;
      }
      .hud-rain {
        position: absolute;
        inset: 0;
        opacity: 0.34;
        color: #35f59a;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.7rem;
        line-height: 1;
        text-shadow: 0 0 9px rgba(53, 245, 154, 0.72);
        pointer-events: none;
      }
      .hud-rain span {
        position: absolute;
        top: -170%;
        writing-mode: vertical-rl;
        text-orientation: upright;
        white-space: nowrap;
        animation: hudRain 4.8s linear infinite;
      }
      .hud-rain span:nth-child(1) { left: 5%; animation-delay: -0.4s; }
      .hud-rain span:nth-child(2) { left: 18%; animation-delay: -2.1s; opacity: 0.82; }
      .hud-rain span:nth-child(3) { left: 31%; animation-delay: -1.2s; opacity: 0.68; }
      .hud-rain span:nth-child(4) { left: 46%; animation-delay: -3.4s; }
      .hud-rain span:nth-child(5) { left: 61%; animation-delay: -0.9s; opacity: 0.76; }
      .hud-rain span:nth-child(6) { left: 73%; animation-delay: -2.8s; opacity: 0.7; }
      .hud-rain span:nth-child(7) { left: 86%; animation-delay: -1.7s; }
      .hud-rain span:nth-child(8) { left: 95%; animation-delay: -3.9s; opacity: 0.62; }
      .hud-rain span:nth-child(2n) {
        animation-duration: 5.9s;
      }
      .hud-scanline {
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, transparent 0%, rgba(45, 212, 191, 0.24) 48%, transparent 100%);
        height: 36%;
        animation: hudScan 3.2s ease-in-out infinite;
        pointer-events: none;
      }
      .hud-content {
        position: relative;
        z-index: 1;
        padding: 18px;
        backdrop-filter: blur(10px);
      }
      .hud-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 14px;
      }
      .hud-kicker {
        color: #8ef7bf;
        font-size: 0.73rem;
        font-weight: 900;
        letter-spacing: 0;
        text-transform: uppercase;
      }
      .hud-title {
        color: #f8fffb;
        font-size: 1.28rem;
        font-weight: 900;
        margin-top: 3px;
        text-shadow: 0 0 18px rgba(53, 245, 154, 0.2);
      }
      .hud-stage {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        min-width: max-content;
        border: 1px solid rgba(142, 247, 191, 0.28);
        border-radius: 999px;
        background: rgba(2, 6, 23, 0.48);
        color: #d9fff0;
        padding: 8px 11px;
        font-size: 0.82rem;
        font-weight: 900;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
      }
      .hud-pulse {
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: #35f59a;
        box-shadow: 0 0 0 rgba(53, 245, 154, 0.72);
        animation: hudPulse 1.5s infinite;
      }
      .hud-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
      }
      .hud-stat {
        border: 1px solid rgba(255, 255, 255, 0.11);
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.46);
        padding: 12px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.07);
      }
      .hud-stat span {
        display: block;
        color: #9fb7b0;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0;
      }
      .hud-stat strong {
        display: block;
        color: #ffffff;
        font-size: 1.35rem;
        line-height: 1.1;
        margin-top: 7px;
      }
      .hud-pipeline {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 13px;
      }
      .hud-chip {
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.52);
        color: #9fb7b0;
        padding: 6px 9px;
        font-size: 0.73rem;
        font-weight: 800;
      }
      .hud-chip.done {
        border-color: rgba(37, 211, 102, 0.22);
        color: #b8f7d1;
      }
      .hud-chip.active {
        border-color: rgba(53, 245, 154, 0.72);
        background: rgba(20, 184, 166, 0.2);
        color: #ffffff;
        box-shadow: 0 0 18px rgba(37, 211, 102, 0.16);
      }
      @keyframes hudRain {
        0% { transform: translateY(-30%); }
        100% { transform: translateY(320%); }
      }
      @keyframes hudScan {
        0%, 100% { transform: translateY(-85%); opacity: 0; }
        18%, 72% { opacity: 0.9; }
        50% { transform: translateY(185%); }
      }
      @keyframes hudSweep {
        0%, 16% { transform: translateX(0) skewX(-14deg); opacity: 0; }
        38%, 62% { opacity: 0.9; }
        100% { transform: translateX(360%) skewX(-14deg); opacity: 0; }
      }
      @keyframes hudPulse {
        0% { box-shadow: 0 0 0 0 rgba(53, 245, 154, 0.72); }
        70% { box-shadow: 0 0 0 8px rgba(53, 245, 154, 0); }
        100% { box-shadow: 0 0 0 0 rgba(53, 245, 154, 0); }
      }
      @media (max-width: 760px) {
        .hud-header {
          flex-direction: column;
        }
        .hud-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
      }
      [data-testid="stMetricValue"] {
        color: var(--rt-ink);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>Recruiter Twin - Intelligent Ranking Console</h1>
  <div class="built">
    Dream Team Engineers<br>
    Secure | Speed | Scalable
  </div>
  <p>
    Rank the Senior AI Engineer candidate pool with evidence-based scoring,
    honeypot filtering, behavioral availability signals and hybrid retrieval.
  </p>
</div>
    """,
    unsafe_allow_html=True,
)

if "candidate_source" not in st.session_state:
    st.session_state["candidate_source"] = "main"

st.markdown('<div class="section-title">Choose Candidate Source</div>', unsafe_allow_html=True)
source_cols = st.columns(3)
with source_cols[0]:
    source_button("Use Main Data File", "main", "Stream candidates.jsonl from the project root.")
with source_cols[1]:
    source_button("Use Sample Data", "sample", "Run quickly against the bundled sample.")
with source_cols[2]:
    source_button("Upload File", "upload", "Upload a JSON or JSONL file with up to 100 candidates.")

source = st.session_state["candidate_source"]
using_main_file = source == "main"
candidates: list[dict[str, Any]] = []

st.markdown('<div class="section-title">Input Status</div>', unsafe_allow_html=True)
if using_main_file:
    if not MAIN_CANDIDATES.exists():
        st.error(f"Missing data file: {MAIN_CANDIDATES}")
        st.stop()
    size_mb = MAIN_CANDIDATES.stat().st_size / 1024 / 1024
    cols = st.columns(4)
    with cols[0]:
        render_metric("Source", "Main JSONL", MAIN_CANDIDATES.name)
    with cols[1]:
        render_metric("File Size", f"{size_mb:.1f} MB", "streamed from disk")
    with cols[2]:
        render_metric("Expected Pool", "100,000", "candidate records")
    with cols[3]:
        render_metric("Output", "Top 100", "submission-ready CSV")
elif source == "sample":
    candidates = json.loads(SAMPLE_CANDIDATES.read_text(encoding="utf-8"))
    cols = st.columns(4)
    with cols[0]:
        render_metric("Source", "Sample", SAMPLE_CANDIDATES.name)
    with cols[1]:
        render_metric("Loaded", f"{len(candidates):,}", "candidate records")
    with cols[2]:
        render_metric("Mode", "Fast Check", "local scoring")
    with cols[3]:
        render_metric("Output", "Preview", "downloadable CSV")
else:
    uploaded = st.file_uploader("Upload candidates", type=["jsonl", "json"])
    if uploaded:
        raw = uploaded.read().decode("utf-8")
        candidates = parse_uploaded_candidates(raw)
        cols = st.columns(4)
        with cols[0]:
            render_metric("Source", "Upload", uploaded.name)
        with cols[1]:
            render_metric("Loaded", f"{len(candidates):,}", "max 100 records")
        with cols[2]:
            render_metric("Mode", "Preview", "local scoring")
        with cols[3]:
            render_metric("Output", "Preview", "downloadable CSV")
    else:
        st.info("Choose a JSON or JSONL file to enable ranking.")

if not using_main_file and not candidates:
    st.stop()

st.markdown('<div class="section-title">Ranking Controls</div>', unsafe_allow_html=True)
controls = st.columns([1, 1, 1.2, 1.3, 2])
with controls[0]:
    top_max = 100 if using_main_file else min(100, len(candidates))
    top_default = 100 if using_main_file else min(20, len(candidates))
    top_n = st.slider("Top-N", 5, top_max, top_default)
with controls[1]:
    shortlist_size = 1500
    if using_main_file:
        shortlist_size = st.number_input(
            "Stage-1 Shortlist",
            min_value=100,
            max_value=5000,
            value=1500,
            step=100,
        )
    else:
        st.number_input("Stage-1 shortlist", value=len(candidates), disabled=True)
with controls[2]:
    speed_mode = "Fast"
    if using_main_file:
        speed_mode = st.selectbox(
            "Ranking mode",
            ["Fast", "Balanced", "Max Precision"],
            help="Fast skips CPU embeddings. Balanced uses lightweight LSA. Max Precision uses MiniLM when available.",
        )
    else:
        st.selectbox("Ranking mode", ["Preview"], disabled=True)
with controls[3]:
    output_choice = st.selectbox(
        "Download format",
        ["Submission CSV", "Detailed CSV"],
        help="Submission CSV is the required candidate_id, rank, score, reasoning format.",
    )
with controls[4]:
    st.write("")
    st.write("")
    run_clicked = st.button("Run Candidate Ranking", type="primary", width="stretch")

if run_clicked:
    clicked_at = time.perf_counter()
    st.markdown('<div class="section-title">Live Ranking Stats</div>', unsafe_allow_html=True)
    if using_main_file:
        embedding_backend = {
            "Fast": "none",
            "Balanced": "lsa",
            "Max Precision": "auto",
        }[speed_mode]
        df, stats = main_file_rank(
            top_n=top_n,
            shortlist_size=int(shortlist_size),
            embedding_backend=embedding_backend,
        )
        pot_df = pd.DataFrame()
    else:
        df, stats, pot_df = sample_or_upload_rank(candidates, top_n)
    ranked_at = time.perf_counter()

    export_started = time.perf_counter()
    download_payload, download_name, download_mime = build_download_payload(df, output_choice)
    export_finished = time.perf_counter()
    total_ready = export_finished - clicked_at

    timings = list(stats.get("timings", []))
    timings.extend([
        {"Task": "Dashboard result table", "Time Taken": f"{max(0.0, ranked_at - clicked_at - float(stats.get('elapsed', 0.0))):.2f}s"},
        {"Task": f"{output_choice} generation", "Time Taken": f"{export_finished - export_started:.2f}s"},
        {"Task": "Total click to file ready", "Time Taken": f"{total_ready:.2f}s"},
    ])

    top_score = float(df.iloc[0]["score"]) if not df.empty else 0.0

    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric("Scanned", f"{int(stats.get('scanned', 0)):,}", "candidate records")
    with metric_cols[1]:
        render_metric("Shortlisted", f"{int(stats.get('shortlisted', 0)):,}", "after Stage 1")
    with metric_cols[2]:
        render_metric("Filtered", f"{int(stats.get('skipped_honeypot', 0)):,}", "honeypot profiles")
    with metric_cols[3]:
        render_metric("Top Score", f"{top_score:.3f}", "best-ranked fit")
    with metric_cols[4]:
        render_metric("File Ready In", f"{total_ready:.1f}s", f"{len(df):,} rows emitted")

    st.markdown('<div class="section-title">Execution Timeline</div>', unsafe_allow_html=True)
    render_timing_table(timings)

    st.markdown('<div class="section-title">Ranked Shortlist</div>', unsafe_allow_html=True)
    display_cols = [col for col in ["rank", "candidate_id", "score", "title", "company", "yoe", "location_fit", "risk_flags", "reasoning"] if col in df.columns]
    render_ranked_shortlist_table(df, display_cols)

    chart_df = df.head(15)[["candidate_id", "score"]].set_index("candidate_id")
    st.markdown('<div class="section-title">Top Candidate Score Profile</div>', unsafe_allow_html=True)
    st.bar_chart(chart_df, width="stretch")

    if not pot_df.empty:
        with st.expander(f"Filtered honeypots ({len(pot_df)})"):
            st.dataframe(pot_df, hide_index=True, width="stretch")

    st.markdown('<div class="section-title">Generated File</div>', unsafe_allow_html=True)
    st.success(f"{download_name} is ready in {output_choice} format.")
    st.download_button(
        f"Download {output_choice}",
        download_payload,
        file_name=download_name,
        mime=download_mime,
        width="stretch",
    )
