"""Feature extraction for a single candidate record.

Three layers, in deliberate order of trust:

1. EVIDENCE — what the career-history descriptions actually say they built.
   This is the primary signal. Skills *listed* but never *evidenced* are
   exactly the keyword-stuffer trap the dataset plants.
2. PLAUSIBILITY — honeypot detection. Internally inconsistent profiles
   (skill used longer than total career, "expert" with zero months, career
   timeline that doesn't add up) are forced out of contention.
3. AVAILABILITY — behavioral signals as a multiplier. A perfect
   profile that never logs in and never replies is not hireable.
"""

from __future__ import annotations

from datetime import date
from typing import Any, TypeAlias, cast

from intelligent_recruiter.job_intelligence import jd_profile as jd

CandidateRecord: TypeAlias = dict[str, Any]
ObjectRecord: TypeAlias = dict[str, Any]

REFERENCE_DATE = date.today()
MUST_CONCEPT_ITEMS = tuple(jd.MUST_HAVE_CONCEPTS.items())
MUST_CONCEPT_TERM_GROUPS = tuple(jd.MUST_HAVE_CONCEPTS.values())
MUST_SKILL_TERM_GROUPS = tuple(
    tuple(term.strip() for term in terms) for terms in jd.MUST_HAVE_CONCEPTS.values()
)
NICE_CONCEPT_ITEMS = tuple(jd.NICE_TO_HAVE_CONCEPTS.items())


def _object(value: Any) -> ObjectRecord:
    return cast(ObjectRecord, value) if isinstance(value, dict) else {}


def _objects(value: Any) -> list[ObjectRecord]:
    if not isinstance(value, list):
        return []
    items = cast(list[Any], value)
    return [cast(ObjectRecord, item) for item in items if isinstance(item, dict)]


def _profile(c: CandidateRecord) -> ObjectRecord:
    return _object(c.get("profile"))


def _career_history(c: CandidateRecord) -> list[ObjectRecord]:
    return _objects(c.get("career_history"))


def _skills(c: CandidateRecord) -> list[ObjectRecord]:
    return _objects(c.get("skills"))


def _signals(c: CandidateRecord) -> ObjectRecord:
    return _object(c.get("redrob_signals"))


def _text(record: ObjectRecord, key: str) -> str:
    value = record.get(key)
    return value if isinstance(value, str) else ""


def _number(record: ObjectRecord, key: str, default: float = 0.0) -> float:
    value = record.get(key)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _integer(record: ObjectRecord, key: str, default: int = 0) -> int:
    return int(_number(record, key, float(default)))


def _boolean(record: ObjectRecord, key: str) -> bool:
    return bool(record.get(key))


# ---------------------------------------------------------------------------
# text assembly
# ---------------------------------------------------------------------------

def candidate_texts(c: CandidateRecord) -> tuple[str, str]:
    """Return (evidence_text, listed_skills_text), both lowercased.

    evidence_text = narrative fields a candidate can't trivially stuff
    (career descriptions, summary, headline, titles).
    """
    p = _profile(c)
    parts: list[str] = [
        _text(p, "headline"),
        _text(p, "summary"),
        _text(p, "current_title"),
    ]
    for job in _career_history(c):
        parts.append(_text(job, "title"))
        parts.append(_text(job, "description"))
    evidence = " \n ".join(x for x in parts if x).lower()
    skills = " , ".join(_text(skill, "name") for skill in _skills(c)).lower()
    return evidence, skills


def _hits(text: str, terms: list[str]) -> int:
    return sum(1 for t in terms if t in text)


def _has_hit(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    return any(term in text for term in terms)


# ---------------------------------------------------------------------------
# 1. evidence features
# ---------------------------------------------------------------------------

def evidence_features(c: CandidateRecord) -> dict[str, Any]:
    evidence, skills_text = candidate_texts(c)
    must = {name: _hits(evidence, terms) for name, terms in MUST_CONCEPT_ITEMS}
    nice = {name: _hits(evidence, terms) for name, terms in NICE_CONCEPT_ITEMS}

    # skills-list-only matches (the stuffer signature): concept appears in the
    # skills list but never in any narrative text.
    listed_only = 0
    for skill_terms, evidence_terms in zip(MUST_SKILL_TERM_GROUPS, MUST_CONCEPT_TERM_GROUPS):
        in_list = _has_hit(skills_text, skill_terms)
        in_evidence = _has_hit(evidence, evidence_terms)
        if in_list and not in_evidence:
            listed_only += 1

    off_domain = _hits(evidence, jd.OFF_DOMAIN_TERMS)
    nlp_ir = _hits(evidence, jd.NLP_IR_TERMS)

    return {
        "must": must,
        "nice": nice,
        "listed_only_concepts": listed_only,
        "off_domain_hits": off_domain,
        "nlp_ir_hits": nlp_ir,
        "evidence_text": evidence,
    }


# ---------------------------------------------------------------------------
# career-shape features (disqualifiers from the JD)
# ---------------------------------------------------------------------------

def career_features(c: CandidateRecord) -> dict[str, Any]:
    p = _profile(c)
    history = _career_history(c)
    title = _text(p, "current_title").lower()

    companies = [_text(j, "company").lower() for j in history]
    titles = [_text(j, "title").lower() for j in history]

    def is_consulting(name: str) -> bool:
        return any(f in name for f in jd.CONSULTING_FIRMS)

    consulting_flags = [is_consulting(x) for x in companies]
    consulting_only = bool(companies) and all(consulting_flags)
    has_product_exp = any(not f for f in consulting_flags)

    def is_research(comp: str, t: str) -> bool:
        return any(x in comp for x in jd.RESEARCH_ORG_TERMS) or any(
            x in t for x in jd.RESEARCH_TITLE_TERMS
        )

    research_only = bool(history) and all(
        is_research(comp, t) for comp, t in zip(companies, titles)
    )

    non_eng_title = any(t in title for t in jd.NON_ENGINEERING_TITLE_TERMS)
    eng_title = any(t in title for t in jd.ENGINEERING_TITLE_TERMS)

    # Title-chaser / job-hopper: average completed-stint tenure.
    durations = [_integer(j, "duration_months") for j in history if not _boolean(j, "is_current")]
    avg_tenure = (sum(durations) / len(durations)) if durations else 36.0
    hopper = len(history) >= 3 and avg_tenure < 16

    return {
        "consulting_only": consulting_only,
        "currently_consulting_with_product_past": (
            bool(consulting_flags) and consulting_flags[0] and has_product_exp
        ),
        "research_only": research_only,
        "non_engineering_title": non_eng_title and not eng_title,
        "job_hopper": hopper,
        "avg_tenure_months": avg_tenure,
    }


# ---------------------------------------------------------------------------
# 2. honeypot / plausibility checks
# ---------------------------------------------------------------------------

def honeypot_flags(c: CandidateRecord) -> list[str]:
    flags: list[str] = []
    p = _profile(c)
    yoe = _number(p, "years_of_experience")
    total_months = yoe * 12.0
    skills = _skills(c)
    history = _career_history(c)

    # Skill used longer than the entire career (with slack).
    over = [skill for skill in skills if _integer(skill, "duration_months") > total_months + 9]
    if over:
        flags.append("skill_duration_exceeds_career")

    # "Expert" proficiency with ~zero actual usage.
    zero_experts = [
        skill for skill in skills
        if _text(skill, "proficiency") == "expert" and _integer(skill, "duration_months") <= 1
    ]
    if len(zero_experts) >= 3:
        flags.append("expert_skills_with_zero_usage")

    # Career timeline wildly inconsistent with claimed experience.
    hist_months = sum(_integer(j, "duration_months") for j in history)
    if history and total_months > 0:
        if hist_months > total_months * 1.9 + 24:
            flags.append("history_exceeds_claimed_experience")
        # Claims many years but documented history is a tiny fraction.
        if yoe >= 6 and hist_months < total_months * 0.25:
            flags.append("claimed_experience_unsupported")

    # Tenure at a single role longer than the claimed total career.
    if any(_integer(j, "duration_months") > total_months + 12 for j in history):
        flags.append("single_role_exceeds_career")

    # Date arithmetic check: stated duration vs actual start/end dates.
    for j in history:
        try:
            start = date.fromisoformat(_text(j, "start_date"))
            end = (
                REFERENCE_DATE
                if _boolean(j, "is_current") or not _text(j, "end_date")
                else date.fromisoformat(_text(j, "end_date"))
            )
            real_months = (end.year - start.year) * 12 + (end.month - start.month)
            stated = _integer(j, "duration_months")
            if abs(real_months - stated) > 18:
                flags.append("duration_date_mismatch")
                break
            if start > REFERENCE_DATE:
                flags.append("future_start_date")
                break
        except Exception:
            continue

    return flags


# ---------------------------------------------------------------------------
# 3. behavioral availability multiplier (0.45 .. 1.10)
# ---------------------------------------------------------------------------

def behavioral_multiplier(c: CandidateRecord) -> tuple[float, dict[str, Any]]:
    s = _signals(c)
    m = 1.0
    info: dict[str, Any] = {}

    # Recency of activity — dominant availability signal.
    days_inactive = 9999
    try:
        last = date.fromisoformat(_text(s, "last_active_date"))
        days_inactive = (REFERENCE_DATE - last).days
    except Exception:
        pass
    info["days_inactive"] = days_inactive
    if days_inactive <= 14:
        m *= 1.05
    elif days_inactive <= 45:
        m *= 1.0
    elif days_inactive <= 90:
        m *= 0.88
    elif days_inactive <= 180:
        m *= 0.72
    else:
        m *= 0.55

    rr = _number(s, "recruiter_response_rate")
    info["response_rate"] = rr
    if rr >= 0.6:
        m *= 1.05
    elif rr >= 0.3:
        m *= 1.0
    elif rr >= 0.15:
        m *= 0.85
    else:
        m *= 0.70

    if _boolean(s, "open_to_work_flag"):
        m *= 1.04
    icr = _number(s, "interview_completion_rate", -1.0)
    if 0 <= icr < 0.5:
        m *= 0.85

    notice = _integer(s, "notice_period_days")
    info["notice_days"] = notice
    if notice <= 30:
        m *= 1.03
    elif notice <= 60:
        m *= 0.97
    else:
        m *= 0.90

    if _boolean(s, "verified_email") and _boolean(s, "verified_phone"):
        m *= 1.01

    return max(0.45, min(m, 1.10)), info


# ---------------------------------------------------------------------------
# location & experience fit
# ---------------------------------------------------------------------------

def location_fit(c: CandidateRecord) -> tuple[float, str]:
    p = _profile(c)
    loc = _text(p, "location").lower()
    country = _text(p, "country").lower()
    relocate = _boolean(_signals(c), "willing_to_relocate")

    if any(city in loc for city in jd.PREFERRED_CITIES):
        return 1.0, "in a preferred hub (Pune/Noida)"
    if country == "india" and any(city in loc for city in jd.TIER1_INDIA_CITIES):
        return 0.85, "Tier-1 India city"
    if country == "india":
        return (0.65 if relocate else 0.45), (
            "in India, willing to relocate" if relocate else "in India, relocation unclear"
        )
    return (0.30 if relocate else 0.10), "outside India (case-by-case per JD)"


def experience_fit(yoe: float) -> float:
    if jd.EXP_IDEAL_LO <= yoe <= jd.EXP_IDEAL_HI:
        return 1.0
    if jd.EXP_MIN <= yoe <= jd.EXP_MAX:
        return 0.9
    if 4.0 <= yoe < jd.EXP_MIN:
        return 0.7
    if jd.EXP_MAX < yoe <= 12.0:
        return 0.65
    if 3.0 <= yoe < 4.0:
        return 0.45
    return 0.25
