#!/usr/bin/env python3
"""Streamlit Numerology Hiring Scorer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import io
import json
import os
import re
from typing import Dict, List, Tuple

import openai
from openai import OpenAI
import pandas as pd
from pypdf import PdfReader
import plotly.express as px
import streamlit as st
from supabase import create_client
from scoring_engine import *
from login import login
from modules.compare_candidates import render as render_compare
from modules.roles_and_candidates import render as render_roles_candidates
from modules.role_master import render as render_role_master
from modules.user_management import render as render_user_mgmt

st.set_page_config(
    page_title="Numerology Hiring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_pages_style = """
<style>
section[data-testid="stSidebarNav"] {
    display: none;
}
</style>
"""

st.markdown(hide_pages_style, unsafe_allow_html=True)

TRAITS = [
    "Execution",
    "Detail Precision",
    "Loyalty",
    "Politics Safe",
    "Ownership",
    "Discipline",
    "Communication",
    "Analytical",
    "Strategy",
]

SKILL_TRAITS = [
    "Execution",
    "Detail Precision",
    "Communication",
    "Analytical",
    "Strategy",
]
IMPORTANCE_THRESHOLDS = {
    "Medium": 6.5,
    "High": 7.0,
    "Very High": 7.5,
}
CORE_TRAITS = [
    "Execution",
    "Detail Precision",
    "Loyalty",
    "Politics Safe",
    "Ownership",
    "Discipline",
    "Communication",
    "Analytical",
    "Strategy",
]
NEW_ATTRIBUTES = [
    "Authority",
    "Dominance",
    "Integrity",
    "Emotional Stability",
    "Aggression",
    "Risk Appetite",
    "Consistency",
    "Obedience",
    "Assertiveness",
    "Enforcement Ability",
]
ATTRIBUTE_KEYS = CORE_TRAITS + NEW_ATTRIBUTES
TRAIT_MAP_FILE = os.path.join(os.path.dirname(__file__), "trait_attribute_map.json")
DEFAULT_TRAITS = [
    "Leadership Ability",
    "Ownership Mentality",
    "Reliability",
    "Integrity",
]

TRAIT_EMOJIS: Dict[str, str] = {
    "Execution": "⚙️",
    "Detail Precision": "🔎",
    "Loyalty": "🤝",
    "Politics Safe": "🛡️",
    "Ownership": "📌",
    "Discipline": "📏",
    "Communication": "🗣️",
    "Analytical": "📊",
    "Strategy": "♟️",
}

ROLE_WEIGHTS = {
    "Execution Focused": {
        "Finish Execution": 0.25,
        "Loyalty": 0.25,
        "Self Ownership": 0.15,
        "Politics Safe": 0.15,
        "Communication": 0.05,
        "Strategy & Analytical": 0.15,
    },
    "Strategy Focused": {
        "Finish Execution": 0.05,
        "Loyalty": 0.20,
        "Self Ownership": 0.15,
        "Politics Safe": 0.15,
        "Communication": 0.20,
        "Strategy & Analytical": 0.25,
    },
}


@dataclass
class ScoreResult:
    candidate_numbers: Dict[str, int]
    leader_numbers: Dict[str, int]
    trait_scores: Dict[str, float]
    composite_scores: Dict[str, float]
    loyalty_meta: Dict[str, float]
    risk_status: str
    risk_flags: List[str]
    overall_score_100: float
    verdict: str
    flags: List[str]


def clamp(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, value))


def digital_root(n: int) -> int:
    n = abs(n)
    if n == 0:
        return 0
    while n > 9:
        n = sum(int(ch) for ch in str(n))
    return n


def validate_dob(dob: str) -> str:
    try:
        parsed = datetime.strptime(dob.strip(), "%d/%m/%Y")
    except ValueError as exc:
        raise ValueError("DOB must be in dd/mm/yyyy format and be a valid date.") from exc
    return parsed.strftime("%d/%m/%Y")


def numerology_numbers(dob: str) -> Dict[str, int]:
    day, month, year = map(int, dob.split("/"))
    birth = digital_root(day)
    month_total = digital_root(month)
    destiny = digital_root(sum(int(ch) for ch in f"{day:02d}{month:02d}{year:04d}"))

    if birth == 0:
        birth = 9
    if month_total == 0:
        month_total = 9
    if destiny == 0:
        destiny = 9

    return {"birth": birth, "month": month_total, "destiny": destiny}


def compute_trait_scores(destiny: int, birth: int, month: int) -> Dict[str, float]:
    base_traits = {
        trait: (
            DESTINY_MATRIX[destiny][trait] * 0.5
            + BIRTH_MATRIX[birth][trait] * 0.3
            + MONTH_MATRIX[month][trait] * 0.2
        )
        for trait in TRAITS
        if trait != "Discipline"
    }

    process_score = (
        DISCIPLINE_MATRIX[destiny]["Process"] * 0.5
        + DISCIPLINE_MATRIX[birth]["Process"] * 0.3
        + DISCIPLINE_MATRIX[month]["Process"] * 0.2
    )
    value_score = (
        DISCIPLINE_MATRIX[destiny]["Value"] * 0.5
        + DISCIPLINE_MATRIX[birth]["Value"] * 0.3
        + DISCIPLINE_MATRIX[month]["Value"] * 0.2
    )
    general_score = (
        DISCIPLINE_MATRIX[destiny]["General"] * 0.5
        + DISCIPLINE_MATRIX[birth]["General"] * 0.3
        + DISCIPLINE_MATRIX[month]["General"] * 0.2
    )
    discipline_score = clamp(
        process_score * 0.4 + value_score * 0.35 + general_score * 0.25
    )

    base_traits["Discipline"] = discipline_score
    return base_traits


def apply_adjustments(
    trait_scores: Dict[str, float],
    candidate_numbers: Dict[str, int],
) -> List[str]:
    flags: List[str] = []

    if (
        candidate_numbers["destiny"] == 9
        and candidate_numbers["birth"] == 9
        and candidate_numbers["month"] == 9
    ):
        flags.append("Triple 9: Soft firing / compassion bias")

    if candidate_numbers["destiny"] in {4, 8, 9}:
        trait_scores["Ownership"] = clamp(trait_scores["Ownership"] + 0.5)
        flags.append("Bonus: Destiny in {4,8,9} -> Ownership +0.5")

    if candidate_numbers["month"] == 1:
        trait_scores["Discipline"] = clamp(trait_scores["Discipline"] + 0.3)
        flags.append("Bonus: Month 1 -> Self-discipline +0.3")

    return flags


def compute_loyalty_pillar(
    traits: Dict[str, float],
    candidate_numbers: Dict[str, int],
    leader_numbers: Dict[str, int],
    attribute_scores: Dict[str, float],
) -> Dict[str, float]:
    baseline_loyalty = (traits["Loyalty"] + traits["Politics Safe"]) / 2

    org_loyalty = baseline_loyalty
    if candidate_numbers["destiny"] in {4, 8, 9}:
        org_loyalty += 0.5
    org_loyalty = clamp(org_loyalty)

    leadership_penalty = 0.0
    no_penalty_triggered = True
    # Leadership loyalty starts from baseline, not org loyalty.
    if candidate_numbers["destiny"] == 1 and leader_numbers["destiny"] == 1:
        leadership_penalty -= 2.0
        no_penalty_triggered = False
    if candidate_numbers["month"] == 7:
        leadership_penalty -= 1.0
        no_penalty_triggered = False
    if candidate_numbers["destiny"] == 5:
        leadership_penalty -= 1.0
        no_penalty_triggered = False
    if candidate_numbers["destiny"] in {4, 8, 9} and no_penalty_triggered:
        leadership_penalty += 0.3

    authority_alignment = calculate_authority_alignment(attribute_scores)
    peer_influence = calculate_peer_influence(attribute_scores)
    leadership_loyalty_base = baseline_loyalty
    leadership_loyalty = (
        baseline_loyalty * 0.50
        + authority_alignment * 0.35
        - peer_influence * 0.18
        + leadership_penalty
    )
    leadership_loyalty = clamp(leadership_loyalty)
    final_loyalty = clamp(org_loyalty * 0.6 + leadership_loyalty * 0.4)

    return {
        "baseline_loyalty": baseline_loyalty,
        "org_loyalty": org_loyalty,
        "leadership_loyalty_base": leadership_loyalty_base,
        "leadership_penalty": leadership_penalty,
        "leadership_loyalty": leadership_loyalty,
        "final_loyalty": final_loyalty,
        "authority_alignment": authority_alignment,
        "peer_influence": peer_influence,
    }


def compute_composites(traits: Dict[str, float], loyalty_score: float) -> Dict[str, float]:
    return {
        "Finish Execution": (traits["Execution"] + traits["Detail Precision"]) / 2,
        "Self Ownership": (traits["Ownership"] + traits["Discipline"]) / 2,
        "Strategy & Analytical": (traits["Strategy"] + traits["Analytical"]) / 2,
        "Loyalty": loyalty_score,
        "Politics Safe": traits["Politics Safe"],
        "Communication": traits["Communication"],
    }


def compute_overall(composites: Dict[str, float], role_type: str) -> float:
    overall_10 = sum(composites[name] * ROLE_WEIGHTS[role_type][name] for name in ROLE_WEIGHTS[role_type])
    return overall_10 * 10.0


def compute_risk_gate(composites: Dict[str, float], traits: Dict[str, float]) -> Tuple[str, List[str]]:
    risk_flags: List[str] = []
    loyalty = composites["Loyalty"]
    politics_safe = composites["Politics Safe"]
    self_ownership = composites["Self Ownership"]
    discipline = traits["Discipline"]

    if loyalty < 6.5:
        risk_flags.append("Low Loyalty")
    if politics_safe < 6.5:
        risk_flags.append("Political Risk")
    if self_ownership < 6.5:
        risk_flags.append("Low Self Ownership")
    if discipline < 6:
        risk_flags.append("Low Discipline")

    risk_status = "FAIL" if risk_flags else "PASS"
    return risk_status, risk_flags


def verdict_from_score(score_100: float) -> str:
    if score_100 >= 80:
        return "Strong Hire"
    if score_100 >= 65:
        return "Hire w/ Guardrails"
    return "Weak / Risk"


def strength_tag(value: float) -> str:
    if value < 4.0:
        return "Very Weak"
    if value < 6.0:
        return "Weak"
    if value < 8.0:
        return "Moderate"
    return "Strong"


def loyalty_detail_tag(value: float) -> str:
    if value < 2.0:
        return "Very Weak"
    if value < 4.0:
        return "Weak"
    if value < 6.0:
        return "Moderate"
    if value < 8.0:
        return "Strong"
    return "Very Strong"


def final_loyalty_tag(value: float) -> str:
    if value < 4.0:
        return "Weak"
    if value < 7.0:
        return "Moderate"
    return "Strong"


def classify_loyalty_profile(
    org_loyalty: float,
    leadership_loyalty: float,
    authority_alignment: float,
    peer_influence: float,
) -> str:
    if leadership_loyalty >= 7.5 and peer_influence <= 3.5:
        return "Leadership Anchored - Highly Trusted"
    if leadership_loyalty >= 6.5 and peer_influence <= 4.5:
        return "Stable and Leadership Aligned"
    if leadership_loyalty >= 5.5 and peer_influence <= 5.5:
        return "Stable but Influenceable (Guardrails Required)"
    if leadership_loyalty < 5.5 and peer_influence >= 4:
        return "Influenceable - Leadership Loyalty Risk"
    return "Mixed Loyalty Profile - Monitor"


def label_score(score: float) -> str:
    if score >= 7.5:
        return "Very Strong"
    if score >= 6.5:
        return "Strong"
    if score >= 5.5:
        return "Moderate"
    if score >= 4.5:
        return "Guarded"
    return "Risk"


def normalize_decimals(text: str) -> str:
    return re.sub(r"\b\d+\.\d+\b", lambda m: f"{float(m.group(0)):.1f}", text)


def calculate_all_attribute_scores(numbers: Dict[str, int], base_traits: Dict[str, float]) -> Dict[str, float]:
    values: Dict[str, float] = {k: float(v) for k, v in base_traits.items()}
    for attribute in NEW_ATTRIBUTES:
        values[attribute] = calculate_attribute_score(
            attribute=attribute,
            birth=numbers["birth"],
            destiny=numbers["destiny"],
            month=numbers["month"],
        )
    return values


def calculate_authority_alignment(attr: Dict[str, float]) -> float:
    score = (
        attr["Authority"] * 0.30
        + attr["Integrity"] * 0.25
        + attr["Obedience"] * 0.20
        + attr["Consistency"] * 0.15
        + attr["Emotional Stability"] * 0.10
    )
    return clamp(score)


def calculate_peer_influence(attr: Dict[str, float]) -> float:
    score = (
        (10 - attr["Emotional Stability"]) * 0.35
        + (10 - attr["Dominance"]) * 0.25
        + (10 - attr["Assertiveness"]) * 0.15
        + attr["Risk Appetite"] * 0.15
        + attr["Aggression"] * 0.10
    )
    return clamp(score)


def _normalize_mapping(mapping: Dict[str, float], allowed_keys: List[str], top_k: int = 4) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for key, value in mapping.items():
        if key in allowed_keys and isinstance(value, (int, float)) and value >= 0:
            cleaned[key] = float(value)
    if not cleaned:
        return {}
    sorted_items = sorted(cleaned.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    total = sum(v for _, v in sorted_items)
    if total <= 0:
        return {}
    return {k: v / total for k, v in sorted_items}


def load_trait_attribute_map() -> Dict[str, Dict[str, float]]:
    merged = {k: dict(v) for k, v in TRAIT_ATTRIBUTE_MAP.items()}
    if os.path.exists(TRAIT_MAP_FILE):
        try:
            with open(TRAIT_MAP_FILE, "r", encoding="utf-8") as f:
                persisted = json.load(f)
            if isinstance(persisted, dict):
                for trait, mapping in persisted.items():
                    if isinstance(mapping, dict):
                        normalized = _normalize_mapping(mapping, ATTRIBUTE_KEYS, top_k=4)
                        if normalized:
                            merged[str(trait)] = normalized
        except Exception:
            pass
    return merged


def persist_trait_attribute_map(trait_map: Dict[str, Dict[str, float]]) -> None:
    try:
        with open(TRAIT_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(trait_map, f, indent=2)
    except Exception:
        pass


def _extract_json_object(raw_text: str) -> Dict[str, object]:
    try:
        data = json.loads(raw_text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            data = json.loads(raw_text[start : end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def generate_role_specific_traits(
    api_key: str,
    role_name: str,
    role_description: str,
) -> List[str]:
    client = OpenAI(api_key=api_key)
    prompt = f"""
You are a hiring role trait engine.
Core traits (must be excluded): {", ".join(CORE_TRAITS)}

Generate ADDITIONAL role-specific traits only.
Output JSON only in format:
{{"traits": ["Trait 1", "Trait 2", "Trait 3", "Trait 4"]}}

Rules:
- Return 4 to 8 traits.
- No trait may match or duplicate core traits.
- Keep traits operational and measurable.

ROLE NAME: {role_name}
ROLE DESCRIPTION: {role_description if role_description.strip() else "Not provided"}
"""
    response = client.responses.create(model="gpt-4.1-mini", input=prompt)
    data = _extract_json_object(response.output_text)
    raw_traits = data.get("traits", [])
    traits: List[str] = []
    if isinstance(raw_traits, list):
        for t in raw_traits:
            t_str = str(t).strip()
            if (
                t_str
                and t_str not in traits
                and t_str not in CORE_TRAITS
            ):
                traits.append(t_str)
    return traits


def call_ai_trait_mapping(api_key: str, trait: str) -> Dict[str, float]:
    client = OpenAI(api_key=api_key)
    prompt = f"""
Map this hiring trait to the attribute system.

Trait: {trait}
Available attributes:
{", ".join(ATTRIBUTE_KEYS)}

Return ONLY valid JSON object:
{{
  "attribute_name": weight
}}

Rules:
- Use 3 attributes max.
- All weights must sum to 1.
- Only use attribute names exactly as provided.
- No extra text.
"""
    response = client.responses.create(model="gpt-4.1-mini", input=prompt)
    data = _extract_json_object(response.output_text)
    mapping = {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}
    return _normalize_mapping(mapping, ATTRIBUTE_KEYS, top_k=3)


def ensure_trait_mapping(
    api_key: str,
    trait: str,
    trait_map: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    if trait in trait_map:
        trait_map[trait] = _normalize_mapping(trait_map[trait], ATTRIBUTE_KEYS, top_k=4) or {
            "Execution": 0.4,
            "Consistency": 0.3,
            "Integrity": 0.3,
        }
        return trait_map
    ai_mapping = call_ai_trait_mapping(api_key, trait)
    if not ai_mapping:
        ai_mapping = {"Execution": 0.4, "Consistency": 0.3, "Integrity": 0.3}
    # Mandatory normalization to ensure mapping weights sum to 1.
    trait_map[trait] = _normalize_mapping(ai_mapping, ATTRIBUTE_KEYS, top_k=4) or {
        "Execution": 0.4,
        "Consistency": 0.3,
        "Integrity": 0.3,
    }
    persist_trait_attribute_map(trait_map)
    return trait_map


def calculate_trait_score(trait: str, attribute_scores: Dict[str, float], trait_map: Dict[str, Dict[str, float]]) -> float:
    mapping = trait_map[trait]
    score = 0.0
    for attribute, weight in mapping.items():
        score += attribute_scores.get(attribute, 0.0) * weight
    return round(clamp(score), 2)


def generate_role_trait_weights(
    api_key: str,
    role_name: str,
    role_description: str,
    role_traits: List[str],
) -> Dict[str, float]:
    client = OpenAI(api_key=api_key)
    prompt = f"""
You are a hiring role analysis engine.
Assign importance weight to each trait.

Role Name:
{role_name}

Role Description:
{role_description if role_description.strip() else "Not provided"}

Traits:
{role_traits}

Rules:
- Return JSON only.
- All weights must sum to 1.
- Higher weight = higher importance.
- Include only provided trait names.

Format:
{{
  "Trait": weight
}}
"""
    response = client.responses.create(model="gpt-4.1-mini", input=prompt)
    data = _extract_json_object(response.output_text)
    weights = {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}
    filtered = {t: max(0.0, weights.get(t, 0.0)) for t in role_traits}
    total = sum(filtered.values())
    if total <= 0:
        filtered = {t: 1.0 / len(role_traits) for t in role_traits}
        total = sum(filtered.values())
    # Mandatory normalization pass.
    normalized = {k: v / total for k, v in filtered.items()}
    return normalized


def score_candidate(candidate_dob: str, leader_dob: str, role_type: str) -> ScoreResult:
    candidate_numbers = numerology_numbers(validate_dob(candidate_dob))
    leader_numbers = numerology_numbers(validate_dob(leader_dob))

    trait_scores = compute_trait_scores(
        destiny=candidate_numbers["destiny"],
        birth=candidate_numbers["birth"],
        month=candidate_numbers["month"],
    )
    attribute_scores = calculate_all_attribute_scores(candidate_numbers, trait_scores)
    loyalty_meta = compute_loyalty_pillar(
        traits=trait_scores,
        candidate_numbers=candidate_numbers,
        leader_numbers=leader_numbers,
        attribute_scores=attribute_scores,
    )
    flags = apply_adjustments(
        trait_scores=trait_scores,
        candidate_numbers=candidate_numbers,
    )

    composite_scores = compute_composites(trait_scores, loyalty_meta["final_loyalty"])
    risk_status, risk_flags = compute_risk_gate(composite_scores, trait_scores)
    overall_100 = compute_overall(composite_scores, role_type)
    if risk_status == "FAIL":
        final_verdict = "High Risk – Do Not Proceed"
    else:
        final_verdict = verdict_from_score(overall_100)

    return ScoreResult(
        candidate_numbers=candidate_numbers,
        leader_numbers=leader_numbers,
        trait_scores=trait_scores,
        composite_scores=composite_scores,
        loyalty_meta=loyalty_meta,
        risk_status=risk_status,
        risk_flags=risk_flags,
        overall_score_100=overall_100,
        verdict=final_verdict,
        flags=flags,
    )


def display_result(result: ScoreResult) -> None:
    st.markdown("### Candidate Result")
    st.markdown(
        f"Birth: **{result.candidate_numbers['birth']}** | "
        f"Month: **{result.candidate_numbers['month']}** | "
        f"Destiny: **{result.candidate_numbers['destiny']}**"
    )
    st.markdown(f"Overall Score: **{result.overall_score_100:.1f}/100**")
    st.markdown(f"Verdict: **{result.verdict}**")


def run_comparison(dob_input: str) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for line in dob_input.splitlines():
        dob = line.strip()
        if not dob:
            continue
        try:
            candidate_dob = validate_dob(dob)
            result = score_candidate(candidate_dob, "03/11/1994", "Execution Focused")
            rows.append(
                {
                    "DOB": candidate_dob,
                    "Overall Score": round(result.overall_score_100, 1),
                    "Verdict": result.verdict,
                }
            )
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    return df


def verdict_badge(verdict: str) -> str:
    if verdict == "Strong Hire":
        return "🟢"
    if verdict == "Hire w/ Guardrails":
        return "🟡"
    return "🔴"


def extract_resume_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if name.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    raise ValueError("Unsupported resume format. Upload .txt or .pdf")


def extract_dob_from_text(text: str) -> str | None:
    # Look for common DOB patterns and normalize to dd/mm/yyyy.
    month_map = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    patterns = [
        r"(?i)\b(?:dob|date of birth)\s*[:\-]?\s*(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b",
        r"\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b",
        r"(?i)\b(?:dob|date of birth)\s*[:\-]?\s*(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b",
        r"(?i)\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            # Numeric date formats: dd/mm/yyyy or dd-mm-yyyy
            if len(match.groups()) == 1:
                candidate = match.group(1).replace("-", "/")
                try:
                    return validate_dob(candidate)
                except ValueError:
                    continue

            # Text month formats: 10 Oct 2026 / 10 October 2026
            if len(match.groups()) == 3:
                day = int(match.group(1))
                month_text = match.group(2).strip().lower()
                year = int(match.group(3))
                if month_text not in month_map:
                    continue
                month = month_map[month_text]
                candidate = f"{day:02d}/{month:02d}/{year:04d}"
                try:
                    return validate_dob(candidate)
                except ValueError:
                    continue
    return None


def extract_candidate_name_from_text(text: str) -> str | None:
    # Prefer explicit labels first.
    label_patterns = [
        r"(?im)^\s*(?:candidate\s*name|name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,80})\s*$",
    ]
    for pattern in label_patterns:
        match = re.search(pattern, text)
        if match:
            candidate = " ".join(match.group(1).split())
            if _looks_like_person_name(candidate):
                return candidate

    # Fallback: first plausible line in top section of resume.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[:20]:
        cleaned = re.sub(r"[^A-Za-z .'-]", "", line).strip()
        if _looks_like_person_name(cleaned):
            return cleaned
    return None


def _looks_like_person_name(value: str) -> bool:
    if not value:
        return False
    lower = value.lower()
    blocked = [
        "resume",
        "curriculum",
        "vitae",
        "profile",
        "summary",
        "email",
        "phone",
        "address",
        "linkedin",
        "github",
        "objective",
        "experience",
        "education",
    ]
    if any(token in lower for token in blocked):
        return False
    if "@" in value or any(ch.isdigit() for ch in value):
        return False

    parts = [p for p in value.split() if p]
    if len(parts) < 2 or len(parts) > 5:
        return False
    for part in parts:
        if len(part) < 2:
            return False
    return True


def run_openai_resume_review(
    api_key: str,
    role_name: str,
    role_description: str,
    role_type: str,
    resume_text: str,
    candidate_numbers: Dict[str, int],
    trait_scores: Dict[str, float],
    composite_scores: Dict[str, float],
    overall_score_100: float,
    verdict: str,
) -> Dict[str, object]:
    client = OpenAI(api_key=api_key)
    rounded_traits = {k: round(v, 1) for k, v in trait_scores.items()}
    rounded_composites = {k: round(v, 1) for k, v in composite_scores.items()}
    prompt = f"""
You are a hiring analyst. Review the candidate resume against the role and combine that with provided numerology result.

Role Name:
{role_name}

Role Description:
{role_description if role_description.strip() else "Not provided"}

Resume:
{resume_text[:12000]}

Numerology Summary:
- Candidate numbers: destiny={candidate_numbers['destiny']}, birth={candidate_numbers['birth']}, month={candidate_numbers['month']}
- Role type: {role_type}
- Trait scores (rounded, 1 decimal): {rounded_traits}
- Composite scores (rounded, 1 decimal): {rounded_composites}
- Overall numerology score: {overall_score_100:.1f}/100
- Verdict: {verdict}

Return strict JSON with keys:
- candidate_assessment_summary (string, max 90 words)
- role_candidate_fit_summary (string, max 120 words)
- strengths_for_role (array of 5 strings, each starting with a short heading)
- traits_to_validate (array of 2 strings; each must start with trait name and score like "Detail Precision (4.8): ...", and explain risk vs role requirements)
- recommended_guardrails (array of 4 strings)

Important rules:
- Do NOT convert role text into a numerology total.
- Base role commentary on role function + provided trait/composite scores.
- Do not include markdown, code fences, or extra prose.
- Keep output concise and practical.
"""
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )
    raw_text = response.output_text
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("OpenAI response was not valid JSON.")
        data = json.loads(raw_text[start : end + 1])
    return data


def conceptual_bar(value: float, width: int = 10) -> str:
    blocks = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    v = max(0.0, min(10.0, value))
    full = int(v)
    frac_idx = int(round((v - full) * 8))
    if frac_idx == 8:
        full += 1
        frac_idx = 0
    bar = ("█" * full + blocks[frac_idx]).ljust(width)
    return bar


def verdict_sentence(verdict: str, role_type: str) -> str:
    role_label = "execution-oriented" if role_type == "Execution Focused" else "strategy-oriented"
    if verdict == "Strong Hire":
        return f"This candidate surpasses the threshold for a strong {role_label} hire."
    if verdict == "Hire w/ Guardrails":
        return f"This candidate is a viable {role_label} hire with clear guardrails."
    return f"This candidate is currently below the threshold for a {role_label} hire."


def trait_chart(traits: Dict[str, float]) -> None:
    df = pd.DataFrame(
        [{"Trait": t, "Score": v, "Tag": strength_tag(v)} for t, v in traits.items()]
    ).sort_values("Score")

    fig = px.bar(
        df,
        x="Score",
        y="Trait",
        orientation="h",
        text=df["Score"].map(lambda x: f"{x:.1f}"),
        color="Score",
        color_continuous_scale=[(0.0, "#d64541"), (0.5, "#f39c12"), (1.0, "#2e8b57")],
        range_color=[0, 10],
    )
    fig.update_layout(height=460, coloraxis_showscale=False, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    fig.update_traces(textposition="outside", hovertemplate="%{y}: %{x:.2f}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df.sort_values("Score", ascending=False),
        use_container_width=False,
        hide_index=True,
    )


def composite_chart(composites: Dict[str, float], traits: Dict[str, float]) -> None:
    breakdown = {
        "Finish Execution": f"Execution: {traits['Execution']:.1f}, Detail Precision: {traits['Detail Precision']:.1f}",
        "Self Ownership": f"Ownership: {traits['Ownership']:.1f}, Discipline: {traits['Discipline']:.1f}",
        "Strategy & Analytical": f"Strategy: {traits['Strategy']:.1f}, Analytical: {traits['Analytical']:.1f}",
        "Loyalty": "Direct trait",
        "Politics Safe": "Direct trait",
        "Communication": "Direct trait",
    }

    rows = []
    for name, value in composites.items():
        rows.append({"Composite": name, "Score": value, "Tag": strength_tag(value), "Breakdown": breakdown[name]})

    df = pd.DataFrame(rows).sort_values("Score")

    fig = px.bar(
        df,
        x="Score",
        y="Composite",
        orientation="h",
        text=df["Score"].map(lambda x: f"{x:.1f}"),
        color="Score",
        color_continuous_scale=[(0.0, "#d64541"), (0.5, "#f39c12"), (1.0, "#2e8b57")],
        range_color=[0, 10],
    )
    fig.update_layout(height=360, coloraxis_showscale=False, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    fig.update_traces(textposition="outside", hovertemplate="%{y}: %{x:.2f}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

    display_df = df.sort_values("Score", ascending=False)
    st.dataframe(display_df, use_container_width=False, hide_index=True)


def get_supabase_client():
    return create_client(
        st.secrets["SUPABASE_URL"].strip(),
        st.secrets["SUPABASE_KEY"].strip(),
    )


def employee_upload() -> None:
    st.title("Upload Candidate")
    supabase = get_supabase_client()
    roles_response = supabase.table("roles").select("*").execute()
    roles_data = roles_response.data or []
    role_names = ["Select Role"] + [r["role_name"] for r in roles_data if r.get("role_name")]

    selected_role = st.selectbox("Role", role_names, index=0)
    selected_role_description = ""
    if selected_role != "Select Role":
        selected_role_description = next(
            r.get("role_description", "")
            for r in roles_data
            if r.get("role_name") == selected_role
        )

    st.text_area("Role Description", value=selected_role_description, disabled=True)
    name = st.text_input("Candidate Name")
    dob = st.text_input("DOB (DD/MM/YYYY)")
    cv = st.file_uploader("Upload CV", type=["pdf", "docx"], key="employee_cv_upload")
    personal_excel = st.file_uploader("Upload Personal Excel", type=["xlsx"], key="employee_excel_upload")

    if st.button("Submit Candidate", key="employee_submit_candidate"):
        if selected_role == "Select Role":
            st.error("Please select role")
            return

        cv_path = ""
        excel_path = ""
        if cv:
            safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", cv.name)
            cv_path = f"resumes/{int(datetime.now().timestamp())}_{safe_name}"
            supabase.storage.from_("candidates-files").upload(cv_path, cv.getvalue())

        if personal_excel:
            safe_excel_name = re.sub(r"[^a-zA-Z0-9._-]", "_", personal_excel.name)
            excel_path = f"excel/{int(datetime.now().timestamp())}_{safe_excel_name}"
            supabase.storage.from_("candidates-files").upload(excel_path, personal_excel.getvalue())

        birth, destiny, month = calculate_numbers_from_dob(dob)
        result = evaluate_candidate_for_role(
            birth,
            destiny,
            month,
            selected_role,
            selected_role_description,
        )

        score = float(getattr(result, "overall_score_100", 0.0))
        verdict = str(getattr(result, "risk_status", "FAIL")).upper()

        supabase.table("Candidates").insert({
            "name": name,
            "dob": dob,
            "role": selected_role,
            "role_description": selected_role_description,
            "cv_url": cv_path,
            "personal_excel_url": excel_path if personal_excel else None,
            "score": score,
            "verdict": verdict,
            "stage": "Review Pending",
        }).execute()

        if verdict == "PASS":
            st.success("😊 Suitable")
        else:
            st.warning("🤔 Needs Review")


def my_candidates_page() -> None:
    st.title("My Candidates")
    supabase = get_supabase_client()
    response = supabase.table("Candidates").select("*").execute()
    if not response.data:
        st.info("No candidates uploaded yet.")
        return
    df = pd.DataFrame(response.data)
    columns = [c for c in ["name", "role", "verdict", "stage"] if c in df.columns]
    if not columns:
        st.info("No candidate data available.")
        return
    df = df[columns]
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    st.dataframe(df, use_container_width=True, hide_index=True)


def database_page() -> None:
    st.title("Candidate Database")
    supabase = get_supabase_client()
    candidates = supabase.table("Candidates").select("*").execute()
    df = pd.DataFrame(candidates.data or [])
    if df.empty:
        st.info("No candidates found.")
        return

    def _to_title(value: object) -> str:
        text = str(value or "").strip()
        return text.title() if text else ""

    if "status" not in df.columns:
        df["status"] = ""

    role_values = sorted(
        {str(role).strip() for role in df.get("role", pd.Series(dtype=str)).dropna() if str(role).strip()}
    )
    if not role_values:
        st.info("No roles found in candidate data.")
        return

    for role_name in role_values:
        role_df = df[df["role"].astype(str).str.strip() == role_name].copy()
        if role_df.empty:
            continue

        display_df = pd.DataFrame(
            {
                "Name": role_df["name"].map(_to_title) if "name" in role_df.columns else "",
                "Role": role_df["role"].map(_to_title) if "role" in role_df.columns else "",
                "Verdict": role_df["verdict"].fillna("").astype(str).str.upper() if "verdict" in role_df.columns else "",
                "Stage": role_df["stage"].map(_to_title) if "stage" in role_df.columns else "",
                "Status": role_df["status"].map(_to_title),
            }
        )

        st.markdown(f"## Role: {_to_title(role_name)}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def single_analysis_page() -> None:
    query_params = st.query_params
    auto_dob = query_params.get("dob", "")
    auto_role = query_params.get("role", "")
    auto_role_desc = query_params.get("role_description", "")
    auto_run = bool(auto_dob)

    st.title("Hiring Scorer")
    st.markdown(
        """
<style>
[data-testid="stTable"] table,
[data-testid="stDataFrame"] table {
    width: auto !important;
    table-layout: auto !important;
}
[data-testid="stTable"] td,
[data-testid="stTable"] th,
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th {
    white-space: nowrap !important;
}

.stMarkdown p,
.stMarkdown li,
.stCaption,
[data-testid="stAlertContent"] {
    font-size: 1.05rem !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Inputs")
        if "candidate_name_input" not in st.session_state:
            st.session_state["candidate_name_input"] = ""
        if "candidate_dob_input" not in st.session_state:
            st.session_state["candidate_dob_input"] = ""
        if auto_dob and not st.session_state["candidate_dob_input"]:
            st.session_state["candidate_dob_input"] = auto_dob
        if "last_parsed_resume" not in st.session_state:
            st.session_state["last_parsed_resume"] = ""

        resume_file = st.file_uploader("Upload Resume (.pdf or .txt)", type=["pdf", "txt"], key="resume_upload")
        if resume_file:
            resume_key = f"{resume_file.name}:{resume_file.size}"
            if st.session_state["last_parsed_resume"] != resume_key:
                try:
                    resume_text_for_prefill = extract_resume_text(resume_file)
                    auto_messages: List[str] = []

                    if not st.session_state["candidate_name_input"].strip():
                        detected_name = extract_candidate_name_from_text(resume_text_for_prefill)
                        if detected_name:
                            st.session_state["candidate_name_input"] = detected_name
                            auto_messages.append(f"Auto-filled candidate name: {detected_name}")

                    if not st.session_state["candidate_dob_input"].strip():
                        detected_dob = extract_dob_from_text(resume_text_for_prefill)
                        if detected_dob:
                            st.session_state["candidate_dob_input"] = detected_dob
                            auto_messages.append(f"Auto-filled candidate DOB: {detected_dob}")

                    for message in auto_messages:
                        st.caption(message)

                    st.session_state["last_parsed_resume"] = resume_key
                except Exception as exc:
                    st.warning(f"Resume auto-parse failed: {exc}")

        candidate_name = st.text_input("Candidate Name", key="candidate_name_input")

        role_name = st.text_input("Role Name", value=auto_role)
        role_description = st.text_area("Role Description", value=auto_role_desc, height=130)
        candidate_dob_input = st.text_input("Candidate DOB (dd/mm/yyyy)", key="candidate_dob_input")
        leader_dob_input = st.text_input("Leader DOB (dd/mm/yyyy)", value="03/11/1994")
        role_type = st.selectbox("Role Type", ["Execution Focused", "Strategy Focused"])

        run_scorecard = st.button("Run Rule-Based Scorecard", type="primary", use_container_width=True)
        run_ai = st.button("Run AI Analysis (OpenAI)", use_container_width=True)
        if auto_run and not run_scorecard and not run_ai:
            run_scorecard = True

    if not run_scorecard and not run_ai:
        st.info("Fill inputs, then run scorecard. Use AI Analysis button only when you want OpenAI output.")
        return
    if auto_run and run_scorecard and not run_ai:
        st.success("Running analysis automatically...")
    if not candidate_name.strip():
        candidate_name = "Candidate"
    if not role_name.strip():
        st.error("Role Name is required.")
        return
    try:
        candidate_dob_text = validate_dob(candidate_dob_input)
        leader_dob_text = validate_dob(leader_dob_input)
    except ValueError as exc:
        st.error(str(exc))
        return

    result = score_candidate(
        candidate_dob=candidate_dob_text,
        leader_dob=leader_dob_text,
        role_type=role_type,
    )

    meta_style = "font-size:17px;"
    st.markdown(f'<span style="{meta_style}">📍 <b>Candidate Name:</b> {candidate_name}</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="{meta_style}">📍 <b>Candidate DOB:</b> {candidate_dob_text}</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="{meta_style}">📍 <b>Role Type:</b> {role_type}</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="{meta_style}">📍 <b>Role:</b> {role_name}</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="{meta_style}">📍 <b>Role Description:</b> {role_description}</span>', unsafe_allow_html=True)
    st.markdown(f'<span style="{meta_style}">📍 <b>Verdict Logic:</b> (80+ strong, 65-79 guardrails, &lt;65 risk)</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## 🧠 Candidate Assessment Summary")
    st.markdown("### 🔢 Summary")
    st.markdown(
        f"Birth Total: **{result.candidate_numbers['birth']}** | "
        f"Month Total: **{result.candidate_numbers['month']}** | "
        f"Destiny Total: **{result.candidate_numbers['destiny']}**"
    )
    st.markdown("---")

    st.markdown("### 📊 Raw Trait Scores (0-10)")
    trait_rows = [
        {
            "Trait": f"{TRAIT_EMOJIS.get(trait, '•')} {trait}",
            "Score": f"{result.trait_scores[trait]:.1f}",
            "Tag": strength_tag(result.trait_scores[trait]),
        }
        for trait in TRAITS
    ]
    raw_rows_html = "".join(
        f"<tr><td>{row['Trait']}</td><td>{row['Score']}</td><td>{row['Tag']}</td></tr>"
        for row in trait_rows
    )
    st.markdown(
        f"""
<style>
.raw-traits-table {{
  border-collapse: collapse;
  width: auto;
}}
.raw-traits-table th, .raw-traits-table td {{
  border: 1px solid #ddd;
  padding: 8px 12px;
  font-size: 19px;
  white-space: nowrap;
}}
.raw-traits-table th {{
  background: #f7f7f7;
  font-weight: 700;
}}
</style>
<table class="raw-traits-table">
  <thead>
    <tr><th>Trait</th><th>Score</th><th>Tag</th></tr>
  </thead>
  <tbody>
    {raw_rows_html}
  </tbody>
</table>
""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### 📊 Composite Bar Chart (Conceptual)")
    chart_order = [
        "Finish Execution",
        "Loyalty",
        "Self Ownership",
        "Politics Safe",
        "Communication",
        "Strategy & Analytical",
    ]
    lines = []
    for name in chart_order:
        score = result.composite_scores[name]
        lines.append(f"{name:<22}{conceptual_bar(score)} {score:.1f} ({strength_tag(score)})")
    st.code("\n".join(lines), language="text")
    st.markdown("---")

    st.markdown("## 🎯 Suitability Verdict")
    if result.risk_status == "FAIL":
        st.error("Risk Status: FAIL")
        st.error("High Risk – Do Not Proceed")
        for rf in result.risk_flags:
            st.markdown(f"- {rf}")
        display_verdict = "High Risk – Do Not Proceed"
        verdict_icon = "🔴"
    else:
        st.success("Core Stability: PASS")
        display_verdict = result.verdict
        verdict_icon = verdict_badge(result.verdict)
    st.markdown(f"### {verdict_icon} {display_verdict}")
    if result.risk_status == "FAIL":
        score_color = "#d64541"
    else:
        score_color = "#2e8b57" if result.verdict == "Strong Hire" else ("#f39c12" if result.verdict == "Hire w/ Guardrails" else "#d64541")
    st.markdown(
        f'<span style="font-size:1.8rem;font-weight:400;color:#000000;">Overall Score ({role_type}):</span> '
        f'<span style="font-size:3rem;font-weight:900;color:{score_color};">{result.overall_score_100:.1f}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(verdict_sentence(result.verdict, role_type))
    st.markdown("---")

    st.markdown("## Loyalty Summary")
    st.markdown("### LOYALTY PROFILE")
    lm = result.loyalty_meta
    org_label = label_score(lm["org_loyalty"])
    leadership_label = label_score(lm["leadership_loyalty"])
    authority_label = label_score(lm["authority_alignment"])
    peer_label = label_score(10 - lm["peer_influence"])
    loyalty_profile_label = classify_loyalty_profile(
        lm["org_loyalty"],
        lm["leadership_loyalty"],
        lm["authority_alignment"],
        lm["peer_influence"],
    )
    st.markdown(f"- **Organizational Loyalty:** {lm['org_loyalty']:.1f} / 10 -> {org_label}")
    st.markdown(f"- **Leadership Loyalty:** {lm['leadership_loyalty']:.1f} / 10 -> {leadership_label}")
    st.markdown("")
    st.markdown(f"- **Authority Alignment:** {lm['authority_alignment']:.1f} -> {authority_label}")
    st.markdown(f"- **Peer Influence Risk:** {lm['peer_influence']:.1f} -> {peer_label}")
    st.markdown("")
    st.markdown(f"- **Overall Loyalty Stability Score:** {lm['final_loyalty']:.1f} / 10")
    st.markdown(f"- **Loyalty Profile Classification:** {loyalty_profile_label}")
    if lm["leadership_loyalty"] < 6 and lm["org_loyalty"] > 7.5:
        st.warning("WARNING: Loyal to system but not leadership")
    st.markdown(f"- **Baseline Calculation:** ({result.trait_scores['Loyalty']:.1f} + {result.trait_scores['Politics Safe']:.1f}) / 2 = {lm['baseline_loyalty']:.1f}")
    st.markdown(f"- **Leadership Penalty/Bonus Applied:** {lm['leadership_penalty']:.1f}")
    st.markdown(f"- **Authority Alignment Index:** {lm['authority_alignment']:.1f}")
    st.markdown(f"- **Peer Influence Susceptibility:** {lm['peer_influence']:.1f}")
    st.markdown(f"- **Final Calculation:** (org {lm['org_loyalty']:.1f} * 0.6) + (leadership {lm['leadership_loyalty']:.1f} * 0.4) = {lm['final_loyalty']:.1f}")
    loyalty_notes: List[str] = []
    penalty_triggered = False
    if result.candidate_numbers["destiny"] == 1 and result.leader_numbers["destiny"] == 1:
        loyalty_notes.append("Authority conflict (1 vs 1): -2")
        penalty_triggered = True
    if result.candidate_numbers["month"] == 7:
        loyalty_notes.append("Silent politics risk (month 7): -1")
        penalty_triggered = True
    if result.candidate_numbers["destiny"] == 5:
        loyalty_notes.append("Volatility risk (destiny 5): -1")
        penalty_triggered = True
    if result.candidate_numbers["destiny"] in {4, 8, 9}:
        loyalty_notes.append("Organizational stability bonus (destiny 4/8/9): +0.5")
        if not penalty_triggered:
            loyalty_notes.append("Leadership stability bonus (destiny 4/8/9, no penalty triggered): +0.3")
    st.markdown("---")

    all_triggered_rules = []
    all_triggered_rules.extend(result.flags)
    all_triggered_rules.extend(loyalty_notes)
    if all_triggered_rules:
        st.warning("⚠ Risk Rules Triggered")
        for item in all_triggered_rules:
            st.markdown(f"- {item}")
        st.markdown("---")

    if not run_ai:
        return

    resume_text = ""
    if resume_file:
        try:
            resume_text = extract_resume_text(resume_file)
        except Exception as exc:
            st.error(f"Resume parsing failed: {exc}")
            return

    openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY", "")

    if not openai_api_key.strip():
        st.info("OpenAI API key not found. AI sections disabled.")
        return
    openai.api_key = openai_api_key

    try:
        ai_result = run_openai_resume_review(
            api_key=openai_api_key,
            role_name=role_name,
            role_description=role_description,
            role_type=role_type,
            resume_text=resume_text,
            candidate_numbers=result.candidate_numbers,
            trait_scores=result.trait_scores,
            composite_scores=result.composite_scores,
            overall_score_100=result.overall_score_100,
            verdict=result.verdict,
        )
        st.markdown("## Role x Candidate Fit")
        role_traits_ai = generate_role_specific_traits(openai_api_key, role_name, role_description)
        role_traits: List[str] = []
        if not role_traits_ai:
            role_traits = list(DEFAULT_TRAITS)
        else:
            for trait in role_traits_ai:
                if trait not in CORE_TRAITS and trait not in role_traits:
                    role_traits.append(trait)
            fallback_traits = [t for t in TRAIT_ATTRIBUTE_MAP.keys() if t not in CORE_TRAITS]
            for trait in fallback_traits:
                if 4 <= len(role_traits) <= 8:
                    break
                if trait not in role_traits:
                    role_traits.append(trait)
            role_traits = role_traits[:8]
            if len(role_traits) < 4:
                role_traits = list(DEFAULT_TRAITS)

        attribute_scores = calculate_all_attribute_scores(result.candidate_numbers, result.trait_scores)
        trait_map = load_trait_attribute_map()
        for trait in role_traits:
            trait_map = ensure_trait_mapping(openai_api_key, trait, trait_map)

        role_trait_scores: Dict[str, float] = {
            trait: calculate_trait_score(trait, attribute_scores, trait_map)
            for trait in role_traits
        }
        role_trait_weights = generate_role_trait_weights(openai_api_key, role_name, role_description, role_traits)
        role_score_10 = calculate_role_score_weighted(role_trait_scores, role_trait_weights)
        role_score_100 = round(role_score_10 * 10.0, 1)

        st.markdown("**Role-Specific Traits (AI generated, validated 4-8):**")
        st.markdown(", ".join(role_traits))
        st.markdown(f"**Role Suitability Score:** {role_score_100:.1f}/100")

        score_calc_rows = []
        total_contribution = 0.0
        for trait in role_traits:
            weight = role_trait_weights.get(trait, 0.0)
            trait_score = role_trait_scores.get(trait, 0.0)
            contribution = trait_score * weight
            total_contribution += contribution
            score_calc_rows.append(
                {
                    "Trait": trait,
                    "Trait Score": f"{conceptual_bar(trait_score)} {trait_score:.1f}",
                    "Weight": f"{weight * 100:.1f}%",
                    "Contribution": f"{contribution:.1f}",
                }
            )
        score_calc_rows.append(
            {
                "Trait": "Total",
                "Trait Score": "-",
                "Weight": f"{sum(role_trait_weights.values()) * 100:.1f}%",
                "Contribution": f"{total_contribution:.1f}",
            }
        )
        st.markdown("**Score Calculation**")
        st.table(pd.DataFrame(score_calc_rows))
        st.markdown("---")

        st.markdown(f"## 🔥 Strengths for {role_name} Role")
        for item in ai_result.get("strengths_for_role", []):
            st.markdown(f"✔ {normalize_decimals(str(item))}")
        st.markdown("---")

        st.markdown("## ⚠ Traits to Validate (Interview Focus)")
        low_traits = sorted(result.trait_scores.items(), key=lambda x: x[1])[:3]
        st.markdown("**Key Raw Trait Risks (from rule engine):**")
        for trait_name, trait_value in low_traits:
            st.markdown(
                f"- **{trait_name} ({trait_value:.1f})**: {strength_tag(trait_value)}"
            )
        st.markdown("")
        for item in ai_result.get("traits_to_validate", []):
            st.markdown(f"🔹 {normalize_decimals(str(item))}")
        st.markdown("---")

        st.markdown("## 🛡 Recommended Guardrails (If Hired)")
        for item in ai_result.get("recommended_guardrails", []):
            st.markdown(f"• {normalize_decimals(str(item))}")
        st.markdown("---")

    except Exception as exc:
        st.error(f"OpenAI analysis failed: {exc}")


def unified_app() -> None:
    if "user" not in st.session_state:
        login()
        st.stop()

    user_role = st.session_state.user["role"]

    if user_role == "admin":
        pages = [
            "Single Analysis",
            "Roles and Candidates",
            "Compare Candidates",
            "Candidate Database",
            "Upload Candidate",
            "Role Master",
            "User Management",
        ]
    else:
        pages = [
            "Upload Candidate",
            "My Candidates",
            "Role Master",
        ]

    with st.sidebar:
        st.markdown(f"**Logged in:** {st.session_state.user.get('email', '')}")
        if st.button("Logout"):
            del st.session_state["user"]
            st.rerun()
        if "nav" in st.session_state:
            selection = st.session_state["nav"]
            del st.session_state["nav"]
        else:
            selection = st.selectbox("Navigation", pages)

    if selection == "Upload Candidate":
        employee_upload()
    elif selection == "My Candidates":
        my_candidates_page()
    elif selection == "Role Master":
        render_role_master()
    elif selection == "Single Analysis":
        single_analysis_page()
    elif selection == "Compare Candidates":
        render_compare()
    elif selection == "Candidate Database":
        database_page()
    elif selection == "User Management":
        render_user_mgmt()
    elif selection == "Roles and Candidates":
        render_roles_candidates()


if __name__ == "__main__":
    unified_app()
