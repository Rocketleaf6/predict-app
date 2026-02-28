#!/usr/bin/env python3
"""Multi-Candidate comparison dashboard."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

if "user" not in st.session_state:
    st.error("Login required.")
    st.stop()

if st.session_state.user.get("role") != "admin":
    st.error("Admin access required.")
    st.stop()


# Import from app.py without modifying app.py
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
spec = importlib.util.spec_from_file_location("numerology_app_main", str(APP_PATH))
app = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["numerology_app_main"] = app
spec.loader.exec_module(app)


def calculate_numbers_from_dob(dob: str) -> Tuple[int, int, int]:
    """Wrapper: validate and compute birth/destiny/month numbers."""
    normalized = app.validate_dob(dob)
    nums = app.numerology_numbers(normalized)
    return nums["birth"], nums["destiny"], nums["month"]


def _build_role_fit_for_candidate(
    role_name: str,
    role_description: str,
    candidate_numbers: Dict[str, int],
    trait_scores: Dict[str, float],
    api_key: str,
) -> Tuple[float, List[str], Dict[str, float], Dict[str, float]]:
    """Deterministic role-fit score with optional AI trait/weight generation."""
    role_traits: List[str] = []

    if api_key:
        try:
            role_traits_ai = app.generate_role_specific_traits(api_key, role_name, role_description)
            for trait in role_traits_ai:
                if trait not in app.CORE_TRAITS and trait not in role_traits:
                    role_traits.append(trait)
        except Exception:
            role_traits = []

    if not role_traits:
        role_traits = list(app.DEFAULT_TRAITS)

    fallback_traits = [t for t in app.TRAIT_ATTRIBUTE_MAP.keys() if t not in app.CORE_TRAITS]
    for trait in fallback_traits:
        if 4 <= len(role_traits) <= 8:
            break
        if trait not in role_traits:
            role_traits.append(trait)
    role_traits = role_traits[:8]

    attribute_scores = app.calculate_all_attribute_scores(candidate_numbers, trait_scores)
    trait_map = app.load_trait_attribute_map()
    for trait in role_traits:
        trait_map = app.ensure_trait_mapping(api_key, trait, trait_map) if api_key else trait_map
        if trait not in trait_map:
            trait_map[trait] = {"Execution": 0.4, "Consistency": 0.3, "Integrity": 0.3}

    role_trait_scores = {
        trait: app.calculate_trait_score(trait, attribute_scores, trait_map)
        for trait in role_traits
    }

    if api_key:
        try:
            role_trait_weights = app.generate_role_trait_weights(api_key, role_name, role_description, role_traits)
        except Exception:
            role_trait_weights = {t: 1.0 / len(role_traits) for t in role_traits}
    else:
        role_trait_weights = {t: 1.0 / len(role_traits) for t in role_traits}

    total = sum(role_trait_weights.values())
    if total <= 0:
        role_trait_weights = {t: 1.0 / len(role_traits) for t in role_traits}
    else:
        role_trait_weights = {k: v / total for k, v in role_trait_weights.items()}

    role_score_10 = app.calculate_role_score_weighted(role_trait_scores, role_trait_weights)
    return role_score_10, role_traits, role_trait_scores, attribute_scores


def evaluate_candidate_for_role(
    dob: str,
    leadership_dob: str,
    role_name: str,
    role_description: str,
) -> Dict[str, object]:
    """Reusable candidate evaluation for comparison page."""
    normalized_dob = app.validate_dob(dob)
    normalized_leader_dob = app.validate_dob(leadership_dob)
    score_result = app.score_candidate(normalized_dob, normalized_leader_dob, "Execution Focused")

    api_key = os.getenv("OPENAI_API_KEY", "")
    role_score_10, role_traits, role_trait_scores, attribute_scores = _build_role_fit_for_candidate(
        role_name=role_name,
        role_description=role_description,
        candidate_numbers=score_result.candidate_numbers,
        trait_scores=score_result.trait_scores,
        api_key=api_key,
    )

    return {
        "dob": normalized_dob,
        "numbers": score_result.candidate_numbers,
        "trait_scores": score_result.trait_scores,
        "composite_scores": score_result.composite_scores,
        "overall_score_100": score_result.overall_score_100,
        "attribute_scores": attribute_scores,
        "role_traits": role_traits,
        "role_trait_scores": role_trait_scores,
        "role_score": role_score_10,
    }


def color_score(val):
    try:
        v = float(val)
    except Exception:
        return ""
    if v >= 7:
        return "background-color: #c6efce; color: #006100;"  # green
    if v >= 5:
        return "background-color: #ffeb9c; color: #9c6500;"  # yellow
    return "background-color: #ffc7ce; color: #9c0006;"  # red


def recommendation_label(score: float) -> str:
    if score >= 8:
        return "Strong Hire Recommendation"
    if score >= 6:
        return "Moderate Hire Recommendation"
    return "Risk Hire Recommendation"


def _short_dob_label(dob: str) -> str:
    parts = dob.split("/")
    if len(parts) == 3:
        return f"{parts[0]}/{parts[1]}"
    return dob


def _display_candidate_label(dob: str, name_map: Dict[str, str]) -> str:
    name = name_map.get(dob, "").strip()
    if name:
        return name
    return _short_dob_label(dob)


def _get_shared_role_traits_and_weights(role_name: str, role_description: str, api_key: str) -> Tuple[List[str], Dict[str, float]]:
    role_traits: List[str] = []
    if api_key:
        try:
            role_traits_ai = app.generate_role_specific_traits(api_key, role_name, role_description)
            for trait in role_traits_ai:
                if trait not in app.CORE_TRAITS and trait not in role_traits:
                    role_traits.append(trait)
        except Exception:
            role_traits = []

    if not role_traits:
        role_traits = list(app.DEFAULT_TRAITS)

    fallback_traits = [t for t in app.TRAIT_ATTRIBUTE_MAP.keys() if t not in app.CORE_TRAITS]
    for trait in fallback_traits:
        if 4 <= len(role_traits) <= 8:
            break
        if trait not in role_traits:
            role_traits.append(trait)
    role_traits = role_traits[:8]
    if len(role_traits) < 4:
        role_traits = list(app.DEFAULT_TRAITS)

    if api_key:
        try:
            role_trait_weights = app.generate_role_trait_weights(api_key, role_name, role_description, role_traits)
        except Exception:
            role_trait_weights = {t: 1.0 / len(role_traits) for t in role_traits}
    else:
        role_trait_weights = {t: 1.0 / len(role_traits) for t in role_traits}

    total = sum(role_trait_weights.values())
    if total <= 0:
        role_trait_weights = {t: 1.0 / len(role_traits) for t in role_traits}
    else:
        role_trait_weights = {k: v / total for k, v in role_trait_weights.items()}
    return role_traits, role_trait_weights


def main() -> None:
    query_params = st.query_params
    auto_dobs = query_params.get("dobs", "")
    auto_names = query_params.get("names", "")
    auto_role = query_params.get("role", "")
    auto_role_desc = query_params.get("role_description", "")
    auto_run_compare = bool(auto_dobs)

    st.set_page_config(page_title="Compare Candidates", layout="wide")
    st.title("Compare Candidates")

    st.subheader("ROLE INPUT")
    role_name = st.text_input("Role Name", value=auto_role)
    role_description = st.text_area("Role Description", value=auto_role_desc, height=120)
    leadership_dob = st.text_input("Leadership DOB (dd/mm/yyyy)", value="03/11/1994")

    st.subheader("CANDIDATE INPUT")
    if "compare_candidates_initialized" not in st.session_state:
        parsed_dobs = [d.strip() for d in auto_dobs.split(",") if d.strip()] if auto_dobs else []
        parsed_names = [n.strip() for n in auto_names.split(",")] if auto_names else []
        if parsed_dobs:
            st.session_state["compare_candidates"] = []
            for i, dob in enumerate(parsed_dobs):
                name = parsed_names[i] if i < len(parsed_names) else ""
                st.session_state["compare_candidates"].append({"name": name, "dob": dob})
        else:
            st.session_state["compare_candidates"] = [{"name": "", "dob": ""}]
        st.session_state["compare_candidates_initialized"] = True

    if st.button("+ Add Candidate"):
        st.session_state["compare_candidates"].append({"name": "", "dob": ""})

    for idx, _candidate in enumerate(st.session_state["compare_candidates"]):
        st.markdown(f"**Candidate {idx + 1}**")
        c1, c2 = st.columns(2)
        name_key = f"compare_candidate_name_{idx}"
        dob_key = f"compare_candidate_dob_{idx}"
        if name_key not in st.session_state:
            st.session_state[name_key] = _candidate["name"]
        if dob_key not in st.session_state:
            st.session_state[dob_key] = _candidate["dob"]
        c1.text_input("Name", key=name_key)
        c2.text_input("DOB (dd/mm/yyyy)", key=dob_key)

    compare = st.button("COMPARE", type="primary")

    if auto_run_compare and not compare:
        st.success("Running comparison automatically...")
    if not compare and not auto_run_compare:
        return

    if not role_name.strip():
        st.error("Please enter role name.")
        return
    if not role_description.strip():
        role_description = "Not provided"

    try:
        leadership_dob = app.validate_dob(leadership_dob)
    except Exception:
        st.error("Please enter a valid Leadership DOB in dd/mm/yyyy format.")
        return

    dobs = []
    names = []
    for idx in range(len(st.session_state["compare_candidates"])):
        dob = st.session_state.get(f"compare_candidate_dob_{idx}", "").strip()
        name = st.session_state.get(f"compare_candidate_name_{idx}", "").strip()
        if dob:
            dobs.append(dob)
            names.append(name)
    if not dobs:
        st.error("Please enter at least one candidate DOB.")
        return
    query_name_map: Dict[str, str] = {}
    for idx, dob in enumerate(dobs):
        if idx < len(names) and names[idx]:
            query_name_map[dob] = names[idx]

    results: List[Dict[str, object]] = []
    invalid_dobs: List[str] = []
    for dob in dobs:
        try:
            _ = calculate_numbers_from_dob(dob)
            eval_result = evaluate_candidate_for_role(dob, leadership_dob, role_name, role_description)
            results.append(eval_result)
        except Exception:
            invalid_dobs.append(dob)

    if invalid_dobs:
        st.warning(f"Skipped invalid DOB(s): {', '.join(invalid_dobs)}")

    if not results:
        st.error("No valid candidates to compare.")
        return

    candidate_dobs = [r["dob"] for r in results]
    api_key = os.getenv("OPENAI_API_KEY", "")
    role_traits, role_trait_weights = _get_shared_role_traits_and_weights(role_name, role_description, api_key)
    trait_map = app.load_trait_attribute_map()
    for trait in role_traits:
        trait_map = app.ensure_trait_mapping(api_key, trait, trait_map) if api_key else trait_map
        if trait not in trait_map:
            trait_map[trait] = {"Execution": 0.4, "Consistency": 0.3, "Integrity": 0.3}

    candidate_role_trait_scores: Dict[str, Dict[str, float]] = {}
    candidate_final_role_scores: Dict[str, float] = {}
    for r in results:
        dob = r["dob"]
        attrs = app.calculate_all_attribute_scores(r["numbers"], r["trait_scores"])
        trait_scores = {trait: app.calculate_trait_score(trait, attrs, trait_map) for trait in role_traits}
        candidate_role_trait_scores[dob] = trait_scores
        candidate_final_role_scores[dob] = app.calculate_role_score_weighted(trait_scores, role_trait_weights)

    st.markdown("---")
    st.subheader("Raw Trait Scores (0-10) - Comparative")
    raw_rows: List[Dict[str, object]] = []
    for trait in app.TRAITS:
        row: Dict[str, object] = {"Trait": f"{app.TRAIT_EMOJIS.get(trait, '•')} {trait}"}
        best_dob = None
        best_score = -1.0
        for r in results:
            dob = r["dob"]
            score = round(float(r["trait_scores"][trait]), 1)
            row[dob] = score
            if score > best_score:
                best_score = score
                best_dob = dob
        row["Winner"] = _display_candidate_label(best_dob or "", query_name_map)
        raw_rows.append(row)
    overall_row: Dict[str, object] = {"Trait": "Overall Score (/100)"}
    best_overall_dob = None
    best_overall_score = -1.0
    for r in results:
        dob = r["dob"]
        overall_score = round(float(r["overall_score_100"]), 1)
        overall_row[dob] = overall_score
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            best_overall_dob = dob
    overall_row["Winner"] = _display_candidate_label(best_overall_dob or "", query_name_map)
    raw_rows.append(overall_row)
    raw_df = pd.DataFrame(raw_rows)
    raw_score_cols = [c for c in raw_df.columns if c not in ("Trait", "Winner")]
    raw_styler = raw_df.style.map(color_score, subset=raw_score_cols).format({c: "{:.1f}" for c in raw_score_cols})
    st.dataframe(raw_styler, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("ROLE SCORE TABLE")
    role_rows: List[Dict[str, object]] = []
    for trait in role_traits:
        row: Dict[str, object] = {"Trait": trait, "Weight": f"{role_trait_weights.get(trait, 0.0) * 100:.0f}%"}
        best_dob = None
        best_score = -1.0
        for dob in candidate_dobs:
            score = round(float(candidate_role_trait_scores[dob].get(trait, 0.0)), 1)
            row[dob] = score
            if score > best_score:
                best_score = score
                best_dob = dob
        row["Weighted Winner"] = _display_candidate_label(best_dob or "", query_name_map)
        role_rows.append(row)

    final_row: Dict[str, object] = {"Trait": "Final weighted score", "Weight": "100%"}
    best_final_dob = None
    best_final_score = -1.0
    for dob in candidate_dobs:
        score = round(float(candidate_final_role_scores[dob]), 1)
        final_row[dob] = score
        if score > best_final_score:
            best_final_score = score
            best_final_dob = dob
    final_row["Weighted Winner"] = _display_candidate_label(best_final_dob or "", query_name_map)
    role_rows.append(final_row)
    final_row_100: Dict[str, object] = {"Trait": "Role Score (/100)", "Weight": "100%"}
    best_final_100_dob = None
    best_final_100_score = -1.0
    for dob in candidate_dobs:
        score_100 = round(float(candidate_final_role_scores[dob] * 10.0), 1)
        final_row_100[dob] = score_100
        if score_100 > best_final_100_score:
            best_final_100_score = score_100
            best_final_100_dob = dob
    final_row_100["Weighted Winner"] = _display_candidate_label(best_final_100_dob or "", query_name_map)
    role_rows.append(final_row_100)

    role_df = pd.DataFrame(role_rows)
    role_score_cols = [c for c in role_df.columns if c not in ("Trait", "Weight", "Weighted Winner")]
    role_styler = role_df.style.map(color_score, subset=role_score_cols).format({c: "{:.1f}" for c in role_score_cols})
    st.dataframe(role_styler, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Role Score Chart (Comparative)")
    role_chart_rows = []
    for trait in role_traits:
        for dob in candidate_dobs:
            role_chart_rows.append(
                {
                    "Trait": trait,
                    "DOB": dob,
                    "Score": round(float(candidate_role_trait_scores[dob].get(trait, 0.0)), 1),
                }
            )
    role_chart_df = pd.DataFrame(role_chart_rows)
    role_fig = px.bar(
        role_chart_df,
        x="Trait",
        y="Score",
        color="DOB",
        barmode="group",
        text=role_chart_df["Score"].map(lambda x: f"{x:.1f}"),
    )
    role_fig.update_traces(textposition="outside")
    st.plotly_chart(role_fig, use_container_width=True)

    st.markdown("---")
    composite_rows = []
    for r in results:
        for comp_name, comp_value in r["composite_scores"].items():
            value = round(float(comp_value), 1)
            composite_rows.append(
                {
                    "DOB": r["dob"],
                    "Composite": comp_name,
                    "Score": value,
                    "Tag": app.strength_tag(value),
                    "Conceptual": f"{app.conceptual_bar(value)} {value:.1f} ({app.strength_tag(value)})",
                }
            )
    composite_df = pd.DataFrame(composite_rows)

    st.markdown("---")
    st.subheader("WINNER DISPLAY")
    winner_dob = max(candidate_final_role_scores, key=candidate_final_role_scores.get)
    winner_score = round(float(candidate_final_role_scores[winner_dob]), 1)
    winner_label = _display_candidate_label(winner_dob, query_name_map)
    st.success(f"Best Candidate: {winner_label} | Final weighted score: {winner_score:.1f}")
    st.markdown(f"**{recommendation_label(float(winner_score))}**")

    st.markdown("---")
    st.subheader("Composite Bar Chart (Comparative)")
    chart_df = composite_df.copy()
    fig = px.bar(
        chart_df,
        x="Composite",
        y="Score",
        color="DOB",
        barmode="group",
        text=chart_df["Score"].map(lambda x: f"{x:.1f}"),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
