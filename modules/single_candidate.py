#!/usr/bin/env python3
"""Single candidate analysis module."""

from __future__ import annotations

import os
from typing import Dict, List

import openai
import pandas as pd
import streamlit as st


def _get_app():
    import app as app_module
    return app_module


def render() -> None:
    app = _get_app()

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
                    resume_text_for_prefill = app.extract_resume_text(resume_file)
                    auto_messages: List[str] = []

                    if not st.session_state["candidate_name_input"].strip():
                        detected_name = app.extract_candidate_name_from_text(resume_text_for_prefill)
                        if detected_name:
                            st.session_state["candidate_name_input"] = detected_name
                            auto_messages.append(f"Auto-filled candidate name: {detected_name}")

                    if not st.session_state["candidate_dob_input"].strip():
                        detected_dob = app.extract_dob_from_text(resume_text_for_prefill)
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
        candidate_dob_text = app.validate_dob(candidate_dob_input)
        leader_dob_text = app.validate_dob(leader_dob_input)
    except ValueError as exc:
        st.error(str(exc))
        return

    result = app.score_candidate(
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
            "Trait": f"{app.TRAIT_EMOJIS.get(trait, '•')} {trait}",
            "Score": f"{result.trait_scores[trait]:.1f}",
            "Tag": app.strength_tag(result.trait_scores[trait]),
        }
        for trait in app.TRAITS
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
        lines.append(f"{name:<22}{app.conceptual_bar(score)} {score:.1f} ({app.strength_tag(score)})")
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
        verdict_icon = app.verdict_badge(result.verdict)
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
    st.markdown(app.verdict_sentence(result.verdict, role_type))
    st.markdown("---")

    st.markdown("## Loyalty Summary")
    st.markdown("### LOYALTY PROFILE")
    lm = result.loyalty_meta
    org_label = app.label_score(lm["org_loyalty"])
    leadership_label = app.label_score(lm["leadership_loyalty"])
    authority_label = app.label_score(lm["authority_alignment"])
    peer_label = app.label_score(10 - lm["peer_influence"])
    loyalty_profile_label = app.classify_loyalty_profile(
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
            resume_text = app.extract_resume_text(resume_file)
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
        ai_result = app.run_openai_resume_review(
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
        role_traits_ai = app.generate_role_specific_traits(openai_api_key, role_name, role_description)
        role_traits: List[str] = []
        if not role_traits_ai:
            role_traits = list(app.DEFAULT_TRAITS)
        else:
            for trait in role_traits_ai:
                if trait not in app.CORE_TRAITS and trait not in role_traits:
                    role_traits.append(trait)
            fallback_traits = [t for t in app.TRAIT_ATTRIBUTE_MAP.keys() if t not in app.CORE_TRAITS]
            for trait in fallback_traits:
                if 4 <= len(role_traits) <= 8:
                    break
                if trait not in role_traits:
                    role_traits.append(trait)
            role_traits = role_traits[:8]
            if len(role_traits) < 4:
                role_traits = list(app.DEFAULT_TRAITS)

        attribute_scores = app.calculate_all_attribute_scores(result.candidate_numbers, result.trait_scores)
        trait_map = app.load_trait_attribute_map()
        for trait in role_traits:
            trait_map = app.ensure_trait_mapping(openai_api_key, trait, trait_map)

        role_trait_scores: Dict[str, float] = {
            trait: app.calculate_trait_score(trait, attribute_scores, trait_map)
            for trait in role_traits
        }
        role_trait_weights = app.generate_role_trait_weights(openai_api_key, role_name, role_description, role_traits)
        role_score_10 = app.calculate_role_score_weighted(role_trait_scores, role_trait_weights)
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
                    "Trait Score": f"{app.conceptual_bar(trait_score)} {trait_score:.1f}",
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
            st.markdown(f"✔ {app.normalize_decimals(str(item))}")
        st.markdown("---")

        st.markdown("## ⚠ Traits to Validate (Interview Focus)")
        low_traits = sorted(result.trait_scores.items(), key=lambda x: x[1])[:3]
        st.markdown("**Key Raw Trait Risks (from rule engine):**")
        for trait_name, trait_value in low_traits:
            st.markdown(f"- **{trait_name} ({trait_value:.1f})**: {app.strength_tag(trait_value)}")
        st.markdown("")
        for item in ai_result.get("traits_to_validate", []):
            st.markdown(f"🔹 {app.normalize_decimals(str(item))}")
        st.markdown("---")

        st.markdown("## 🛡 Recommended Guardrails (If Hired)")
        for item in ai_result.get("recommended_guardrails", []):
            st.markdown(f"• {app.normalize_decimals(str(item))}")
        st.markdown("---")
    except Exception as exc:
        st.error(f"OpenAI analysis failed: {exc}")
