#!/usr/bin/env python3
"""Candidate database module."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def _get_app():
    import app as app_module
    return app_module


def _to_title(value: object) -> str:
    text = str(value or "").strip()
    return text.title() if text else ""


def render() -> None:
    if "user" not in st.session_state:
        st.error("Login required.")
        st.stop()
    if st.session_state.user.get("role") != "admin":
        st.error("Admin access required.")
        st.stop()

    app = _get_app()
    st.title("Candidate Database")
    supabase = app.get_supabase_client()
    candidates = supabase.table("Candidates").select("*").execute()
    df = pd.DataFrame(candidates.data or [])
    if df.empty:
        st.info("No candidates found.")
        return

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
