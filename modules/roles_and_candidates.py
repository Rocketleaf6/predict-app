#!/usr/bin/env python3
"""Roles and candidates dashboard module."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from supabase import create_client


DISPLAY_COLUMNS = ["Name", "Role", "Verdict", "Stage", "Status", "Action"]


def _to_title(value: object) -> str:
    text = str(value or "").strip()
    return text.title() if text else ""


def _load_candidates(supabase) -> pd.DataFrame:
    response = None
    for table_name in ("Candidates", "candidates"):
        try:
            response = supabase.table(table_name).select("*").execute()
            if response is not None:
                break
        except Exception:
            response = None
    if response is None:
        return pd.DataFrame()
    df = pd.DataFrame(response.data or [])
    if df.empty:
        return df
    if "status" not in df.columns:
        df["status"] = ""
    return df


def _update_candidate_status(supabase, candidate_id: object, status: str) -> None:
    for table_name in ("Candidates", "candidates"):
        try:
            supabase.table(table_name).update(
                {
                    "status": status,
                    "stage": "Finalized",
                }
            ).eq("id", candidate_id).execute()
            return
        except Exception:
            continue
    st.error("Could not update candidate status in Supabase.")


def _render_actions(supabase, candidate_id: object, candidate_name: str) -> None:
    st.caption(f"Update status for {_to_title(candidate_name)}")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Go Ahead", key=f"go_ahead_{candidate_id}", use_container_width=True):
            _update_candidate_status(supabase, candidate_id, "Go Ahead")
            st.rerun()

    with col2:
        if st.button("Waitlist", key=f"waitlist_{candidate_id}", use_container_width=True):
            _update_candidate_status(supabase, candidate_id, "Waitlist")
            st.rerun()

    with col3:
        if st.button("Reject", key=f"reject_{candidate_id}", use_container_width=True):
            _update_candidate_status(supabase, candidate_id, "Reject")
            st.rerun()


def render() -> None:
    if "user" not in st.session_state:
        st.error("Login required.")
        st.stop()
    if st.session_state.user.get("role") != "admin":
        st.error("Admin access required.")
        st.stop()

    st.title("Roles and Candidates Dashboard")

    supabase = create_client(
        st.secrets["SUPABASE_URL"].strip(),
        st.secrets["SUPABASE_KEY"].strip(),
    )

    df = _load_candidates(supabase)
    if df.empty:
        st.info("No candidates found.")
        return

    required_cols = {"id", "name", "role", "verdict", "stage", "status"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns in candidates table: {', '.join(missing)}")
        return

    role_values = sorted(
        {str(role).strip() for role in df["role"].dropna() if str(role).strip()}
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
                "Name": role_df["name"].map(_to_title),
                "Role": role_df["role"].map(_to_title),
                "Verdict": role_df["verdict"].fillna("").astype(str).str.upper(),
                "Stage": role_df["stage"].map(_to_title),
                "Status": role_df["status"].map(_to_title),
                "Action": "Use Buttons Below",
            }
        )

        st.markdown(f"## Role: {_to_title(role_name)}")
        st.dataframe(display_df[DISPLAY_COLUMNS], use_container_width=True, hide_index=True)

        for _, row in role_df.iterrows():
            st.caption(
                f"{_to_title(row['name'])} | "
                f"{_to_title(row['role'])} | "
                f"{str(row.get('verdict', '') or '').upper()} | "
                f"{_to_title(row.get('stage', ''))} | "
                f"{_to_title(row.get('status', ''))}"
            )
            _render_actions(supabase, row["id"], row["name"])
            st.markdown("---")
