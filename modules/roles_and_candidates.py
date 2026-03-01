#!/usr/bin/env python3
"""Roles and candidates dashboard module."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from supabase import create_client


DISPLAY_COLUMNS = ["Name", "Role", "Verdict", "Stage", "Status"]


def _to_proper_case(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(part.capitalize() for part in text.split())


def _load_candidates(supabase):
    response = None
    table_name_used = None
    for table_name in ("Candidates", "candidates"):
        try:
            response = supabase.table(table_name).select("*").execute()
            table_name_used = table_name
            break
        except Exception:
            response = None
    return response, table_name_used


def _update_candidate_status(supabase, table_name: str, candidate_id, status: str) -> None:
    supabase.table(table_name).update({
        "status": status,
        "stage": "Reviewed",
    }).eq("id", candidate_id).execute()


def _render_role_table(role_name: str, role_df: pd.DataFrame, supabase, table_name: str) -> None:
    st.markdown(f"## {role_name}")

    display_df = role_df.copy()
    display_df["name"] = display_df["name"].map(_to_proper_case)
    display_df["role"] = display_df["role"].map(_to_proper_case)
    display_df["verdict"] = display_df["verdict"].map(_to_proper_case)
    display_df["stage"] = display_df["stage"].map(_to_proper_case)
    display_df["status"] = display_df["status"].map(_to_proper_case)

    table_df = display_df[["name", "role", "verdict", "stage", "status"]].copy()
    table_df.columns = DISPLAY_COLUMNS
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    header_cols = st.columns([2.2, 2.0, 1.4, 1.4, 1.4, 1.2])
    header_cols[0].markdown("**Name**")
    header_cols[1].markdown("**Role**")
    header_cols[2].markdown("**Verdict**")
    header_cols[3].markdown("**Stage**")
    header_cols[4].markdown("**Status**")
    header_cols[5].markdown("**Action**")

    for _, candidate in display_df.iterrows():
        candidate_id = candidate.get("id")
        row_cols = st.columns([2.2, 2.0, 1.4, 1.4, 1.4, 1.2])
        row_cols[0].write(candidate["name"])
        row_cols[1].write(candidate["role"])
        row_cols[2].write(candidate["verdict"])
        row_cols[3].write(candidate["stage"])
        row_cols[4].write(candidate["status"])

        with row_cols[5]:
            action_cols = st.columns(3)
            if action_cols[0].button("Go Ahead", key=f"go_ahead_{candidate_id}"):
                _update_candidate_status(supabase, table_name, candidate_id, "Go Ahead")
                st.rerun()
            if action_cols[1].button("Waitlist", key=f"waitlist_{candidate_id}"):
                _update_candidate_status(supabase, table_name, candidate_id, "Waitlist")
                st.rerun()
            if action_cols[2].button("Reject", key=f"reject_{candidate_id}"):
                _update_candidate_status(supabase, table_name, candidate_id, "Rejected")
                st.rerun()

    st.markdown("---")


def render() -> None:
    if "user" not in st.session_state:
        st.error("Login required.")
        st.stop()
    if st.session_state.user.get("role") != "admin":
        st.error("Admin access required.")
        st.stop()

    st.title("Roles and Candidates Dashboard")
    supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

    response, table_name = _load_candidates(supabase)
    if response is None or table_name is None:
        st.error("Could not read table from Supabase. Tried: Candidates, candidates")
        return

    df = pd.DataFrame(response.data or [])
    if df.empty:
        st.warning("No candidates found in Supabase table: candidates")
        return

    required_cols = {"id", "name", "role", "verdict", "stage"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in Supabase candidates table: {', '.join(missing)}")
        return

    if "status" not in df.columns:
        df["status"] = ""

    df["role"] = df["role"].fillna("")
    grouped_roles = sorted([role for role in df["role"].unique().tolist() if str(role).strip()])

    if not grouped_roles:
        st.info("No roles found in candidate data.")
        return

    for role_name in grouped_roles:
        role_df = df[df["role"] == role_name].copy()
        _render_role_table(_to_proper_case(role_name), role_df, supabase, table_name)
