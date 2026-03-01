#!/usr/bin/env python3
"""Roles and candidates dashboard module."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from supabase import create_client


DECISION_OPTIONS = ["Review Pending", "Go Ahead", "Waitlist", "Reject"]


def _to_title(value: object) -> str:
    text = str(value or "").strip()
    return text.title() if text else ""


def _load_candidates(supabase):
    response = None
    table_used = None
    for table_name in ("Candidates", "candidates"):
        try:
            response = supabase.table(table_name).select("*").execute()
            table_used = table_name
            break
        except Exception:
            response = None
    return pd.DataFrame(response.data or []), table_used


def _update_candidate_status(supabase, candidate_id: object, action_value: str) -> bool:
    for table_name in ("Candidates", "candidates"):
        try:
            supabase.table(table_name).update(
                {
                    "status": action_value,
                }
            ).eq("id", candidate_id).execute()
            return True
        except Exception:
            continue
    return False


def _prepare_compare_session(selected_rows: pd.DataFrame, role_name: str, role_description: str) -> None:
    dobs = [str(value).strip() for value in selected_rows["dob"].tolist() if str(value).strip()]
    names = [str(value or "").strip() for value in selected_rows["name"].tolist()]
    ids = [value for value in selected_rows["id"].tolist()]

    st.session_state["compare_candidate_ids"] = ids
    st.session_state["compare_candidates"] = [
        {"name": names[idx], "dob": dobs[idx]}
        for idx in range(min(len(dobs), len(names)))
    ]
    st.session_state["compare_candidates_initialized"] = True

    st.query_params["dobs"] = ",".join(dobs)
    st.query_params["names"] = ",".join(names)
    st.query_params["role"] = role_name
    st.query_params["role_description"] = role_description


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

    df, table_used = _load_candidates(supabase)
    if df.empty:
        st.info("No candidates found.")
        return
    if table_used is None:
        st.error("Could not read candidates table from Supabase.")
        return

    required_cols = {"id", "name", "role", "dob", "verdict", "stage"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns in candidates table: {', '.join(missing)}")
        return

    if "status" not in df.columns:
        df["status"] = ""
    if "role_description" not in df.columns:
        df["role_description"] = ""

    role_values = sorted(
        {str(role).strip() for role in df["role"].dropna() if str(role).strip()}
    )

    for role_name in role_values:
        role_df = df[df["role"].astype(str).str.strip() == role_name].copy().reset_index(drop=True)
        if role_df.empty:
            continue

        st.markdown(f"## Role: {_to_title(role_name)}")
        display_df = role_df.copy()
        display_df["Select"] = False
        display_df["Decision"] = display_df["status"].fillna("Review Pending")
        display_df["Decision"] = display_df["Decision"].replace("", "Review Pending")
        display_df["name"] = display_df["name"].map(_to_title)
        display_df["role"] = display_df["role"].map(_to_title)
        display_df["verdict"] = display_df["verdict"].fillna("").astype(str).str.upper()
        display_df["stage"] = display_df["stage"].map(_to_title)

        edited_df = st.data_editor(
            display_df[
                [
                    "Select",
                    "name",
                    "role",
                    "verdict",
                    "stage",
                    "Decision",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            key=f"role_editor_{role_name}",
            column_config={
                "Select": st.column_config.CheckboxColumn("Select"),
                "Decision": st.column_config.SelectboxColumn(
                    "Decision",
                    options=DECISION_OPTIONS,
                    required=True,
                ),
            },
            disabled=["name", "role", "verdict", "stage"],
        )

        for idx, row in edited_df.iterrows():
            candidate_id = role_df.iloc[idx]["id"]
            new_decision = str(row.get("Decision", "Review Pending")).strip() or "Review Pending"
            old_decision = str(role_df.iloc[idx].get("status", "") or "").strip() or "Review Pending"

            if new_decision == old_decision:
                continue

            if _update_candidate_status(supabase, candidate_id, new_decision):
                st.rerun()
            else:
                st.error("Could not update candidate status.")
                return

        selected_rows = edited_df[edited_df["Select"] == True]

        if st.button("Analyze Selected Candidates", key=f"analyze_selected_{role_name}"):
            if len(selected_rows) == 0:
                st.warning("Please select candidates")
            else:
                selected_ids = role_df.iloc[selected_rows.index]["id"]
                selected_full = role_df[role_df["id"].isin(selected_ids)].copy()
                selected_role_name = str(selected_full.iloc[0]["role"])
                role_description = str(selected_full.iloc[0]["role_description"] or "")

                _prepare_compare_session(
                    selected_full,
                    selected_role_name,
                    role_description,
                )

                st.session_state["nav"] = "Compare Candidates"
                st.session_state["compare_ready"] = True
                st.rerun()
