#!/usr/bin/env python3
"""Roles and candidates dashboard module."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from supabase import create_client


ACTION_OPTIONS = ["Select Action", "Go Ahead", "Waitlist", "Reject"]


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
    status_value = "Rejected" if action_value == "Reject" else action_value
    for table_name in ("Candidates", "candidates"):
        try:
            supabase.table(table_name).update(
                {
                    "status": status_value,
                    "stage": "Finalized",
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

        selected_ids = []
        for idx, row in role_df.iterrows():
            checkbox_label = _to_title(row["name"]) or f"Candidate {idx + 1}"
            if st.checkbox(checkbox_label, key=f"compare_select_{role_name}_{row['id']}"):
                selected_ids.append(row["id"])

        if st.button("Analyze Selected Candidates", key=f"analyze_selected_{role_name}"):
            if not selected_ids:
                st.warning("Select at least one candidate.")
            else:
                selected_rows = role_df[role_df["id"].isin(selected_ids)].copy()
                role_description = str(selected_rows.iloc[0].get("role_description", "") or "")
                _prepare_compare_session(selected_rows, role_name, role_description)
                st.switch_page("modules/compare_candidates.py")

        editor_df = pd.DataFrame(
            {
                "Name": role_df["name"].map(_to_title),
                "Role": role_df["role"].map(_to_title),
                "Verdict": role_df["verdict"].fillna("").astype(str).str.upper(),
                "Stage": role_df["stage"].map(_to_title),
                "Status": role_df["status"].map(_to_title),
                "Action": "Select Action",
            }
        )

        edited_df = st.data_editor(
            editor_df,
            use_container_width=True,
            hide_index=True,
            key=f"role_editor_{role_name}",
            column_config={
                "Action": st.column_config.SelectboxColumn(
                    "Action",
                    options=ACTION_OPTIONS,
                )
            },
            disabled=["Name", "Role", "Verdict", "Stage", "Status"],
        )

        for idx, row in edited_df.iterrows():
            action_value = str(row.get("Action", "Select Action")).strip()
            if action_value == "Select Action":
                continue

            candidate_id = role_df.iloc[idx]["id"]
            if _update_candidate_status(supabase, candidate_id, action_value):
                st.rerun()
            else:
                st.error("Could not update candidate status.")
                return
