#!/usr/bin/env python3
"""Role master module."""

from __future__ import annotations

import streamlit as st
from supabase import create_client


def render() -> None:
    if "user" not in st.session_state:
        st.error("Login required.")
        st.stop()

    user_role = st.session_state.user.get("role")
    st.title("Role Master")
    supabase = create_client(st.secrets["SUPABASE_URL"].strip(), st.secrets["SUPABASE_KEY"].strip())

    st.subheader("Create New Role")
    role_name_input = st.text_input("Role Name", key="module_role_name_input")
    role_description_input = st.text_area("Role Description", key="module_role_description_input")

    if st.button("Save Role", key="module_save_role"):
        if not role_name_input or not role_description_input:
            st.error("Enter role name and description")
        else:
            supabase.table("roles").insert({"role_name": role_name_input, "role_description": role_description_input}).execute()
            st.success("Role saved successfully")

    st.subheader("Existing Roles")
    roles_response = supabase.table("roles").select("*").execute()
    roles_data = roles_response.data or []

    for role in roles_data:
        if st.button(role["role_name"], key=f"module_edit_role_{role['id']}"):
            st.session_state["module_edit_role_id"] = role["id"]

    if "module_edit_role_id" in st.session_state:
        role_id = st.session_state["module_edit_role_id"]
        role_data = next(r for r in roles_data if r["id"] == role_id)

        @st.dialog("Edit Role")
        def edit_role_dialog():
            new_name = st.text_input("Role Name", value=role_data["role_name"])
            new_desc = st.text_area("Role Description", value=role_data["role_description"])
            col1, col2 = st.columns(2)
            with col1:
                if user_role == "admin" and st.button("Save", key=f"module_save_edit_{role_id}"):
                    supabase.table("roles").update({"role_name": new_name, "role_description": new_desc}).eq("id", role_id).execute()
                    del st.session_state["module_edit_role_id"]
                    st.rerun()
            with col2:
                if st.button("Cancel", key=f"module_cancel_edit_{role_id}"):
                    del st.session_state["module_edit_role_id"]
                    st.rerun()
            if user_role == "admin" and st.button("Delete Role", key=f"module_delete_role_{role_id}"):
                supabase.table("roles").delete().eq("id", role_id).execute()
                del st.session_state["module_edit_role_id"]
                st.rerun()

        edit_role_dialog()
