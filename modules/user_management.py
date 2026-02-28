#!/usr/bin/env python3
"""User management module."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from supabase import create_client


def render() -> None:
    if "user" not in st.session_state:
        st.error("Login required.")
        st.stop()
    if st.session_state.user.get("role") != "admin":
        st.error("Admin access required.")
        st.stop()

    st.title("User Management")
    supabase = create_client(st.secrets["SUPABASE_URL"].strip(), st.secrets["SUPABASE_KEY"].strip())

    st.subheader("Create User")
    email = st.text_input("Email")
    password = st.text_input("Password")
    role = st.selectbox("Role", ["employee", "admin"])

    if st.button("Create User"):
        supabase.table("users").insert({"email": email, "password": password, "role": role}).execute()
        st.success("User created")

    st.subheader("Existing Users")
    users = supabase.table("users").select("*").execute()
    st.dataframe(pd.DataFrame(users.data or []), use_container_width=True, hide_index=True)
