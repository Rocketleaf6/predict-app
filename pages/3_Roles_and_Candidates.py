#!/usr/bin/env python3

import urllib.parse

import pandas as pd
import streamlit as st
try:
    from supabase import create_client
except ModuleNotFoundError:
    st.set_page_config(page_title="Roles and Candidates Dashboard", layout="wide")
    st.title("Roles and Candidates Dashboard")
    st.error("Missing dependency: supabase. Install with: `pip install supabase`")
    st.stop()

if "user" not in st.session_state:
    st.error("Login required.")
    st.stop()

if st.session_state.user.get("role") != "admin":
    st.error("Admin access required.")
    st.stop()


st.set_page_config(page_title="Roles and Candidates Dashboard", layout="wide")
st.title("Roles and Candidates Dashboard")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

response = None
for table_name in ("Candidates", "candidates"):
    try:
        response = supabase.table(table_name).select("*").execute()
        break
    except Exception:
        response = None

if response is None:
    st.error("Could not read table from Supabase. Tried: Candidates, candidates")
    st.stop()

df = pd.DataFrame(response.data)

if df.empty:
    st.warning("No candidates found in Supabase table: candidates")
    st.stop()

required_cols = {"role", "name", "dob", "role_description", "cv_url", "personal_excel_url"}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in Supabase candidates table: {', '.join(missing)}")
    st.stop()

roles = df["role"].dropna().unique()
selected_role = st.selectbox("Select Role", roles)
role_df = df[df["role"] == selected_role]

st.header(f"Candidates for {selected_role}")

selected_dobs = []
selected_names = []

for i, row in role_df.iterrows():
    col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 3, 1, 1, 1])

    col1.write(row["name"])
    col2.write(row["dob"])
    col3.write(row["role_description"])
    cv_url = supabase.storage.from_("files").get_public_url(row["cv_url"]) if row.get("cv_url") else ""
    excel_url = (
        supabase.storage.from_("files").get_public_url(row["personal_excel_url"])
        if row.get("personal_excel_url")
        else ""
    )
    col4.markdown(f"[Open CV]({cv_url})" if cv_url else "-")
    col5.markdown(f"[Open Data]({excel_url})" if excel_url else "-")

    params = urllib.parse.urlencode(
        {
            "dob": row["dob"],
            "role": row["role"],
            "role_description": row["role_description"],
        }
    )
    analyze_url = f"/?{params}"

    col6.markdown(
        f'<a href="{analyze_url}" target="_blank">Analyze</a>',
        unsafe_allow_html=True,
    )

    if st.checkbox("Select", key=f"candidate_select_{i}"):
        selected_dobs.append(str(row["dob"]))
        selected_names.append(str(row["name"]))

if selected_dobs:
    role_desc_for_compare = ""
    if not role_df.empty:
        role_desc_for_compare = str(role_df.iloc[0]["role_description"])
    params = urllib.parse.urlencode(
        {
            "dobs": ",".join(selected_dobs),
            "names": ",".join(selected_names),
            "role": str(selected_role),
            "role_description": role_desc_for_compare,
        }
    )
    compare_url = f"/Compare_Candidates?{params}"
    st.markdown(
        f'<a href="{compare_url}" target="_blank">Compare Selected</a>',
        unsafe_allow_html=True,
    )
