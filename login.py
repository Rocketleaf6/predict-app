import extra_streamlit_components as stx
import streamlit as st
from supabase import create_client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


def get_cookie_manager():
    if "global_cookie_manager" not in st.session_state:
        st.session_state["global_cookie_manager"] = stx.CookieManager(key="global_cookie_manager")
    return st.session_state["global_cookie_manager"]


def restore_login_from_cookie():
    if "logged_in" in st.session_state and st.session_state.logged_in:
        return

    cookie_manager = get_cookie_manager()
    username = cookie_manager.get("username")

    if username:
        st.session_state.logged_in = True
        st.session_state.username = username

        try:
            res = (
                supabase.table("users")
                .select("*")
                .eq("email", username)
                .limit(1)
                .execute()
            )
            if res.data:
                st.session_state.user = res.data[0]
            else:
                st.session_state.logged_in = False
                st.session_state.username = None
                cookie_manager.delete("username")
        except Exception:
            st.session_state.logged_in = False
            st.session_state.username = None


def logout():
    cookie_manager = get_cookie_manager()
    cookie_manager.delete("username")

    st.session_state.logged_in = False
    st.session_state.username = None
    if "user" in st.session_state:
        del st.session_state["user"]

    st.rerun()


def login():
    st.title("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        res = (
            supabase.table("users")
            .select("*")
            .eq("email", email)
            .eq("password", password)
            .execute()
        )

        if res.data:
            username = email
            cookie_manager = get_cookie_manager()
            cookie_manager.set("username", username)
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.user = res.data[0]
            st.rerun()
        else:
            st.error("Invalid login")
