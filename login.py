import base64
import datetime as dt
import hashlib
import hmac
import json
import time

import extra_streamlit_components as stx
import streamlit as st
from supabase import create_client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
COOKIE_NAME = "numerology_auth"
COOKIE_TTL_DAYS = 14

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


@st.cache_resource
def get_cookie_manager():
    return stx.CookieManager()


def _auth_secret() -> str:
    return (
        st.secrets.get("AUTH_COOKIE_SECRET", "")
        or st.secrets.get("APP_PASSWORD", "")
        or SUPABASE_KEY
    )


def _encode_segment(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")


def _decode_segment(payload: str) -> bytes:
    padding = "=" * (-len(payload) % 4)
    return base64.urlsafe_b64decode(payload + padding)


def _sign_segment(segment: str) -> str:
    digest = hmac.new(
        _auth_secret().encode("utf-8"),
        segment.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _encode_segment(digest)


def _create_auth_token(user: dict) -> str:
    payload = {
        "id": user["id"],
        "email": user.get("email", ""),
        "role": user.get("role", ""),
        "exp": int(time.time()) + COOKIE_TTL_DAYS * 24 * 60 * 60,
    }
    payload_segment = _encode_segment(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _sign_segment(payload_segment)
    return f"{payload_segment}.{signature}"


def _verify_auth_token(token: str) -> dict | None:
    try:
        payload_segment, signature = token.split(".", 1)
        expected_signature = _sign_segment(payload_segment)
        if not hmac.compare_digest(signature, expected_signature):
            return None
        payload = json.loads(_decode_segment(payload_segment).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def restore_login_from_cookie() -> None:
    if "user" in st.session_state:
        return
    cookies = get_cookie_manager()
    token = cookies.get(COOKIE_NAME)
    if not token:
        return

    payload = _verify_auth_token(token)
    if not payload:
        cookies.delete(COOKIE_NAME)
        return

    try:
        res = (
            supabase.table("users")
            .select("*")
            .eq("id", payload["id"])
            .limit(1)
            .execute()
        )
    except Exception:
        return

    if res.data:
        st.session_state["user"] = res.data[0]
    else:
        cookies.delete(COOKIE_NAME)


def logout_user() -> None:
    get_cookie_manager().delete(COOKIE_NAME)
    for key in ("user", "nav", "selected_nav"):
        if key in st.session_state:
            del st.session_state[key]


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
            st.session_state.user = res.data[0]
            get_cookie_manager().set(
                COOKIE_NAME,
                _create_auth_token(res.data[0]),
                expires_at=dt.datetime.utcnow() + dt.timedelta(days=COOKIE_TTL_DAYS),
            )
            st.rerun()
        else:
            st.error("Invalid login")
