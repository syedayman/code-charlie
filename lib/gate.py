"""
Password gate for the Code Charlie Streamlit app.

Single shared password (settings.GATE_PASSWORD). On match, sets
st.session_state["authenticated"] so subsequent reruns skip the prompt.
"""

import streamlit as st

from core.config import settings


def require_password() -> bool:
    """Render gate UI if needed. Returns True if user is authenticated.

    When False, the caller should st.stop() so nothing else renders.
    """
    if st.session_state.get("authenticated"):
        return True

    st.markdown("## Code Charlie")
    st.caption("Building-code compliance research chatbot. Enter the access password to continue.")

    with st.form("gate_form", clear_on_submit=False):
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Unlock")

    if submitted:
        if password and password == settings.GATE_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False
