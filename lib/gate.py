"""
Password gate for the Code Charlie Streamlit app.

Single shared password (settings.GATE_PASSWORD). On match, sets
st.session_state["authenticated"] so subsequent reruns skip the prompt.
"""

import base64
from pathlib import Path

import streamlit as st

from core.config import settings

INTRO_IMAGE_PATH = Path(__file__).resolve().parents[1] / "code-charlie.png"
DISPLAY_FONT_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "fonts"
    / "ClashGrotesk-Variable.woff2"
)


def _file_data_uri(path: Path, mime_type: str) -> str | None:
    if not path.exists():
        return None

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def require_password() -> bool:
    """Render gate UI if needed. Returns True if user is authenticated.

    When False, the caller should st.stop() so nothing else renders.
    """
    if st.session_state.get("authenticated"):
        return True

    font_data_uri = _file_data_uri(DISPLAY_FONT_PATH, "font/woff2")
    if font_data_uri:
        st.html(
            f"""
<style>
@font-face {{
    font-family: 'Clash Grotesk';
    src: url('{font_data_uri}') format('woff2');
    font-weight: 200 700;
    font-display: swap;
}}
</style>
"""
        )

    st.html(
        """
<style>
.stApp {
    background:
        radial-gradient(ellipse at top, rgba(99, 102, 241, 0.09), transparent 48%),
        radial-gradient(ellipse at bottom right, rgba(168, 85, 247, 0.08), transparent 58%),
        #020617 !important;
    color: #e2e8f0 !important;
}
#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}
html, body, .stApp, .stApp [data-testid="stAppViewContainer"] {
    height: 100vh;
    overflow: hidden;
}
.stApp [data-testid="stAppViewContainer"] > .main,
.stApp section.main {
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
.block-container {
    max-width: 48rem !important;
    height: 100vh !important;
    max-height: 100vh !important;
    padding-top: 0.75rem !important;
    padding-bottom: 0.75rem !important;
    display: flex !important;
    flex-direction: column;
    justify-content: center;
    gap: 0.4rem;
    overflow: hidden;
}
.stApp h1 {
    color: #f8fafc !important;
    text-align: center;
    font-weight: 700 !important;
    letter-spacing: 0;
    margin-bottom: 0.25rem !important;
}
.stApp [data-testid="stCaptionContainer"],
.stApp .stCaption,
.stApp p,
.stApp label {
    color: #94a3b8 !important;
}
.stApp [data-testid="stImage"] {
    display: flex;
    justify-content: center;
}
.stApp [data-testid="stImage"] img {
    filter: drop-shadow(0 0 42px rgba(56, 189, 248, 0.22));
}
.stApp [data-testid="stElementContainer"]:has(.gate-logo-wrap) {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    width: 100%;
}
.gate-logo-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    height: 100%;
    margin: 0;
}
.gate-logo-wrap img {
    display: block;
    width: auto;
    height: 100%;
    max-height: min(calc(100vh - 24rem), 90vw);
    max-width: 90vw;
    object-fit: contain;
    filter: drop-shadow(0 0 60px rgba(56, 189, 248, 0.28));
}
.gate-title-wrap {
    display: flex;
    justify-content: center;
    width: 100%;
    text-align: center;
    margin: 0;
}
.gate-title-wrap .gate-title {
    color: #f8fafc !important;
    text-align: center !important;
    font-family: 'Clash Grotesk', Inter, system-ui, sans-serif !important;
    font-size: 2.25rem;
    line-height: 1.05;
    font-weight: 700;
    letter-spacing: 0;
    margin: 0 !important;
}
.gate-subtitle {
    margin: 0 !important;
    font-size: 0.9rem;
}
.gate-description {
    margin: 0 0 0.25rem !important;
    font-size: 0.85rem;
}
.gate-subtitle,
.gate-description {
    text-align: center;
}
.stApp [data-testid="stForm"] {
    background: rgba(15, 23, 42, 0.62) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 1rem !important;
    backdrop-filter: blur(14px);
    padding: 0.9rem !important;
}
.stApp [data-testid="stTextInput"] input {
    background: rgba(2, 6, 23, 0.62) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #f8fafc !important;
    border-radius: 0.6rem !important;
    padding-right: 3.5rem !important;
}
.stApp [data-testid="stTextInput"] input::placeholder {
    color: #94a3b8 !important;
    opacity: 0.85 !important;
}
.stApp [data-testid="stTextInput"] [data-testid="InputInstructions"],
.stApp [data-testid="stTextInput"] [class*="InputInstructions"],
.stApp [data-testid="stTextInput"] [aria-live="polite"] {
    display: none !important;
}
.stApp [data-testid="stTextInput"] button {
    margin-right: 0.35rem !important;
}
.stApp [data-testid="stTextInput"] input:focus {
    border-color: rgba(129, 140, 248, 0.48) !important;
    box-shadow: none !important;
}
.stApp [data-testid="stForm"] .stButton button {
    background: rgba(99, 102, 241, 0.82) !important;
    border: 1px solid rgba(129, 140, 248, 0.5) !important;
    color: #ffffff !important;
    border-radius: 0.65rem !important;
    width: 100%;
}
</style>
""",
    )

    image_data_uri = _file_data_uri(INTRO_IMAGE_PATH, "image/png")
    if image_data_uri:
        st.html(f'<div class="gate-logo-wrap"><img src="{image_data_uri}" alt=""></div>')

    st.html('<div class="gate-title-wrap"><h1 class="gate-title">CODE CHARLIE</h1></div>')
    st.markdown(
        '<p class="gate-subtitle">Powered by Medha 1.0 - KARR AI</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='gate-description'>"
        "Vertical Transportation (VT) Building-code Compliance Research"
        "</p>",
        unsafe_allow_html=True,
    )

    with st.form("gate_form", clear_on_submit=False):
        password = st.text_input(
            "Password",
            type="password",
            autocomplete="current-password",
            placeholder="Enter password",
        )
        submitted = st.form_submit_button("Unlock")

    if submitted:
        if password and password == settings.GATE_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False
