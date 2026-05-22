"""
Code Charlie — Streamlit UI

Single-file app:
  - password gate
  - sidebar with multi-session list + new chat button
  - chat history replayed from LangGraph checkpoint
  - sources expander under each assistant message
  - clarification chips when classifier asks "which code?"
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import streamlit as st

from agent.graph import (
    build_code_charlie_graph,
    get_code_charlie_state,
    initialize_code_charlie_session_checkpoint,
    invoke_code_charlie_graph,
)
from agent.messages import add_user_message
from agent.state import (
    INTRO_MESSAGE_CONTENT,
    create_initial_code_charlie_state,
)
from core.config import settings
from lib.gate import require_password
from lib.sessions import (
    auto_title_session,
    create_session,
    delete_session,
    get_session,
    list_sessions,
    rename_session,
    update_after_message,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTRO_IMAGE_PATH = Path(__file__).with_name("code-charlie.png")
DISPLAY_FONT_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "fonts"
    / "ClashGrotesk-Variable.woff2"
)


def _file_data_uri(path: Path, mime_type: str) -> str | None:
    if not path.exists():
        return None

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(
    page_title="Code Charlie",
    page_icon=str(INTRO_IMAGE_PATH) if INTRO_IMAGE_PATH.exists() else "🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Gate
# =============================================================================

if not require_password():
    st.stop()


# =============================================================================
# Eager warm-up after auth (mask first-render DB/pool latency)
# =============================================================================


def _checkpoint_error_message(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


@st.cache_resource(show_spinner=False)
def _warm_graph() -> bool:
    """Build the LangGraph + open PostgresSaver pool once per worker."""
    build_code_charlie_graph()
    return True


if not st.session_state.get("_warmed"):
    splash = st.empty()
    with splash.container():
        st.markdown("### Code Charlie")
        st.caption("Connecting…")
        with st.spinner(""):
            try:
                _warm_graph()
                st.session_state.pop("_checkpoint_error", None)
            except Exception as exc:
                logger.exception("Code Charlie database warm-up failed")
                st.session_state["_checkpoint_error"] = _checkpoint_error_message(exc)
    splash.empty()
    st.session_state["_warmed"] = True


# =============================================================================
# Global CSS — sidebar polish + hover-only actions
# =============================================================================

display_font_data_uri = _file_data_uri(DISPLAY_FONT_PATH, "font/woff2")
if display_font_data_uri:
    st.html(
        f"""
<style>
@font-face {{
    font-family: 'Clash Grotesk';
    src: url('{display_font_data_uri}') format('woff2');
    font-weight: 200 700;
    font-display: swap;
}}
</style>
"""
    )

st.html(
    """
<script>
/* Hard-remove the Streamlit Cloud "profile chip" + "Hosted with Streamlit"
   badge. CSS hides may lose to overlay inline styles; nuking the nodes
   wins. MutationObserver re-runs on every DOM change since the cloud
   overlay can re-inject these elements after navigation. */
(function () {
  var SELECTORS = [
    '[class*="_profileContainer_"]',
    '[class*="_viewerBadge_"]',
    'a[href*="streamlit.io/cloud"]'
  ].join(',');
  function nuke() {
    document.querySelectorAll(SELECTORS).forEach(function (el) { el.remove(); });
  }
  nuke();
  try {
    new MutationObserver(nuke).observe(document.documentElement, {
      childList: true,
      subtree: true
    });
  } catch (e) {}
})();
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* =========================================================================
   KARR AI Code Charlie reskin — dark slate base, indigo accents, glassmorph.
   Streamlit 1.40+ exposes `st-key-{key}` classes on keyed widgets which we
   target along with the stable `data-testid` attrs.
   ========================================================================= */

/* ---- Global base ---- */
/* Set font on root containers only — wildcard `*` would clobber Material
   Symbols font and break `:material/icon:` rendering. Children inherit. */
html, body, .stApp, .block-container,
section[data-testid="stSidebar"] {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top, rgba(99, 102, 241, 0.08), transparent 50%),
        radial-gradient(ellipse at bottom right, rgba(168, 85, 247, 0.06), transparent 60%),
        #020617 !important;
    color: #e2e8f0 !important;
}

/* Hide default Streamlit chrome, but keep the header alive for sidebar reopen. */
#MainMenu,
footer {
    visibility: hidden;
    height: 0;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.75rem !important;
    visibility: visible !important;
    z-index: 999999 !important;
    pointer-events: auto !important;
}
header[data-testid="stHeader"] [data-testid="stDecoration"],
header[data-testid="stHeader"] [data-testid="stStatusWidget"] {
    display: none !important;
}

/* Streamlit Community Cloud deploy toolbar siblings (Fork / Share / GitHub /
   Deploy / kebab). Keep parent stToolbar visible — the sidebar reopen
   chevron (stExpandSidebarButton) lives inside it. Only nuke its noisy
   children. */
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stMainMenuButton"],
[data-testid="stActionButton"],
[data-testid="stAppViewerBadge"],
[data-testid="stForkButton"],
[data-testid="stForkAppButton"],
[data-testid="stAppForkButton"],
[data-testid="stShareButton"],
[data-testid="stStarButton"],
[data-testid="stProfileWidget"],
[data-testid="stOwnerBadge"],
[data-testid="stHostedDeploymentBadge"],
[data-testid="viewerBadge"],
[data-testid="stViewerBadge"],
[class*="viewerBadge"],
[class*="ViewerBadge"],
[class*="ForkButton"],
[class*="ShareButton"],
[class*="ProfileBadge"],
[class*="ProfileWidget"],
[class*="OwnerBadge"],
[class*="_profileContainer_"],
[class*="_viewerBadge_"],
a[href*="streamlit.io/cloud"],
.viewerBadge_link__qRIco,
.viewerBadge_container__1QSob,
.viewerBadge_text__1JaDK,
.styles_terminalButton__JBj5T,
header[data-testid="stHeader"] a[href*="github.com"],
header[data-testid="stHeader"] a[href*="streamlit.io"],
header[data-testid="stHeader"] a[href*="share.streamlit.io"],
header[data-testid="stHeader"] button[data-testid="stBaseButton-header"],
a[href*="streamlit.io"][data-testid*="Badge"],
a[href*="github.com"][class*="viewer"],
a[href*="github.com/streamlit"],
.stApp > div > a[href*="github.com"],
.stApp > div > a[href*="streamlit.io"],
.stApp > a[href*="github.com"],
.stApp > a[href*="streamlit.io"] {
    display: none !important;
    visibility: hidden !important;
}

/* Force the toolbar + sidebar expand button visible. The sidebar reopen
   chevron (stExpandSidebarButton) is nested inside stToolbar — without
   this rule the broader hide above would catch it. */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarHeader"],
button[data-testid="stExpandSidebarButton"],
button[data-testid="stBaseButton-headerNoPadding"] {
    display: flex !important;
    visibility: visible !important;
    pointer-events: auto !important;
    opacity: 1 !important;
}
[data-testid="stExpandSidebarButton"] *,
[data-testid="stSidebarCollapseButton"] * {
    visibility: visible !important;
    opacity: 1 !important;
}
header[data-testid="stHeader"] button {
    background: rgba(99, 102, 241, 0.2) !important;
    border: 1px solid rgba(129, 140, 248, 0.35) !important;
    border-radius: 0.7rem !important;
    color: #e2e8f0 !important;
    margin-left: 0.6rem !important;
    opacity: 1 !important;
    visibility: visible !important;
}
header[data-testid="stHeader"] button:hover {
    background: rgba(99, 102, 241, 0.32) !important;
}
header[data-testid="stHeader"] svg {
    color: #e2e8f0 !important;
    fill: currentColor !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 6rem !important;
    max-width: 78rem !important;
}

/* Headings + captions in main area */
.stApp h1, .stApp h2, .stApp h3 {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}
.stApp h1 { font-size: 1.875rem !important; }
.stApp [data-testid="stCaptionContainer"],
.stApp .stCaption,
.stApp small {
    color: #94a3b8 !important;
}

.stApp p, .stApp li, .stApp span, .stApp div {
    color: #e2e8f0;
}

/* ---- Sidebar — glassmorph dark ---- */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.55) !important;
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
section[data-testid="stSidebar"] > div {
    background: transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] [data-testid="stImage"] img {
    display: block !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
.sidebar-brand-image {
    display: block !important;
    width: 220px !important;
    max-width: 100% !important;
    height: auto !important;
    margin-left: auto !important;
    margin-right: auto !important;
    margin-bottom: 0.5rem !important;
    border-radius: 0.5rem !important;
    image-rendering: -webkit-optimize-contrast;
    image-rendering: auto;
}
/* Center the stHtml wrapper that holds the brand image. */
section[data-testid="stSidebar"] [data-testid="stHtml"]:has(> .sidebar-brand-image) {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}
.sidebar-brand-title {
    color: #f1f5f9 !important;
    font-family: 'Clash Grotesk', 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    line-height: 1.2 !important;
    margin: 0.4rem 0 0.2rem !important;
}
.sidebar-brand-subtitle {
    color: #ffffff !important;
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    letter-spacing: 0 !important;
    line-height: 1.3 !important;
    margin: 0 0 0.95rem !important;
    text-transform: none !important;
}
.main-chat-title {
    color: #f1f5f9 !important;
    font-family: 'Clash Grotesk', 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    font-size: 1.875rem !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    line-height: 1.2 !important;
    margin: 0 0 1rem !important;
}
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h1 {
    color: #f1f5f9 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.25rem !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #64748b !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
section[data-testid="stSidebar"] hr,
section[data-testid="stSidebar"] [data-testid="stDivider"] {
    border-color: rgba(255, 255, 255, 0.08) !important;
    background: rgba(255, 255, 255, 0.08) !important;
    margin: 0.6rem 0 !important;
}

/* ---- Primary "New chat" button (sidebar primary) ---- */
section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: rgba(99, 102, 241, 0.18) !important;
    border: 1px solid rgba(129, 140, 248, 0.35) !important;
    color: #e0e7ff !important;
    font-weight: 500 !important;
    border-radius: 0.75rem !important;
    box-shadow: none !important;
    padding: 0.55rem 1rem !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
    background: rgba(99, 102, 241, 0.28) !important;
    border-color: rgba(129, 140, 248, 0.5) !important;
}

/* ---- Chat row buttons ---- */
.stApp [class*="st-key-open_"] button,
.stApp [class*="st-key-open-"] button {
    background: transparent !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    color: #f1f5f9 !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    text-align: left !important;
    width: 100% !important;
    padding: 0.55rem 0.7rem !important;
    border-radius: 0.5rem !important;
    line-height: 1.35 !important;
}
.stApp [class*="st-key-open_"] button > *,
.stApp [class*="st-key-open-"] button > * {
    display: block !important;
    width: 100% !important;
    min-width: 0 !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    text-align: left !important;
}
.stApp [class*="st-key-open_"] button:hover,
.stApp [class*="st-key-open-"] button:hover {
    background: rgba(255, 255, 255, 0.05) !important;
}
.stApp [class*="st-key-open_active_"] button,
.stApp [class*="st-key-open-active-"] button {
    background: rgba(99, 102, 241, 0.18) !important;
    border-color: rgba(129, 140, 248, 0.4) !important;
    box-shadow: inset 3px 0 0 0 rgba(129, 140, 248, 0.9) !important;
}
.stApp [class*="st-key-open_active_"] button:hover,
.stApp [class*="st-key-open-active-"] button:hover {
    background: rgba(99, 102, 241, 0.26) !important;
    border-color: rgba(129, 140, 248, 0.55) !important;
}
.stApp [class*="st-key-open_active_"] button strong,
.stApp [class*="st-key-open-active-"] button strong {
    color: #ffffff !important;
}
.stApp [class*="st-key-open_"] button [data-testid="stMarkdownContainer"],
.stApp [class*="st-key-open-"] button [data-testid="stMarkdownContainer"],
.stApp [class*="st-key-open_"] button [data-testid="stMarkdownContainer"] *,
.stApp [class*="st-key-open-"] button [data-testid="stMarkdownContainer"] * {
    display: block !important;
    text-align: left !important;
    width: 100% !important;
    max-width: 100% !important;
    margin-left: 0 !important;
    margin-right: auto !important;
}

/* High-specificity override using actual DOM testids. Streamlit wraps the
   tertiary button content as:
     button[data-testid="stBaseButton-tertiary"]
       > div  > span  > div[data-testid="stMarkdownContainer"]
   The inner div + span are inline-flex with center alignment via emotion
   hashed classes. Need to flatten every level to block + left-align. */
section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] {
    display: flex !important;
    flex-direction: column !important;
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] > div,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] > div > span,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] [data-testid="stMarkdownContainer"] {
    display: block !important;
    width: 100% !important;
    max-width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] [data-testid="stMarkdownContainer"] p {
    text-align: left !important;
    margin: 0 !important;
}

/* ---- Icon buttons (rename / delete) ---- */
.stApp [class*="st-key-rename_"],
.stApp [class*="st-key-delete_"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: auto !important;
}
.stApp [class*="st-key-rename_"] button,
.stApp [class*="st-key-delete_"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #64748b !important;
    padding: 0 !important;
    min-width: 32px !important;
    width: 32px !important;
    max-width: 32px !important;
    min-height: 32px !important;
    height: 32px !important;
    max-height: 32px !important;
    font-size: 14px !important;
    line-height: 1 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 0.375rem !important;
    margin: 0 !important;
    flex: 0 0 32px !important;
    align-self: center !important;
    transition: background-color 0.12s ease, color 0.12s ease !important;
}
/* Center the icon button wrapper within its column row so the 32×32 button
   doesn't stretch when the adjacent open-button col is taller (active row). */
.stApp [class*="st-key-rename_"],
.stApp [class*="st-key-delete_"] {
    align-self: center !important;
}
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"],
section[data-testid="stSidebar"] [data-testid="stColumn"] {
    align-items: center !important;
}

/* Icon button compact 32×32, right-aligned in its column so spacing
   between the title block and the icons stays even when the sidebar is
   resized. No hover bg, so no need to stretch. */
section[data-testid="stSidebar"] [class*="st-key-rename_"] button[data-testid="stBaseButton-secondary"],
section[data-testid="stSidebar"] [class*="st-key-delete_"] button[data-testid="stBaseButton-secondary"],
section[data-testid="stSidebar"] [class*="st-key-rename_"] button,
section[data-testid="stSidebar"] [class*="st-key-delete_"] button {
    width: 32px !important;
    min-width: 32px !important;
    max-width: 32px !important;
    height: 32px !important;
    min-height: 32px !important;
    max-height: 32px !important;
    padding: 0 !important;
    line-height: 1 !important;
    flex: 0 0 32px !important;
    align-self: center !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
/* Wrapper centers button vertically + pushes it to the right edge of its
   column (flex-end) so the gap is equal on both sides of the icon row. */
section[data-testid="stSidebar"] [class*="st-key-rename_"],
section[data-testid="stSidebar"] [class*="st-key-delete_"] {
    align-self: center !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    width: 100% !important;
}
/* Add breathing room between the chat title and the icon buttons so a
   long title doesn't run flush into the pencil icon when the sidebar is
   narrow. Padding lives on the open button + a left margin on the first
   icon column. */
section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] {
    padding-right: 0.5rem !important;
}
section[data-testid="stSidebar"] [class*="st-key-rename_"] {
    margin-left: 1rem !important;
}
section[data-testid="stSidebar"] [class*="st-key-delete_"] {
    margin-left: 0.35rem !important;
}
/* Center the icon glyph inside the now-stretched icon button. The button
   wraps: button > div > span[stIconMaterial]. Streamlit positions the
   inner div top-left by default; flatten every level to center. */
section[data-testid="stSidebar"] [class*="st-key-rename_"] button > *,
section[data-testid="stSidebar"] [class*="st-key-delete_"] button > *,
section[data-testid="stSidebar"] [class*="st-key-rename_"] button > * > *,
section[data-testid="stSidebar"] [class*="st-key-delete_"] button > * > * {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    height: 100% !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] [class*="st-key-rename_"] [data-testid="stIconMaterial"],
section[data-testid="stSidebar"] [class*="st-key-delete_"] [data-testid="stIconMaterial"] {
    margin: 0 auto !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.stApp [class*="st-key-rename_"] button:hover,
.stApp [class*="st-key-rename_"] button:focus,
.stApp [class*="st-key-rename_"] button:active {
    background: transparent !important;
    color: #e2e8f0 !important;
    box-shadow: none !important;
}
.stApp [class*="st-key-delete_"] button:hover,
.stApp [class*="st-key-delete_"] button:focus,
.stApp [class*="st-key-delete_"] button:active {
    background: transparent !important;
    color: #fca5a5 !important;
    box-shadow: none !important;
}
.stApp [class*="st-key-rename_"] button p,
.stApp [class*="st-key-delete_"] button p {
    line-height: 1 !important;
    font-size: 14px !important;
    margin: 0 !important;
    text-align: center !important;
    justify-content: center !important;
    color: inherit !important;
}

.stApp [class*="st-key-rename-"] button,
.stApp [class*="st-key-delete-"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #64748b !important;
    padding: 0 !important;
    min-width: 32px !important;
    width: 32px !important;
    max-width: 32px !important;
    height: 32px !important;
    min-height: 32px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 0.375rem !important;
    margin: 0 !important;
}
.stApp [class*="st-key-rename-"] button p,
.stApp [class*="st-key-delete-"] button p,
.stApp [class*="st-key-rename-"] button span,
.stApp [class*="st-key-delete-"] button span {
    margin: 0 !important;
    width: auto !important;
    text-align: center !important;
    line-height: 1 !important;
    color: inherit !important;
}
section[data-testid="stSidebar"] button:has(span[class*="material"]) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-width: 32px !important;
    width: 32px !important;
    max-width: 32px !important;
    height: 32px !important;
    min-height: 32px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
section[data-testid="stSidebar"] button:has(span[class*="material"]) [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] button:has(span[class*="material"]) p,
section[data-testid="stSidebar"] button:has(span[class*="material"]) span {
    margin: 0 !important;
    width: auto !important;
    text-align: center !important;
    line-height: 1 !important;
}

section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    align-items: center;
    background: transparent !important;
    gap: 0.25rem !important;
}

/* Sidebar search input — compact, slate-950 bg */
section[data-testid="stSidebar"] [data-testid="stTextInput"] {
    margin-top: 0.5rem !important;
    margin-bottom: 0.25rem !important;
}
section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: rgba(2, 6, 23, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #f1f5f9 !important;
    border-radius: 0.5rem !important;
    padding: 0.45rem 0.7rem !important;
    font-size: 0.85rem !important;
}
section[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
    color: #64748b !important;
}
section[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: rgba(255, 255, 255, 0.2) !important;
    box-shadow: none !important;
}

/* ---- Chat messages ---- */
.stApp [data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
    margin-bottom: 0.5rem !important;
    gap: 0.35rem !important;
    column-gap: 0.35rem !important;
    align-items: flex-start !important;
}
.stApp [data-testid="stChatMessage"] > * {
    margin-left: 0 !important;
    margin-right: 0 !important;
}

/* Default bubble look (assistant — slate glass). Covers all messages,
   then user-specific selectors below override with indigo + row-reverse. */
.stApp [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #f1f5f9 !important;
    border-radius: 1rem !important;
    padding: 0.85rem 1.1rem !important;
    max-width: min(88%, 58rem) !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}

/* User bubble — indigo tint, right-aligned. Cover every known Streamlit
   chat-message DOM shape: testid variants + class-based fallbacks. */
.stApp [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
.stApp [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
.stApp [data-testid="stChatMessage"]:has([aria-label="user avatar"]),
.stApp .stChatMessage--user,
.stApp [data-testid="stChatMessageUser"] {
    flex-direction: row-reverse !important;
    justify-content: flex-start !important;
}
.stApp [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
.stApp [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"],
.stApp [data-testid="stChatMessage"]:has([aria-label="user avatar"]) [data-testid="stChatMessageContent"],
.stApp .stChatMessage--user [data-testid="stChatMessageContent"],
.stApp [data-testid="stChatMessageUser"] [data-testid="stChatMessageContent"] {
    background: rgba(99, 102, 241, 0.2) !important;
    border: 1px solid rgba(129, 140, 248, 0.25) !important;
    color: #ffffff !important;
}

/* Avatars — gradient circle */
.stApp [data-testid="stChatMessageAvatarAssistant"],
.stApp [data-testid="stChatMessageAvatarUser"],
.stApp [data-testid="stChatMessageAvatarCustom"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(168, 85, 247, 0.3)) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #c7d2fe !important;
    width: 2rem !important;
    height: 2rem !important;
    border-radius: 50% !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 0.85rem !important;
    margin: 0 !important;
    flex: none !important;
}

/* Message text — markdown inside */
.stApp [data-testid="stChatMessageContent"] p,
.stApp [data-testid="stChatMessageContent"] li,
.stApp [data-testid="stChatMessageContent"] span {
    color: inherit !important;
    line-height: 1.55 !important;
}
.stApp [data-testid="stChatMessageContent"] code {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #c7d2fe !important;
    padding: 0.1rem 0.35rem !important;
    border-radius: 0.25rem !important;
    font-size: 0.85em !important;
}
.stApp [data-testid="stChatMessageContent"] pre {
    background: rgba(2, 6, 23, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 0.5rem !important;
}
.stApp [data-testid="stChatMessageContent"] a {
    color: #93c5fd !important;
}
.stApp [data-testid="stChatMessageContent"] table {
    background: rgba(2, 6, 23, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 0.75rem;
    overflow: hidden;
}
.stApp [data-testid="stChatMessageContent"] th {
    background: rgba(255, 255, 255, 0.06) !important;
    color: #cbd5e1 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.7rem !important;
}
.stApp [data-testid="stChatMessageContent"] td {
    color: #e2e8f0 !important;
    border-color: rgba(255, 255, 255, 0.06) !important;
}

/* ---- Chat input (bottom composer) ---- */
.stApp [data-testid="stChatInput"] {
    background: rgba(15, 23, 42, 0.5) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 1rem !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.stApp [data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #f1f5f9 !important;
    caret-color: #c7d2fe !important;
}
.stApp [data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
}
.stApp [data-testid="stChatInput"] button {
    background: rgba(99, 102, 241, 0.2) !important;
    border: 1px solid rgba(129, 140, 248, 0.35) !important;
    color: #c7d2fe !important;
    border-radius: 0.6rem !important;
    transition: background-color 0.15s ease;
}
.stApp [data-testid="stChatInput"] button:hover {
    background: rgba(99, 102, 241, 0.3) !important;
}
.stApp [data-testid="stChatInput"] svg {
    fill: #c7d2fe !important;
}

/* Sticky bottom anchoring for composer */
.stApp [data-testid="stBottomBlockContainer"] {
    background: transparent !important;
    max-width: 78rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* ---- Clarification chips (main-area buttons) ---- */
.stApp [class*="st-key-chip_"] button {
    background: rgba(99, 102, 241, 0.15) !important;
    border: 1px solid rgba(129, 140, 248, 0.3) !important;
    color: #e0e7ff !important;
    border-radius: 9999px !important;
    padding: 0.35rem 1rem !important;
    font-size: 0.85rem !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
.stApp [class*="st-key-chip_"] button:hover {
    background: rgba(99, 102, 241, 0.25) !important;
    border-color: rgba(129, 140, 248, 0.5) !important;
}

/* ---- Expanders (Sources / Flagged claims) ---- */
.stApp [data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 0.75rem !important;
    margin-top: 0.5rem;
}
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
    color: #c7d2fe !important;
    font-size: 0.8rem !important;
}
.stApp [data-testid="stExpander"] p {
    color: #cbd5e1 !important;
}
.stApp [data-testid="stExpander"] hr {
    border-color: rgba(255, 255, 255, 0.06) !important;
}

/* ---- Dialog (rename / delete modals) ---- */
.stApp [role="dialog"],
.stApp [data-testid="stDialog"] {
    background: rgba(15, 23, 42, 0.95) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 1rem !important;
}
.stApp [role="dialog"] h2,
.stApp [data-testid="stDialog"] h2 {
    color: #f1f5f9 !important;
}
.stApp [data-testid="stTextInput"] input {
    background: rgba(2, 6, 23, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #f1f5f9 !important;
    border-radius: 0.5rem !important;
}
.stApp [data-testid="stTextInput"] input:focus {
    border-color: rgba(255, 255, 255, 0.25) !important;
    box-shadow: none !important;
}

/* Generic primary buttons in dialogs (Save / Delete) */
.stApp [role="dialog"] .stButton button[kind="primary"],
.stApp [data-testid="stDialog"] .stButton button[kind="primary"] {
    background: rgba(99, 102, 241, 0.25) !important;
    border: 1px solid rgba(129, 140, 248, 0.4) !important;
    color: #e0e7ff !important;
    border-radius: 0.6rem !important;
    box-shadow: none !important;
}
.stApp [role="dialog"] .stButton button[kind="primary"]:hover,
.stApp [data-testid="stDialog"] .stButton button[kind="primary"]:hover {
    background: rgba(99, 102, 241, 0.35) !important;
}
/* Secondary dialog buttons */
.stApp [role="dialog"] .stButton button:not([kind="primary"]),
.stApp [data-testid="stDialog"] .stButton button:not([kind="primary"]) {
    background: transparent !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #cbd5e1 !important;
    border-radius: 0.6rem !important;
    box-shadow: none !important;
}
.stApp [role="dialog"] .stButton button:not([kind="primary"]):hover,
.stApp [data-testid="stDialog"] .stButton button:not([kind="primary"]):hover {
    background: rgba(255, 255, 255, 0.05) !important;
}

/* ---- Spinner color ---- */
.stApp .stSpinner > div {
    border-top-color: #818cf8 !important;
}

/* ---- Scrollbars ---- */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.15);
}

/* ---- Intro image — soft glow ring ---- */
.stApp [data-testid="stImage"] img {
    filter: drop-shadow(0 0 40px rgba(56, 189, 248, 0.18));
}

/* ---- Gate / login form — match dark theme ---- */
.stApp [data-testid="stForm"] {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 1rem !important;
    backdrop-filter: blur(12px);
}
.stApp [data-testid="stForm"] .stButton button {
    background: rgba(99, 102, 241, 0.25) !important;
    border: 1px solid rgba(129, 140, 248, 0.4) !important;
    color: #e0e7ff !important;
    border-radius: 0.6rem !important;
    width: 100%;
}

/* ---- Markdown emphasis colors in main area ---- */
.stApp strong { color: #f8fafc !important; }
.stApp em { color: #e2e8f0 !important; }
.stApp blockquote {
    border-left: 3px solid rgba(129, 140, 248, 0.4) !important;
    background: rgba(99, 102, 241, 0.05);
    color: #cbd5e1 !important;
}
</style>
""",
)


# =============================================================================
# Helpers
# =============================================================================

def _normalize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """LangChain message objects or dicts → plain dicts."""
    out: List[Dict[str, Any]] = []
    for msg in messages or []:
        if hasattr(msg, "content"):
            role = getattr(msg, "type", "user")
            role = "user" if role == "human" else ("assistant" if role == "ai" else role)
            content = msg.content
            metadata = getattr(msg, "metadata", None) or getattr(msg, "additional_kwargs", None) or {}
        else:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            metadata = msg.get("metadata") or {}
        out.append({"role": role, "content": content or "", "metadata": metadata})
    return out


_DISPLAY_TZ = ZoneInfo("Asia/Dubai")


def _fmt_time(iso_str: Optional[str]) -> str:
    """Format a Supabase ISO timestamp into the app's display timezone."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone(_DISPLAY_TZ).strftime("%b %d, %H:%M")
    except Exception:
        return iso_str


def _relative_time(iso_str: Optional[str]) -> str:
    """Human-friendly relative time: 'just now', '5m ago', '2h ago', '3d ago'."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(_DISPLAY_TZ)
    except Exception:
        return iso_str
    now = datetime.now(_DISPLAY_TZ)
    diff = now - dt
    secs = int(diff.total_seconds())
    if secs < 60:
        return "just now"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m ago"
    hrs = mins // 60
    if hrs < 24:
        return f"{hrs}h ago"
    days = hrs // 24
    if days < 7:
        return f"{days}d ago"
    return dt.strftime("%b %d")


def _bucket_session(iso_str: Optional[str]) -> str:
    """Group a session by last-message recency. Matches KARR AI bucketing."""
    if not iso_str:
        return "Older"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(_DISPLAY_TZ)
    except Exception:
        return "Older"
    now = datetime.now(_DISPLAY_TZ)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    week_start = today_start - timedelta(days=7)
    if dt >= today_start:
        return "Today"
    if dt >= yesterday_start:
        return "Yesterday"
    if dt >= week_start:
        return "Previous 7 days"
    return "Older"


_GROUP_ORDER = ("Today", "Yesterday", "Previous 7 days", "Older")


def _ensure_current_session() -> str:
    """Return the current session id.

    Priority:
      1. The session id stored in session_state, IF it still exists in the DB
         (otherwise it's stale — e.g. session was soft-deleted).
      2. The most recently active existing session for this gate user.
      3. Only if none exist at all, create a fresh empty session.

    This avoids spawning a new "Untitled chat" on every page reload, after
    deleting the active chat, etc.
    """
    sid = st.session_state.get("current_session_id")
    if sid and get_session(sid):
        return sid
    # Stale or missing — drop it.
    if sid:
        st.session_state.pop("current_session_id", None)

    # Fall back to the most recent existing session.
    recent = list_sessions(limit=1)
    if recent:
        sid = recent[0]["id"]
        st.session_state["current_session_id"] = sid
        return sid

    # No sessions exist for this gate user — create the very first one.
    row = create_session()
    sid = row["id"]
    initialize_code_charlie_session_checkpoint(session_id=sid, user_id=settings.GATE_USER_ID)
    st.session_state["current_session_id"] = sid
    return sid


def _render_assistant_meta(metadata: Dict[str, Any]) -> None:
    """Render sources + flagged claims under an assistant message."""
    sources = metadata.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for src in sources:
                section = src.get("section") or "—"
                doc = src.get("document") or "—"
                title = src.get("title") or ""
                text = src.get("text") or ""
                similarity = src.get("similarity")
                sim_label = f" · {similarity:.0%}" if isinstance(similarity, (int, float)) else ""
                header = f"**{doc}** — Section {section}{sim_label}"
                if title:
                    header += f"  \n_{title}_"
                st.markdown(header)
                if text:
                    st.caption(text)
                st.markdown("---")

    if metadata.get("dropped_claims"):
        with st.expander("Potential issues flagged"):
            for claim in metadata["dropped_claims"]:
                st.caption(claim)


def _invoke_for_pending(session_id: str, text: str) -> None:
    """Run the graph for the pending user message and persist sidebar metadata."""
    row = get_session(session_id)
    if not row:
        st.error("Session not found.")
        return

    current_state = get_code_charlie_state(session_id)
    if current_state is None:
        current_state = create_initial_code_charlie_state(
            user_id=settings.GATE_USER_ID,
            session_id=session_id,
            initial_message=text,
        )
    else:
        current_state = {
            **current_state,
            "messages": current_state.get("messages", []) + [add_user_message(text)],
            "intent": None,
            "error": None,
        }

    was_first_user_turn = (row.get("message_count") or 0) == 0
    result_state = invoke_code_charlie_graph(current_state, thread_id=session_id)
    update_after_message(session_id, row, result_state.get("scope"))

    if was_first_user_turn and not row.get("title"):
        msgs = _normalize_messages(result_state.get("messages", []))
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        first_assistant = next(
            (
                m["content"]
                for m in msgs
                if m["role"] == "assistant" and m["content"] != INTRO_MESSAGE_CONTENT
            ),
            "",
        )
        if first_user:
            auto_title_session(session_id, first_user, first_assistant)


# =============================================================================
# Sidebar — session list
# =============================================================================

# Resolve the active session BEFORE rendering the sidebar so the sidebar can
# highlight it on the very first render. Without this hoist, the sidebar
# reads `current_session_id` from session_state while it is still None on a
# fresh page load; only the main area below would compute it, leaving the
# auto-selected chat unhighlighted until the user clicks something.
if not st.session_state.get("_checkpoint_error"):
    try:
        _ensure_current_session()
    except Exception as exc:
        logger.exception("Failed to resolve Code Charlie session for sidebar")
        st.session_state["_checkpoint_error"] = _checkpoint_error_message(exc)

with st.sidebar:
    # Render the brand image as a raw <img> with a base64 data URI instead
    # of `st.image()`. Streamlit's media server re-encodes uploaded images
    # which can soften details — inlining the original PNG bytes preserves
    # source quality. Crisp rendering hints help on hi-DPI displays.
    intro_data_uri = _file_data_uri(INTRO_IMAGE_PATH, "image/png")
    if intro_data_uri:
        st.html(
            f'<img class="sidebar-brand-image" src="{intro_data_uri}" alt="Code Charlie" />'
        )

    st.html(
        '<div class="sidebar-brand-title">Code Charlie</div>'
        '<div class="sidebar-brand-subtitle">Powered by Medha 1.0 — KARR AI</div>'
    )

    if st.button("＋ New chat", use_container_width=True, type="primary"):
        try:
            row = create_session()
            sid = row["id"]
            initialize_code_charlie_session_checkpoint(session_id=sid, user_id=settings.GATE_USER_ID)
            st.session_state["current_session_id"] = sid
            st.session_state.pop("pending_prompt", None)
            st.session_state.pop("_checkpoint_error", None)
            st.rerun()
        except Exception as exc:
            logger.exception("Failed to create Code Charlie session")
            st.session_state["_checkpoint_error"] = _checkpoint_error_message(exc)
            st.rerun()

    search_query = st.text_input(
        "Search",
        key="sidebar_search",
        placeholder="Search chats…",
        label_visibility="collapsed",
    )

    st.divider()

    sessions = list_sessions(limit=50)
    current_id = st.session_state.get("current_session_id")

    q = (search_query or "").strip().lower()
    if q:
        sessions = [s for s in sessions if q in (s.get("title") or "").lower()]

    if not sessions:
        st.caption("_No matching chats._" if q else "_No sessions yet._")
    else:
        grouped: Dict[str, List[Dict[str, Any]]] = {g: [] for g in _GROUP_ORDER}
        for s in sessions:
            grouped[_bucket_session(s.get("last_message_at"))].append(s)

        for group in _GROUP_ORDER:
            bucket = grouped[group]
            if not bucket:
                continue
            st.caption(group)
            for s in bucket:
                sid = s["id"]
                is_current = sid == current_id
                title = s.get("title") or "Untitled chat"
                subtitle_bits = []
                if s.get("scope_code"):
                    subtitle_bits.append(s["scope_code"])
                subtitle_bits.append(_relative_time(s.get("last_message_at")))
                subtitle = " · ".join(b for b in subtitle_bits if b)

                row_cols = st.columns([0.74, 0.13, 0.13])
                with row_cols[0]:
                    label = f"**{title}**\n\n{subtitle}" if subtitle else f"**{title}**"
                    btn_key = f"open_active_{sid}" if is_current else f"open_{sid}"
                    if st.button(
                        label,
                        key=btn_key,
                        use_container_width=True,
                        type="tertiary",
                    ):
                        st.session_state["current_session_id"] = sid
                        st.session_state.pop("pending_prompt", None)
                        st.rerun()
                with row_cols[1]:
                    if st.button(":material/edit:", key=f"rename_{sid}", help="Rename", type="tertiary"):
                        st.session_state["_rename_target"] = {"sid": sid, "title": s.get("title") or ""}
                        st.rerun()
                with row_cols[2]:
                    if st.button(":material/delete:", key=f"delete_{sid}", help="Delete", type="tertiary"):
                        st.session_state["_delete_target"] = {"sid": sid, "title": title}
                        st.rerun()


# Rename + delete modals (dialogs) — open when a target session is queued.
_rename_target = st.session_state.get("_rename_target")
if _rename_target:
    @st.dialog("Rename chat")
    def _rename_dialog():
        target_sid = _rename_target["sid"]
        current_title = _rename_target["title"]
        new_title = st.text_input("New title", value=current_title, key="_rename_field")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Cancel", use_container_width=True, key="_rename_cancel"):
                st.session_state.pop("_rename_target", None)
                st.rerun()
        with col_b:
            if st.button("Save", type="primary", use_container_width=True, key="_rename_save"):
                if new_title.strip():
                    rename_session(target_sid, new_title.strip())
                st.session_state.pop("_rename_target", None)
                st.rerun()
    _rename_dialog()

_delete_target = st.session_state.get("_delete_target")
if _delete_target:
    @st.dialog("Delete chat?")
    def _delete_dialog():
        target_sid = _delete_target["sid"]
        target_title = _delete_target["title"]
        st.write(f"Permanently delete **{target_title}**? This cannot be undone.")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Cancel", use_container_width=True, key="_delete_cancel"):
                st.session_state.pop("_delete_target", None)
                st.rerun()
        with col_b:
            if st.button("Delete", type="primary", use_container_width=True, key="_delete_confirm"):
                delete_session(target_sid)
                if target_sid == st.session_state.get("current_session_id"):
                    st.session_state.pop("current_session_id", None)
                st.session_state.pop("_delete_target", None)
                st.rerun()
    _delete_dialog()


def _render_checkpoint_error(details: str) -> None:
    st.title("Code Charlie")
    st.error("Code Charlie cannot connect to the chat database right now.")
    st.caption(
        "This usually means the Supabase/Postgres pooler is unreachable, "
        "temporarily out of connections, or the DATABASE_URL credentials need checking."
    )
    st.code(details, language="text")
    if st.button("Retry database connection", type="primary"):
        _warm_graph.clear()
        st.session_state.pop("_checkpoint_error", None)
        st.session_state.pop("_warmed", None)
        st.rerun()


# =============================================================================
# Main chat area
# =============================================================================

checkpoint_error = st.session_state.get("_checkpoint_error")
if checkpoint_error:
    _render_checkpoint_error(checkpoint_error)
    st.stop()

try:
    current_session_id = _ensure_current_session()
    state = get_code_charlie_state(current_session_id) or {}
except Exception as exc:
    logger.exception("Failed to load Code Charlie checkpoint state")
    _render_checkpoint_error(_checkpoint_error_message(exc))
    st.stop()

messages = _normalize_messages(state.get("messages", []))

st.html('<h1 class="main-chat-title">Code Charlie</h1>')
#st.caption("Ask compliance questions across DBC, CIBSE, EN 81, BCO, ASME/ADA/IBC, HTM, ISO, DoH, BMU, CSI, Machinery Directive.")

# Render existing history
for m in messages:
    role = m["role"]
    content = m["content"]
    metadata = m.get("metadata") or {}

    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)
        if role == "assistant" and metadata:
            _render_assistant_meta(metadata)


# Pending message: show user bubble + spinner immediately while graph runs.
pending_prompt = st.session_state.get("pending_prompt")
if pending_prompt and pending_prompt.get("session_id") == current_session_id:
    text = pending_prompt["text"]
    # Render the user's message bubble at once so they can see what they sent.
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        with st.spinner("Researching…"):
            try:
                _invoke_for_pending(current_session_id, text)
            except Exception as exc:
                logger.exception("Code Charlie turn failed")
                st.session_state["_checkpoint_error"] = _checkpoint_error_message(exc)
                st.error("The chat database connection failed while processing this message.")
    st.session_state.pop("pending_prompt", None)
    st.rerun()


# Pending clarification chips
pending_clar = state.get("pending_clarification") if state else None
if pending_clar and pending_clar.get("options"):
    st.markdown("**Pick one to continue:**")
    options = pending_clar["options"]
    chip_cols = st.columns(min(4, max(2, len(options))))
    for i, opt in enumerate(options):
        with chip_cols[i % len(chip_cols)]:
            if st.button(opt, key=f"chip_{current_session_id}_{i}_{opt}", use_container_width=True):
                st.session_state["pending_prompt"] = {
                    "session_id": current_session_id,
                    "text": opt,
                }
                st.rerun()


# Chat input
prompt = st.chat_input("Ask about a building code…")
if prompt:
    st.session_state["pending_prompt"] = {
        "session_id": current_session_id,
        "text": prompt,
    }
    st.rerun()
