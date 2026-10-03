"""Global light "industrial metrology dashboard" CSS theme for the Streamlit
app. Presentation only: color tokens, font, and element styling. No
measurement, calibration, or detection logic lives here.
"""
from __future__ import annotations

import streamlit as st

COLOR_BG = "#F5F7FA"
COLOR_SURFACE = "#FFFFFF"
COLOR_TEXT = "#17202A"
COLOR_TEXT_MUTED = "#667085"
COLOR_BORDER = "#D9DEE7"

COLOR_PASS = "#16A34A"
COLOR_PASS_BG = "#DCFCE7"
COLOR_FAIL = "#DC2626"
COLOR_FAIL_BG = "#FEE2E2"
COLOR_WARN = "#D97706"
COLOR_WARN_BG = "#FEF3C7"
COLOR_INFO = "#2563EB"
COLOR_INFO_BG = "#DBEAFE"

# Dark "HMI kiosk" palette for the compact default result card
# (ui/inspection_view.build_hmi_result_html). Deliberately NOT part of the
# global `_CSS`/`inject_theme_css` below — the rest of the app (including
# the Advanced / Detailed View) keeps the light theme above unchanged; the
# HMI card carries these as its own scoped <style> block instead.
HMI_BG = "#14171C"
HMI_SURFACE = "#1D2128"
HMI_TEXT = "#F2F4F7"
HMI_TEXT_MUTED = "#8B93A1"
HMI_BORDER = "#2A2F38"
HMI_PASS = "#22C55E"
HMI_FAIL = "#EF4444"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.stApp {{
    background-color: {COLOR_BG};
}}

section[data-testid="stSidebar"] {{
    background-color: {COLOR_SURFACE};
    border-right: 1px solid {COLOR_BORDER};
}}
section[data-testid="stSidebar"] .stButton > button {{
    text-align: left;
    justify-content: flex-start;
    font-weight: 500;
}}

h1, h2, h3, h4, h5, h6 {{
    color: {COLOR_TEXT} !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}}

p, span, label, li, .stMarkdown {{
    color: {COLOR_TEXT};
}}

[data-testid="stMetricValue"] {{
    color: {COLOR_TEXT} !important;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {COLOR_TEXT_MUTED} !important;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
}}

.stButton > button {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    font-weight: 500;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
    transition: border-color 0.15s ease, background-color 0.15s ease;
}}
.stButton > button:hover {{
    border-color: {COLOR_INFO};
    color: {COLOR_INFO};
}}
.stButton > button[kind="primary"] {{
    background-color: {COLOR_INFO};
    color: #FFFFFF;
    border: none;
    font-weight: 600;
}}
.stButton > button[kind="primary"]:hover {{
    background-color: #1D4ED8;
}}
.stDownloadButton > button {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    font-weight: 500;
}}
.stDownloadButton > button:hover {{
    border-color: {COLOR_INFO};
    color: {COLOR_INFO};
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
}}

button[data-baseweb="tab"] {{
    color: {COLOR_TEXT_MUTED};
    font-weight: 500;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {COLOR_TEXT};
    font-weight: 700;
}}
div[data-baseweb="tab-highlight"] {{
    background-color: {COLOR_INFO} !important;
}}
div[data-baseweb="tab-border"] {{
    background-color: {COLOR_BORDER} !important;
}}

div[role="radiogroup"] label {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    padding: 4px 12px;
    margin-right: 4px;
}}

hr {{
    border-color: {COLOR_BORDER};
}}

/* ---- Structural containers (targeted via st.container(key=...)) ---- */
.st-key-mv_toolbar {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    padding: 8px 10px 2px 10px;
    margin-bottom: 12px;
}}
.st-key-mv_toolbar .stButton > button {{
    padding: 3px 12px;
    font-size: 0.8rem;
}}

/* ---- Panels / headers / chips ---- */
.mv-panel {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    animation: mvFadeUp 0.3s ease both;
}}
.mv-header-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}}
.mv-header-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {COLOR_TEXT};
}}
.mv-header-sub {{
    color: {COLOR_TEXT_MUTED};
    font-size: 0.8rem;
    margin-top: 2px;
}}
.mv-chip-row {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 6px;
}}
.mv-chip {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: {COLOR_TEXT_MUTED};
    background: {COLOR_BG};
    border: 1px solid {COLOR_BORDER};
    border-radius: 3px;
    padding: 2px 8px;
}}

.mv-status-dot {{
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}}
.mv-status-dot.pass {{ background: {COLOR_PASS}; }}
.mv-status-dot.fail {{ background: {COLOR_FAIL}; }}
.mv-status-dot.warn {{ background: {COLOR_WARN}; }}
.mv-status-dot.neutral {{ background: {COLOR_TEXT_MUTED}; }}

/* ---- Badges (small, inline) ---- */
.mv-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    border-radius: 5px;
    padding: 6px 14px;
    animation: mvScaleIn 0.3s ease both;
}}
.mv-badge-pass {{ background: {COLOR_PASS_BG}; color: {COLOR_PASS}; border: 1px solid rgba(22,163,74,0.35); }}
.mv-badge-fail {{ background: {COLOR_FAIL_BG}; color: {COLOR_FAIL}; border: 1px solid rgba(220,38,38,0.35); }}
.mv-badge-incomplete {{ background: {COLOR_WARN_BG}; color: {COLOR_WARN}; border: 1px solid rgba(217,119,6,0.35); }}
.mv-badge-neutral {{ background: {COLOR_BG}; color: {COLOR_TEXT_MUTED}; border: 1px solid {COLOR_BORDER}; font-size: 0.78rem; }}

/* ---- Hero banner (big PASS/FAIL) ---- */
.mv-hero {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    border-radius: 8px;
    padding: 22px 20px;
    margin-bottom: 12px;
    border: 1px solid;
    animation: mvFadeUp 0.35s ease both;
}}
.mv-hero-title {{
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    animation: mvScaleIn 0.35s cubic-bezier(0.2,0.8,0.2,1) both;
}}
.mv-hero-sub {{
    font-size: 0.88rem;
    margin-top: 4px;
    font-weight: 500;
}}
.mv-hero-pass {{ background: {COLOR_PASS_BG}; border-color: rgba(22,163,74,0.35); }}
.mv-hero-pass .mv-hero-title {{ color: {COLOR_PASS}; }}
.mv-hero-pass .mv-hero-sub {{ color: #15803D; }}
.mv-hero-fail {{ background: {COLOR_FAIL_BG}; border-color: rgba(220,38,38,0.35); }}
.mv-hero-fail .mv-hero-title {{ color: {COLOR_FAIL}; }}
.mv-hero-fail .mv-hero-sub {{ color: #B91C1C; }}
.mv-hero-incomplete {{ background: {COLOR_WARN_BG}; border-color: rgba(217,119,6,0.35); }}
.mv-hero-incomplete .mv-hero-title {{ color: {COLOR_WARN}; }}
.mv-hero-incomplete .mv-hero-sub {{ color: #B45309; }}
.mv-hero-neutral {{ background: {COLOR_BG}; border-color: {COLOR_BORDER}; }}
.mv-hero-neutral .mv-hero-title {{ color: {COLOR_TEXT_MUTED}; font-size: 1.3rem; }}
.mv-hero-neutral .mv-hero-sub {{ color: {COLOR_TEXT_MUTED}; }}

/* ---- Side panel ---- */
.mv-side-panel {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    padding: 14px 16px;
}}
.mv-side-metric {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 6px 0;
    border-bottom: 1px solid {COLOR_BORDER};
}}
.mv-side-metric:last-child {{ border-bottom: none; }}
.mv-side-metric .lbl {{ color: {COLOR_TEXT_MUTED}; font-size: 0.78rem; }}
.mv-side-metric .val {{
    color: {COLOR_TEXT};
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 0.88rem;
}}

/* ---- Per-feature cards (always visible, one per detected feature) ---- */
.mv-feature-card {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-left: 3px solid {COLOR_TEXT_MUTED};
    border-radius: 5px;
    padding: 8px 12px;
    margin-bottom: 8px;
    animation: mvFadeUp 0.3s ease both;
}}
.mv-feature-card.pass {{ border-left-color: {COLOR_PASS}; }}
.mv-feature-card.fail {{ border-left-color: {COLOR_FAIL}; background: {COLOR_FAIL_BG}; }}
.mv-feature-card.warn {{ border-left-color: {COLOR_WARN}; background: {COLOR_WARN_BG}; }}
.mv-feature-card.selected {{ border-color: {COLOR_INFO}; box-shadow: 0 0 0 1px {COLOR_INFO}; }}
.mv-feature-card-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 700;
    font-size: 0.85rem;
    color: {COLOR_TEXT};
}}
.mv-feature-card-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 4px 10px;
    margin-top: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: {COLOR_TEXT_MUTED};
}}
.mv-feature-card-grid b {{ color: {COLOR_TEXT}; font-weight: 600; }}

/* ---- Sidebar nav ---- */
.mv-nav-heading {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {COLOR_TEXT_MUTED};
    margin: 14px 0 4px 2px;
}}
.mv-nav-item-disabled {{
    color: {COLOR_TEXT_MUTED};
    border: 1px dashed {COLOR_BORDER};
    border-radius: 5px;
    padding: 5px 10px;
    font-size: 0.85rem;
    opacity: 0.65;
    margin-bottom: 2px;
}}

/* ---- Measurement table ---- */
table.mv-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}}
table.mv-table th {{
    text-align: left;
    color: {COLOR_TEXT_MUTED};
    text-transform: uppercase;
    font-size: 0.66rem;
    letter-spacing: 0.05em;
    font-weight: 700;
    padding: 8px 10px;
    border-bottom: 1px solid {COLOR_BORDER};
    background: {COLOR_BG};
}}
table.mv-table td {{
    padding: 7px 10px;
    border-bottom: 1px solid {COLOR_BORDER};
    color: {COLOR_TEXT};
    font-family: 'IBM Plex Mono', monospace;
}}
table.mv-table tr:last-child td {{ border-bottom: none; }}
table.mv-table tr:hover td {{ background: #FAFBFC; }}
table.mv-table tr.mv-row-selected td {{ background: {COLOR_INFO_BG}; }}

.mv-dot-pass, .mv-dot-fail, .mv-dot-warn, .mv-dot-neutral {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
}}
.mv-dot-pass {{ background: {COLOR_PASS}; }}
.mv-dot-fail {{ background: {COLOR_FAIL}; }}
.mv-dot-warn {{ background: {COLOR_WARN}; }}
.mv-dot-neutral {{ background: {COLOR_TEXT_MUTED}; }}

.mv-val-strong {{
    font-variant-numeric: tabular-nums;
    color: {COLOR_TEXT};
    font-weight: 700;
}}
.mv-conf-low {{
    color: {COLOR_WARN} !important;
    font-weight: 600;
}}
.mv-conflict-block {{
    background: {COLOR_WARN_BG};
    border: 1px solid rgba(217,119,6,0.4);
    border-radius: 4px;
    padding: 6px 8px;
    color: {COLOR_WARN};
    font-family: 'Inter', sans-serif;
    font-size: 0.74rem;
    line-height: 1.5;
    white-space: normal;
    max-width: 320px;
}}
.mv-conflict-title {{
    font-weight: 800;
    letter-spacing: 0.02em;
    margin-bottom: 2px;
}}

@keyframes mvFadeUp {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes mvScaleIn {{
    from {{ opacity: 0; transform: scale(0.9); }}
    to {{ opacity: 1; transform: scale(1); }}
}}

/* =====================  MODERN PRO SKIN (overrides above)  ===================== */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.stApp {{
    background:
        radial-gradient(900px 400px at 85% -10%, rgba(99,102,241,0.10), transparent 60%),
        radial-gradient(700px 360px at -5% 0%, rgba(14,165,233,0.08), transparent 60%),
        #F4F6FB;
}}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}

/* Title */
h1 {{
    font-size: 1.9rem !important;
    background: linear-gradient(90deg, #0F172A 0%, #4338CA 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent !important;
    letter-spacing: -0.02em;
}}
h2 {{ font-size: 1.35rem !important; }}

/* Dark sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0B1220 0%, #111A2E 100%);
    border-right: none;
}}
section[data-testid="stSidebar"] * {{ color: #CBD5E1; }}
section[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    color: #E2E8F0; border-radius: 10px; font-weight: 600;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(99,102,241,0.25); border-color: #818CF8; color: #FFFFFF;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 10px;
}}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.10); }}

/* Buttons */
.stButton > button {{
    border-radius: 10px; padding: 0.5rem 1rem; font-weight: 600;
    border: 1px solid #E2E8F0; box-shadow: 0 1px 2px rgba(15,23,42,0.05);
    transition: all 0.18s ease;
}}
.stButton > button:hover {{
    border-color: #6366F1; color: #4338CA;
    box-shadow: 0 6px 16px rgba(99,102,241,0.18); transform: translateY(-1px);
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 55%, #0EA5E9 100%);
    color: #fff; border: none; box-shadow: 0 6px 18px rgba(79,70,229,0.35);
}}
.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #4338CA 0%, #4F46E5 55%, #0284C7 100%);
    color: #fff; box-shadow: 0 10px 24px rgba(79,70,229,0.45);
}}

/* Cards: bordered containers, panels, expanders, uploader */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #FFFFFF; border: 1px solid #E5E9F2 !important;
    border-radius: 16px !important; box-shadow: 0 4px 24px rgba(15,23,42,0.05);
}}
.mv-panel, .mv-side-panel {{
    border-radius: 14px; border: 1px solid #E5E9F2;
    box-shadow: 0 4px 20px rgba(15,23,42,0.05);
}}
[data-testid="stExpander"] {{
    border-radius: 12px; border: 1px solid #E5E9F2; background: #FFFFFF;
}}
[data-testid="stFileUploader"] section {{
    border: 2px dashed #A5B4FC; background: rgba(238,242,255,0.6); border-radius: 16px;
}}
[data-testid="stFileUploader"] section:hover {{ border-color: #6366F1; background: rgba(238,242,255,0.95); }}
[data-testid="stAlert"] {{ border-radius: 12px; border: none; }}
[data-testid="stStatusWidget"], [data-testid="stStatus"] {{ border-radius: 12px; }}

/* Step bar as segmented pills */
.st-key-mv_steps .stButton > button {{
    border-radius: 999px; padding: 0.55rem 1rem; font-weight: 700;
    background: #FFFFFF; border: 1px solid #E2E8F0; color: #475569;
}}
.st-key-mv_steps .stButton > button[kind="primary"] {{ color: #fff; }}

/* Hero banner (verdict) */
.mv-hero {{
    border-radius: 18px; padding: 28px 24px; border: none;
    box-shadow: 0 10px 30px rgba(15,23,42,0.08);
}}
.mv-hero-title {{ font-size: 2.4rem; letter-spacing: 0.04em; }}
.mv-hero-pass {{ background: linear-gradient(135deg, #DCFCE7 0%, #ECFDF5 100%); box-shadow: 0 10px 30px rgba(22,163,74,0.18); }}
.mv-hero-fail {{ background: linear-gradient(135deg, #FEE2E2 0%, #FFF1F2 100%); box-shadow: 0 10px 30px rgba(220,38,38,0.18); }}
.mv-hero-incomplete {{ background: linear-gradient(135deg, #FEF3C7 0%, #FFFBEB 100%); }}
.mv-hero-neutral {{ background: linear-gradient(135deg, #EEF2FF 0%, #F8FAFC 100%); }}

/* Tables */
table.mv-table {{ border-collapse: separate; border-spacing: 0; overflow: hidden; border-radius: 12px; }}
table.mv-table th {{ background: #F1F5F9; color: #475569; padding: 10px 12px; }}
table.mv-table td {{ padding: 10px 12px; }}
table.mv-table tr:hover td {{ background: #F8FAFF; }}
.mv-chip {{ border-radius: 999px; background: #EEF2FF; color: #4338CA; border-color: #C7D2FE; padding: 3px 12px; }}
.mv-badge {{ border-radius: 999px; }}

/* Tabs */
button[data-baseweb="tab"][aria-selected="true"] {{ color: #4338CA; }}
div[data-baseweb="tab-highlight"] {{ background-color: #6366F1 !important; height: 3px; border-radius: 3px; }}
</style>
"""


def inject_theme_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
