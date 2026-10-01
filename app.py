"""Machine Vision Measurement — Streamlit UI.

This file only orchestrates the `vision`/`utils` packages; all detection,
calibration, geometry, and export logic lives there. See README.md for the
accuracy/calibration ground rules this app follows.

The UI is a 3-step guided wizard (Calibrate -> Measure -> Export) so a
first-time user only sees what's relevant right now. Power-user controls
(checkerboard/perspective calibration, the full measurement table, tolerances,
coordinate-origin choice) live behind the "Show advanced options" sidebar
toggle instead of being on screen by default.
"""
from __future__ import annotations

import hashlib
import logging
import traceback
from datetime import datetime
from html import escape as esc_html

import cv2
import numpy as np
import streamlit as st

from ui import inspection_view, theme
from utils import export as export_utils
from vision import annotations, calibration, geometry, measurement, preprocessing, qr, segmentation, validation
from vision import shifter_fork_geometry, shifter_fork_measurement
from vision.inspection_spec import INSPECTION_SPEC
from vision.shifter_fork_recipes import SHIFTER_FORK_RECIPES
from vision.types import CalibrationProfile, ToleranceSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("machine_vision_measurement")

st.set_page_config(page_title="Machine Vision Measurement", page_icon="📐", layout="wide")
theme.inject_theme_css()

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("wizard_step", 1)
st.session_state.setdefault("active_profile_id", None)
st.session_state.setdefault("pending_profile", None)
st.session_state.setdefault("pending_warp_preview", None)
st.session_state.setdefault("batch_results", {})
st.session_state.setdefault("tolerance_specs", {})
st.session_state.setdefault("selected_feature", {})
st.session_state.setdefault("inspection_spec_overrides", {})
st.session_state.setdefault("shifter_fork_recipe", {})

# "One-time machine calibration": a fresh session (no active profile chosen
# yet) auto-resumes the most recently saved calibration if one exists on
# disk, so the operator isn't forced to re-select it for every new session —
# they only ever calibrate again if they explicitly choose to.
if st.session_state["active_profile_id"] is None:
    _saved = calibration.list_profiles()
    if _saved:
        st.session_state["active_profile_id"] = _saved[0].profile_id


def get_active_profile() -> CalibrationProfile | None:
    pid = st.session_state.get("active_profile_id")
    if not pid:
        return None
    try:
        return calibration.load_profile(pid)
    except (FileNotFoundError, OSError):
        st.session_state["active_profile_id"] = None
        return None


def go_to_step(step: int) -> None:
    st.session_state["wizard_step"] = step
    st.rerun()


def fmt_value(px_value: float | None, mm_value: float | None) -> str:
    if mm_value is not None:
        return f"{mm_value:.2f} mm"
    if px_value is not None:
        return f"{px_value:.1f} px"
    return "—"


def compute_aggregate_status(tolerance_results: list) -> tuple[str, str]:
    """The single tri-state (+neutral) aggregate used everywhere a PASS/FAIL
    summary is shown (hero banner, feature cards, history, system status):
    FAIL if anything failed; INCOMPLETE if nothing failed but a configured
    spec couldn't be measured (N/A); PASS if at least one real PASS and
    nothing worse; else a neutral "measured, no tolerances configured" state
    — never a fabricated PASS/FAIL when no spec was ever set."""
    if not tolerance_results:
        return "neutral", "MEASURED"
    if any(t.status == "FAIL" for t in tolerance_results):
        return "fail", "FAIL"
    if any(t.status == "N/A" for t in tolerance_results):
        return "incomplete", "INCOMPLETE"
    if any(t.status == "PASS" for t in tolerance_results):
        return "pass", "PASS"
    return "neutral", "MEASURED"


def compute_inspection_spec_aggregate(results: list) -> tuple[str, str, dict]:
    """The product-spec's own INSPECTION RESULT, per the exact rule given:
    any FAIL -> overall FAIL; no FAIL but something couldn't be measured,
    lacks an approved tolerance, or is a specification conflict -> overall
    INCOMPLETE; every parameter measured and within tolerance -> overall
    PASS. A SPECIFICATION CONFLICT is deliberately never silently absorbed
    into a passing result. Separate from `compute_aggregate_status` (which
    drives the generic ad-hoc tolerance system's hero banner, kept
    unchanged) — this is the fork bracket's own named inspection outcome."""
    n = len(results)
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_incomplete = sum(1 for r in results if r.status == "INCOMPLETE")
    n_not_detected = sum(1 for r in results if r.status == "NOT DETECTED")
    n_not_specified = sum(1 for r in results if r.status == "NOT SPECIFIED")
    n_conflict = sum(1 for r in results if r.status == "SPECIFICATION CONFLICT")
    unresolved = n_incomplete + n_not_detected + n_not_specified + n_conflict
    counts = {
        "total": n, "measured": n_pass + n_fail, "pass": n_pass, "fail": n_fail,
        "incomplete": n_incomplete, "not_detected": n_not_detected,
        "not_specified": n_not_specified, "conflict": n_conflict, "unresolved": unresolved,
    }

    if n_fail > 0:
        return "fail", "FAIL", counts
    if unresolved > 0:
        return "incomplete", "INCOMPLETE", counts
    if n_pass > 0:
        return "pass", "PASS", counts
    return "neutral", "MEASURED", counts


def compute_shifter_fork_aggregate(sf_reject: bool, sf_reject_reason: str) -> tuple[str, str]:
    """The Shifter Fork recipe's single boolean accept/reject signal, as a
    (status_key, label) pair matching the hero-banner CSS keys used
    elsewhere (`mv-hero-pass` / `mv-hero-fail`)."""
    if sf_reject:
        return "fail", f"REJECT — {sf_reject_reason}" if sf_reject_reason else "REJECT"
    return "pass", "ACCEPT"


# ---------------------------------------------------------------------------
# Sidebar: grouped navigation (Inspection / Setup / Diagnostics / Reports)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="mv-nav-heading">Inspection</div>', unsafe_allow_html=True)
    if st.button("New Inspection", use_container_width=True, key="nav_new_inspection"):
        st.session_state["batch_results"] = {}
        st.session_state["tolerance_specs"] = {}
        st.session_state["selected_feature"] = {}
        go_to_step(1)
    if st.button("Measure Part", use_container_width=True, key="nav_measure_part"):
        go_to_step(2)
    with st.expander("Inspection History", expanded=False):
        if st.session_state["batch_results"]:
            for fname, data in st.session_state["batch_results"].items():
                status_key, status_label = compute_aggregate_status(data.get("tolerance_results", []))
                dot = {"pass": "🟢", "fail": "🔴", "incomplete": "🟠"}.get(status_key, "⚪")
                st.caption(f"{dot} **{fname}** — {status_label} · {data.get('timestamp', '—')}")
        else:
            st.caption("No inspections yet this session.")

    st.markdown('<div class="mv-nav-heading">Setup</div>', unsafe_allow_html=True)
    if st.button("Calibration", use_container_width=True, key="nav_calibration"):
        go_to_step(1)
    advanced_mode = st.checkbox(
        "Inspection Settings", value=False, key="advanced_mode",
        help="Full measurement table, tolerance configuration, and coordinate-origin choice.",
    )
    st.markdown(
        '<div class="mv-nav-item-disabled">Feature Configuration '
        '<span style="float:right;">Coming soon</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="mv-nav-heading">Diagnostics</div>', unsafe_allow_html=True)
    debug_mode = st.checkbox(
        "Vision Debug", value=False, key="debug_mode",
        help="Preprocessed image, main-part mask, and every hole candidate "
             "(accepted/rejected, with the reason).",
    )
    with st.expander("System Status", expanded=False):
        _sys_profile = get_active_profile()
        st.caption(f"Calibration: {'ACTIVE — ' + _sys_profile.name if _sys_profile else 'NOT ACTIVE'}")
        st.caption(f"Images measured this session: {len(st.session_state['batch_results'])}")
        _statuses = [compute_aggregate_status(d.get("tolerance_results", []))[0]
                     for d in st.session_state["batch_results"].values()]
        st.caption(f"Pass: {_statuses.count('pass')}  ·  Fail: {_statuses.count('fail')}  ·  "
                   f"Incomplete: {_statuses.count('incomplete')}")

    st.markdown('<div class="mv-nav-heading">Reports</div>', unsafe_allow_html=True)
    if st.button("Inspection Reports / Export", use_container_width=True, key="nav_export"):
        go_to_step(3)

    st.divider()
    st.caption(
        "A normal photograph has no inherent physical scale. Real-world mm measurements "
        "require an explicit calibration. This app never assumes 1 pixel = 1 mm."
    )

active_profile = get_active_profile()

_title_col, _status_col = st.columns([3, 2])
with _title_col:
    st.title("Machine Vision Inspection")
    st.caption("Dimensional Quality Inspection System")
with _status_col:
    _cal_line = (f'<span class="mv-status-dot pass"></span>Calibration: <b>ACTIVE</b> — {esc_html(active_profile.name)}'
                 if active_profile else '<span class="mv-status-dot warn"></span>Calibration: <b>NOT ACTIVE</b>')
    st.markdown(
        f"""<div style="text-align:right; padding-top:10px;">
            <div><span class="mv-status-dot pass"></span><b>SYSTEM READY</b></div>
            <div class="mv-header-sub">{_cal_line}</div>
            <div class="mv-header-sub">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Toolbar — every button routes to a real action (no decorative controls)
# ---------------------------------------------------------------------------
with st.container(key="mv_toolbar"):
    tb = st.columns(7)
    with tb[0]:
        if st.button("New Inspection", key="tb_new", use_container_width=True):
            st.session_state["batch_results"] = {}
            st.session_state["tolerance_specs"] = {}
            st.session_state["selected_feature"] = {}
            go_to_step(1)
    with tb[1]:
        if st.button("Load Image", key="tb_load", use_container_width=True):
            go_to_step(2)
    with tb[2]:
        if st.button("Re-measure", key="tb_remeasure", use_container_width=True):
            st.rerun()
    with tb[3]:
        if st.button("Calibration", key="tb_cal", use_container_width=True):
            go_to_step(1)
    with tb[4]:
        if st.button("Debug", key="tb_debug", use_container_width=True):
            st.session_state["debug_mode"] = not st.session_state.get("debug_mode", False)
            st.rerun()
    with tb[5]:
        if st.button("Export Report", key="tb_export", use_container_width=True):
            go_to_step(3)
    with tb[6]:
        st.markdown(
            '<button onclick="window.print()" style="width:100%;padding:3px 10px;'
            'font-size:0.8rem;border:1px solid #D9DEE7;border-radius:5px;background:#FFFFFF;'
            'color:#17202A;cursor:pointer;font-family:Inter,sans-serif;margin-top:1px;">'
            '\U0001F5A8 Print</button>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Step navigation bar
# ---------------------------------------------------------------------------
STEP_LABELS = ["① Calibrate", "② Measure Parts", "③ Export Results"]
nav_cols = st.columns(3)
for i, col in enumerate(nav_cols, start=1):
    with col:
        if st.button(STEP_LABELS[i - 1], key=f"nav_step_{i}", type=("primary" if i == st.session_state["wizard_step"] else "secondary"), use_container_width=True):
            go_to_step(i)
st.divider()

step = st.session_state["wizard_step"]

# ===========================================================================
# STEP 1: Calibrate
# ===========================================================================
if step == 1:
    st.header("Step 1: Calibrate")
    st.write(
        "To show real-world millimetre measurements, the app needs to know your photo's scale. "
        "**If you skip this step, you'll still get accurate pixel measurements — just no mm.**"
    )

    if active_profile is not None:
        st.success(
            f"✅ Calibration active: **{active_profile.name}** "
            f"({active_profile.mm_per_pixel:.4f} mm/pixel, {active_profile.confidence} confidence)"
        )

    saved_profiles = calibration.list_profiles()
    if saved_profiles:
        with st.expander(f"Use a saved calibration ({len(saved_profiles)} available)", expanded=active_profile is None):
            for p in saved_profiles:
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.write(f"**{p.name}** — {p.method.replace('_', ' ')}, {p.confidence} confidence, {p.mm_per_pixel:.4f} mm/px")
                with c2:
                    if st.button("Use", key=f"wiz_use_{p.profile_id}"):
                        st.session_state["active_profile_id"] = p.profile_id
                        st.rerun()
                with c3:
                    if st.button("🗑️", key=f"wiz_del_{p.profile_id}"):
                        calibration.delete_profile(p.profile_id)
                        if st.session_state["active_profile_id"] == p.profile_id:
                            st.session_state["active_profile_id"] = None
                        st.rerun()

    st.subheader("Create a new calibration")
    st.write(
        "**Easiest option:** photograph something with a precisely known size — a coin, ruler, "
        "an ID/credit card (85.60 × 53.98 mm), or a printed sheet — on a plain background."
    )

    with st.form("simple_known_dim_form"):
        ref_file = st.file_uploader("Photo of your reference object", type=IMAGE_TYPES)
        col1, col2 = st.columns(2)
        with col1:
            known_mm = st.number_input("Its real size (mm)", min_value=0.001, value=50.0, step=0.1, format="%.3f")
        with col2:
            direction_choice = st.radio("Which side did you measure?", ["Left to right (horizontal)", "Top to bottom (vertical)"])
        cal_name = st.text_input("Give this calibration a name (optional)")
        submitted = st.form_submit_button("Calculate calibration", type="primary")

    if submitted:
        if ref_file is None:
            st.error("Upload a photo of your reference object first.")
        else:
            direction = "horizontal" if direction_choice.startswith("Left") else "vertical"
            try:
                img = preprocessing.decode_image_bytes(ref_file.getvalue())
                img, _ = preprocessing.resize_for_processing(img)
                profile = calibration.known_dimension_calibrate(img, known_mm, direction, name=cal_name)
                st.session_state["pending_profile"] = profile
                st.session_state["pending_warp_preview"] = None
            except ValueError as e:
                st.error(str(e))
                st.session_state["pending_profile"] = None

    # --- Advanced calibration methods -----------------------------------------
    if advanced_mode:
        with st.expander("Advanced: Checkerboard or Perspective correction"):
            method = st.radio("Method", ["Checkerboard", "Perspective / 4-point"], horizontal=True)

            if method == "Checkerboard":
                st.info(
                    "Upload one or more checkerboard images. One image gives a scale-only "
                    "calibration. Three or more images from different angles/positions also "
                    "estimate the camera's lens distortion."
                )
                with st.form("checkerboard_form"):
                    cb_files = st.file_uploader("Checkerboard image(s)", type=IMAGE_TYPES, accept_multiple_files=True)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        pattern_cols = st.number_input("Inner corners (columns)", min_value=2, value=9, step=1)
                    with c2:
                        pattern_rows = st.number_input("Inner corners (rows)", min_value=2, value=6, step=1)
                    with c3:
                        square_mm = st.number_input("Square size (mm)", min_value=0.001, value=25.0, step=0.5, format="%.3f")
                    cb_name = st.text_input("Calibration name (optional)", key="cb_name")
                    cb_submitted = st.form_submit_button("Detect checkerboard & compute calibration")

                if cb_submitted:
                    if not cb_files:
                        st.error("Upload at least one checkerboard image.")
                    else:
                        try:
                            imgs = [preprocessing.decode_image_bytes(f.getvalue()) for f in cb_files]
                            profile = calibration.checkerboard_calibrate(
                                imgs, int(pattern_cols), int(pattern_rows), square_mm, name=cb_name,
                            )
                            st.session_state["pending_profile"] = profile
                            st.session_state["pending_warp_preview"] = None
                        except ValueError as e:
                            st.error(str(e))
                            st.session_state["pending_profile"] = None

            else:
                st.info(
                    "Select an image and enter the pixel coordinates of the 4 corners of a "
                    "rectangle of known real-world size, in order: top-left, top-right, "
                    "bottom-right, bottom-left. The image will be warped to correct perspective skew."
                )
                persp_file = st.file_uploader("Reference image", type=IMAGE_TYPES, key="persp_file")

                if persp_file is not None:
                    p_img = preprocessing.decode_image_bytes(persp_file.getvalue())
                    p_img, _ = preprocessing.resize_for_processing(p_img)
                    H, W = p_img.shape[:2]
                    inset_x, inset_y = int(0.1 * W), int(0.1 * H)
                    default_pts = [(inset_x, inset_y), (W - inset_x, inset_y), (W - inset_x, H - inset_y), (inset_x, H - inset_y)]

                    st.markdown("**Corner points (pixels)**")
                    labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
                    pts = []
                    pcols = st.columns(4)
                    for i, (col, label) in enumerate(zip(pcols, labels)):
                        with col:
                            px = st.number_input(f"{label} X", min_value=0, max_value=W - 1, value=default_pts[i][0], key=f"persp_x{i}")
                            py = st.number_input(f"{label} Y", min_value=0, max_value=H - 1, value=default_pts[i][1], key=f"persp_y{i}")
                            pts.append((float(px), float(py)))

                    preview = p_img.copy()
                    pts_arr = np.array(pts, dtype=np.int32)
                    cv2.polylines(preview, [pts_arr], isClosed=True, color=(0, 255, 255), thickness=2)
                    for p in pts:
                        cv2.circle(preview, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
                    st.image(preprocessing.to_rgb(preview), caption="Selected quadrilateral", width=500)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        known_w = st.number_input("Known width (mm)", min_value=0.001, value=50.0, step=0.5, format="%.3f")
                    with c2:
                        known_h = st.number_input("Known height (mm)", min_value=0.001, value=50.0, step=0.5, format="%.3f")
                    with c3:
                        px_per_mm_target = st.number_input("Warped output resolution (px/mm)", min_value=1.0, value=10.0, step=1.0)
                    persp_name = st.text_input("Calibration name (optional)", key="persp_name")

                    if st.button("Compute perspective calibration"):
                        try:
                            profile, warped = calibration.perspective_calibrate(
                                p_img, pts, known_w, known_h, px_per_mm_target, name=persp_name,
                            )
                            st.session_state["pending_profile"] = profile
                            st.session_state["pending_warp_preview"] = warped
                        except ValueError as e:
                            st.error(str(e))
                            st.session_state["pending_profile"] = None

    # --- Pending profile preview + save ----------------------------------------
    pending: CalibrationProfile | None = st.session_state.get("pending_profile")
    if pending is not None:
        st.divider()
        st.markdown("### Calibration result (not yet saved)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Detected reference", f"{pending.detected_pixels:.2f} px" if pending.detected_pixels else "—")
        c2.metric("Calibration", f"{pending.mm_per_pixel:.5f} mm/px")
        c3.metric("Confidence", pending.confidence)
        if advanced_mode:
            for n in pending.notes:
                st.caption(f"• {n}")
        if st.session_state.get("pending_warp_preview") is not None:
            st.image(preprocessing.to_rgb(st.session_state["pending_warp_preview"]),
                      caption="Warped (fronto-parallel) preview", width=400)

        save_col1, save_col2 = st.columns(2)
        with save_col1:
            if st.button("💾 Save & use this calibration", type="primary", use_container_width=True):
                calibration.save_profile(pending)
                st.session_state["active_profile_id"] = pending.profile_id
                st.session_state["pending_profile"] = None
                st.success("Calibration saved and active.")
                st.rerun()
        with save_col2:
            if st.button("Discard", use_container_width=True):
                st.session_state["pending_profile"] = None
                st.rerun()

    st.divider()
    next_label = "Next: Measure Parts →" if active_profile else "Continue without calibration (pixels only) →"
    if st.button(next_label, type="primary"):
        go_to_step(2)

# ===========================================================================
# STEP 2: Measure
# ===========================================================================
elif step == 2:
    st.header("Step 2: Upload & Measure")

    if active_profile is not None:
        st.success(f"Using calibration **{active_profile.name}** — results will show mm.")
    else:
        st.warning("No calibration active — results will show pixels only. Go to Step 1 to calibrate.")

    uploaded_files = st.file_uploader(
        "Upload one or more part images",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        help="For reliable measurement, use a fixed camera, controlled lighting and a plain, contrasting background.",
    )

    if advanced_mode:
        origin_choice = st.radio("MM coordinate origin", ["Bounding box top-left", "Part centroid"], horizontal=True)
        st.checkbox("Y axis increases upward (engineering convention)", value=True)
        decimals = st.selectbox("Decimal places for mm values", [0, 1, 2, 3, 4], index=2, key="decimals_precision")
    else:
        origin_choice = "Bounding box top-left"
        decimals = 2

    if not uploaded_files:
        st.info("Upload at least one image to begin measurement.")
    else:
        for idx, uf in enumerate(uploaded_files):
            st.divider()

            sf_recipe_options = ["None (legacy Fork/Bracket spec only)"] + list(SHIFTER_FORK_RECIPES)
            sf_choice = st.selectbox(
                "Shifter Fork recipe",
                sf_recipe_options,
                format_func=lambda pid: pid if pid == sf_recipe_options[0] else f"{pid} — {SHIFTER_FORK_RECIPES[pid].title}",
                key=f"sf_recipe_{uf.name}",
                help="Select a part number to run the Shifter Fork Dimensional Inspection recipe on this image, "
                     "in addition to the existing Fork/Bracket inspection parameters below.",
            )
            st.session_state["shifter_fork_recipe"][uf.name] = None if sf_choice == sf_recipe_options[0] else sf_choice

            try:
                original = preprocessing.decode_image_bytes(uf.getvalue())
            except ValueError as e:
                st.error(str(e))
                continue

            img, _ = preprocessing.resize_for_processing(original)

            profile_for_image = active_profile
            display_img = img
            part = quality = seg_diag = gray = circles = candidates = None
            tips = orientation = qr_result = None
            records = feature_table_rows = tolerance_results = origin_px = inspection_results = None
            sf_landmarks = sf_results = None
            sf_reject, sf_reject_reason = True, ""
            pipeline_ok = False

            # -----------------------------------------------------------------
            # Run the unchanged detection/measurement pipeline, staged so the
            # status label reflects real work already being done (no artificial
            # delay is added anywhere below).
            # -----------------------------------------------------------------
            with st.status(f"Inspecting {uf.name}", expanded=False) as status:
                try:
                    if active_profile is not None and active_profile.method == "perspective":
                        status.update(label="Applying perspective correction...")
                        try:
                            display_img = calibration.warp_with_profile(img, active_profile)
                        except ValueError as e:
                            st.error(f"Could not apply perspective calibration: {e}")
                            display_img = img
                            profile_for_image = None

                    status.update(label="Detecting part boundary...")
                    part = segmentation.build_detected_part(display_img)
                    quality = validation.build_quality_report(display_img, part, profile_for_image)

                    # --- Hard gate: refuse to measure rather than mislead ------
                    if quality.has_errors():
                        status.update(label="Image quality insufficient", state="error")
                        st.error("Image quality insufficient for reliable measurement.")
                        for issue in quality.issues:
                            if issue.severity == "error":
                                st.caption(f"• {issue.message}")
                    else:
                        status.update(label="Validating outer boundary...")
                        seg_diag = validation.evaluate_segmentation(part, display_img.shape[:2])

                        if debug_mode:
                            with st.expander(f"🔍 Vision Debug — outer boundary ({uf.name})", expanded=True):
                                if part is not None:
                                    boundary_img = annotations.draw_boundary_debug(display_img, part)
                                    st.image(preprocessing.to_rgb(boundary_img), caption="Outer boundary diagnostics", use_container_width=True)
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Contour area", f"{seg_diag.area_px2:,.0f} px²")
                                m2.metric("Bounding-box area", f"{seg_diag.bbox_area_px2:,.0f} px²")
                                m3.metric("Extent (informational)", f"{seg_diag.extent:.0%}")
                                m4.metric("Perimeter", f"{seg_diag.perimeter_px:,.0f} px")
                                m5, m6, m7 = st.columns(3)
                                m5.metric("Convex hull area", f"{seg_diag.convex_hull_area_px2:,.0f} px²")
                                m6.metric("Solidity", f"{seg_diag.solidity:.0%}",
                                          help=f"Must be ≥ {validation.MIN_SOLIDITY_FOR_RELIABLE_CONTOUR:.0%} to accept.")
                                m7.metric("Compactness", f"{seg_diag.compactness:.2f}",
                                          help=f"Must be ≤ {validation.MAX_COMPACTNESS_FOR_RELIABLE_CONTOUR:.1f} to accept "
                                               "(1.0 = a perfect circle; higher = more jagged/noisy).")
                                if seg_diag.accepted:
                                    st.success("Boundary ACCEPTED — extent is low fill of the bounding rectangle "
                                               "shown above for information only; it is never used to reject a shape.")
                                else:
                                    st.error("Boundary REJECTED")
                                    for r in seg_diag.reasons:
                                        st.caption(f"• {r}")

                        # --- Hard gate: refuse to measure rather than mislead --
                        if not seg_diag.accepted:
                            status.update(label="Boundary rejected", state="error")
                            st.error("Part boundary could not be detected reliably.")
                            st.caption(" ".join(seg_diag.reasons))
                        else:
                            warnings_only = [i for i in quality.issues if i.severity == "warning"]
                            if warnings_only or advanced_mode:
                                with st.expander("⚠️ Image quality warnings", expanded=bool(warnings_only)):
                                    if not warnings_only:
                                        st.success("No quality issues detected.")
                                    for issue in quality.issues:
                                        if issue.severity == "warning":
                                            st.warning(issue.message)
                                        elif issue.severity == "info" and advanced_mode:
                                            st.info(issue.message)

                            status.update(label="Detecting features (holes / bores)...")
                            gray = preprocessing.to_gray(display_img)
                            circles, candidates = geometry.detect_holes(
                                gray, part.mask, part.hole_contours, part.bbox, part.area_px2, part.contour,
                            )
                            circles = measurement.classify_and_label_circles(circles, part)

                            # Orientation + QR are independent diagnostics —
                            # neither gates or changes any measurement below
                            # (see vision.geometry.estimate_part_orientation
                            # and vision.qr.detect_qr module docstrings).
                            tips = geometry.find_fork_tips(part.contour, part.area_px2)
                            orientation = geometry.estimate_part_orientation(part, circles, tips)
                            qr_result = qr.detect_qr(gray)

                            status.update(label="Measuring dimensions...")
                            x, y, w, h = part.bbox
                            if origin_choice == "Part centroid":
                                m = cv2.moments(part.contour)
                                origin_px = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (x + w / 2, y + h / 2)
                            else:
                                origin_px = (float(x), float(y))

                            records = measurement.build_measurement_records(part, circles, profile_for_image, origin_px)

                            status.update(label="Checking tolerances...")
                            tolerance_results = measurement.apply_tolerances(
                                records, st.session_state["tolerance_specs"].get(uf.name, {}),
                            )
                            feature_table_rows = measurement.build_feature_summary_table(
                                circles, profile_for_image, origin_px, tolerance_results,
                            )

                            status.update(label="Evaluating inspection parameters...")
                            spec_overrides = st.session_state["inspection_spec_overrides"].get(uf.name, {})
                            inspection_results = measurement.evaluate_inspection_spec(
                                part, circles, records, profile_for_image, spec_overrides,
                            )

                            sf_recipe_id = st.session_state["shifter_fork_recipe"].get(uf.name)
                            if sf_recipe_id is not None:
                                status.update(label="Evaluating Shifter Fork recipe...")
                                sf_recipe = SHIFTER_FORK_RECIPES[sf_recipe_id]
                                sf_landmarks = shifter_fork_geometry.extract_shifter_fork_landmarks(part, circles)
                                sf_results, sf_reject, sf_reject_reason = shifter_fork_measurement.evaluate_shifter_fork(
                                    sf_landmarks, sf_recipe, profile_for_image,
                                )

                            pipeline_ok = True
                            status.update(label="Inspection complete", state="complete")
                except Exception:
                    logger.exception("Unexpected error while processing %s", uf.name)
                    status.update(label="Internal error", state="error")
                    st.error("An internal error occurred while processing this image. This has been logged; "
                             "try a different image or report the issue.")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())

            if not pipeline_ok:
                continue

            # ===================================================================
            # Result screen — light industrial metrology dashboard layout
            # ===================================================================
            key_results = inspection_view.build_hmi_key_results(sf_results, sf_landmarks, inspection_results, part.bbox, decimals)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if sf_results is not None:
                hmi_verdict_key, hmi_verdict_label = compute_shifter_fork_aggregate(sf_reject, sf_reject_reason)
            else:
                # Judged from exactly the rows the compact card shows, not
                # the full 22-parameter spec -- see
                # inspection_view.compute_hmi_verdict's docstring for why
                # (the full-spec aggregate is still shown, unchanged, in
                # Advanced / Detailed View's own "INSPECTION RESULT" hero).
                hmi_verdict_key, hmi_verdict_label = inspection_view.compute_hmi_verdict(key_results)

            # --- POS DEMO ONLY: remove this block after the demo ----------------
            # The real verdict is always computed above; only when it is FAIL do we
            # offer a button that swaps BOTH views to a hardcoded all-PASS result.
            demo_key = f"pos_demo_pass_{uf.name}"
            demo_on = st.session_state.get(demo_key, False)
            if hmi_verdict_key == "fail" or demo_on:
                if st.button("Back to real result" if demo_on else "Show PASS result (POS demo)",
                             key=f"{demo_key}_btn"):
                    st.session_state[demo_key] = not demo_on
                    st.rerun()
            if demo_on:
                inspection_results, sf_results, sf_reject, sf_reject_reason = inspection_view.demo_force_pass_all(
                    inspection_results, sf_results,
                )
                key_results = inspection_view.build_hmi_key_results(sf_results, sf_landmarks, inspection_results, part.bbox, decimals)
                key_results, hmi_verdict_key, hmi_verdict_label = inspection_view.demo_force_pass(key_results)
            # --- end POS DEMO ----------------------------------------------------

            advanced_view = st.checkbox(
                "Advanced / Detailed View", key=f"advanced_view_{uf.name}", value=False,
                help="Off: compact HMI-style summary. On: the full detailed dashboard (tabs, tables, tolerance "
                     "configuration, feature cards, debug mode).",
            )

            if advanced_view:
                tol_by_feature = {t.feature: t for t in tolerance_results}
                agg_key, agg_label = compute_aggregate_status(tolerance_results)

                inspection_id = hashlib.md5(uf.getvalue()).hexdigest()[:8].upper()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                n_bores = sum(1 for c in circles if c.feature_type == "bore")
                n_holes = len(circles) - n_bores
                n_pass = sum(1 for t in tolerance_results if t.status == "PASS")

                # --- PART INSPECTION RESULT: big PASS/FAIL/INCOMPLETE hero -----------
                hero_sub = (
                    f"Inspection Complete &nbsp;·&nbsp; Part <b>{esc_html(uf.name)}</b> "
                    f"&nbsp;·&nbsp; ID {inspection_id} &nbsp;·&nbsp; {timestamp}<br/>"
                    f"Calibration: <b>{esc_html(profile_for_image.name) if profile_for_image else 'Not calibrated'}</b>"
                    + (f" &nbsp;·&nbsp; {len(tolerance_results)} characteristic(s) inspected, "
                       f"{n_pass}/{len(tolerance_results)} passed" if tolerance_results else
                       " &nbsp;·&nbsp; 0 characteristics configured — showing measured values only")
                )
                st.markdown(
                    f"""<div class="mv-hero mv-hero-{agg_key}">
                        <div class="mv-hero-title">{agg_label}</div>
                        <div class="mv-hero-sub">{hero_sub}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""<div class="mv-panel" style="padding:10px 16px;">
                        <div class="mv-chip-row" style="margin-top:0;">
                            <span class="mv-chip">{len(circles)} feature(s)</span>
                            <span class="mv-chip">{n_holes} hole(s)</span>
                            <span class="mv-chip">{n_bores} bore(s)</span>
                            <span class="mv-chip">Boundary: {esc_html(part.segmentation_method)}</span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # --- INSPECTION RESULT: the 10-parameter fork-bracket spec -----------
                spec_key, spec_label, spec_counts = compute_inspection_spec_aggregate(inspection_results)
                spec_sub = (
                    f"{spec_counts['total']} Parameters &nbsp;·&nbsp; {spec_counts['measured']} Measured "
                    f"&nbsp;·&nbsp; {spec_counts['pass']} PASS &nbsp;·&nbsp; {spec_counts['fail']} FAIL "
                    f"&nbsp;·&nbsp; {spec_counts['incomplete'] + spec_counts['not_detected']} Incomplete/Not Detected "
                    f"&nbsp;·&nbsp; {spec_counts['not_specified']} No Tolerance"
                    + (f" &nbsp;·&nbsp; ⚠ {spec_counts['conflict']} Specification Conflict" if spec_counts['conflict'] else "")
                )
                st.markdown(
                    f"""<div class="mv-hero mv-hero-{spec_key}" style="padding:14px 20px;">
                        <div class="mv-hero-title" style="font-size:1.5rem;">INSPECTION RESULT: {spec_label}</div>
                        <div class="mv-hero-sub">{spec_sub}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                with st.container(border=True):
                    st.markdown("#### Inspection Parameters")
                    st.caption(
                        "Product-specific spec for this fork bracket. Nominal/tolerance values are "
                        "configurable placeholders (see Tolerance Configuration below), not "
                        "engineering-approved limits."
                    )
                    spec_table_html = inspection_view.build_inspection_parameters_table_html(inspection_results, decimals)
                    st.markdown(f'<div class="mv-panel">{spec_table_html}</div>', unsafe_allow_html=True)

                    with st.expander("Tolerance Configuration (Inspection Parameters)"):
                        st.caption(
                            "Override the nominal/tolerance for any parameter — an override always applies "
                            "a bilateral (±) tolerance and is the explicit confirmation a SPECIFICATION "
                            "CONFLICT row is waiting for. Changes apply immediately to this image's "
                            "PASS/FAIL/INCOMPLETE/NOT SPECIFIED result above."
                        )
                        overrides_for_image = st.session_state["inspection_spec_overrides"].setdefault(uf.name, {})
                        for spec in INSPECTION_SPEC:
                            default_tol_mm = spec.tolerance.plus_mm or spec.tolerance.minus_mm
                            cur_nominal, cur_tol = overrides_for_image.get(
                                spec.parameter_id, (spec.nominal_mm, default_tol_mm),
                            )
                            oc1, oc2, oc3 = st.columns([3, 1, 1])
                            with oc1:
                                st.caption(f"{spec.name}  ·  spec: {spec.tolerance.display()}")
                            with oc2:
                                new_nominal = st.number_input(
                                    "Nominal (mm)", value=float(cur_nominal), key=f"spec_nom_{uf.name}_{spec.parameter_id}",
                                    format="%.3f", label_visibility="collapsed",
                                )
                            with oc3:
                                new_tol = st.number_input(
                                    "± Tolerance (mm)", value=float(cur_tol), min_value=0.0,
                                    key=f"spec_tol_{uf.name}_{spec.parameter_id}", format="%.3f", label_visibility="collapsed",
                                )
                            overrides_for_image[spec.parameter_id] = (new_nominal, new_tol)

                # --- Shifter Fork Dimensional Inspection recipe (opt-in) -------------
                if sf_results is not None:
                    sf_key, sf_label = compute_shifter_fork_aggregate(sf_reject, sf_reject_reason)
                    st.markdown(
                        f"""<div class="mv-hero mv-hero-{sf_key}" style="padding:14px 20px;">
                            <div class="mv-hero-title" style="font-size:1.5rem;">SHIFTER FORK: {esc_html(sf_label)}</div>
                            <div class="mv-hero-sub">Recipe {esc_html(sf_recipe_id)} — {esc_html(SHIFTER_FORK_RECIPES[sf_recipe_id].title)}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    with st.container(border=True):
                        st.markdown("#### Shifter Fork Inspection Parameters")
                        st.caption(
                            "CONFIRMED rows decide ACCEPT/REJECT. ASSUMED rows (best-guess mapping "
                            "to the drawing, pending sign-off) and INFO ONLY rows (e.g. Bore Diameter — "
                            "its H7 tolerance is tighter than a camera can resolve) are always measured "
                            "and shown, but never affect the verdict."
                        )
                        sf_table_html = inspection_view.build_shifter_fork_table_html(sf_results, decimals)
                        st.markdown(f'<div class="mv-panel">{sf_table_html}</div>', unsafe_allow_html=True)

                sel_map = st.session_state["selected_feature"]
                selected_short_id = sel_map.get(uf.name)

                # --- Main workspace: ~65% image viewer / ~35% measurement panel ------
                col_view, col_side = st.columns([13, 7])

                with col_view:
                    tab_labels = ["Original", "Processed", "Measurement", "Overlay"] + (["Debug"] if debug_mode else [])
                    tabs = st.tabs(tab_labels)

                    # Built once, used by both the Measurement tab (render) and
                    # the Debug tab (placement diagnostics) in this same run.
                    img_h, img_w = display_img.shape[:2]
                    svg, placement_diag = inspection_view.build_svg_overlay(
                        part, circles, records, profile_for_image, origin_px, img_w, img_h,
                        tol_by_feature, selected_short_id, decimals=decimals,
                    )
                    inspection_results_by_id = {r.parameter_id: r for r in inspection_results}
                    fork_svg, fork_placement_diag = inspection_view.build_fork_annotations_svg(
                        inspection_results_by_id, [d["box"] for d in placement_diag], img_w, img_h, decimals,
                    )
                    svg = svg[:-len("</svg>")] + fork_svg + "</svg>"
                    placement_diag = placement_diag + fork_placement_diag
                    focus = None
                    if selected_short_id:
                        focus_circle = next((c for c in circles if c.short_id == selected_short_id), None)
                        if focus_circle is not None:
                            focus = (focus_circle.cx, focus_circle.cy, focus_circle.r)

                    with tabs[0]:
                        st.image(preprocessing.to_rgb(original), caption="Original (as uploaded)", use_container_width=True)

                    with tabs[1]:
                        st.image(preprocessing.to_rgb(display_img), caption="Preprocessed / working image", use_container_width=True)
                        st.caption(f"Segmentation method: {part.segmentation_method}")

                    with tabs[2]:
                        html_component = inspection_view.build_inspection_html(
                            display_img, svg, img_w, img_h, viewport_id=f"mvViewport{idx}", focus=focus,
                        )
                        st.components.v1.html(html_component, height=inspection_view.COMPONENT_HEIGHT_PX, scrolling=False)
                        if profile_for_image is None:
                            st.caption("⚠️ Showing pixels only — calibrate in Step 1 to see mm.")
                        perspective_notes = [i.message for i in quality.issues if "perspective" in i.message.lower() or "aspect ratio" in i.message.lower()]
                        for note in perspective_notes:
                            st.caption(f"ℹ️ {note}")

                    with tabs[3]:
                        overlay_img = annotations.draw_full_annotation(display_img, part, circles, profile_for_image, origin_px)
                        st.image(preprocessing.to_rgb(overlay_img), caption="Detection overlay (boundary + circles)", use_container_width=True)

                    if debug_mode:
                        with tabs[4]:
                            st.markdown("##### 1–2 · Original / preprocessed")
                            d1, d2 = st.columns(2)
                            with d1:
                                st.image(preprocessing.to_rgb(original), caption="Original (as uploaded)", use_container_width=True)
                            with d2:
                                st.image(preprocessing.to_rgb(display_img), caption="Preprocessed / working image", use_container_width=True)

                            st.markdown("##### 3–6 · Intermediate hole-candidate masks")
                            hole_masks = segmentation.debug_hole_masks(gray, part.mask)
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            with mc1:
                                st.image(hole_masks.get("otsu_dark"), caption="Otsu threshold (dark class)", use_container_width=True)
                            with mc2:
                                st.image(hole_masks.get("adaptive_raw"), caption="Adaptive threshold (raw)", use_container_width=True)
                            with mc3:
                                st.image(hole_masks.get("adaptive_after_morphology"), caption="Adaptive threshold (after morphological open+close)", use_container_width=True)
                            with mc4:
                                st.image(hole_masks.get("canny"), caption="Canny edges (within part mask)", use_container_width=True)
                            st.image(part.mask, caption=f"Main-part mask (via {part.segmentation_method})", width=300)

                            st.markdown("##### 7–9 · Hole candidates")
                            debug_img = annotations.draw_hole_candidates_debug(display_img, candidates)
                            st.image(preprocessing.to_rgb(debug_img), caption="Hole candidates — green = accepted, red = rejected", use_container_width=True)

                            accepted = [c for c in candidates if c.accepted]
                            rejected = [c for c in candidates if not c.accepted]
                            st.write(f"**{len(accepted)} accepted, {len(rejected)} rejected** out of {len(candidates)} candidate region(s).")

                            for i, c in enumerate(accepted, start=1):
                                st.caption(
                                    f"Hole #{i}: confidence {c.confidence_score * 100:.0f}% ({c.confidence_label}) · "
                                    f"center ({c.cx:.1f}, {c.cy:.1f}) · equivalent diameter {c.equiv_diameter_px:.1f}px"
                                    + (" · Hough-confirmed" if c.hough_confirmed else "")
                                )

                            if rejected:
                                st.write("Rejected candidates:")
                                for c in rejected[:50]:
                                    st.caption(
                                        f"- center ({c.cx:.1f}, {c.cy:.1f}), diameter {c.equiv_diameter_px:.1f}px — "
                                        + "; ".join(c.rejection_reasons)
                                    )
                                if len(rejected) > 50:
                                    st.caption(f"...and {len(rejected) - 50} more rejected candidates.")

                            st.markdown("##### 10 · Label-placement diagnostics")
                            st.caption(
                                "Every dimension/feature/corner label's chosen position out of the 8 "
                                "candidates (N/S/E/W/NE/NW/SE/SW), its collision score (lower is better; "
                                "0 = no overlap with anything already placed), and its final bounding box "
                                "in image pixels."
                            )
                            placement_rows = [{
                                "Label": d["id"], "Kind": d["kind"], "Direction": d["direction"],
                                "Collision score": d["score"],
                                "Box center (px)": f"({d['box']['cx']:.1f}, {d['box']['cy']:.1f})",
                                "Box size (px)": f"{d['box']['w']:.0f} × {d['box']['h']:.0f}",
                            } for d in placement_diag]
                            st.dataframe(placement_rows, use_container_width=True, hide_index=True)

                            st.markdown("##### 11 · Dimension / feature anchor points")
                            anchor_rows = [{"Feature": c.short_id or c.label, "Anchor X (px)": round(c.cx, 1),
                                             "Anchor Y (px)": round(c.cy, 1), "Radius (px)": round(c.r, 1)} for c in circles]
                            st.dataframe(anchor_rows, use_container_width=True, hide_index=True)

                            st.markdown("##### 12 · Calibration scale")
                            if profile_for_image is not None:
                                st.caption(f"{profile_for_image.mm_per_pixel:.6f} mm/pixel — method: {profile_for_image.method}")
                            else:
                                st.caption("No calibration active — all values above are in pixels only.")

                            st.markdown("##### 13 · Inspection Parameters — per-parameter breakdown")
                            st.caption(
                                "How every one of the 10 parameters was computed: detection method, "
                                "source pixel geometry, confidence, and — for anything not PASS/FAIL — "
                                "exactly why."
                            )
                            for r in inspection_results:
                                with st.expander(f"{r.name} — {r.status}"):
                                    st.caption(f"Method: {r.method}")
                                    st.caption(f"Confidence: {r.confidence}")
                                    if r.reason:
                                        st.caption(f"Reason: {r.reason}")
                                    debug_rows = [{"Field": k, "Value": str(v)} for k, v in r.debug.items()]
                                    if debug_rows:
                                        st.dataframe(debug_rows, use_container_width=True, hide_index=True)

                            st.markdown("##### 14 · Orientation & QR diagnostics")
                            st.caption(
                                "Reported for visualization only — no measurement above depends on this "
                                "(diameters/radii come from fitted circles, distances from Euclidean "
                                "geometry between detected points, and Overall Fork Width/Height from "
                                "the part's own rotated bounding rectangle, all rotation-invariant already)."
                            )
                            oc1, oc2 = st.columns(2)
                            with oc1:
                                st.markdown("**Orientation**")
                                if orientation is not None and orientation["confidence"] == "HIGH":
                                    st.caption(f"Direction (fork opening → main hole): {orientation['angle_deg']:.1f}°")
                                    st.caption(f"Rotated-rect angle (raw): {orientation['rect_angle_deg']:.1f}°")
                                    st.caption("Confidence: HIGH")
                                    pts = orientation["reference_points"]
                                    st.caption(
                                        f"Reference points — fork-tip midpoint: "
                                        f"({pts['fork_tip_midpoint'][0]:.1f}, {pts['fork_tip_midpoint'][1]:.1f}), "
                                        f"main-hole center: ({pts['main_hole_center'][0]:.1f}, {pts['main_hole_center'][1]:.1f})"
                                    )
                                elif orientation is not None:
                                    st.caption("Orientation ambiguous — inspection not evaluated for direction.")
                                    st.caption(f"Reason: {orientation['reason']}")
                                    st.caption(f"Rotated-rect angle (raw, always available): {orientation['rect_angle_deg']:.1f}°")
                                else:
                                    st.caption("Not computed for this image.")
                            with oc2:
                                st.markdown("**QR code**")
                                if qr_result is not None:
                                    st.caption(f"Status: {qr_result['status']}")
                                    if qr_result["status"] == qr.QR_STATUS_DETECTED:
                                        st.caption(f"Data: {qr_result['data']}")
                                    st.caption("Independent diagnostic — never gates or fails inspection.")
                                else:
                                    st.caption("Not computed for this image.")

                            if sf_landmarks is not None:
                                st.markdown("##### 15 · Shifter Fork landmarks")
                                if sf_landmarks.reject_reason:
                                    st.error(f"Hard failure: {sf_landmarks.reject_reason}")
                                sf_debug_img = annotations.draw_shifter_fork_debug(display_img, sf_landmarks)
                                st.image(preprocessing.to_rgb(sf_debug_img), caption="Shifter Fork geometry diagnostics", use_container_width=True)
                                st.caption(
                                    "Axis is the part's own rotated-rectangle long axis through its true area "
                                    "centroid, oriented away from the tip — an implementation assumption, "
                                    "not a verified drawing convention; flagged pending validation against a "
                                    "real part."
                                )

                details_key = f"view_details_{uf.name}"

                with col_side:
                    width_r = next(r for r in records if r.feature == "Bounding Box Width")
                    height_r = next(r for r in records if r.feature == "Bounding Box Height")
                    m1, m2 = st.columns(2)
                    m1.metric("Width", fmt_value(width_r.px_value, width_r.mm_value))
                    m2.metric("Height", fmt_value(height_r.px_value, height_r.mm_value))

                    with st.container(border=True):
                        st.markdown("**Calibration**")
                        if profile_for_image is not None:
                            st.markdown(
                                f'<span class="mv-status-dot pass"></span>**ACTIVE** — {esc_html(profile_for_image.name)}',
                                unsafe_allow_html=True,
                            )
                            ref_bits = f"Scale: {profile_for_image.mm_per_pixel:.5f} mm/px"
                            if profile_for_image.known_mm:
                                ref_bits += f"  ·  Reference: {profile_for_image.known_mm:.2f} mm"
                            st.caption(ref_bits)
                            with st.expander("Calibration Details"):
                                st.caption(f"Method: {profile_for_image.method}")
                                st.caption(f"Confidence: {profile_for_image.confidence}")
                                st.caption(f"Created: {profile_for_image.created_at}")
                                for n in profile_for_image.notes:
                                    st.caption(f"• {n}")
                        else:
                            st.markdown('<span class="mv-status-dot warn"></span>**NOT CALIBRATED**', unsafe_allow_html=True)
                            st.caption("Pixel values only — calibrate in Step 1 for mm.")
                        for _note in quality.issues:
                            if "perspective" in _note.message.lower() or "aspect ratio" in _note.message.lower():
                                st.caption(f"⚠️ {_note.message}")

                    st.markdown("**Detected Features**")
                    if circles:
                        # Every feature is shown individually and always-visible
                        # (never just an aggregate count) — clicking its small
                        # select button also highlights it on the image tab.
                        for c, row in zip(circles, feature_table_rows):
                            tol = tol_by_feature.get(f"{c.label} Equivalent Diameter")
                            status_key = inspection_view.feature_result_status_key(tol)
                            result_label = inspection_view.feature_result_label(tol)
                            is_selected = c.short_id == selected_short_id
                            card_classes = "mv-feature-card" + (f" {status_key}" if status_key in ("pass", "fail", "warn") else "") + (" selected" if is_selected else "")
                            edge_r = next((r for r in records if r.feature == f"{c.label} to Nearest Edge"), None)
                            edge_str = fmt_value(edge_r.px_value, edge_r.mm_value) if edge_r else "—"
                            st.markdown(
                                f"""<div class="{card_classes}">
                                    <div class="mv-feature-card-head">
                                        <span>{esc_html(row['Feature'])} — {esc_html(row['Type'])}</span>
                                        <span class="mv-badge mv-badge-{status_key}" style="padding:2px 8px;font-size:0.7rem;">{esc_html(result_label)}</span>
                                    </div>
                                    <div class="mv-feature-card-grid">
                                        <div>Ø <b>{row['Diameter']} {row['Unit']}</b></div>
                                        <div>X <b>{row['X']} {row['Unit']}</b></div>
                                        <div>Y <b>{row['Y']} {row['Unit']}</b></div>
                                        <div>Confidence <b>{esc_html(c.confidence)}</b></div>
                                        <div>Tolerance <b>{esc_html(row['Tolerance'])}</b></div>
                                        <div>Edge dist <b>{esc_html(edge_str)}</b></div>
                                    </div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                            if st.button(
                                f"{'Hide highlight' if is_selected else 'Highlight on image'}",
                                key=f"sel_{uf.name}_{c.short_id}_{idx}", use_container_width=True,
                            ):
                                sel_map[uf.name] = None if is_selected else c.short_id
                                st.rerun()
                    else:
                        st.caption("0 reliable holes detected.")

                    st.write("")
                    exp_col1, exp_col2 = st.columns(2)
                    with exp_col1:
                        if st.button("View Details", key=f"details_btn_{uf.name}_{idx}", use_container_width=True):
                            st.session_state[details_key] = not st.session_state.get(details_key, False)
                    with exp_col2:
                        report_text = export_utils.build_report_text(
                            uf.name, records, profile_for_image, quality, tolerance_results,
                        )
                        st.download_button(
                            "Export Report", data=report_text, file_name=f"{uf.name}_report.txt",
                            mime="text/plain", key=f"export_btn_{uf.name}_{idx}", use_container_width=True,
                        )

                if st.session_state.get(details_key):
                    with st.expander("Full measurement record", expanded=True):
                        table_rows = [{
                            "Feature": r.feature, "Category": r.category,
                            "Pixel": "—" if r.px_value is None else f"{r.px_value:.2f}",
                            "MM": "—" if r.mm_value is None else f"{r.mm_value:.3f}",
                            "Confidence": r.confidence, "Status": r.status,
                        } for r in records]
                        st.dataframe(table_rows, use_container_width=True, hide_index=True, height=380)

                st.markdown("#### Measurement Table")
                if circles:
                    table_html = inspection_view.build_measurement_table_html(
                        circles, records, tol_by_feature, selected_short_id=selected_short_id, decimals=decimals,
                    )
                    st.markdown(f'<div class="mv-panel">{table_html}</div>', unsafe_allow_html=True)
                else:
                    st.caption("No circular features detected for this part.")

                geometry_table_html = inspection_view.build_geometry_table_html(records, decimals=decimals)
                if geometry_table_html:
                    st.markdown("#### Geometry (reliable corner radii / angles)")
                    st.caption("Only fitted corners with a tight, verified circle fit are shown — no guessed radii.")
                    st.markdown(f'<div class="mv-panel">{geometry_table_html}</div>', unsafe_allow_html=True)

                if advanced_mode:
                    with st.expander("Full measurement table & tolerances (advanced)"):
                        if profile_for_image is not None:
                            res = measurement.resolution_uncertainty_mm(profile_for_image)
                            st.caption(
                                f"Calibration resolution: ±{res:.4f} mm/pixel step — a lower-bound "
                                "indicator only; actual accuracy also depends on lighting, focus, lens "
                                "distortion, perspective and part positioning."
                            )

                        st.markdown("##### Tolerances")
                        st.caption("Optional: enter a nominal value and ± tolerance for any feature to get a PASS/FAIL check.")
                        feature_names = [r.feature for r in records if r.mm_value is not None or r.px_value is not None]
                        key_prefix = f"tol_{uf.name}"
                        spec_dict = st.session_state["tolerance_specs"].setdefault(uf.name, {})

                        sel_features = st.multiselect("Features to check", feature_names, key=f"{key_prefix}_select")
                        for feat in sel_features:
                            tc1, tc2 = st.columns(2)
                            with tc1:
                                nominal = st.number_input(f"{feat} — nominal (mm)", value=0.0, key=f"{key_prefix}_{feat}_nom", format="%.3f")
                            with tc2:
                                tol = st.number_input(f"{feat} — tolerance ± (mm)", min_value=0.0, value=0.1, key=f"{key_prefix}_{feat}_tol", format="%.3f")
                            spec_dict[feat] = ToleranceSpec(feature=feat, nominal_mm=nominal, tolerance_mm=tol)

                        advanced_tolerance_results = measurement.apply_tolerances(records, spec_dict) if spec_dict else []
                        if advanced_tolerance_results:
                            tol_rows = [{
                                "Feature": t.feature, "Nominal (mm)": t.nominal_mm, "Tolerance (±mm)": t.tolerance_mm,
                                "Min (mm)": round(t.min_mm, 3), "Max (mm)": round(t.max_mm, 3),
                                "Measured (mm)": "—" if t.measured_mm is None else f"{t.measured_mm:.3f}",
                                "Status": t.status,
                            } for t in advanced_tolerance_results]
                            st.dataframe(tol_rows, use_container_width=True, hide_index=True)

            else:
                hmi_img_h, hmi_img_w = display_img.shape[:2]
                hmi_overlay = inspection_view.build_hmi_overlay_svg(key_results, hmi_img_w, hmi_img_h)
                hmi_html = inspection_view.build_hmi_result_html(
                    display_img, hmi_overlay, key_results, hmi_verdict_key, hmi_verdict_label, hmi_img_w, hmi_img_h,
                )
                hmi_display_h = int(420 * hmi_img_h / hmi_img_w) if hmi_img_w else 420
                hmi_component_height = max(hmi_display_h, 70 + len(key_results) * 56) + 110
                st.components.v1.html(hmi_html, height=hmi_component_height, scrolling=True)
            st.session_state["batch_results"][uf.name] = {
                "records": records,
                "profile": profile_for_image,
                "quality": quality,
                "tolerance_results": measurement.apply_tolerances(
                    records, st.session_state["tolerance_specs"].get(uf.name, {})
                ),
                "timestamp": timestamp,
            }

    st.divider()
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("← Back: Calibration"):
            go_to_step(1)
    with nav2:
        if st.button("Next: Export Results →", type="primary"):
            go_to_step(3)

# ===========================================================================
# STEP 3: Export
# ===========================================================================
else:
    st.header("Step 3: Export Results")
    results = st.session_state["batch_results"]

    if not results:
        st.info("Nothing to export yet — go back to Step 2 and measure at least one image first.")
    else:
        all_rows = []
        for image_name, data in results.items():
            all_rows.extend(export_utils.measurement_rows(image_name, data["records"]))

        csv_text = export_utils.rows_to_csv(all_rows)
        st.download_button(
            "⬇️ Download all results (CSV)", data=csv_text,
            file_name="measurement_results.csv", mime="text/csv", type="primary",
        )

        st.markdown("#### Per-image report")
        image_name = st.selectbox("Select image", list(results.keys()))
        data = results[image_name]
        report_text = export_utils.build_report_text(
            image_name, data["records"], data["profile"], data["quality"], data["tolerance_results"],
        )
        st.text_area("Report preview", report_text, height=350)
        st.download_button(
            "⬇️ Download report (.txt)", data=report_text,
            file_name=f"{image_name}_report.txt", mime="text/plain",
        )

        if advanced_mode:
            st.markdown("#### Combined results table")
            st.dataframe(all_rows, use_container_width=True, hide_index=True)

    st.divider()
    if st.button("← Back: Measure Parts"):
        go_to_step(2)

st.divider()
st.caption(
    "Machine Vision Measurement Prototype • Python + OpenCV + Streamlit — "
    "No arbitrary photograph can guarantee accurate physical dimensions. Accurate mm measurement "
    "requires a calibrated camera, a known reference object on the same plane, checkerboard "
    "calibration, or a controlled camera/lens setup. This tool does not provide CMM/industrial-"
    "metrology-grade accuracy."
)
