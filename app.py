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

import cv2
import numpy as np
import streamlit as st

from vision import annotations, calibration, geometry, measurement, preprocessing, segmentation, validation
from vision.types import CalibrationProfile, ToleranceSpec

st.set_page_config(page_title="Machine Vision Measurement", page_icon="📐", layout="wide")

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("wizard_step", 1)
st.session_state.setdefault("active_profile_id", None)
st.session_state.setdefault("pending_profile", None)
st.session_state.setdefault("pending_warp_preview", None)
st.session_state.setdefault("batch_results", {})
st.session_state.setdefault("tolerance_specs", {})


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


# ---------------------------------------------------------------------------
# Sidebar: calibration status + advanced-mode toggle
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Calibration status")
    active_profile = get_active_profile()

    if active_profile is not None:
        st.success(f"CALIBRATED — {active_profile.name}")
        st.caption(f"{active_profile.mm_per_pixel:.5f} mm/pixel · confidence {active_profile.confidence}")
    else:
        st.error("NOT CALIBRATED")
        st.caption("Only pixel measurements will be produced until a calibration profile is active.")

    st.divider()
    advanced_mode = st.checkbox(
        "Show advanced options", value=False,
        help="Checkerboard/perspective calibration, the full measurement table, tolerances, "
             "and coordinate-origin choice.",
    )

    st.divider()
    if st.button("🔄 Start over", use_container_width=True):
        st.session_state["batch_results"] = {}
        st.session_state["tolerance_specs"] = {}
        go_to_step(1)

    st.divider()
    st.caption(
        "A normal photograph has no inherent physical scale. Real-world mm measurements "
        "require an explicit calibration. This app never assumes 1 pixel = 1 mm."
    )

active_profile = get_active_profile()

st.title("📐 Machine Vision Measurement")
st.caption("Calibrated part measurement — pixel and millimetre values are always shown separately.")

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
    else:
        origin_choice = "Bounding box top-left"

    if not uploaded_files:
        st.info("Upload at least one image to begin measurement.")
    else:
        for uf in uploaded_files:
            st.divider()
            st.markdown(f"### {uf.name}")

            try:
                original = preprocessing.decode_image_bytes(uf.getvalue())
            except ValueError as e:
                st.error(str(e))
                continue

            img, _ = preprocessing.resize_for_processing(original)

            profile_for_image = active_profile
            display_img = img
            if active_profile is not None and active_profile.method == "perspective":
                st.caption("Perspective calibration active — measuring on the corrected (warped) image.")
                try:
                    display_img = calibration.warp_with_profile(img, active_profile)
                except ValueError as e:
                    st.error(f"Could not apply perspective calibration: {e}")
                    display_img = img
                    profile_for_image = None

            part = segmentation.build_detected_part(display_img)
            quality = validation.build_quality_report(display_img, part, profile_for_image)

            blocking_issues = [i for i in quality.issues if i.severity in ("warning", "error")]
            if blocking_issues:
                with st.expander("⚠️ Image quality warnings", expanded=True):
                    for issue in quality.issues:
                        if issue.severity == "error":
                            st.error(issue.message)
                        elif issue.severity == "warning":
                            st.warning(issue.message)
                        elif advanced_mode:
                            st.info(issue.message)
            elif advanced_mode:
                st.caption("✅ No image quality issues detected.")

            if part is None:
                st.error("Could not confidently detect a part in this image. Try a clearer image with a plain, contrasting background.")
                continue

            gray = preprocessing.to_gray(display_img)
            hough = geometry.detect_circles_hough(gray, part.mask)
            contour_circles = geometry.detect_circles_contour(part.hole_contours)
            circles = geometry.merge_circle_detections(hough, contour_circles)

            x, y, w, h = part.bbox
            if origin_choice == "Part centroid":
                m = cv2.moments(part.contour)
                origin_px = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (x + w / 2, y + h / 2)
            else:
                origin_px = (float(x), float(y))

            records = measurement.build_measurement_records(part, circles, profile_for_image, origin_px)
            annotated = annotations.draw_full_annotation(display_img, part, circles, profile_for_image, origin_px)

            col_img, col_summary = st.columns([3, 2])
            with col_img:
                st.image(preprocessing.to_rgb(annotated), caption="Annotated measurement", use_container_width=True)
                if advanced_mode:
                    st.caption(f"Segmentation method: {part.segmentation_method}")
            with col_summary:
                st.markdown("#### Results")
                width_r = next(r for r in records if r.feature == "Bounding Box Width")
                height_r = next(r for r in records if r.feature == "Bounding Box Height")
                m1, m2 = st.columns(2)
                m1.metric("Width", fmt_value(width_r.px_value, width_r.mm_value))
                m2.metric("Height", fmt_value(height_r.px_value, height_r.mm_value))

                if circles:
                    st.write(f"**{len(circles)} hole(s) detected:**")
                    for c in circles:
                        d_record = next(r for r in records if r.feature == f"Hole {c.circle_id} Diameter")
                        st.write(f"- Hole {c.circle_id}: Ø {fmt_value(d_record.px_value, d_record.mm_value)} ({c.confidence} confidence)")
                else:
                    st.caption("No holes detected.")

                if profile_for_image is None:
                    st.caption("⚠️ Showing pixels only — calibrate in Step 1 to see mm.")

            if advanced_mode:
                with st.expander("Full measurement table & tolerances"):
                    table_rows = [{
                        "Feature": r.feature, "Category": r.category,
                        "Pixel": "—" if r.px_value is None else f"{r.px_value:.2f}",
                        "MM": "—" if r.mm_value is None else f"{r.mm_value:.3f}",
                        "Confidence": r.confidence, "Status": r.status,
                    } for r in records]
                    st.dataframe(table_rows, use_container_width=True, hide_index=True, height=380)

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

                    tolerance_results = measurement.apply_tolerances(records, spec_dict) if spec_dict else []
                    if tolerance_results:
                        tol_rows = [{
                            "Feature": t.feature, "Nominal (mm)": t.nominal_mm, "Tolerance (±mm)": t.tolerance_mm,
                            "Min (mm)": round(t.min_mm, 3), "Max (mm)": round(t.max_mm, 3),
                            "Measured (mm)": "—" if t.measured_mm is None else f"{t.measured_mm:.3f}",
                            "Status": t.status,
                        } for t in tolerance_results]
                        st.dataframe(tol_rows, use_container_width=True, hide_index=True)

            st.session_state["batch_results"][uf.name] = {
                "records": records,
                "profile": profile_for_image,
                "quality": quality,
                "tolerance_results": measurement.apply_tolerances(
                    records, st.session_state["tolerance_specs"].get(uf.name, {})
                ),
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
        from utils import export

        all_rows = []
        for image_name, data in results.items():
            all_rows.extend(export.measurement_rows(image_name, data["records"]))

        csv_text = export.rows_to_csv(all_rows)
        st.download_button(
            "⬇️ Download all results (CSV)", data=csv_text,
            file_name="measurement_results.csv", mime="text/csv", type="primary",
        )

        st.markdown("#### Per-image report")
        image_name = st.selectbox("Select image", list(results.keys()))
        data = results[image_name]
        report_text = export.build_report_text(
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
