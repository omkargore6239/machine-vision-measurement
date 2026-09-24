
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Machine Vision Measurement", page_icon="📐", layout="wide")

st.title("📐 Machine Vision Measurement")
st.caption("Upload a part image → detect the main part → measure visible 2D geometry.")

# ---------- Helpers ----------
def read_uploaded_image(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def resize_for_processing(img, max_side=1800):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return img, 1.0
    out = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return out, scale

def make_foreground_mask(img, method="auto"):
    """
    Tries several segmentation methods and selects a plausible large object contour.
    This is intended for controlled/clean images. Arbitrary backgrounds may require
    manual masking or a better industrial lighting setup.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []

    # Otsu dark and light foregrounds
    for mode in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
        _, m = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
        candidates.append(m)

    # Adaptive threshold
    m = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 7
    )
    candidates.append(m)

    # Edge based mask
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    candidates.append(edges)

    H, W = gray.shape
    image_area = H * W

    best = None
    best_score = -1

    for mask in candidates:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < image_area * 0.01 or area > image_area * 0.95:
                continue

            x, y, w, h = cv2.boundingRect(c)
            rect_area = w * h
            if rect_area <= 0:
                continue

            extent = area / rect_area
            center_x = x + w / 2
            center_y = y + h / 2
            center_penalty = abs(center_x - W/2)/(W/2) + abs(center_y - H/2)/(H/2)

            # Prefer reasonably large, compact objects, not the whole frame.
            score = (area / image_area) * 4 + extent * 1.5 - center_penalty * 0.25

            if score > best_score:
                best_score = score
                best = c

    if best is None:
        return None

    final_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(final_mask, [best], -1, 255, thickness=cv2.FILLED)

    # Smooth the mask
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return final_mask

def contour_measurements(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)

    rect = cv2.minAreaRect(c)
    rw, rh = rect[1]
    if rw and rh:
        rotated_w = max(rw, rh)
        rotated_h = min(rw, rh)
    else:
        rotated_w = rotated_h = 0

    return {
        "contour": c,
        "area_px2": area,
        "perimeter_px": perimeter,
        "x": x, "y": y, "w": w, "h": h,
        "rotated_w": rotated_w,
        "rotated_h": rotated_h
    }

def detect_circles(img, mask=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min(gray.shape[:2]) // 12),
        param1=100,
        param2=30,
        minRadius=max(5, min(gray.shape[:2]) // 100),
        maxRadius=max(10, min(gray.shape[:2]) // 3)
    )

    results = []
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        H, W = gray.shape
        for x, y, r in circles:
            if not (0 <= x < W and 0 <= y < H):
                continue

            # Keep circles whose center is inside the detected object when a mask exists.
            if mask is not None and mask[y, x] == 0:
                continue

            # Reject circles very close to image boundary.
            if x-r < 0 or y-r < 0 or x+r >= W or y+r >= H:
                continue

            results.append({"x": int(x), "y": int(y), "r": int(r),
                            "diameter": int(2*r)})

    # Remove near-duplicates
    unique = []
    for item in results:
        if all((item["x"]-u["x"])**2 + (item["y"]-u["y"])**2 > 20**2 for u in unique):
            unique.append(item)

    return unique[:20]

def annotate(img, measurements, circles, px_per_mm=None):
    out = img.copy()
    c = measurements["contour"]
    cv2.drawContours(out, [c], -1, (0, 255, 0), 3)

    x, y, w, h = measurements["x"], measurements["y"], measurements["w"], measurements["h"]
    cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), 2)

    label = f"W: {w}px | H: {h}px"
    cv2.putText(out, label, (max(10, x), max(30, y-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    for i, circ in enumerate(circles, start=1):
        cx, cy, r = circ["x"], circ["y"], circ["r"]
        cv2.circle(out, (cx, cy), r, (0, 165, 255), 3)
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
        text = f"C{i}: D={2*r}px"
        cv2.putText(out, text, (max(5, cx-r), max(20, cy-r-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

    return out

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------- UI ----------
uploaded = st.file_uploader(
    "Upload a part image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="For reliable industrial measurement, use a fixed camera, fixed lighting and a plain contrasting background."
)

with st.sidebar:
    st.header("Calibration")
    use_cal = st.checkbox("Convert pixels to mm", value=False)
    known_length = st.number_input("Known reference length (mm)", min_value=0.001, value=10.0, step=1.0)
    known_pixels = st.number_input("Reference length in image (pixels)", min_value=0.001, value=100.0, step=1.0)
    if use_cal:
        px_per_mm = known_pixels / known_length
        st.info(f"Scale: {px_per_mm:.4f} px/mm")
    else:
        px_per_mm = None
        st.info("Without calibration, measurements are shown in pixels.")

if uploaded:
    original = read_uploaded_image(uploaded)
    processed, scale = resize_for_processing(original)

    st.subheader("Input image")
    st.image(to_rgb(processed), use_container_width=True)

    mask = make_foreground_mask(processed)
    if mask is None:
        st.error("Could not confidently detect the main part. Try a clearer image with a plain contrasting background.")
        st.stop()

    measurements = contour_measurements(mask)
    if measurements is None:
        st.error("No measurable object was detected.")
        st.stop()

    circles = detect_circles(processed, mask)
    annotated = annotate(processed, measurements, circles, px_per_mm)

    # The displayed pixel measurements correspond to the processed image.
    # Convert back to original-image pixel scale.
    proc_to_original = 1 / scale
    w_px = measurements["w"] * proc_to_original
    h_px = measurements["h"] * proc_to_original
    area_px2 = measurements["area_px2"] * proc_to_original**2
    perimeter_px = measurements["perimeter_px"] * proc_to_original

    st.subheader("Detected result")
    col1, col2 = st.columns(2)
    with col1:
        st.image(to_rgb(annotated), caption="Detected measurements", use_container_width=True)
    with col2:
        st.markdown("### Overall dimensions")
        st.metric("Width", f"{w_px:.2f} px")
        st.metric("Height", f"{h_px:.2f} px")
        st.metric("Area", f"{area_px2:.2f} px²")
        st.metric("Perimeter", f"{perimeter_px:.2f} px")

        if px_per_mm:
            st.markdown("### Calibrated dimensions")
            st.metric("Width", f"{w_px / px_per_mm:.2f} mm")
            st.metric("Height", f"{h_px / px_per_mm:.2f} mm")
            st.metric("Area", f"{area_px2 / (px_per_mm**2):.2f} mm²")
            st.metric("Perimeter", f"{perimeter_px / px_per_mm:.2f} mm")

    if circles:
        st.subheader("Detected circular features")
        rows = []
        for i, c in enumerate(circles, start=1):
            d_px = c["diameter"] * proc_to_original
            row = {
                "Feature": f"Circle {i}",
                "Center X (px)": round(c["x"] * proc_to_original, 2),
                "Center Y (px)": round(c["y"] * proc_to_original, 2),
                "Diameter (px)": round(d_px, 2),
            }
            if px_per_mm:
                row["Diameter (mm)"] = round(d_px / px_per_mm, 2)
            rows.append(row)
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No circular features were confidently detected in this image.")

    st.warning(
        "This is a learning/prototype tool. For production industrial measurement, "
        "use controlled lighting, fixed camera geometry and a proper calibration target. "
        "Do not use arbitrary phone photos as a metrology system without validation."
    )
else:
    st.info("Upload a JPG/PNG image above to start measuring.")
    st.markdown("""
    ### What this first version measures
    - Overall visible width
    - Overall visible height
    - Object area and perimeter
    - Detectable circular features / holes
    - Optional pixel → mm conversion using a known reference

    ### Not included yet
    - PLC
    - Conveyor control
    - Reject mechanism
    - PASS/FAIL tolerances
    - 3D/thickness measurement
    """)

st.divider()
st.caption("Machine Vision Measurement Prototype • Python + OpenCV + Streamlit")
