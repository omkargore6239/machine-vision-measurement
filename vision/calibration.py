"""Calibration: turning pixel measurements into millimetres.

Three methods:
  - known_dimension_calibrate: detect a reference object of known size, derive
    an isotropic mm/px scale. Simple, but assumes a fronto-parallel camera and
    a shared reference/part plane.
  - checkerboard_calibrate: OpenCV chessboard corner detection gives a
    scale from known square size; with >=3 images, also runs full camera
    calibration (camera matrix + distortion coefficients).
  - perspective_calibrate: a 4-point homography warps the image to a
    canonical fronto-parallel view at an exact chosen mm/px scale, correcting
    for perspective skew — but only when the user explicitly supplies the
    four reference points, never automatically.

Every profile is written to disk as JSON so it can be reloaded later. NOTE:
on ephemeral hosting (e.g. Streamlit Community Cloud) this directory does not
persist across redeploys/restarts — see README.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from vision import geometry, segmentation
from vision.types import CalibrationProfile, CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM

CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "calibration_profiles"

# known-dimension confidence thresholds
KNOWN_DIM_HIGH_EXTENT = 0.85
KNOWN_DIM_MEDIUM_EXTENT = 0.65
KNOWN_DIM_MIN_AREA_FRACTION = 0.02

# checkerboard confidence thresholds
CHECKERBOARD_HIGH_MIN_CORNERS = 40
CHECKERBOARD_MEDIUM_MIN_CORNERS = 15
CHECKERBOARD_HIGH_MAX_REPROJ_ERROR = 0.5


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------
# Method A: known reference dimension
# --------------------------------------------------------------------------------

def known_dimension_calibrate(
    img: np.ndarray, known_mm: float, direction: str, name: str = "",
) -> CalibrationProfile:
    """`direction` is "horizontal" or "vertical". Detects the reference
    object's outline and uses its bounding-box width/height in pixels."""
    if known_mm <= 0:
        raise ValueError("Known reference dimension must be greater than zero.")

    part = segmentation.build_detected_part(img)
    if part is None:
        raise ValueError(
            "Could not detect a reference object in the calibration image. "
            "Use a clear, plain-contrast background."
        )

    x, y, w, h = part.bbox
    measured_px = float(w if direction == "horizontal" else h)
    if measured_px <= 0:
        raise ValueError("Detected reference object has zero size — cannot calibrate.")

    H, W = img.shape[:2]
    rect_area = w * h
    extent = (part.area_px2 / rect_area) if rect_area > 0 else 0.0
    area_fraction = rect_area / (H * W)

    notes = [
        f"Reference object detected via '{part.segmentation_method}' segmentation.",
        f"Bounding box: {w}x{h}px; using {direction} = {measured_px:.1f}px for {known_mm:.3f}mm.",
        "Assumes isotropic scale (square pixels) and a fronto-parallel camera — "
        "not corrected for perspective distortion.",
    ]

    if part.touches_border:
        confidence = CONFIDENCE_LOW
        notes.append("LOW confidence: reference outline touches the image border.")
    elif extent >= KNOWN_DIM_HIGH_EXTENT and area_fraction >= KNOWN_DIM_MIN_AREA_FRACTION:
        confidence = CONFIDENCE_HIGH
        notes.append(f"HIGH confidence: rectangularity (extent) {extent:.2f}, covers {area_fraction*100:.1f}% of frame.")
    elif extent >= KNOWN_DIM_MEDIUM_EXTENT:
        confidence = CONFIDENCE_MEDIUM
        notes.append(f"MEDIUM confidence: rectangularity (extent) {extent:.2f}.")
    else:
        confidence = CONFIDENCE_LOW
        notes.append(f"LOW confidence: rectangularity (extent) {extent:.2f} is poor — detection may be unreliable.")

    mm_per_pixel = known_mm / measured_px

    return CalibrationProfile(
        profile_id=_new_id(),
        name=name or f"Known-dimension {known_mm:g}mm",
        method="known_dimension",
        created_at=_now_iso(),
        confidence=confidence,
        notes=notes,
        reference_image_size=(H, W),
        mm_per_pixel=mm_per_pixel,
        known_mm=known_mm,
        detected_pixels=measured_px,
    )


def known_feature_calibrate(
    feature_diameter_px: float, known_mm: float, image_size: tuple[int, int], name: str = "",
) -> CalibrationProfile:
    """Scale from a round feature of known diameter that is ON the part (e.g.
    its drilled hole), measured at the app's processing resolution. Same
    plane as the part by construction, so no reference/part height mismatch.
    `image_size` is the (H, W) of the processed image the pixels came from."""
    if known_mm <= 0:
        raise ValueError("Known feature diameter must be greater than zero.")
    if feature_diameter_px <= 0:
        raise ValueError("Detected feature has zero size - cannot calibrate.")
    return CalibrationProfile(
        profile_id=_new_id(),
        name=name or f"Known hole {known_mm:g}mm",
        method="known_dimension",
        created_at=_now_iso(),
        confidence=CONFIDENCE_MEDIUM,
        notes=[
            f"Scale from a detected round feature: {feature_diameter_px:.1f}px = {known_mm:.3f}mm.",
            "Assumes isotropic scale, a fronto-parallel camera, and the same camera/lens/height for later photos.",
        ],
        reference_image_size=image_size,
        mm_per_pixel=known_mm / feature_diameter_px,
        known_mm=known_mm,
        detected_pixels=float(feature_diameter_px),
    )


# --------------------------------------------------------------------------------
# Method B: checkerboard
# --------------------------------------------------------------------------------

def checkerboard_calibrate(
    images: list[np.ndarray], pattern_cols: int, pattern_rows: int,
    square_size_mm: float, name: str = "",
) -> CalibrationProfile:
    """`pattern_cols`/`pattern_rows` = number of INNER corners (i.e. one less
    than the number of squares) along each axis."""
    if square_size_mm <= 0:
        raise ValueError("Checkerboard square size must be greater than zero.")
    if not images:
        raise ValueError("At least one checkerboard image is required.")

    pattern_size = (pattern_cols, pattern_rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2) * square_size_mm

    obj_points, img_points = [], []
    corners_found_counts = []
    last_gray_shape = None

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        last_gray_shape = gray.shape
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        corners_found_counts.append(len(corners))

    if not img_points:
        raise ValueError(
            f"No checkerboard with {pattern_cols}x{pattern_rows} inner corners was found in "
            "the supplied image(s). Check the pattern size and image quality."
        )

    # Scale from mean spacing between adjacent detected corners in one image
    # (always available, even with a single image).
    corners = img_points[0].reshape(pattern_rows, pattern_cols, 2)
    h_dists = np.linalg.norm(np.diff(corners, axis=1), axis=2)
    v_dists = np.linalg.norm(np.diff(corners, axis=0), axis=2)
    mean_spacing_px = float(np.mean(np.concatenate([h_dists.ravel(), v_dists.ravel()])))
    mm_per_pixel = square_size_mm / mean_spacing_px

    camera_matrix = None
    dist_coeffs = None
    reprojection_error = None

    if len(img_points) >= 3:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, last_gray_shape[::-1], None, None
        )
        reprojection_error = float(ret)
        camera_matrix = mtx.tolist()
        dist_coeffs = dist.ravel().tolist()

    total_corners = pattern_cols * pattern_rows
    best_found = max(corners_found_counts)
    notes = [
        f"{len(img_points)}/{len(images)} supplied image(s) had a detected "
        f"{pattern_cols}x{pattern_rows} checkerboard.",
        f"Scale derived from mean inter-corner spacing ({mean_spacing_px:.2f}px per {square_size_mm:g}mm square).",
    ]
    if camera_matrix is not None:
        notes.append(f"Camera matrix + distortion coefficients estimated from {len(img_points)} images "
                      f"(reprojection error {reprojection_error:.3f}px).")
    else:
        notes.append("Only scale calibration performed — supply 3+ checkerboard images from different "
                      "poses to also estimate lens distortion.")

    if reprojection_error is not None and reprojection_error <= CHECKERBOARD_HIGH_MAX_REPROJ_ERROR \
            and best_found >= CHECKERBOARD_HIGH_MIN_CORNERS:
        confidence = CONFIDENCE_HIGH
    elif best_found >= CHECKERBOARD_MEDIUM_MIN_CORNERS:
        confidence = CONFIDENCE_MEDIUM
    else:
        confidence = CONFIDENCE_LOW

    H, W = last_gray_shape
    return CalibrationProfile(
        profile_id=_new_id(),
        name=name or f"Checkerboard {square_size_mm:g}mm squares",
        method="checkerboard",
        created_at=_now_iso(),
        confidence=confidence,
        notes=notes,
        reference_image_size=(H, W),
        mm_per_pixel=mm_per_pixel,
        known_mm=square_size_mm,
        detected_pixels=mean_spacing_px,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        reprojection_error_px=reprojection_error,
        checkerboard_images_used=len(img_points),
    )


def undistort_image(img: np.ndarray, profile: CalibrationProfile) -> np.ndarray:
    if not profile.camera_matrix or not profile.dist_coeffs:
        raise ValueError("This calibration profile has no camera matrix/distortion coefficients.")
    mtx = np.array(profile.camera_matrix)
    dist = np.array(profile.dist_coeffs)
    return cv2.undistort(img, mtx, dist)


# --------------------------------------------------------------------------------
# Method C: 4-point perspective correction
# --------------------------------------------------------------------------------

def perspective_calibrate(
    img: np.ndarray, src_points: list[tuple[float, float]],
    known_width_mm: float, known_height_mm: float,
    px_per_mm_target: float = 10.0, name: str = "",
) -> tuple[CalibrationProfile, np.ndarray]:
    """`src_points` are 4 image-pixel (x, y) corners of a known-size rectangle,
    given in order: top-left, top-right, bottom-right, bottom-left. Returns
    the profile AND the warped canonical (fronto-parallel) image, since
    downstream detection should run on the warped image for this method."""
    if known_width_mm <= 0 or known_height_mm <= 0:
        raise ValueError("Known width/height must be greater than zero.")
    if len(src_points) != 4:
        raise ValueError("Exactly 4 points are required for perspective calibration.")

    out_w = max(2, int(round(known_width_mm * px_per_mm_target)))
    out_h = max(2, int(round(known_height_mm * px_per_mm_target)))

    src = np.array(src_points, dtype=np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

    H_matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H_matrix, (out_w, out_h))

    mm_per_pixel = 1.0 / px_per_mm_target

    # Confidence: penalize degenerate/near-collinear point selection.
    side_lengths = [
        geometry.euclidean_distance(tuple(src[i]), tuple(src[(i + 1) % 4])) for i in range(4)
    ]
    min_side = min(side_lengths)
    confidence = CONFIDENCE_HIGH if min_side >= 40 else (CONFIDENCE_MEDIUM if min_side >= 15 else CONFIDENCE_LOW)

    notes = [
        f"4-point homography warps the source quadrilateral to a {out_w}x{out_h}px canonical view "
        f"representing {known_width_mm:g}x{known_height_mm:g}mm ({px_per_mm_target:g}px/mm).",
        "Run part detection on the WARPED image, not the original, when using this calibration.",
        "Only corrects perspective skew of the plane containing the 4 selected points — features "
        "not on that plane will still be distorted.",
    ]

    Him, Wim = img.shape[:2]
    profile = CalibrationProfile(
        profile_id=_new_id(),
        name=name or f"Perspective {known_width_mm:g}x{known_height_mm:g}mm",
        method="perspective",
        created_at=_now_iso(),
        confidence=confidence,
        notes=notes,
        reference_image_size=(Him, Wim),
        mm_per_pixel=mm_per_pixel,
        known_width_mm=known_width_mm,
        known_height_mm=known_height_mm,
        homography=H_matrix.tolist(),
        warped_size=(out_w, out_h),
    )
    return profile, warped


def warp_with_profile(img: np.ndarray, profile: CalibrationProfile) -> np.ndarray:
    if profile.method != "perspective" or not profile.homography:
        raise ValueError("This profile does not contain a perspective homography.")
    H_matrix = np.array(profile.homography)
    w, h = profile.warped_size
    return cv2.warpPerspective(img, H_matrix, (w, h))


# --------------------------------------------------------------------------------
# Conversion + persistence
# --------------------------------------------------------------------------------

def apply_calibration(px_value: float, profile: CalibrationProfile | None, power: int = 1) -> float | None:
    """Convert a pixel length (power=1) or area (power=2) to mm / mm^2. Returns
    None if no profile is supplied — callers must never invent a scale."""
    if profile is None or profile.mm_per_pixel is None:
        return None
    return px_value * (profile.mm_per_pixel ** power)


def save_profile(profile: CalibrationProfile) -> Path:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    path = CALIBRATION_DIR / f"{profile.profile_id}.json"
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return path


def load_profile(profile_id: str) -> CalibrationProfile:
    path = CALIBRATION_DIR / f"{profile_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationProfile.from_dict(data)


def list_profiles() -> list[CalibrationProfile]:
    if not CALIBRATION_DIR.exists():
        return []
    profiles = []
    for path in sorted(CALIBRATION_DIR.glob("*.json")):
        try:
            profiles.append(CalibrationProfile.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return sorted(profiles, key=lambda p: p.created_at, reverse=True)


def delete_profile(profile_id: str) -> None:
    path = CALIBRATION_DIR / f"{profile_id}.json"
    if path.exists():
        path.unlink()
