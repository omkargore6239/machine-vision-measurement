"""Shared data structures for the measurement pipeline.

These are plain dataclasses so calibration profiles can be serialized to JSON
and so every other module has one shared vocabulary for "a part", "a circle",
"a measurement" etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

# Confidence labels used everywhere. Never assign HIGH without measured evidence.
CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"

MM_UNAVAILABLE_STATUS = "MM measurement unavailable — calibration required"

# In real industrial inspection, what gets measured is driven by an
# engineering drawing/spec, not "detect everything the algorithm can find".
# These names are the vocabulary a future characteristic-driven inspection
# config would use (e.g. "require exactly 4 HOLE_DIAMETER features within
# tolerance X, plus 1 INNER_DIAMETER"). Nothing currently enforces or
# consumes these beyond naming — this is deliberately just the shared
# vocabulary/scaffolding asked for, not a full spec-driven inspection engine
# (that's a materially larger feature, out of scope for this change).
CHARACTERISTIC_OVERALL_WIDTH = "overall_width"
CHARACTERISTIC_OVERALL_HEIGHT = "overall_height"
CHARACTERISTIC_OUTER_DIAMETER = "outer_diameter"
CHARACTERISTIC_INNER_DIAMETER = "inner_diameter"
CHARACTERISTIC_HOLE_DIAMETER = "hole_diameter"
CHARACTERISTIC_HOLE_COUNT = "hole_count"
CHARACTERISTIC_HOLE_PITCH = "hole_pitch"
CHARACTERISTIC_HOLE_POSITION = "hole_position"
CHARACTERISTIC_EDGE_DISTANCE = "edge_distance"
CHARACTERISTIC_RADIUS = "radius"
CHARACTERISTIC_ANGLE = "angle"


@dataclass
class InspectionCharacteristic:
    """One required-to-inspect characteristic, as it would come from an
    engineering drawing/spec — the intended hook point for a future
    "configure what to check" UI. `feature_ref` optionally ties it to a
    specific feature's short_id (e.g. "H1", "CB"); left None for whole-part
    characteristics like overall width. Not yet wired into the pipeline."""

    characteristic_type: str
    feature_ref: Optional[str] = None
    nominal_mm: Optional[float] = None
    tolerance_mm: Optional[float] = None


@dataclass
class CalibrationProfile:
    """A saved calibration. `mm_per_pixel` is an isotropic scale (same in X/Y),
    valid for the `known_dimension` and `checkerboard` methods. The
    `perspective` method instead warps images to a canonical view where
    `mm_per_pixel` also applies, via the stored homography.
    """

    profile_id: str
    name: str
    method: str  # "known_dimension" | "checkerboard" | "perspective"
    created_at: str
    confidence: str
    notes: list[str] = field(default_factory=list)
    reference_image_size: Optional[tuple[int, int]] = None  # (h, w)

    mm_per_pixel: Optional[float] = None
    known_mm: Optional[float] = None
    detected_pixels: Optional[float] = None

    # checkerboard camera-calibration extras (optional)
    camera_matrix: Optional[list[list[float]]] = None
    dist_coeffs: Optional[list[float]] = None
    reprojection_error_px: Optional[float] = None
    checkerboard_images_used: Optional[int] = None

    # perspective extras (optional)
    homography: Optional[list[list[float]]] = None  # 3x3, image px -> warped px
    warped_size: Optional[tuple[int, int]] = None  # (w, h)
    known_width_mm: Optional[float] = None
    known_height_mm: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CalibrationProfile":
        return CalibrationProfile(**data)


@dataclass
class DetectedPart:
    contour: np.ndarray
    mask: np.ndarray
    segmentation_method: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    rotated_rect: tuple[float, float, float, float, float]  # cx, cy, w, h, angle
    area_px2: float
    perimeter_px: float
    hole_contours: list[np.ndarray] = field(default_factory=list)
    touches_border: bool = False


@dataclass
class SegmentationDiagnostics:
    """The full evidence behind the outer-boundary accept/reject decision —
    Vision Debug Mode's boundary panel shows all of this. `extent` (fill
    ratio vs. the rotated bounding rectangle) is deliberately NOT part of the
    accept/reject logic: it's informational only, since a legitimately
    concave/open part (a C-bracket, a fork, an L-shape) has a lot of
    intentional empty space in its own bounding rectangle. Reliability is
    instead judged from solidity (vs. the part's own convex hull, which
    only "fills in" genuine concavities) and boundary compactness (a proxy
    for how jagged/noisy the contour is), neither of which penalize shape."""

    area_px2: float
    perimeter_px: float
    bbox_area_px2: float
    rotated_rect_area_px2: float
    extent: float                  # informational only — see class docstring
    convex_hull_area_px2: float
    solidity: float                # area / convex-hull area
    compactness: float             # perimeter / (2*sqrt(pi*area)); 1.0 = perfect circle
    area_fraction_of_image: float
    touches_border: bool
    accepted: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class CircleFeature:
    """A hole/circular feature that has already passed every acceptance gate
    in `vision.geometry` — this is the reported, trustworthy result. See
    `HoleCandidate` for the full diagnostic record (including rejected
    candidates) used by Vision Debug Mode."""

    circle_id: int
    cx: float
    cy: float
    r: float  # equivalent radius, from measured area (2*sqrt(area/pi)/2)
    method: str  # "contour" | "hough+contour"
    circularity: Optional[float]
    confidence: str
    major_px: Optional[float] = None  # ellipse major axis (full length), if fit
    minor_px: Optional[float] = None  # ellipse minor axis (full length), if fit
    ellipse_angle_deg: Optional[float] = None
    solidity: Optional[float] = None
    confidence_score: Optional[float] = None  # 0-1 continuous score behind `confidence`
    hough_confirmed: bool = False

    # Set by `measurement.classify_and_label_circles` (not by detection
    # itself — geometry has no notion of "which hole is the bore", only
    # measurement/reporting does). Empty/"hole" until then.
    feature_type: str = "hole"  # "hole" | "bore"
    label: str = ""             # e.g. "Hole 1", "Center Bore"
    short_id: str = ""          # e.g. "H1", "CB"


@dataclass
class HoleCandidate:
    """A candidate hole region with the full diagnostic evidence used to
    accept or reject it. Every candidate is kept (not just accepted ones) so
    Vision Debug Mode can show what was rejected and why."""

    contour: np.ndarray
    cx: float
    cy: float
    area_px2: float
    equiv_diameter_px: float
    circularity: float
    solidity: float
    aspect_ratio: float  # major/minor from an ellipse fit; 1.0 if no fit
    major_axis_px: Optional[float]
    minor_axis_px: Optional[float]
    ellipse_angle_deg: Optional[float]
    contrast: float
    interior_std: float
    edge_strength_px: float
    hough_confirmed: bool = False
    confidence_score: float = 0.0
    confidence_label: str = CONFIDENCE_LOW
    accepted: bool = False
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class MeasurementRecord:
    feature: str
    category: str
    px_value: Optional[float]
    mm_value: Optional[float]
    unit_px: str
    unit_mm: str
    confidence: str
    status: str


@dataclass
class QualityIssue:
    message: str
    severity: str  # "info" | "warning" | "error"


@dataclass
class QualityReport:
    issues: list[QualityIssue]
    metrics: dict[str, Any]

    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)


@dataclass
class ToleranceSpec:
    feature: str
    nominal_mm: float
    tolerance_mm: float


@dataclass
class ToleranceResult:
    feature: str
    nominal_mm: float
    tolerance_mm: float
    min_mm: float
    max_mm: float
    measured_mm: Optional[float]
    status: str  # "PASS" | "FAIL" | "N/A"
