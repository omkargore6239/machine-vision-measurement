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
