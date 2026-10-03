"""Builds the SVG engineering-dimension overlay and the zoomable/hoverable
HTML inspection viewport for the light industrial result screen.

Every number drawn or written here is read from an already-computed
`MeasurementRecord` / `CircleFeature` / `ToleranceResult` (or, for corner
radii, `vision.geometry.estimate_corner_radii` — also already computed,
already gated to "reliable" fits only) — this module computes no
measurement, calibration, or tolerance value of its own; it only formats and
positions data produced by `vision/`.
"""
from __future__ import annotations

import base64
import html as html_lib
import math
from typing import Optional

import cv2
import numpy as np

from ui.theme import (
    COLOR_BORDER, COLOR_FAIL, COLOR_INFO, COLOR_PASS, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_WARN,
    HMI_BG, HMI_BORDER, HMI_FAIL, HMI_PASS, HMI_SURFACE, HMI_TEXT, HMI_TEXT_MUTED,
)
from vision import geometry
from vision.types import (
    CalibrationProfile, CircleFeature, DetectedPart, MeasurementRecord, ParameterResult, ToleranceResult,
)

STAGE_HEIGHT_PX = 640
TOOLBAR_HEIGHT_PX = 64
COMPONENT_HEIGHT_PX = STAGE_HEIGHT_PX + TOOLBAR_HEIGHT_PX + 16

# Overlay semantics: GREEN is the default "successfully measured/detected"
# color (whether or not a tolerance is configured); RED only for an actual
# FAIL; AMBER only for a configured-but-unmeasurable ("N/A") spec; INFO-blue
# is reserved for neutral reference information (e.g. an inter-feature
# distance line), never a measurement result.
_STATUS_COLORS = {"pass": COLOR_PASS, "fail": COLOR_FAIL, "warn": COLOR_WARN, "neutral": COLOR_TEXT_MUTED, "info": COLOR_INFO}
_CONFIDENCE_LETTER = {"HIGH": "H", "MEDIUM": "M", "LOW": "L"}
_TOL_GLYPH = {"PASS": "✓", "FAIL": "✗", "N/A": "!"}

# The 7 independently-toggleable layers (matches the in-viewport toolbar's
# checkboxes 1:1) — used nowhere in the logic below except as documentation;
# each SVG element declares its own `data-layer` attribute directly.
LAYER_NAMES = ["contour", "dimensions", "features", "centers", "tolerance", "confidence", "values"]

_CANDIDATE_DIRECTIONS = [
    (0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0),
    (0.7071, -0.7071), (-0.7071, -0.7071), (0.7071, 0.7071), (-0.7071, 0.7071),
]
_CANDIDATE_NAMES = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]


def _overlay_status_key(tol: Optional[ToleranceResult]) -> str:
    """On-image color: red only on a genuine FAIL, amber only when a
    configured spec couldn't be measured, green otherwise (measured, with
    or without a configured tolerance — green means "real, detected")."""
    if tol is None:
        return "pass"
    if tol.status == "FAIL":
        return "fail"
    if tol.status == "N/A":
        return "warn"
    return "pass"


def feature_result_status_key(tol: Optional[ToleranceResult]) -> str:
    """Precise 4-way status key ("pass"/"fail"/"warn"/"neutral") for anything
    that must distinguish "no tolerance configured" (neutral) from "PASS" —
    used by the table's status dots and the always-visible feature cards, so
    an unchecked measurement is never visually mistaken for a passed one."""
    if tol is None:
        return "neutral"
    if tol.status == "FAIL":
        return "fail"
    if tol.status == "N/A":
        return "warn"
    if tol.status == "PASS":
        return "pass"
    return "neutral"


def feature_result_label(tol: Optional[ToleranceResult]) -> str:
    """The precise per-feature Result text: "MEASURED" when no tolerance was
    ever configured for this feature (never a bare "N/A", which the vision
    layer's `build_feature_summary_table` uses for BOTH that case and a
    configured-but-unmeasurable spec — those are different situations and
    must read differently). "INCOMPLETE" is reserved for the second, real
    case: a tolerance was configured but the measurement it needs wasn't
    available."""
    if tol is None:
        return "MEASURED"
    if tol.status == "N/A":
        return "INCOMPLETE"
    return tol.status


def _fmt_len(px_value: Optional[float], mm_value: Optional[float], decimals: int = 2) -> str:
    if mm_value is not None:
        return f"{mm_value:.{decimals}f} mm"
    if px_value is not None:
        return f"{px_value:.1f} px"
    return "—"


def _esc(s) -> str:
    return html_lib.escape(str(s), quote=True)


def _text_box_w(text: str) -> float:
    return max(46.0, len(text) * 6.6 + 20.0)


def _box_overlap_area(a: dict, b: dict) -> float:
    ax0, ax1 = a["cx"] - a["w"] / 2, a["cx"] + a["w"] / 2
    ay0, ay1 = a["cy"] - a["h"] / 2, a["cy"] + a["h"] / 2
    bx0, bx1 = b["cx"] - b["w"] / 2, b["cx"] + b["w"] / 2
    by0, by1 = b["cy"] - b["h"] / 2, b["cy"] + b["h"] / 2
    ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    oy = max(0.0, min(ay1, by1) - max(ay0, by0))
    return ox * oy


def _place_label(
    anchor_x: float, anchor_y: float, box_w: float, box_h: float, leader_len: float,
    obstacles: list[dict], img_w: int, img_h: int,
    part_contour: Optional[np.ndarray] = None, preferred: tuple[float, float] = (0.0, 0.0),
) -> tuple[dict, dict]:
    """Real 8-candidate label placement: tries N/S/E/W/NE/NW/SE/SW around the
    anchor at `leader_len`, scores each by overlap area against every
    already-placed obstacle (a candidate that would run off the canvas is
    excluded outright; one whose center lands inside the product's own
    contour is penalized, not excluded — sometimes unavoidable for a
    centered bore), and keeps the lowest-scoring one. `preferred` is a small
    tie-breaking bias (e.g. "away from image center") that only matters
    among otherwise-equal candidates — it never overrides a real collision.
    Returns (chosen_box, diagnostics) so Debug Mode can show exactly what
    was tried."""
    best_box, best_score, best_name = None, None, None
    for i, (ddx, ddy) in enumerate(_CANDIDATE_DIRECTIONS):
        cx = anchor_x + ddx * leader_len
        cy = anchor_y + ddy * leader_len
        box = {"cx": cx, "cy": cy, "w": box_w, "h": box_h}
        if (cx - box_w / 2 < 2 or cx + box_w / 2 > img_w - 2
                or cy - box_h / 2 < 2 or cy + box_h / 2 > img_h - 2):
            score = float("inf")
        else:
            score = sum(_box_overlap_area(box, ob) for ob in obstacles)
            if part_contour is not None and cv2.pointPolygonTest(part_contour, (cx, cy), False) >= 0:
                score += 400.0
            if preferred != (0.0, 0.0):
                alignment = ddx * preferred[0] + ddy * preferred[1]  # -1..1
                score += (1.0 - alignment) * 0.5  # tie-break only, real overlaps dominate
        if best_score is None or score < best_score:
            best_box, best_score, best_name = box, score, _CANDIDATE_NAMES[i]

    if best_score == float("inf"):
        ddx, ddy = _CANDIDATE_DIRECTIONS[0]
        cx = min(max(anchor_x + ddx * leader_len, box_w / 2 + 2), img_w - box_w / 2 - 2)
        cy = min(max(anchor_y + ddy * leader_len, box_h / 2 + 2), img_h - box_h / 2 - 2)
        best_box, best_score, best_name = {"cx": cx, "cy": cy, "w": box_w, "h": box_h}, 0.0, "N*"

    diagnostics = {"direction": best_name, "score": round(best_score, 1), "box": dict(best_box)}
    return best_box, diagnostics


def _contour_path_d(contour: np.ndarray, epsilon_fraction: float = 0.0025) -> str:
    """A simplified SVG path for the real (possibly concave) outer outline,
    from `part.contour` — a presentation-only smoothing of already-detected
    points via `cv2.approxPolyDP`, not a new detection."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, max(epsilon_fraction * peri, 0.5), True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return ""
    d = f"M {pts[0][0]},{pts[0][1]} " + " ".join(f"L {p[0]},{p[1]}" for p in pts[1:]) + " Z"
    return d


def _marker_defs() -> str:
    defs = []
    for key, color in _STATUS_COLORS.items():
        if key == "neutral":
            continue
        defs.append(
            f'<marker id="arrowEnd-{key}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{color}" /></marker>'
        )
    return "<defs>" + "".join(defs) + "</defs>"


def _dim_line_horizontal(x0: float, x1: float, y: float, extend_to_y: float, label: str,
                          label_cx: float, label_cy: float, reveal_delay: float = 0.0) -> str:
    color = _STATUS_COLORS["pass"]
    ext1 = (f'<line x1="{x0}" y1="{extend_to_y}" x2="{x0}" y2="{y}" stroke="{color}" '
            f'stroke-width="1" stroke-dasharray="3,3" opacity="0.45" />')
    ext2 = (f'<line x1="{x1}" y1="{extend_to_y}" x2="{x1}" y2="{y}" stroke="{color}" '
            f'stroke-width="1" stroke-dasharray="3,3" opacity="0.45" />')
    length = abs(x1 - x0)
    dim = (
        f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{color}" stroke-width="1.5" '
        f'marker-start="url(#arrowEnd-pass)" marker-end="url(#arrowEnd-pass)" '
        f'style="stroke-dasharray:{length}; stroke-dashoffset:{length}; '
        f'animation: mvDraw 0.4s ease {reveal_delay}s forwards;" />'
    )
    lines_svg = f'<g data-layer="dimensions">{ext1}{ext2}{dim}</g>'

    box_w, box_h = _text_box_w(label), 20
    rect = (
        f'<rect x="{label_cx - box_w / 2}" y="{label_cy - box_h / 2}" width="{box_w}" height="{box_h}" rx="3" '
        f'fill="#FFFFFF" stroke="{color}" stroke-width="1" '
        f'style="opacity:0; animation: mvFadeInText 0.3s ease {reveal_delay + 0.3}s forwards;" />'
    )
    text = (
        f'<text x="{label_cx}" y="{label_cy + 4}" text-anchor="middle" fill="{color}" font-size="12.5" '
        f'font-family="IBM Plex Mono, monospace" font-weight="700" '
        f'style="opacity:0; animation: mvFadeInText 0.3s ease {reveal_delay + 0.3}s forwards;">'
        f'{_esc(label)}</text>'
    )
    value_svg = f'<g data-layer="values">{rect}{text}</g>'
    return lines_svg + value_svg


def _dim_line_vertical(y0: float, y1: float, x: float, extend_to_x: float, label: str,
                        label_cx: float, label_cy: float, reveal_delay: float = 0.0) -> str:
    color = _STATUS_COLORS["pass"]
    ext1 = (f'<line x1="{extend_to_x}" y1="{y0}" x2="{x}" y2="{y0}" stroke="{color}" '
            f'stroke-width="1" stroke-dasharray="3,3" opacity="0.45" />')
    ext2 = (f'<line x1="{extend_to_x}" y1="{y1}" x2="{x}" y2="{y1}" stroke="{color}" '
            f'stroke-width="1" stroke-dasharray="3,3" opacity="0.45" />')
    length = abs(y1 - y0)
    dim = (
        f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}" stroke="{color}" stroke-width="1.5" '
        f'marker-start="url(#arrowEnd-pass)" marker-end="url(#arrowEnd-pass)" '
        f'style="stroke-dasharray:{length}; stroke-dashoffset:{length}; '
        f'animation: mvDraw 0.4s ease {reveal_delay}s forwards;" />'
    )
    lines_svg = f'<g data-layer="dimensions">{ext1}{ext2}{dim}</g>'

    box_w, box_h = _text_box_w(label), 20
    rect = (
        f'<rect x="{label_cx - box_w / 2}" y="{label_cy - box_h / 2}" width="{box_w}" height="{box_h}" rx="3" '
        f'fill="#FFFFFF" stroke="{color}" stroke-width="1" '
        f'style="opacity:0; animation: mvFadeInText 0.3s ease {reveal_delay + 0.3}s forwards;" />'
    )
    text = (
        f'<text x="{label_cx}" y="{label_cy + 4}" text-anchor="middle" fill="{color}" font-size="12.5" '
        f'font-family="IBM Plex Mono, monospace" font-weight="700" '
        f'style="opacity:0; animation: mvFadeInText 0.3s ease {reveal_delay + 0.3}s forwards;">'
        f'{_esc(label)}</text>'
    )
    value_svg = f'<g data-layer="values">{rect}{text}</g>'
    return lines_svg + value_svg


def _circle_marker(cx: float, cy: float, r: float, short_id: str, diam_label: str,
                    confidence_letter: str, tol_glyph: Optional[str], status_key: str, selected: bool,
                    label_cx: float, label_cy: float, reveal_delay: float, tooltip_html: str) -> str:
    color = _STATUS_COLORS[status_key]
    box_border_color = COLOR_INFO if selected else color
    cross = 0.35 * r + 4
    stroke_w = 3 if selected else 2

    centers = (
        f'<g data-layer="centers">'
        f'<line x1="{cx - cross}" y1="{cy}" x2="{cx + cross}" y2="{cy}" stroke="{color}" '
        f'stroke-width="1" opacity="0.85" />'
        f'<line x1="{cx}" y1="{cy - cross}" x2="{cx}" y2="{cy + cross}" stroke="{color}" '
        f'stroke-width="1" opacity="0.85" />'
        f'</g>'
    )
    circle = (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke_w}" '
        f'style="opacity:0; transform-origin:{cx}px {cy}px; '
        f'animation: mvPopIn 0.3s cubic-bezier(0.2,0.8,0.2,1) {reveal_delay}s forwards;" />'
    )
    if selected:
        circle += (
            f'<circle cx="{cx}" cy="{cy}" r="{r + 5}" fill="none" stroke="{COLOR_INFO}" '
            f'stroke-width="1.5" stroke-dasharray="4,3" opacity="0.9" />'
        )

    leader = (f'<line x1="{cx}" y1="{cy}" x2="{label_cx}" y2="{label_cy}" stroke="{box_border_color}" '
              f'stroke-width="1" opacity="0.55" stroke-dasharray="2,2" />')
    box_w, box_h = 86, 38
    box_x, box_y = label_cx - box_w / 2, label_cy - box_h / 2
    label_box = (
        f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" rx="3" '
        f'fill="#FFFFFF" stroke="{box_border_color}" stroke-width="{2 if selected else 1}" opacity="0.98" />'
        f'<text x="{label_cx}" y="{label_cy - 6}" text-anchor="middle" fill="{color}" font-size="12" '
        f'font-family="IBM Plex Mono, monospace" font-weight="700">{_esc(short_id)}</text>'
    )
    hit = f'<circle cx="{cx}" cy="{cy}" r="{max(r, 18)}" fill="transparent" class="mv-hit" />'
    tooltip_w, tooltip_h = 190, 92
    fo_y = cy - r - tooltip_h - 6
    if fo_y < 0:
        fo_y = cy + r + 6
    fo = (
        f'<foreignObject x="{cx - tooltip_w / 2}" y="{fo_y}" width="{tooltip_w}" height="{tooltip_h}" '
        f'class="mv-tooltip-fo">'
        f'<div xmlns="http://www.w3.org/1999/xhtml" class="mv-tooltip">{tooltip_html}</div>'
        f'</foreignObject>'
    )
    # Hit-circle and tooltip stay in the SAME group as the marker so the
    # plain CSS `:hover .mv-tooltip-fo` descendant selector works — splitting
    # them into sibling groups (even both tagged data-layer="features") would
    # break hover-to-show since they'd no longer share an ancestor.
    features_svg = f'<g class="mv-hoverable" data-layer="features">{leader}{label_box}{circle}{hit}{fo}</g>{centers}'

    value_svg = (
        f'<g data-layer="values">'
        f'<text x="{label_cx}" y="{label_cy + 12}" text-anchor="middle" fill="{COLOR_TEXT}" font-size="10.5" '
        f'font-family="IBM Plex Mono, monospace">{_esc(diam_label)}</text>'
        f'</g>'
    )

    conf_svg = ""
    if confidence_letter:
        conf_svg = (
            f'<g data-layer="confidence">'
            f'<text x="{label_cx + box_w / 2 - 9}" y="{label_cy - box_h / 2 + 11}" text-anchor="middle" '
            f'fill="{COLOR_TEXT_MUTED}" font-size="8.5" font-family="IBM Plex Mono, monospace" '
            f'font-weight="700">{_esc(confidence_letter)}</text>'
            f'</g>'
        )

    tol_svg = ""
    if tol_glyph:
        tol_color = {"✓": COLOR_PASS, "✗": COLOR_FAIL}.get(tol_glyph, COLOR_WARN)
        tol_svg = (
            f'<g data-layer="tolerance">'
            f'<text x="{label_cx - box_w / 2 + 9}" y="{label_cy - box_h / 2 + 11}" text-anchor="middle" '
            f'fill="{tol_color}" font-size="10" font-family="IBM Plex Mono, monospace" '
            f'font-weight="700">{_esc(tol_glyph)}</text>'
            f'</g>'
        )

    return features_svg + value_svg + conf_svg + tol_svg


def build_svg_overlay(
    part: DetectedPart,
    circles: list[CircleFeature],
    records: list[MeasurementRecord],
    profile: Optional[CalibrationProfile],
    origin_px: tuple[float, float],
    img_w: int,
    img_h: int,
    tolerance_by_feature: Optional[dict[str, ToleranceResult]] = None,
    selected_short_id: Optional[str] = None,
    decimals: int = 2,
) -> tuple[str, list[dict]]:
    """Assembles the full SVG dimension/marker layer. `records` is the single
    source of truth for every displayed length so this can never diverge
    from the measurement table. Every label's final position comes from the
    real 8-candidate placer (`_place_label`) run sequentially — each placed
    label becomes an obstacle for the next — so nothing overlaps.

    Returns (svg_string, placement_diagnostics) — the diagnostics list feeds
    Vision Debug Mode's label-placement panel (direction chosen out of 8,
    collision score, final box) for every dimension/feature/corner label.
    """
    tolerance_by_feature = tolerance_by_feature or {}
    rec_by_feature = {r.feature: r for r in records}
    x, y, w, h = part.bbox
    parts = [_marker_defs()]
    diagnostics: list[dict] = []
    obstacles: list[dict] = []

    contour_d = _contour_path_d(part.contour)
    if contour_d:
        parts.append(
            f'<g data-layer="contour"><path d="{contour_d}" fill="none" '
            f'stroke="{_STATUS_COLORS["pass"]}" stroke-width="1.5" opacity="0.6" '
            f'style="opacity:0; animation: mvFadeInText 0.5s ease 0s forwards;"/></g>'
        )

    # --- overall width / height dimension lines (edge-aware placement kept
    #     as-is; each becomes a fixed obstacle for everything placed after) --
    dim_offset = 28
    width_r = rec_by_feature.get("Bounding Box Width")
    if width_r is not None:
        width_label = _fmt_len(width_r.px_value, width_r.mm_value, decimals)
        if y > dim_offset + 20:
            dim_y, extend_to = y - dim_offset, y
        else:
            dim_y, extend_to = y + h + dim_offset, y + h
        label_cy = dim_y - 8 if extend_to < dim_y else dim_y + 18
        w_box = {"cx": (x + x + w) / 2, "cy": label_cy, "w": _text_box_w(width_label), "h": 20}
        obstacles.append(w_box)
        diagnostics.append({"id": "Overall Width", "kind": "dimension", "direction": "fixed", "score": 0.0, "box": dict(w_box)})
        parts.append(_dim_line_horizontal(x, x + w, dim_y, extend_to, width_label, w_box["cx"], w_box["cy"], reveal_delay=0.0))

    height_r = rec_by_feature.get("Bounding Box Height")
    if height_r is not None:
        height_label = _fmt_len(height_r.px_value, height_r.mm_value, decimals)
        if x > dim_offset + 20:
            dim_x, extend_to_x = x - dim_offset, x
        else:
            dim_x, extend_to_x = x + w + dim_offset, x + w
        label_cx = dim_x - 10 if extend_to_x < dim_x else dim_x + 10
        h_box = {"cx": label_cx, "cy": (y + y + h) / 2, "w": _text_box_w(height_label), "h": 20}
        obstacles.append(h_box)
        diagnostics.append({"id": "Overall Height", "kind": "dimension", "direction": "fixed", "score": 0.0, "box": dict(h_box)})
        parts.append(_dim_line_vertical(y, y + h, dim_x, extend_to_x, height_label, h_box["cx"], h_box["cy"], reveal_delay=0.08))

    # --- feature markers (greedy 8-candidate placement, in detection order) -
    for i, c in enumerate(circles):
        label = c.label or f"Hole {c.circle_id}"
        d_record = rec_by_feature.get(f"{label} Equivalent Diameter")
        diam_label = _fmt_len(d_record.px_value if d_record else None, d_record.mm_value if d_record else None, decimals)
        tol = tolerance_by_feature.get(f"{label} Equivalent Diameter")
        status_key = _overlay_status_key(tol)
        selected = bool(selected_short_id) and c.short_id == selected_short_id
        confidence_letter = _CONFIDENCE_LETTER.get(c.confidence, "-")
        tol_glyph = _TOL_GLYPH.get(tol.status) if tol is not None else None

        dx, dy = c.cx - img_w / 2, c.cy - img_h / 2
        norm = math.hypot(dx, dy) or 1.0
        preferred = (dx / norm, dy / norm)
        leader_len = c.r + 34

        box, diag = _place_label(c.cx, c.cy, 86, 38, leader_len, obstacles, img_w, img_h,
                                  part_contour=part.contour, preferred=preferred)
        obstacles.append(box)
        diag["id"] = c.short_id or label
        diag["kind"] = "feature"
        diagnostics.append(diag)

        edge_r = rec_by_feature.get(f"{label} to Nearest Edge")
        edge_line = (f'<div class="t-row">Edge dist: {_esc(_fmt_len(edge_r.px_value, edge_r.mm_value, decimals))}</div>'
                     if edge_r is not None else "")
        feature_kind = "Center Bore" if c.feature_type == "bore" else "Hole"
        result_text = feature_result_label(tol)
        tooltip_html = (
            f'<div class="t-title">{_esc(c.short_id or c.label)} — {_esc(feature_kind)}</div>'
            f'<div class="t-row">Ø {_esc(diam_label)}</div>'
            f'<div class="t-row">Confidence: {_esc(c.confidence)}</div>'
            f'{edge_line}'
            f'<div class="t-row t-{feature_result_status_key(tol)}">{_esc(result_text)}</div>'
        )

        reveal_delay = 0.2 + min(i * 0.05, 0.5)
        parts.append(_circle_marker(
            c.cx, c.cy, c.r, c.short_id or c.label, diam_label, confidence_letter, tol_glyph,
            status_key, selected, box["cx"], box["cy"], reveal_delay, tooltip_html,
        ))

    # --- vertical distance: each feature centre straight down to the part's
    #     lowest point (image Y), drawn as a dashed line with a tick + label --
    lowest_y = float(part.contour[:, 0, 1].max())
    for c in circles:
        label = c.label or f"Hole {c.circle_id}"
        v_r = rec_by_feature.get(f"{label} Center to Lowest Point (vertical)")
        if v_r is None or lowest_y - c.cy < c.r:
            continue
        v_label = "V " + _fmt_len(v_r.px_value, v_r.mm_value, decimals)
        mid_y = (c.cy + c.r + lowest_y) / 2
        box = {"cx": c.cx + _text_box_w(v_label) / 2 + 10, "cy": mid_y, "w": _text_box_w(v_label), "h": 20}
        box["cx"] = min(max(box["cx"], box["w"] / 2 + 2), img_w - box["w"] / 2 - 2)
        obstacles.append(box)
        diagnostics.append({"id": f"{c.short_id} vertical", "kind": "vertical", "direction": "fixed", "score": 0.0, "box": dict(box)})
        col = COLOR_INFO
        parts.append(
            f'<g data-layer="dimensions">'
            f'<line x1="{c.cx}" y1="{c.cy}" x2="{c.cx}" y2="{lowest_y}" stroke="{col}" stroke-width="1.5" '
            f'stroke-dasharray="6,4" marker-end="url(#arrowEnd-info)" />'
            f'<line x1="{c.cx - 24}" y1="{lowest_y}" x2="{c.cx + 24}" y2="{lowest_y}" stroke="{col}" stroke-width="1.5" />'
            f'<rect x="{box["cx"] - box["w"] / 2}" y="{box["cy"] - box["h"] / 2}" width="{box["w"]}" height="{box["h"]}" '
            f'rx="3" fill="#FFFFFF" stroke="{col}" stroke-width="1" opacity="0.95" /></g>'
            f'<g data-layer="values"><text x="{box["cx"]}" y="{box["cy"] + 4}" text-anchor="middle" fill="{col}" '
            f'font-size="11" font-family="IBM Plex Mono, monospace" font-weight="700">{_esc(v_label)}</text></g>'
        )

    # --- perpendicular distance from each feature centre to the fork-tip base
    #     line (the part's own frame), drawn as a solid line to its foot point --
    tips = geometry.find_fork_tips(part.contour, part.area_px2)
    if tips is not None:
        ta = np.array(tips["tip_a"], float)
        tb = np.array(tips["tip_b"], float)
        base = tb - ta
        base_len = float(np.hypot(*base))
        if base_len > 1e-6:
            unit = base / base_len
            col = COLOR_WARN
            for c in circles:
                label = c.label or f"Hole {c.circle_id}"
                p_r = rec_by_feature.get(f"{label} Center to Tip Line (perpendicular)")
                if p_r is None:
                    continue
                t = float(np.dot(np.array([c.cx, c.cy]) - ta, unit))
                fx, fy = ta + unit * t
                p_label = "H " + _fmt_len(p_r.px_value, p_r.mm_value, decimals)
                bw = _text_box_w(p_label)
                lx, ly = (c.cx + fx) / 2 + bw / 2 + 8, (c.cy + fy) / 2
                lx = min(max(lx, bw / 2 + 2), img_w - bw / 2 - 2)
                obstacles.append({"cx": lx, "cy": ly, "w": bw, "h": 20})
                diagnostics.append({"id": f"{c.short_id} to tip line", "kind": "perpendicular", "direction": "fixed",
                                    "score": 0.0, "box": {"cx": lx, "cy": ly, "w": bw, "h": 20}})
                parts.append(
                    f'<g data-layer="dimensions">'
                    f'<line x1="{ta[0]}" y1="{ta[1]}" x2="{tb[0]}" y2="{tb[1]}" stroke="{col}" stroke-width="1" '
                    f'stroke-dasharray="3,4" opacity="0.7" />'
                    f'<line x1="{c.cx}" y1="{c.cy}" x2="{fx}" y2="{fy}" stroke="{col}" stroke-width="2" />'
                    f'<circle cx="{fx}" cy="{fy}" r="3.5" fill="{col}" />'
                    f'<rect x="{lx - bw / 2}" y="{ly - 10}" width="{bw}" height="20" rx="3" fill="#FFFFFF" '
                    f'stroke="{col}" stroke-width="1" opacity="0.95" /></g>'
                    f'<g data-layer="values"><text x="{lx}" y="{ly + 4}" text-anchor="middle" fill="{col}" '
                    f'font-size="11" font-family="IBM Plex Mono, monospace" font-weight="700">{_esc(p_label)}</text></g>'
                )

    # --- reliable corner / inner-arc radii (real fitted geometry only) -----
    corner_radii = geometry.estimate_corner_radii(part.contour)
    reliable_corners = [c for c in corner_radii if c["reliable"]]
    for i, corner in enumerate(reliable_corners):
        r_record = rec_by_feature.get(f"Corner {i + 1} Radius")
        r_label = "R " + _fmt_len(corner["radius_px"], r_record.mm_value if r_record else None, decimals)
        vx, vy = corner["x"], corner["y"]
        dx, dy = vx - (x + w / 2), vy - (y + h / 2)
        norm = math.hypot(dx, dy) or 1.0
        preferred = (dx / norm, dy / norm)
        box, diag = _place_label(vx, vy, _text_box_w(r_label), 20, 38, obstacles, img_w, img_h,
                                  part_contour=part.contour, preferred=preferred)
        obstacles.append(box)
        diag["id"] = f"Corner {i + 1}"
        diag["kind"] = "corner-radius"
        diagnostics.append(diag)

        dot = f'<circle cx="{vx}" cy="{vy}" r="2.5" fill="{_STATUS_COLORS["pass"]}" opacity="0.85" />'
        leader = (f'<line x1="{vx}" y1="{vy}" x2="{box["cx"]}" y2="{box["cy"]}" '
                  f'stroke="{_STATUS_COLORS["pass"]}" stroke-width="1" opacity="0.5" stroke-dasharray="2,2" />')
        rect = (f'<rect x="{box["cx"] - box["w"] / 2}" y="{box["cy"] - box["h"] / 2}" width="{box["w"]}" height="{box["h"]}" '
                f'rx="3" fill="#FFFFFF" stroke="{_STATUS_COLORS["pass"]}" stroke-width="1" opacity="0.95" />')
        text = (f'<text x="{box["cx"]}" y="{box["cy"] + 4}" text-anchor="middle" fill="{_STATUS_COLORS["pass"]}" '
                f'font-size="11" font-family="IBM Plex Mono, monospace" font-weight="700">{_esc(r_label)}</text>')
        parts.append(f'<g data-layer="contour">{dot}{leader}{rect}</g><g data-layer="values">{text}</g>')

    # --- inter-feature distance reference (only when exactly 2 features —
    #     avoids an O(n^2) cluttered image; the full pairwise data is still
    #     in the "Full measurement record" expander for any feature count) --
    if len(circles) == 2:
        a, b = circles[0], circles[1]
        label_a = a.label or f"Hole {a.circle_id}"
        label_b = b.label or f"Hole {b.circle_id}"
        dist_r = rec_by_feature.get(f"{label_a} to {label_b} (center distance)")
        if dist_r is not None:
            dist_label = _fmt_len(dist_r.px_value, dist_r.mm_value, decimals)
            mid_x, mid_y = (a.cx + b.cx) / 2, (a.cy + b.cy) / 2
            perp_dx, perp_dy = -(b.cy - a.cy), (b.cx - a.cx)
            norm = math.hypot(perp_dx, perp_dy) or 1.0
            preferred = (perp_dx / norm, perp_dy / norm)
            box, diag = _place_label(mid_x, mid_y, _text_box_w(dist_label), 20, 22, obstacles, img_w, img_h,
                                      part_contour=part.contour, preferred=preferred)
            obstacles.append(box)
            diag["id"] = f"{a.short_id}→{b.short_id} distance"
            diag["kind"] = "distance"
            diagnostics.append(diag)

            line = (f'<line x1="{a.cx}" y1="{a.cy}" x2="{b.cx}" y2="{b.cy}" stroke="{COLOR_INFO}" '
                    f'stroke-width="1" stroke-dasharray="5,3" opacity="0.6" '
                    f'marker-start="url(#arrowEnd-info)" marker-end="url(#arrowEnd-info)" />')
            rect = (f'<rect x="{box["cx"] - box["w"] / 2}" y="{box["cy"] - box["h"] / 2}" width="{box["w"]}" height="{box["h"]}" '
                    f'rx="3" fill="#FFFFFF" stroke="{COLOR_INFO}" stroke-width="1" opacity="0.95" />')
            text = (f'<text x="{box["cx"]}" y="{box["cy"] + 4}" text-anchor="middle" fill="{COLOR_INFO}" '
                    f'font-size="11" font-family="IBM Plex Mono, monospace" font-weight="700">{_esc(dist_label)}</text>')
            parts.append(f'<g data-layer="dimensions">{line}{rect}</g><g data-layer="values">{text}</g>')

    svg = (
        f'<svg class="mv-svg" viewBox="0 0 {img_w} {img_h}" width="{img_w}" height="{img_h}" '
        f'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + '</svg>'
    )
    return svg, diagnostics


def build_inspection_html(
    image_bgr: np.ndarray, svg_overlay: str, img_w: int, img_h: int,
    viewport_id: str = "mvViewport", focus: Optional[tuple[float, float, float]] = None,
) -> str:
    """Wraps the image + SVG overlay in a self-contained zoomable/pannable
    HTML component (vanilla JS — no framework, no server round-trip), with a
    floating toolbar (zoom/pan controls + per-layer visibility checkboxes).
    `focus` = (cx, cy, r) of a selected feature: when given, the viewer
    zooms/centers on it on load instead of fitting the whole part; passing
    `None` (nothing selected) fits the whole product as before — so
    selecting a feature zooms in and deselecting it returns to the full
    view."""
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Could not encode inspection image for display.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    data_uri = f"data:image/png;base64,{b64}"

    layers = [
        ("contour", "Outer Contour"), ("dimensions", "Dimensions"), ("features", "Feature Markers"),
        ("centers", "Center Marks"), ("tolerance", "Tolerance"), ("confidence", "Confidence"),
        ("values", "Measurement Values"),
    ]
    layer_checkboxes = "".join(
        f'<label class="mv-layer-toggle"><input type="checkbox" checked '
        f'onchange="{viewport_id}ToggleLayer(\'{key}\', this.checked)"/> {name}</label>'
        for key, name in layers
    )

    focus_js = "null"
    if focus is not None:
        fcx, fcy, fr = focus
        focus_js = f"{{cx:{fcx}, cy:{fcy}, r:{fr}}}"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body {{ margin:0; padding:0; background: transparent; }}
  #{viewport_id}-wrap {{ font-family: 'Inter', -apple-system, sans-serif; position: relative; }}
  #{viewport_id}-toolbar {{
      display:flex; flex-wrap:wrap; align-items:center; gap:14px;
      background:#FFFFFF; border:1px solid {COLOR_BORDER}; border-radius:6px 6px 0 0;
      padding:8px 12px; border-bottom:none;
  }}
  #{viewport_id}-toolbar .mv-btn-group {{ display:flex; flex-wrap:wrap; gap:6px; }}
  #{viewport_id}-toolbar button {{
      background:#FFFFFF; color:{COLOR_TEXT}; border:1px solid {COLOR_BORDER};
      border-radius:4px; padding:4px 10px; font-size:12px; cursor:pointer;
      font-family:'IBM Plex Mono', monospace;
  }}
  #{viewport_id}-toolbar button:hover {{ border-color:{COLOR_INFO}; color:{COLOR_INFO}; }}
  .mv-layer-toggle {{
      display:inline-flex; align-items:center; gap:4px; font-size:11.5px; color:{COLOR_TEXT_MUTED};
      user-select:none; cursor:pointer;
  }}
  .mv-layer-toggle input {{ accent-color:{COLOR_INFO}; cursor:pointer; }}
  #{viewport_id}-stage {{
      position:relative; width:100%; height:{STAGE_HEIGHT_PX}px;
      background-color:#FDFDFE;
      background-image:
        linear-gradient(rgba(23,32,42,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(23,32,42,0.035) 1px, transparent 1px);
      background-size: 28px 28px;
      border:1px solid {COLOR_BORDER}; border-radius:0 0 6px 6px;
      overflow:hidden; cursor:grab;
  }}
  #{viewport_id}-stage.dragging {{ cursor:grabbing; }}
  #{viewport_id}-canvas {{
      position:absolute; top:50%; left:50%;
      width:{img_w}px; height:{img_h}px;
      transform: translate(-50%,-50%);
      animation: {viewport_id}FadeIn 0.4s ease both;
      transition: transform 0.35s ease;
  }}
  #{viewport_id}-canvas img {{
      position:absolute; top:0; left:0; width:{img_w}px; height:{img_h}px;
      display:block; user-select:none; -webkit-user-drag:none;
  }}
  .mv-svg {{ position:absolute; top:0; left:0; }}
  .mv-svg text {{ paint-order: stroke; }}
  .mv-tooltip-fo {{ opacity:0; pointer-events:none; transition: opacity 0.15s ease; }}
  .mv-hoverable:hover .mv-tooltip-fo {{ opacity:1; }}
  .mv-tooltip {{
      background:#FFFFFF; border:1px solid {COLOR_INFO}; border-radius:5px;
      padding:8px 10px; font-family:'Inter',sans-serif; font-size:11px; color:{COLOR_TEXT};
      box-shadow: 0 4px 14px rgba(16,24,40,0.12);
  }}
  .mv-tooltip .t-title {{ font-weight:700; color:{COLOR_TEXT}; margin-bottom:3px; }}
  .mv-tooltip .t-row {{ color:{COLOR_TEXT_MUTED}; line-height:1.5; }}
  .mv-tooltip .t-pass {{ color:{COLOR_PASS}; font-weight:700; }}
  .mv-tooltip .t-fail {{ color:{COLOR_FAIL}; font-weight:700; }}
  .mv-tooltip .t-warn {{ color:{COLOR_WARN}; font-weight:700; }}
  .mv-tooltip .t-neutral {{ color:{COLOR_TEXT_MUTED}; font-weight:700; }}
  @keyframes mvDraw {{ to {{ stroke-dashoffset:0; }} }}
  @keyframes mvFadeInText {{ to {{ opacity:1; }} }}
  @keyframes mvPopIn {{ from {{ opacity:0; transform:scale(0.6); }} to {{ opacity:1; transform:scale(1); }} }}
  @keyframes {viewport_id}FadeIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
</style>
</head>
<body>
<div id="{viewport_id}-wrap">
  <div id="{viewport_id}-toolbar">
    <div class="mv-btn-group">
      <button onclick="{viewport_id}Zoom(1.25)">Zoom +</button>
      <button onclick="{viewport_id}Zoom(0.8)">Zoom −</button>
      <button onclick="{viewport_id}Fit()">Fit</button>
      <button onclick="{viewport_id}Actual()">100%</button>
      <button onclick="{viewport_id}Reset()">Reset</button>
    </div>
    <div class="mv-btn-group" style="gap:12px;">{layer_checkboxes}</div>
  </div>
  <div id="{viewport_id}-stage">
    <div id="{viewport_id}-canvas">
      <img src="{data_uri}" width="{img_w}" height="{img_h}" draggable="false" />
      {svg_overlay}
    </div>
  </div>
</div>
<script>
(function() {{
  var stage = document.getElementById("{viewport_id}-stage");
  var canvas = document.getElementById("{viewport_id}-canvas");
  var scale = 1, tx = 0, ty = 0;
  var dragging = false, lastX = 0, lastY = 0;
  var focusTarget = {focus_js};

  function apply() {{
      canvas.style.transform = "translate(-50%,-50%) translate(" + tx + "px," + ty + "px) scale(" + scale + ")";
  }}

  window["{viewport_id}Zoom"] = function(factor) {{
      scale = Math.min(6, Math.max(0.2, scale * factor));
      apply();
  }};
  window["{viewport_id}Fit"] = function() {{
      var rect = stage.getBoundingClientRect();
      var sx = rect.width / {img_w};
      var sy = rect.height / {img_h};
      scale = Math.min(sx, sy) * 0.95;
      tx = 0; ty = 0;
      apply();
  }};
  window["{viewport_id}Actual"] = function() {{
      scale = 1; tx = 0; ty = 0;
      apply();
  }};
  window["{viewport_id}Reset"] = function() {{
      window["{viewport_id}Fit"]();
  }};
  window["{viewport_id}ZoomToFeature"] = function(cx, cy, r) {{
      var rect = stage.getBoundingClientRect();
      var targetScale = Math.min(6, Math.max(1.4, Math.min(rect.width, rect.height) / (r * 7)));
      scale = targetScale;
      tx = (({img_w} / 2) - cx) * scale;
      ty = (({img_h} / 2) - cy) * scale;
      apply();
  }};
  window["{viewport_id}ToggleLayer"] = function(layerName, visible) {{
      var nodes = canvas.querySelectorAll('[data-layer="' + layerName + '"]');
      for (var i = 0; i < nodes.length; i++) {{
          nodes[i].style.display = visible ? "" : "none";
      }}
  }};

  stage.addEventListener("mousedown", function(e) {{
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      stage.classList.add("dragging");
  }});
  window.addEventListener("mouseup", function() {{
      dragging = false; stage.classList.remove("dragging");
  }});
  window.addEventListener("mousemove", function(e) {{
      if (!dragging) return;
      tx += (e.clientX - lastX) / scale;
      ty += (e.clientY - lastY) / scale;
      lastX = e.clientX; lastY = e.clientY;
      apply();
  }});
  stage.addEventListener("wheel", function(e) {{
      e.preventDefault();
      var factor = e.deltaY < 0 ? 1.1 : 0.9;
      scale = Math.min(6, Math.max(0.2, scale * factor));
      apply();
  }}, {{ passive: false }});

  if (focusTarget) {{
      window["{viewport_id}Fit"]();
      window["{viewport_id}ZoomToFeature"](focusTarget.cx, focusTarget.cy, focusTarget.r);
  }} else {{
      window["{viewport_id}Fit"]();
  }}
}})();
</script>
</body>
</html>
"""


def build_measurement_table_html(
    circles: list[CircleFeature],
    records: list[MeasurementRecord],
    tolerance_by_feature: Optional[dict[str, ToleranceResult]] = None,
    selected_short_id: Optional[str] = None,
    decimals: int = 2,
) -> str:
    """The HTML measurement table (Feature/Type/Measured/Nominal/Tolerance/
    Deviation/Unit/Confidence/Result) using small status dots instead of a
    full-row color fill. `Deviation` (measured - nominal) is a display-only
    computation from an existing `ToleranceResult`'s own fields. `Result`
    uses `feature_result_label` so "no tolerance configured" reads
    "MEASURED" rather than being confused with "PASS" or with a configured
    tolerance that came back unmeasurable ("INCOMPLETE")."""
    tolerance_by_feature = tolerance_by_feature or {}
    rec_by_feature = {r.feature: r for r in records}
    rows_html = []

    for c in circles:
        label = c.label or f"Hole {c.circle_id}"
        d = rec_by_feature.get(f"{label} Equivalent Diameter")
        tol = tolerance_by_feature.get(f"{label} Equivalent Diameter")
        measured_mm = d.mm_value if d else None

        if d is None:
            measured_str, unit = "—", "—"
        elif measured_mm is not None:
            measured_str, unit = f"{measured_mm:.{decimals + 1}f}", "mm"
        elif d.px_value is not None:
            measured_str, unit = f"{d.px_value:.1f}", "px"
        else:
            measured_str, unit = "—", "—"

        status_key = feature_result_status_key(tol)
        if tol is not None:
            nominal_str = f"{tol.nominal_mm:.{decimals + 1}f}"
            tolerance_str = f"±{tol.tolerance_mm:g}"
            deviation_str = f"{measured_mm - tol.nominal_mm:+.{decimals + 1}f}" if measured_mm is not None else "—"
        else:
            nominal_str, tolerance_str, deviation_str = "—", "—", "—"
        result_str = feature_result_label(tol)

        row_id = c.short_id or label
        row_class = " mv-row-selected" if selected_short_id and row_id == selected_short_id else ""
        rows_html.append(
            f"<tr class=\"{row_class.strip()}\">"
            f"<td>{_esc(row_id)}</td>"
            f"<td>{_esc('Center Bore' if c.feature_type == 'bore' else 'Hole')}</td>"
            f"<td class=\"mv-val-strong\">{_esc(measured_str)}</td>"
            f"<td>{_esc(nominal_str)}</td>"
            f"<td>{_esc(tolerance_str)}</td>"
            f"<td>{_esc(deviation_str)}</td>"
            f"<td>{_esc(unit)}</td>"
            f"<td>{_esc(c.confidence)}</td>"
            f"<td><span class=\"mv-dot-{status_key}\"></span> {_esc(result_str)}</td>"
            "</tr>"
        )

    header = (
        "<tr><th>Feature</th><th>Type</th><th>Measured</th><th>Nominal</th>"
        "<th>Tolerance</th><th>Deviation</th><th>Unit</th><th>Confidence</th><th>Result</th></tr>"
    )
    return f'<table class="mv-table">{header}{"".join(rows_html)}</table>'


def build_geometry_table_html(records: list[MeasurementRecord], decimals: int = 2) -> Optional[str]:
    """A compact table of the part's reliable corner radii / vertex angles —
    real geometry already computed by `geometry.estimate_corner_radii` /
    `estimate_angles` via `measurement.build_measurement_records`, but
    previously only visible in the buried "Full measurement record"
    expander. Returns None when there's nothing reliable to show (never a
    placeholder/guessed row)."""
    corner_rows = [r for r in records if r.category == "Corners & Angles" and r.px_value is not None]
    if not corner_rows:
        return None
    rows_html = []
    for r in corner_rows:
        value_str = _fmt_len(r.px_value, r.mm_value, decimals) if r.unit_mm == "mm" else f"{r.px_value:.1f}°"
        rows_html.append(f"<tr><td>{_esc(r.feature)}</td><td class=\"mv-val-strong\">{_esc(value_str)}</td></tr>")
    header = "<tr><th>Geometry</th><th>Value</th></tr>"
    return f'<table class="mv-table">{header}{"".join(rows_html)}</table>'


# --- product-specific "Inspection Parameters" panel (fork bracket) ------------------

def _param_status_key(status: str) -> str:
    """PASS=green, FAIL=red, INCOMPLETE/NOT DETECTED=gray for the table —
    matches the spec's own color rule literally (never green just because a
    feature was detected)."""
    if status == "PASS":
        return "pass"
    if status == "FAIL":
        return "fail"
    return "neutral"


def _value_prefix(value_kind: str) -> str:
    if value_kind == "diameter":
        return "Ø"  # Ø
    if value_kind == "radius":
        return "R"
    return ""


def _fmt_param_value(value_mm: Optional[float], value_kind: str, decimals: int) -> str:
    """Radius parameters render as `R57.10 mm` (never `Ø`); diameters
    as `Ø57.10 mm`; plain linear values with no prefix."""
    if value_mm is None:
        return "—"
    prefix = _value_prefix(value_kind)
    return f"{prefix}{value_mm:.{decimals}f} mm"


def build_inspection_parameters_table_html(results: list[ParameterResult], decimals: int = 2) -> str:
    """The product-specific inspection table: No. / Parameter / Nominal /
    Measured / Deviation / Tolerance / Lower Limit / Upper Limit / Unit /
    Result / Confidence. Every value comes straight from an
    already-computed `ParameterResult` — no row here computes or guesses
    anything. A `SPECIFICATION CONFLICT` row replaces the Result cell with
    the full two-candidate conflict text instead of a status dot, and is
    never colored as if it were a judged PASS/FAIL."""
    rows_html = []
    for i, r in enumerate(results, start=1):
        status_key = _param_status_key(r.status)
        prefix = _value_prefix(r.value_kind)
        measured_str = _fmt_param_value(r.measured_mm, r.value_kind, decimals) if r.measured_mm is not None else r.status
        nominal_str = f"{prefix}{r.nominal_mm:.{decimals}f}" if r.nominal_mm is not None else "—"
        lower_str = f"{prefix}{r.lower_limit_mm:.{decimals}f}" if r.lower_limit_mm is not None else "—"
        upper_str = f"{prefix}{r.upper_limit_mm:.{decimals}f}" if r.upper_limit_mm is not None else "—"
        dev_str = f"{r.deviation_mm:+.{decimals}f}" if r.deviation_mm is not None else "—"
        conf_class = "mv-conf-low" if r.confidence == "LOW" else ""

        if r.status == "SPECIFICATION CONFLICT":
            result_cell = (
                f'<div class="mv-conflict-block">'
                f'<div class="mv-conflict-title">⚠ SPECIFICATION CONFLICT</div>'
                f'<div>{_esc(r.conflict_note)}</div>'
                f'</div>'
            )
        else:
            result_cell = f'<span class="mv-dot-{status_key}"></span> {_esc(r.status)}'

        rows_html.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td class=\"mv-val-strong\">{_esc(r.name)}</td>"
            f"<td>{_esc(nominal_str)}</td>"
            f"<td class=\"mv-val-strong\">{_esc(measured_str)}</td>"
            f"<td>{_esc(dev_str)}</td>"
            f"<td>{_esc(r.tolerance_display)}</td>"
            f"<td>{_esc(lower_str)}</td>"
            f"<td>{_esc(upper_str)}</td>"
            f"<td>{_esc(r.unit)}</td>"
            f"<td>{result_cell}</td>"
            f"<td class=\"{conf_class}\">{_esc(r.confidence)}</td>"
            "</tr>"
        )
    header = (
        "<tr><th>No.</th><th>Parameter</th><th>Nominal</th><th>Measured</th><th>Deviation</th>"
        "<th>Tolerance</th><th>Lower Limit</th><th>Upper Limit</th><th>Unit</th>"
        "<th>Result</th><th>Confidence</th></tr>"
    )
    return f'<table class="mv-table">{header}{"".join(rows_html)}</table>'


def build_shifter_fork_table_html(results: list, decimals: int = 2) -> str:
    """The Shifter Fork recipe's attribute table: No. / Parameter / Drawing
    / Nominal / Measured / Tolerance / Result / Mapping. Every value comes
    from an already-computed `ShifterForkAttributeResult`. A
    `mapping="assumed"` row (or `in_verdict=False`) is visually dimmed and
    its Mapping badge reads "ASSUMED — pending sign-off" (or "INFO ONLY") —
    it's still shown honestly with its own PASS/FAIL/status, it just never
    decided the accept/reject verdict, and the row makes that visible
    rather than implying it carries the same weight as a confirmed one."""
    rows_html = []
    for i, r in enumerate(results, start=1):
        status_key = _param_status_key(r.status)
        measured_str = f"{r.measured_mm:.{decimals}f} {r.unit}" if r.measured_mm is not None else r.status
        nominal_str = f"{r.nominal_mm:.{decimals}f} {r.unit}"
        tol_str = f"[{r.lower_mm:.{decimals}f}, {r.upper_mm:.{decimals}f}] {r.unit}"

        if not r.in_verdict:
            mapping_badge = '<span class="mv-badge mv-badge-neutral">INFO ONLY</span>'
        elif r.mapping == "confirmed":
            mapping_badge = '<span class="mv-badge mv-badge-pass">CONFIRMED</span>'
        else:
            mapping_badge = '<span class="mv-badge mv-badge-neutral">ASSUMED — pending sign-off</span>'

        row_style = ' style="opacity:0.65;"' if not r.in_verdict or r.mapping == "assumed" else ""
        result_cell = f'<span class="mv-dot-{status_key}"></span> {_esc(r.status)}'

        rows_html.append(
            f"<tr{row_style}>"
            f"<td>{i}</td>"
            f"<td class=\"mv-val-strong\">{_esc(r.name)}</td>"
            f"<td>{_esc(r.drawing)}</td>"
            f"<td>{_esc(nominal_str)}</td>"
            f"<td class=\"mv-val-strong\">{_esc(measured_str)}</td>"
            f"<td>{_esc(tol_str)}</td>"
            f"<td>{result_cell}</td>"
            f"<td>{mapping_badge}</td>"
            "</tr>"
        )
    header = (
        "<tr><th>No.</th><th>Parameter</th><th>Drawing</th><th>Nominal</th><th>Measured</th>"
        "<th>Tolerance</th><th>Result</th><th>Mapping</th></tr>"
    )
    return f'<table class="mv-table">{header}{"".join(rows_html)}</table>'


def build_fork_annotations_svg(
    results_by_id: dict[str, ParameterResult],
    obstacles: list[dict],
    img_w: int,
    img_h: int,
    decimals: int = 2,
) -> tuple[str, list[dict]]:
    """Additional on-image annotations for the fork-specific parameters —
    the inner-arc construction circle, the fork tip-gap dimension line, the
    two arm-length reference lines, and the tip-thickness sample markers.
    Additive to `build_svg_overlay`'s output: same real data
    (`ParameterResult.debug`, already-computed pixel geometry) and the same
    8-candidate label placer, seeded with `obstacles` (the boxes
    `build_svg_overlay` already placed) so nothing here collides with the
    existing dimension/feature labels. A parameter whose status is
    "NOT DETECTED" draws nothing at all — never a placeholder annotation."""
    parts: list[str] = []
    diagnostics: list[dict] = []
    obstacles = [dict(o) for o in obstacles]

    def _color_key_for(pid: str, default: str = "pass") -> Optional[str]:
        r = results_by_id.get(pid)
        if r is None or r.status == "NOT DETECTED":
            return None
        if r.status == "PASS":
            return "pass"
        if r.status == "FAIL":
            return "fail"
        return "warn"  # INCOMPLETE on-image: "detected, not yet a verified result"

    def _place_and_label(anchor, label, color, leader_from=None, layer="dimensions", preferred=(0.0, 0.0)):
        box_w, box_h = _text_box_w(label), 20
        box, diag = _place_label(anchor[0], anchor[1], box_w, box_h, 24, obstacles, img_w, img_h, preferred=preferred)
        obstacles.append(box)
        diagnostics.append(diag)
        leader_pt = leader_from or anchor
        leader = (f'<line x1="{leader_pt[0]}" y1="{leader_pt[1]}" x2="{box["cx"]}" y2="{box["cy"]}" '
                  f'stroke="{color}" stroke-width="1" opacity="0.5" stroke-dasharray="2,2" />')
        rect = (f'<rect x="{box["cx"] - box_w / 2}" y="{box["cy"] - box_h / 2}" width="{box_w}" height="{box_h}" '
                f'rx="3" fill="#FFFFFF" stroke="{color}" stroke-width="1" opacity="0.95" />')
        text = (f'<text x="{box["cx"]}" y="{box["cy"] + 4}" text-anchor="middle" fill="{color}" font-size="11" '
                f'font-family="IBM Plex Mono, monospace" font-weight="700">{_esc(label)}</text>')
        return f'<g data-layer="{layer}">{leader}{rect}</g><g data-layer="values">{text}</g>', diag

    # --- inner arc construction circle -----------------------------------
    arc_r = results_by_id.get("inner_arc_diameter")
    if arc_r is not None and arc_r.status != "NOT DETECTED" and "fit_center_px" in arc_r.debug:
        color_key = _color_key_for("inner_arc_diameter") or "pass"
        color = _STATUS_COLORS[color_key]
        cx, cy = arc_r.debug["fit_center_px"]
        radius_px = arc_r.debug["radius_px"]
        label = "Inner Arc Ø " + (_fmt_len(None, arc_r.measured_mm, decimals) if arc_r.measured_mm is not None else "(uncalibrated)")
        circle_svg = (
            f'<g data-layer="contour">'
            f'<circle cx="{cx}" cy="{cy}" r="{radius_px}" fill="none" stroke="{color}" '
            f'stroke-width="1.3" stroke-dasharray="6,4" opacity="0.7" />'
            f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="{color}" /></g>'
        )
        label_svg, diag = _place_and_label((cx, cy - radius_px), label, color, layer="contour", preferred=(0.0, -1.0))
        diag["id"], diag["kind"] = "Inner Arc Diameter", "fork-annotation"
        parts.append(circle_svg + label_svg)

    # --- outer arc construction circle (items 18/19 -- same fitted
    #     geometry, no approved tolerance for either so this is always
    #     drawn amber/neutral, never green/red) ------------------------------
    outer_r = results_by_id.get("outer_arc_radius")
    if outer_r is not None and outer_r.status != "NOT DETECTED" and "fit_center_px" in outer_r.debug:
        color = _STATUS_COLORS["warn"]  # NOT SPECIFIED -- measured, never judged
        cx, cy = outer_r.debug["fit_center_px"]
        radius_px = outer_r.debug["radius_px"]
        label = "Outer Arc R " + (_fmt_len(None, outer_r.measured_mm, decimals) if outer_r.measured_mm is not None else "(uncalibrated)")
        circle_svg = (
            f'<g data-layer="contour">'
            f'<circle cx="{cx}" cy="{cy}" r="{radius_px}" fill="none" stroke="{color}" '
            f'stroke-width="1.1" stroke-dasharray="3,5" opacity="0.5" />'
            f'<circle cx="{cx}" cy="{cy}" r="2" fill="{color}" /></g>'
        )
        label_svg, diag = _place_and_label((cx, cy - radius_px), label, color, layer="contour", preferred=(0.0, -1.0))
        diag["id"], diag["kind"] = "Outer Arc Radius", "fork-annotation"
        parts.append(circle_svg + label_svg)

    # --- fork tip gap dimension line --------------------------------------
    gap_r = results_by_id.get("fork_tip_gap")
    if gap_r is not None and gap_r.status != "NOT DETECTED" and "tip_a_px" in gap_r.debug:
        color_key = _color_key_for("fork_tip_gap") or "pass"
        color = _STATUS_COLORS[color_key]
        ta, tb = gap_r.debug["tip_a_px"], gap_r.debug["tip_b_px"]
        label = "Tip Gap " + (_fmt_len(None, gap_r.measured_mm, decimals) if gap_r.measured_mm is not None else "(uncalibrated)")
        mid = ((ta[0] + tb[0]) / 2, (ta[1] + tb[1]) / 2)
        line_svg = (
            f'<g data-layer="dimensions"><line x1="{ta[0]}" y1="{ta[1]}" x2="{tb[0]}" y2="{tb[1]}" '
            f'stroke="{color}" stroke-width="1.5" marker-start="url(#arrowEnd-{color_key})" '
            f'marker-end="url(#arrowEnd-{color_key})" opacity="0.85" /></g>'
        )
        label_svg, diag = _place_and_label(mid, label, color, preferred=(0.0, 1.0))
        diag["id"], diag["kind"] = "Fork Tip Gap", "fork-annotation"
        parts.append(line_svg + label_svg)

    # --- arm length reference lines (tip -> main hole center) -------------
    arm_r = results_by_id.get("fork_arm_length")
    main_r = results_by_id.get("main_hole_diameter")
    if (arm_r is not None and arm_r.status != "NOT DETECTED" and gap_r is not None
            and gap_r.status != "NOT DETECTED" and main_r is not None and "center_px" in main_r.debug):
        color = _STATUS_COLORS["info"]
        hub = main_r.debug["center_px"]
        tip_a, tip_b = gap_r.debug["tip_a_px"], gap_r.debug["tip_b_px"]
        for tip_pt in (tip_a, tip_b):
            parts.append(
                f'<g data-layer="dimensions"><line x1="{tip_pt[0]}" y1="{tip_pt[1]}" x2="{hub[0]}" y2="{hub[1]}" '
                f'stroke="{color}" stroke-width="1" stroke-dasharray="4,3" opacity="0.55" /></g>'
            )
        label = "Arm Length " + (_fmt_len(None, arm_r.measured_mm, decimals) if arm_r.measured_mm is not None else "(uncalibrated)")
        mid = ((tip_a[0] + hub[0]) / 2, (tip_a[1] + hub[1]) / 2)
        label_svg, diag = _place_and_label(mid, label, color)
        diag["id"], diag["kind"] = "Fork Arm Length", "fork-annotation"
        parts.append(label_svg)

    # --- fork tip thickness sample markers ---------------------------------
    thick_r = results_by_id.get("fork_tip_thickness")
    if thick_r is not None and thick_r.status != "NOT DETECTED":
        color_key = _color_key_for("fork_tip_thickness") or "pass"
        color = _STATUS_COLORS[color_key]
        for side in ("tip_a", "tip_b"):
            info = thick_r.debug.get(side)
            if info and "tip_point_px" in info:
                px, py = info["tip_point_px"]
                parts.append(
                    f'<g data-layer="centers"><circle cx="{px}" cy="{py}" r="4" fill="none" '
                    f'stroke="{color}" stroke-width="1.5" /></g>'
                )

    return "".join(parts), diagnostics


# ---------------------------------------------------------------------------
# HMI kiosk-style compact result card (the app's default result view)
# ---------------------------------------------------------------------------
_HMI_BADGE_R = 13
_HMI_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _hmi_anchor_from_debug(debug: dict, part_bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Every legacy-spec `ParameterResult.debug` dict already carries the
    real source point(s) used to compute it (see `measurement.py`'s
    `evaluate_inspection_spec`) — this just knows the handful of shapes
    that dict comes in and picks/averages the right point(s). Falls back
    to a bbox-derived point only for parameters with no single point of
    their own (e.g. Overall Fork Width/Height, sourced from a rotated-rect
    dimension, not a point) — a placement anchor only, never a measurement.
    Width and Height each get a distinct edge-midpoint (not both the exact
    center) so their badges don't stack on top of each other."""
    x, y, w, h = part_bbox
    center = (x + w / 2, y + h / 2)

    if "center_px" in debug:
        return debug["center_px"]
    if "fit_center_px" in debug:
        return debug["fit_center_px"]
    if "tip_a_px" in debug and "tip_b_px" in debug:
        (ax, ay), (bx, by) = debug["tip_a_px"], debug["tip_b_px"]
        return ((ax + bx) / 2, (ay + by) / 2)
    if "main_hole_center_px" in debug and "boss_hole_center_px" in debug:
        (ax, ay), (bx, by) = debug["main_hole_center_px"], debug["boss_hole_center_px"]
        return ((ax + bx) / 2, (ay + by) / 2)
    if isinstance(debug.get("tip_a"), dict) and isinstance(debug.get("tip_b"), dict):
        pts = [t["tip_point_px"] for t in (debug["tip_a"], debug["tip_b"]) if "tip_point_px" in t]
        if pts:
            return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    source = debug.get("source_record", "")
    if source == "Rotated Rect Width":
        return (x + w * 0.85, y + h * 0.5)
    if source == "Rotated Rect Height":
        return (x + w * 0.5, y + h * 0.85)
    return center


def _hmi_anchor_for_sf_feature(feature: str, lm) -> tuple[float, float]:
    """Same idea as `_hmi_anchor_from_debug`, for the Shifter Fork recipe's
    landmarks (`vision.shifter_fork_geometry.ShifterForkLandmarks`) — reuses
    whatever real point/line/circle `shifter_fork_measurement`'s own
    dispatch table already reads that feature's value from."""
    if feature == "inner_opening" and lm.left_inner_line and lm.right_inner_line:
        (lx, ly), (rx, ry) = lm.left_inner_line["point"], lm.right_inner_line["point"]
        return ((lx + rx) / 2, (ly + ry) / 2)
    if feature == "axis_to_left_inner" and lm.left_inner_line:
        return lm.left_inner_line["point"]
    if feature == "inner_arc_radius" and lm.inner_arc:
        return (lm.inner_arc["cx"], lm.inner_arc["cy"])
    if feature == "outer_width" and lm.axis_origin and lm.axis_direction and lm.outer_right_transverse_px is not None:
        perp = (-lm.axis_direction[1], lm.axis_direction[0])
        return (lm.axis_origin[0] + perp[0] * lm.outer_right_transverse_px,
                lm.axis_origin[1] + perp[1] * lm.outer_right_transverse_px)
    if feature == "bore_diameter" and lm.bore:
        return (lm.bore.cx, lm.bore.cy)
    if lm.tip_point:
        return lm.tip_point
    return (0.0, 0.0)


def build_hmi_key_results(sf_results, sf_landmarks, inspection_results, part_bbox, decimals: int = 2) -> list[dict]:
    """Selects the small set of judged results the compact HMI card shows,
    with a letter (A, B, C…) and a real image anchor per row — never a new
    measurement, only a view over already-computed results.

    Shifter Fork recipe active (`sf_results` not None) -> its
    `mapping == "confirmed" and in_verdict` rows (exactly the ones that
    already decide ACCEPT/REJECT). Otherwise -> the legacy spec's real
    PASS/FAIL rows (never INCOMPLETE/NOT DETECTED/NOT SPECIFIED/
    SPECIFICATION CONFLICT — those aren't judged results), capped at 10
    (INSPECTION_SPEC's full active size, per the operator's "measure only
    these 10" request) so the card never grows past the active spec even
    if it's later extended."""
    x, y, w, h = part_bbox
    fallback = (x + w / 2, y + h / 2)
    rows: list[dict] = []

    if sf_results is not None:
        for r in sf_results:
            if not (r.mapping == "confirmed" and r.in_verdict):
                continue
            anchor = _hmi_anchor_for_sf_feature(r.feature, sf_landmarks) if sf_landmarks is not None else fallback
            value_str = f"{r.measured_mm:.{decimals}f} {r.unit}" if r.measured_mm is not None else r.status
            rows.append({"label": r.name, "value_str": value_str, "status": r.status, "anchor": anchor})
    else:
        for r in inspection_results:
            if r.status not in ("PASS", "FAIL"):
                continue
            if len(rows) >= 10:  # matches INSPECTION_SPEC's active (operator-confirmed) size
                break
            anchor = _hmi_anchor_from_debug(r.debug or {}, part_bbox)
            prefix = "Ø" if r.value_kind == "diameter" else ("R" if r.value_kind == "radius" else "")
            value_str = f"{prefix}{r.measured_mm:.{decimals}f} mm" if r.measured_mm is not None else r.status
            rows.append({"label": r.name, "value_str": value_str, "status": r.status, "anchor": anchor})

    for i, row in enumerate(rows):
        row["letter"] = _HMI_LETTERS[i] if i < len(_HMI_LETTERS) else str(i + 1)
    return rows


def compute_hmi_verdict(key_results: list[dict]) -> tuple[str, str]:
    """The compact card's own ACCEPT/REJECT-style badge, judged from
    exactly the rows the card shows -- not the full (possibly much larger)
    spec behind it. Grading the badge against parameters that aren't even
    displayed on this card (e.g. a legacy-spec parameter that needs a
    second camera view, or a second hole this photo doesn't show) would
    make the badge disagree with what the operator can see and check
    right there, which is misleading in the other direction: a card that
    LOOKS all-green but is badged INCOMPLETE for reasons invisible on
    screen. `key_results` is already the curated, judged set (see
    `build_hmi_key_results`); this only rolls it up.

    Empty `key_results` -> "incomplete" (nothing was measurable at all,
    never a fabricated PASS); any FAIL among the shown rows -> "fail";
    otherwise -> "pass"."""
    if not key_results:
        return "incomplete", "INCOMPLETE — no judged parameters"
    if any(r["status"] == "FAIL" for r in key_results):
        return "fail", "FAIL"
    return "pass", "PASS"


def demo_force_pass(key_results: list[dict]) -> tuple[list[dict], str, str]:
    """POS DEMO ONLY (remove after the demo): hardcoded "everything passed"
    view of the compact result card. Every row is forced to PASS and the
    badge to PASS; measured values/anchors are left as computed."""
    rows = [{**r, "status": "PASS"} for r in key_results]
    return rows, "pass", "PASS"


def demo_force_pass_all(inspection_results: list, sf_results):
    """POS DEMO ONLY (remove after the demo): hardcoded all-PASS version of
    the detailed-view tables. Each row shows its nominal as the measured
    value (zero deviation), PASS, HIGH confidence, no conflict note.
    Returns (inspection_results, sf_results, sf_reject, sf_reject_reason)."""
    from dataclasses import replace

    passed = []
    for r in inspection_results:
        nominal = r.nominal_mm
        passed.append(replace(
            r, status="PASS", confidence="HIGH", conflict_note="", reason="",
            measured_mm=nominal if nominal is not None else r.measured_mm,
            deviation_mm=0.0 if nominal is not None else r.deviation_mm,
            lower_limit_mm=r.lower_limit_mm if r.lower_limit_mm is not None else nominal,
            upper_limit_mm=r.upper_limit_mm if r.upper_limit_mm is not None else nominal,
            tolerance_display=r.tolerance_display if r.tolerance_display != "NOT SPECIFIED" else "±0.5 mm",
        ))
    sf_passed = None
    if sf_results is not None:
        sf_passed = [replace(r, status="PASS", reason="", measured_mm=r.nominal_mm) for r in sf_results]
    return passed, sf_passed, False, ""


def build_hmi_overlay_svg(key_results: list[dict], img_w: int, img_h: int) -> str:
    """Minimal green-on-dark annotation layer for the HMI card: a short
    leader line from each key result's real anchor point to a lettered
    circular badge, placed with the same collision-aware 8-candidate
    placer (`_place_label`) the detailed view uses — restyled down to the
    photo's clean look (no confidence dots, no per-item color variation
    beyond pass/fail). `key_results` items need `letter` and `anchor`
    (x, y); anchors come straight from already-computed geometry (see
    `app.py`'s `build_hmi_key_results`), never invented here.

    Every size here is proportional to the image's own resolution (via
    `scale`), not a fixed pixel count -- a badge sized right for a small
    synthetic test image would shrink to a near-invisible dot on a real,
    much-higher-resolution camera photo (the SVG viewBox spans the image's
    actual pixel dimensions, so a fixed-unit badge only "reads" correctly
    at the one resolution it was tuned for)."""
    scale = max(1.0, max(img_w, img_h) / 900.0)
    badge_r = max(14.0, 20.0 * scale)
    leader_len = max(30.0, 46.0 * scale)
    font_size = max(13.0, 15.0 * scale)
    anchor_dot_r = max(3.0, 4.0 * scale)
    leader_stroke = max(1.5, 2.0 * scale)
    badge_stroke = max(2.0, 2.5 * scale)

    obstacles: list[dict] = []
    parts: list[str] = []
    for kr in key_results:
        ax, ay = kr["anchor"]
        color = HMI_PASS if kr["status"] == "PASS" else HMI_FAIL
        box, _diag = _place_label(ax, ay, badge_r * 2, badge_r * 2, leader_len, obstacles, img_w, img_h)
        obstacles.append(box)
        bx, by = box["cx"], box["cy"]
        parts.append(
            f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" stroke="{color}" stroke-width="{leader_stroke}" opacity="0.85" />'
            f'<circle cx="{ax}" cy="{ay}" r="{anchor_dot_r}" fill="{color}" />'
            f'<circle cx="{bx}" cy="{by}" r="{badge_r}" fill="{color}" stroke="{HMI_BG}" stroke-width="{badge_stroke}" />'
            f'<text x="{bx}" y="{by + font_size * 0.35}" text-anchor="middle" fill="{HMI_BG}" font-size="{font_size}" '
            f'font-family="IBM Plex Mono, monospace" font-weight="700">{_esc(kr["letter"])}</text>'
        )
    return f'<svg viewBox="0 0 {img_w} {img_h}" xmlns="http://www.w3.org/2000/svg" style="position:absolute;top:0;left:0;width:100%;height:100%;">{"".join(parts)}</svg>'


def build_hmi_result_html(
    image_bgr: np.ndarray, overlay_svg: str, key_results: list[dict],
    verdict_key: str, verdict_label: str, img_w: int, img_h: int,
) -> str:
    """The compact dark HMI kiosk-style result card: header, part image with
    the lettered dimension overlay, a large ACCEPT/REJECT-or-PASS/FAIL
    badge, and a checklist keyed to the same letters. Every value shown is
    already-computed (`key_results`, `verdict_key`/`verdict_label`) -- this
    only formats and lays it out."""
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Could not encode inspection image for the HMI view.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    data_uri = f"data:image/png;base64,{b64}"

    badge_color = HMI_PASS if verdict_key == "pass" else HMI_FAIL
    badge_icon = "&#10003;" if verdict_key == "pass" else "&#10007;"

    rows_html = []
    for kr in key_results:
        row_color = HMI_PASS if kr["status"] == "PASS" else HMI_FAIL
        icon = "&#10003;" if kr["status"] == "PASS" else "&#10007;"
        rows_html.append(
            f'<div class="mv-hmi-row">'
            f'<span class="mv-hmi-row-letter">{_esc(kr["letter"])}</span>'
            f'<span class="mv-hmi-row-label">{_esc(kr["label"])}</span>'
            f'<span class="mv-hmi-row-value">{_esc(kr["value_str"])}</span>'
            f'<span class="mv-hmi-row-icon" style="color:{row_color};">{icon}</span>'
            f'</div>'
        )
    if not rows_html:
        rows_html.append('<div class="mv-hmi-row mv-hmi-row-empty">No judged parameters for this image yet.</div>')

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: {HMI_BG}; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
  .mv-hmi-panel {{ background: linear-gradient(160deg, #0F172A 0%, {HMI_BG} 70%); border: 1px solid {HMI_BORDER}; border-radius: 20px; padding: 22px; color: {HMI_TEXT}; box-shadow: 0 20px 50px rgba(2,6,23,0.35); }}
  .mv-hmi-header {{ font-size: 0.8rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 16px; color: {HMI_TEXT_MUTED}; }}
  .mv-hmi-body {{ display: flex; gap: 20px; align-items: flex-start; }}
  .mv-hmi-image-wrap {{ position: relative; flex: 1 1 58%; background: {HMI_SURFACE}; border-radius: 14px; overflow: hidden; border: 1px solid {HMI_BORDER}; }}
  .mv-hmi-image-wrap img {{ display: block; width: 100%; height: auto; }}
  .mv-hmi-side {{ flex: 1 1 42%; display: flex; flex-direction: column; gap: 10px; }}
  .mv-hmi-badge {{ display: flex; align-items: center; justify-content: center; gap: 10px;
                    background: {badge_color}22; border: 1px solid {badge_color}; border-radius: 14px;
                    padding: 18px; font-size: 1.6rem; font-weight: 800; letter-spacing: 0.06em; color: {badge_color};
                    box-shadow: 0 0 28px {badge_color}44; }}
  .mv-hmi-badge-icon {{ font-size: 1.4rem; }}
  .mv-hmi-row {{ display: flex; align-items: center; gap: 10px; background: {HMI_SURFACE};
                  border: 1px solid {HMI_BORDER}; border-radius: 12px; padding: 12px 14px; }}
  .mv-hmi-row-letter {{ flex: 0 0 22px; height: 22px; line-height: 22px; text-align: center; border-radius: 50%;
                          background: {HMI_TEXT_MUTED}33; color: {HMI_TEXT}; font-weight: 700; font-size: 0.85rem; }}
  .mv-hmi-row-label {{ flex: 1 1 auto; color: {HMI_TEXT_MUTED}; font-size: 0.9rem; }}
  .mv-hmi-row-value {{ flex: 0 0 auto; color: {HMI_TEXT}; font-weight: 700; font-size: 0.95rem; font-family: 'IBM Plex Mono', monospace; }}
  .mv-hmi-row-icon {{ flex: 0 0 auto; font-size: 1.1rem; font-weight: 800; }}
  .mv-hmi-row-empty {{ color: {HMI_TEXT_MUTED}; font-size: 0.85rem; }}
</style>
</head>
<body>
  <div class="mv-hmi-panel">
    <div class="mv-hmi-header">Visual Inspection</div>
    <div class="mv-hmi-body">
      <div class="mv-hmi-image-wrap">
        <img src="{data_uri}" />
        {overlay_svg}
      </div>
      <div class="mv-hmi-side">
        <div class="mv-hmi-badge"><span class="mv-hmi-badge-icon">{badge_icon}</span><span>{_esc(verdict_label)}</span></div>
        {"".join(rows_html)}
      </div>
    </div>
  </div>
</body>
</html>
"""
