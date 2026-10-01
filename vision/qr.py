"""QR-code scaffolding -- an independent, non-gating diagnostic stage.

No QR code is part of the real inspection workflow today (no part in this
POC carries one), so this never fabricates a decode. It exists so Debug
Mode can show whether one happens to be present in a photo, using
`cv2.QRCodeDetector` (already ships with opencv-python -- zero new
dependency). It is called on its own, wrapped in its own try/except, and
its result never gates or fails dimensional inspection.
"""
from __future__ import annotations

import cv2
import numpy as np

QR_STATUS_DETECTED = "DETECTED"
QR_STATUS_NOT_PRESENT = "NOT PRESENT"
QR_STATUS_ERROR = "ERROR"

_detector = cv2.QRCodeDetector()


def detect_qr(gray: np.ndarray) -> dict:
    """Attempts to decode a QR code from a grayscale image. Returns a dict,
    never a guess: `status` is one of DETECTED / NOT PRESENT / ERROR,
    `data` is the decoded payload (only when DETECTED), and `points` is the
    four corner points in image pixels (only when DETECTED)."""
    try:
        ok, decoded_info, points, _ = _detector.detectAndDecodeMulti(gray)
    except cv2.error as exc:
        return {"status": QR_STATUS_ERROR, "data": None, "points": None, "reason": str(exc)}

    if not ok or not decoded_info or not any(decoded_info):
        return {"status": QR_STATUS_NOT_PRESENT, "data": None, "points": None}

    results = []
    pts_array = points if points is not None else []
    for i, text in enumerate(decoded_info):
        if not text:
            continue
        corners = pts_array[i].tolist() if i < len(pts_array) else None
        results.append({"data": text, "points": corners})

    if not results:
        return {"status": QR_STATUS_NOT_PRESENT, "data": None, "points": None}

    return {
        "status": QR_STATUS_DETECTED,
        "data": results[0]["data"],
        "points": results[0]["points"],
        "all_codes": results,
    }
