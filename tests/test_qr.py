"""QR scaffolding is independent and honest: it must report a real decode
when one is present, and NOT PRESENT (never a guess or a fabricated code)
when it isn't -- e.g. on the plain synthetic part fixtures used everywhere
else in this suite, none of which contain a QR code."""
import cv2
import numpy as np
import pytest

from tests.fixtures import make_axis_aligned_part
from vision import qr


def _qr_image(text: str, canvas_size: int = 500, qr_size: int = 300, margin: int = 100) -> np.ndarray:
    encoder = cv2.QRCodeEncoder.create()
    modules = encoder.encode(text)  # already 0/255, not 0/1
    qr_img = cv2.resize(modules, (qr_size, qr_size), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
    canvas[margin:margin + qr_size, margin:margin + qr_size] = qr_img
    return canvas


def test_detect_qr_decodes_a_real_code():
    gray = _qr_image("PART-12345")
    result = qr.detect_qr(gray)
    assert result["status"] == qr.QR_STATUS_DETECTED
    assert result["data"] == "PART-12345"
    assert result["points"] is not None


def test_detect_qr_reports_not_present_honestly():
    img, _ = make_axis_aligned_part()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = qr.detect_qr(gray)
    assert result["status"] == qr.QR_STATUS_NOT_PRESENT
    assert result["data"] is None


def test_detect_qr_never_raises_on_blank_image():
    gray = np.zeros((50, 50), dtype=np.uint8)
    result = qr.detect_qr(gray)
    assert result["status"] in (qr.QR_STATUS_NOT_PRESENT, qr.QR_STATUS_ERROR)
