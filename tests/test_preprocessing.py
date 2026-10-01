"""BMP/TIFF are first-class supported input formats (real camera images are
BMP, not just PNG/JPEG) — these tests prove `decode_image_bytes` reads them
losslessly, and that a large real-camera-sized image (5312x3648, matching
the real BMP dataset this was written for) decodes and resizes safely."""
import time

import cv2
import numpy as np
import pytest

from vision import preprocessing


def _random_image(h=200, w=300, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.rand(h, w, 3) * 255).astype(np.uint8)


@pytest.mark.parametrize("ext", [".bmp", ".png", ".jpg", ".tif", ".tiff", ".webp"])
def test_decode_image_bytes_supports_every_accepted_format(ext):
    img = _random_image()
    ok, buf = cv2.imencode(ext, img)
    assert ok, f"cv2 could not encode a {ext} test fixture"
    decoded = preprocessing.decode_image_bytes(buf.tobytes())
    assert decoded.shape == img.shape
    assert decoded.dtype == img.dtype


def test_bmp_decode_is_lossless_not_routed_through_jpeg():
    # BMP must never be silently re-encoded through a lossy codec on the
    # way in -- pixel-for-pixel identical round-trip proves that.
    img = _random_image(seed=1)
    ok, buf = cv2.imencode(".bmp", img)
    assert ok
    decoded = preprocessing.decode_image_bytes(buf.tobytes())
    assert np.array_equal(decoded, img)


def test_tiff_decode_is_lossless():
    img = _random_image(seed=2)
    ok, buf = cv2.imencode(".tif", img)
    assert ok
    decoded = preprocessing.decode_image_bytes(buf.tobytes())
    assert np.array_equal(decoded, img)


def test_decode_image_bytes_rejects_corrupt_data():
    with pytest.raises(ValueError):
        preprocessing.decode_image_bytes(b"not a real image file")


def test_large_real_camera_resolution_bmp_decodes_and_resizes_safely():
    # Matches the real dataset's actual resolution (5312x3648 BMP) -- must
    # decode correctly and downscale for processing well within a couple
    # of seconds, not hang or run out of memory.
    img = _random_image(h=3648, w=5312, seed=3)
    ok, buf = cv2.imencode(".bmp", img)
    assert ok

    t0 = time.time()
    decoded = preprocessing.decode_image_bytes(buf.tobytes())
    decode_time = time.time() - t0
    assert decoded.shape == (3648, 5312, 3)
    assert decode_time < 5.0

    t0 = time.time()
    resized, scale = preprocessing.resize_for_processing(decoded)
    resize_time = time.time() - t0
    assert resize_time < 5.0
    # downscaled to the configured max_side (1800) on the long edge, aspect preserved
    assert max(resized.shape[:2]) <= 1800
    assert abs(resized.shape[1] / resized.shape[0] - 5312 / 3648) < 0.01
    assert 0 < scale < 1.0


def test_original_image_preserved_separately_from_working_copy():
    # app.py keeps `original` (as-uploaded) separate from the resized
    # working copy used for detection -- resizing must not mutate the input.
    img = _random_image(h=4000, w=3000, seed=4)
    original_copy = img.copy()
    resized, scale = preprocessing.resize_for_processing(img)
    assert np.array_equal(img, original_copy)  # input untouched
    assert resized.shape != img.shape or scale == 1.0
