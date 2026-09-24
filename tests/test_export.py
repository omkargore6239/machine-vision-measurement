"""Regression coverage for utils/export.py.

A previous bug: table rows used "" for a missing value in a column that was
otherwise floats (e.g. MM is blank for un-calibrated length records but a
real float for angle records, which don't need calibration). That mixed
str/float column crashes when Streamlit converts it to an Arrow table for
st.dataframe. These tests build a DataFrame the same way Streamlit does and
convert it to Arrow, so a regression fails here instead of only in the UI.
"""
from tests.fixtures import make_axis_aligned_part
from vision import calibration, geometry, measurement, segmentation
from vision.preprocessing import to_gray
from utils import export


def _build_records(with_calibration: bool):
    holes = [(320, 260, 20)]
    img, _ = make_axis_aligned_part(rect_xywh=(250, 200, 300, 200), holes=holes)
    part = segmentation.build_detected_part(img)
    gray = to_gray(img)
    circles, _ = geometry.detect_holes(gray, part.mask, part.hole_contours, part.bbox, part.area_px2)

    profile = None
    if with_calibration:
        profile = calibration.known_dimension_calibrate(img, known_mm=50.0, direction="horizontal")
    return measurement.build_measurement_records(part, circles, profile)


def _assert_arrow_serializable(rows: list[dict]) -> None:
    import pandas as pd
    import pyarrow as pa

    df = pd.DataFrame(rows)
    pa.Table.from_pandas(df)  # raises pyarrow.ArrowTypeError on a mixed-type column


def test_measurement_rows_arrow_serializable_without_calibration():
    # This is the exact scenario that previously crashed: no calibration
    # means length/area records get a blank MM, but angle records (which
    # never need calibration) still carry a real float in that same column.
    records = _build_records(with_calibration=False)
    rows = export.measurement_rows("test.png", records)
    assert any(r["Feature"] == "Rotated Rect Angle" for r in rows)
    _assert_arrow_serializable(rows)


def test_measurement_rows_arrow_serializable_with_calibration():
    records = _build_records(with_calibration=True)
    rows = export.measurement_rows("test.png", records)
    _assert_arrow_serializable(rows)


def test_rows_to_csv_has_one_line_per_row_plus_header():
    records = _build_records(with_calibration=False)
    rows = export.measurement_rows("test.png", records)
    csv_text = export.rows_to_csv(rows)
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("Image,")
    assert len(lines) == len(rows) + 1


def test_rows_to_csv_empty_input():
    assert export.rows_to_csv([]) == ""
