"""Dataset-report utility -- proves it runs the real pipeline (not a stub)
end-to-end over a folder of image files and produces both a per-image and
an aggregate report, using the synthetic fork-bracket fixture written to
disk as stand-in images (no real camera dataset was available to this
session). Also proves it reports a real failure honestly for an image the
pipeline can't process, rather than skipping or fabricating a result."""
import cv2
import numpy as np
import pytest

from tests.fixtures import make_fork_bracket_part
from utils import dataset_report


@pytest.fixture
def image_folder(tmp_path):
    img, _ = make_fork_bracket_part()
    cv2.imwrite(str(tmp_path / "part_01.bmp"), img)
    cv2.imwrite(str(tmp_path / "part_02.png"), img)
    # A genuinely undetectable image (blank canvas) -- must be reported as
    # a real failure, not silently dropped from the report.
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "blank.png"), blank)
    # Not a supported/recognized extension -- must be skipped from the scan.
    (tmp_path / "notes.txt").write_text("not an image")
    return tmp_path


def test_run_dataset_report_processes_every_supported_image(image_folder):
    report = dataset_report.run_dataset_report(image_folder)
    filenames = {r.filename for r in report["images"]}
    assert filenames == {"part_01.bmp", "part_02.png", "blank.png"}


def test_run_dataset_report_succeeds_on_real_geometry(image_folder):
    report = dataset_report.run_dataset_report(image_folder)
    by_name = {r.filename: r for r in report["images"]}

    for name in ("part_01.bmp", "part_02.png"):
        r = by_name[name]
        assert r.loaded
        assert r.part_detected
        assert r.boundary_accepted
        assert r.parameter_statuses  # inspection spec was evaluated
        assert r.orientation_confidence == "HIGH"
        assert 0.0 <= r.orientation_angle_deg < 360.0


def test_run_dataset_report_reports_failure_honestly_not_silently(image_folder):
    report = dataset_report.run_dataset_report(image_folder)
    blank = next(r for r in report["images"] if r.filename == "blank.png")
    assert blank.loaded
    assert not blank.part_detected
    assert blank.errors  # a real reason is recorded
    assert blank.parameter_statuses == {}  # never fabricated past the failed stage


def test_aggregate_counts_are_consistent_with_per_image_results(image_folder):
    report = dataset_report.run_dataset_report(image_folder)
    agg = report["aggregate"]
    assert agg["total_images"] == 3
    assert agg["successfully_loaded"] == 3
    assert agg["part_detected"] == 2
    assert agg["boundary_accepted"] == 2
    assert "main_hole_diameter" in agg["per_parameter_status_counts"]


def test_format_report_text_is_readable(image_folder):
    report = dataset_report.run_dataset_report(image_folder)
    text = dataset_report.format_report_text(report)
    assert "MACHINE VISION DATASET VALIDATION REPORT" in text
    assert "AGGREGATE SUMMARY" in text
    assert "part_01.bmp" in text
    assert "blank.png" in text
