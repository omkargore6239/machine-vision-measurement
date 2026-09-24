# Machine Vision Measurement

A Python + OpenCV + Streamlit application for measuring parts from photographs,
with **calibrated millimetre measurement** as a first-class, explicit feature —
never an assumption.

## The core rule

> A normal photograph has no inherent physical scale. **This app never assumes
> 1 pixel = 1 mm.** Every millimetre value shown anywhere in the app traces
> back to an explicit calibration profile you created. If no calibration is
> active, you will see pixel measurements and the message
> **"MM measurement unavailable — calibration required"** — never a fabricated
> mm number.

Accurate real-world measurement from a photograph additionally requires **one**
of:
- a detected reference object of known size, on the same plane as the part
  (fast, but assumes a roughly perpendicular camera and no perspective skew)
- a checkerboard camera calibration (optionally with lens-distortion correction,
  if you supply 3+ images from different poses)
- a 4-point perspective correction against a known-size rectangle in the
  part's plane

None of these — nor this app in general — provide CMM/industrial-metrology-grade
accuracy. Real accuracy also depends on camera resolution, lens distortion,
lighting, focus, perspective, calibration quality, edge quality, and part
positioning. Treat every measurement as an estimate with the stated confidence
level, not a certified dimension.

## What it does

1. **Calibration** — create, save, load, and delete calibration profiles using
   any of the three methods above. Each profile records an `mm/pixel` scale,
   a confidence level (HIGH/MEDIUM/LOW) derived from actual detection quality
   (not asserted), and the evidence behind that confidence.
2. **Measure** — upload one or more part images. For each image the app:
   - checks image quality (resolution, blur, contrast, exposure, clipping,
     perspective hints) and shows warnings, not silent failures
   - detects the part outline (trying several segmentation methods and
     keeping the best-scoring plausible one, then tightening the boundary
     with a local re-threshold refinement pass)
   - detects circular holes (Hough transform + contour/circularity
     cross-check; a circle is only reported when the geometry actually
     supports it)
   - measures bounding-box and rotated-rect width/height, area, perimeter,
     hole diameters/radii/centers, hole-to-hole and hole-to-edge distances,
     corner radii (only when the fit is tight — otherwise it says so),
     and vertex angles
   - converts every measurement to mm using the active calibration profile,
     or leaves mm blank if none is active
   - lets you enter optional nominal + tolerance values for any feature to
     get a PASS/FAIL check
   - draws an annotated image (outline, bounding boxes, circles, dimension
     text, coordinate axes, calibration status)
3. **Export** — download all results as CSV, or a per-image text report with
   calibration info, measurements, quality warnings, and PASS/FAIL results.

## Project structure

```
app.py                  Streamlit UI — orchestrates the packages below only
vision/
    types.py             Shared dataclasses (CalibrationProfile, DetectedPart, ...)
    preprocessing.py     Image decode/resize + quality metrics (blur, contrast, exposure)
    validation.py        Turns quality metrics into warnings
    segmentation.py       Part detection (multi-method + local refinement)
    calibration.py         Known-dimension / checkerboard / perspective calibration + persistence
    geometry.py             Bounding boxes, circles, distances, corner radius, angles
    measurement.py         Assembles the measurement table (the only place px -> mm happens)
    annotations.py         Draws the overlay image
utils/
    export.py             CSV + text report builders
calibration_profiles/     Saved calibration profiles (JSON) — see note below
tests/                    pytest suite using synthetic images with known ground truth
```

## Install

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```
Linux/macOS:
```bash
source .venv/bin/activate
```

Then:
```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Testing

There's no physical camera or part available for automated testing, so
correctness is verified against synthetically generated images with exactly
known ground truth (see `tests/fixtures.py`) rather than real photos:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The suite checks: detected dimensions/circle sizes match ground truth within
tolerance; calibration arithmetic (`mm_per_pixel = known_mm / measured_px`) is
exact; **no measurement ever produces an mm value without an explicit
calibration profile**; and tolerance PASS/FAIL logic is correct at the
boundary. This validates the measurement *logic*; it is not a substitute for
testing with real camera photos through the browser UI.

## Calibration walkthrough (known-dimension)

Suppose a reference object in the image is physically 50 mm wide and the app
detects its bounding box as 300 px wide:

```
mm_per_pixel = 50 / 300 = 0.16667 mm/px
```

A part measuring 500 px wide is then reported as `500 x 0.16667 = 83.33 mm`.
This is only reliable when the camera is roughly perpendicular to the plane
containing both the reference object and the part — for skewed setups, use
the perspective (4-point) or checkerboard method instead.

## Notes on calibration persistence

Calibration profiles are saved as JSON files under `calibration_profiles/`.
On most local setups this persists across restarts. **On ephemeral hosting
(e.g. Streamlit Community Cloud), this directory does not survive a redeploy
or app restart** — recreate calibration profiles after such an event, or
adapt the app to an external store if you need durable calibration.

## Deployment

Uses `opencv-python-headless` (not `opencv-python`) for compatibility with
Streamlit Community Cloud, which lacks the system libGL dependency the full
`opencv-python` package needs.

## Limitations (read before trusting a measurement)

- Segmentation is a heuristic multi-method prototype, not a certified
  industrial vision pipeline — it can fail on cluttered backgrounds, low
  contrast, or unusual part shapes. Quality warnings flag likely problems,
  but the absence of a warning is not a correctness guarantee.
- Confidence labels (HIGH/MEDIUM/LOW) are derived from measurable evidence
  (rectangularity, corner-detection count, reprojection error, circularity)
  but are still heuristic thresholds, not statistical guarantees.
- Corner radius and angle detection are only reported when the local fit is
  tight; otherwise the app says the value could not be reliably determined
  rather than guessing.
- No arbitrary photograph — however this app is configured — can substitute
  for a calibrated camera and controlled imaging setup when true
  industrial-grade accuracy is required.
