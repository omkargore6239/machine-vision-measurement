
# Machine Vision Measurement Prototype

A beginner-friendly Python + OpenCV + Streamlit application for learning industrial
machine-vision measurement.

## What it does

1. Upload a JPG/PNG/BMP/WEBP part image.
2. Attempts to find the main visible part.
3. Measures:
   - Overall width in pixels
   - Overall height in pixels
   - Area
   - Perimeter
   - Detectable circular features
4. Optional calibration converts pixels to millimetres.

## Important

A normal photo does NOT contain enough information to know real-world millimetres.
You must provide a known reference length or use a proper calibration target.

For industrial use, use:
- fixed camera position
- fixed lens
- controlled lighting
- stable part position
- calibration target
- validation/repeatability testing

The segmentation in this project is a prototype and is not guaranteed to work
on every arbitrary background or every part shape.

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

The browser will open the application.

## Calibration example

Suppose a reference object in the image is physically 50 mm and measures 500 pixels.

Enter:
- Known reference length = 50 mm
- Reference length in image = 500 pixels

The application uses:

`pixels per mm = 500 / 50 = 10 px/mm`

Then a 520-pixel width becomes:

`520 / 10 = 52 mm`

## Next development stages

1. Better industrial segmentation
2. Perspective correction
3. Camera calibration with checkerboard/dot-grid
4. Robust hole/edge measurement
5. Tolerance table and PASS/FAIL
6. Image/data logging
7. PLC communication
8. Camera trigger and conveyor integration
