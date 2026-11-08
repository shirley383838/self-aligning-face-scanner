"""
face_align.py  -  eye-based face alignment and region crops (Member 3)

The rest of the Member 3 code (glasses / hat / mask classifiers and ID matching)
imports align_face() and crop_region() from here, so every image goes through
exactly the same geometry.

This is the "face alignment" step of the Week 11 recognition pipeline
(detection -> alignment -> recognition): the eye landmarks from face_detect.py
are used to normalise the size and orientation of every face.

Alignment: rotate so the eyes are level, scale so the eye spacing is D0 pixels,
and move the eye midpoint to EYE_MID on a fixed CANVAS_W x CANVAS_H canvas.
The transform is a similarity (rotation + uniform scale + translation), built as
a 3x3 matrix from the Week 5 rotation matrix and applied with the Week 3
perspective warp.

Running this file:
    1. runs a self-test on a synthetic face with known eye positions
    2. aligns the frames in data/own/eyes.csv (from face_detect.py), draws the
       four crop boxes and saves a check figure to figures_member3/01_align_check.png
Look at that figure before building anything on top of the crops: if a box is
in the wrong place, change REGIONS below and re-run.
"""

import os
import csv
import math

import numpy as np
import cv2

# ---------------------------------------------------------------- figure settings
SHOW_FIGURES = False   # True also opens the figure window
FIGURE_DPI = 150
if not SHOW_FIGURES:
    import matplotlib
    matplotlib.use("Agg")   # must be set before pyplot is imported, or windows still open
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- geometry
CANVAS_W, CANVAS_H = 192, 256     # aligned face canvas (pixels)
EYE_MID = (96.0, 128.0)           # where the eye midpoint lands
D0 = 64.0                         # eye spacing after alignment (pixels)

# Crop boxes in units of D0, measured from EYE_MID: (x_min, x_max, y_min, y_max).
# Image y points DOWN, so negative y is above the eyes.
REGIONS = {
    "eyes":  (-1.0, 1.0, -0.4, 0.4),   # eye band  -> glasses / sunglasses
    "hat":   (-1.3, 1.3, -1.9, -0.6),  # forehead and above -> hat / cap
    "lower": (-0.9, 0.9,  0.6, 1.9),   # nose tip to chin -> mask
    "face":  (-1.1, 1.1, -0.9, 1.9),   # whole face -> ID matching
}
REGION_COLOURS = {"eyes": (0, 255, 0), "hat": (255, 0, 0), "lower": (0, 0, 255), "face": (0, 255, 255)}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "own", "raw")
EYES_CSV = os.path.join(BASE_DIR, "data", "own", "eyes.csv")
FIGURE_DIR = os.path.join(BASE_DIR, "figures_member3")


def similarity_matrix(left_eye, right_eye, eye_mid=EYE_MID, d0=D0):
    """
    3x3 matrix H mapping image pixels to the aligned canvas.

    theta = angle of the eye line, d = eye spacing, s = D0 / d.
    The rotation by -theta levels the eyes; t places the eye midpoint at EYE_MID:
        H = [[ s cos(theta),  s sin(theta), tx],
             [-s sin(theta),  s cos(theta), ty],
             [ 0,             0,            1 ]]
        t = EYE_MID - s R(-theta) m
    """
    (xl, yl), (xr, yr) = left_eye, right_eye
    if xr < xl:                                   # make sure "left" really is the image-left eye
        (xl, yl), (xr, yr) = (xr, yr), (xl, yl)
    theta = math.atan2(yr - yl, xr - xl)
    d = math.hypot(xr - xl, yr - yl)
    s = d0 / d
    c, sn = math.cos(theta), math.sin(theta)
    R = np.array([[c, sn], [-sn, c]])             # R(-theta)
    m = np.array([(xl + xr) / 2.0, (yl + yr) / 2.0])
    t = np.array(eye_mid) - s * R @ m
    H = np.array([[s * c,  s * sn, t[0]],
                  [-s * sn, s * c, t[1]],
                  [0.0,     0.0,   1.0]])
    return H


def align_face(img, left_eye, right_eye, canvas=(CANVAS_W, CANVAS_H), eye_mid=EYE_MID, d0=D0,
               border=cv2.BORDER_CONSTANT):
    """
    Warp a full frame to the aligned face canvas. The defaults are the canvas every
    classifier uses; the ID card maker passes its own size and framing for a passport photo.
    """
    H = similarity_matrix(left_eye, right_eye, eye_mid, d0)
    return cv2.warpPerspective(img, H, canvas, borderMode=border)


def region_box(name):
    """Pixel box (x0, y0, x1, y1) of a named region on the aligned canvas."""
    x_min, x_max, y_min, y_max = REGIONS[name]
    cx, cy = EYE_MID
    return (int(round(cx + x_min * D0)), int(round(cy + y_min * D0)),
            int(round(cx + x_max * D0)), int(round(cy + y_max * D0)))


def crop_region(aligned, name):
    """Crop one named region out of an aligned face."""
    x0, y0, x1, y1 = region_box(name)
    return aligned[y0:y1, x0:x1]


def draw_regions(aligned):
    """Copy of the aligned face with all region boxes and the eye positions drawn on."""
    out = aligned.copy()
    for name in REGIONS:
        x0, y0, x1, y1 = region_box(name)
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), REGION_COLOURS[name], 1)
    for x in (EYE_MID[0] - D0 / 2, EYE_MID[0] + D0 / 2):
        cv2.circle(out, (int(x), int(EYE_MID[1])), 2, (255, 255, 255), -1)
    return out


# ---------------------------------------------------------------- checks
def self_test():
    """
    Draw two dots on a blank image at known positions, rotated and scaled, align,
    and check the dots land at EYE_MID -/+ D0/2. Prints the worst error in pixels.
    """
    img = np.zeros((720, 1280, 3), np.uint8)
    worst = 0.0
    for angle_deg, spacing, centre in [(0, 100, (640, 300)), (18, 140, (500, 420)),
                                       (-25, 70, (900, 250)), (40, 200, (300, 500))]:
        a = math.radians(angle_deg)
        half = np.array([math.cos(a), math.sin(a)]) * spacing / 2
        le, re = np.array(centre) - half, np.array(centre) + half
        H = similarity_matrix(le, re)
        for eye, target in [(le, (EYE_MID[0] - D0 / 2, EYE_MID[1])),
                            (re, (EYE_MID[0] + D0 / 2, EYE_MID[1]))]:
            p = H @ np.array([eye[0], eye[1], 1.0])
            worst = max(worst, math.hypot(p[0] - target[0], p[1] - target[1]))
        # also check the warp itself: draw dots, warp, find them again
        test = img.copy()
        for eye in (le, re):
            cv2.circle(test, (int(round(eye[0])), int(round(eye[1]))), 4, (255, 255, 255), -1)
        warped = cv2.cvtColor(cv2.warpPerspective(test, H, (CANVAS_W, CANVAS_H)), cv2.COLOR_BGR2GRAY)
        n, _, _, cents = cv2.connectedComponentsWithStats((warped > 100).astype(np.uint8))
        found = sorted(cents[1:].tolist())
        for (fx, fy), tx in zip(found, (EYE_MID[0] - D0 / 2, EYE_MID[0] + D0 / 2)):
            worst = max(worst, math.hypot(fx - tx, fy - EYE_MID[1]))
    status = "passed" if worst < 2.0 else "FAILED"
    print(f"alignment self-test {status}: worst eye position error {worst:.2f} px")
    return worst < 2.0


def read_eyes():
    """Labelled eye positions from eyes.csv, skipping images marked -1 (skipped)."""
    if not os.path.exists(EYES_CSV):
        return []
    with open(EYES_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if float(r["left_x"]) >= 0]   # -1 = no eyes found
    return rows


def check_figure(max_faces=12):
    """Align up to max_faces labelled frames, draw the boxes, save one grid figure."""
    rows = read_eyes()
    if not rows:
        print("No eye positions yet: run capture.py then face_detect.py first.")
        return
    # spread the picks across the list so several people and conditions show up
    picks = [rows[i] for i in np.linspace(0, len(rows) - 1, min(max_faces, len(rows))).astype(int)]

    cols = 6
    nrows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(2.0 * cols, 2.8 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), picks):
        img = cv2.imread(os.path.join(RAW_DIR, r["file"]))
        if img is None:
            continue
        aligned = align_face(img, (float(r["left_x"]), float(r["left_y"])),
                             (float(r["right_x"]), float(r["right_y"])))
        ax.imshow(cv2.cvtColor(draw_regions(aligned), cv2.COLOR_BGR2RGB))
        ax.set_title(r["file"].split("_", 1)[1].rsplit(".", 1)[0] + f"\n({r.get('method', '')})", fontsize=7)
    fig.suptitle("Aligned faces: green eyes, blue hat, red lower face, yellow whole face", fontsize=9)
    fig.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, "01_align_check.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print("check figure saved:", path)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    self_test()
    check_figure()
