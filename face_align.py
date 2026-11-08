"""
Eye-based face alignment and region crops.

A similarity transform levels the eyes, scales the eye spacing to D0 pixels and moves
the eye midpoint to EYE_MID on a CANVAS_W x CANVAS_H canvas, so each region (eye band,
hat, face) lands in the same place for every person.

Running this file runs a self-test on a synthetic face.
"""

import math

import numpy as np
import cv2

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


def similarity_matrix(left_eye, right_eye, eye_mid=EYE_MID, d0=D0):
    """
    3x3 matrix mapping image pixels to the aligned canvas:
    H = [[s cos t, s sin t, tx], [-s sin t, s cos t, ty], [0, 0, 1]], with t the eye-line angle,
    s = D0 / eye spacing and (tx, ty) placing the eye midpoint at EYE_MID.
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


if __name__ == "__main__":
    self_test()
