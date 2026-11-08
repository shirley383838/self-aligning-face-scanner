"""
capture.py  -  webcam capture tool for the Member 3 dataset (accessories + ID matching)

Shows a live preview with a framing guide, and saves the RAW frame (no overlay)
every time you press SPACE. Each saved frame gets a row in labels.csv with the
person, condition, lighting and camera, plus 0/1 flags for glasses / sunglasses /
hat / mask so the classifiers can read the labels straight from the CSV.

Keys (click on the preview window first so it has focus):
    1 plain        2 glasses      3 sunglasses    4 hat         5 mask
    6 turn L 15    7 turn R 15    8 turn L 30     9 turn R 30
    0 ID photo (passport style, plain background)
    c ID card (photo of the printed mock ID card)
    n / p   next / previous person
    l       cycle lighting label (light1 -> light2 -> light3)
    r       turn the picture 90 degrees (camera mounted on its side)
    SPACE   save the current frame
    q / Esc quit

Turn directions are from the CAMERA's point of view: "turn L" means the face
turns towards the left side of the saved image.

Next step after capturing: run face_detect.py to find the eyes in every frame.

Run it with the spare webcam now (CAMERA_TAG = "camA"), then again with the
final presentation webcam later (CAMERA_TAG = "camB"). Keeping the tag in every
filename means the two cameras never get mixed up, and the camA images become a
free cross-camera test set.
"""

import os
import csv
import time
from datetime import datetime

import cv2
import numpy as np

# ---------------------------------------------------------------- settings
CAMERA_INDEX = 0          # 0 = default webcam, try 1 if the wrong camera opens
CAMERA_TAG = "camA"       # camA = spare webcam, camB = final presentation webcam
FRAME_W, FRAME_H = 1280, 720   # requested resolution (camera may give something else)
WARMUP_FRAMES = 30        # frames thrown away at start so auto exposure can settle
MIRROR_PREVIEW = True     # mirror the PREVIEW only (easier to line up), saved frames are not mirrored
START_PERSON = 1
ROTATION = 0              # 0, 90, 180 or 270: turn the camera picture clockwise (camera on its side)
DISPLAY_MAX_H = 900       # shrink the preview if the picture is taller than this
ROTATE_CODES = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
# On Windows, the DirectShow backend opens USB webcams much faster than the default
# (the default can sit for a long time on "opening" with an external camera)
CAMERA_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "own", "raw")
LABELS_CSV = os.path.join(BASE_DIR, "data", "own", "labels.csv")

# key -> (condition name, glasses, sunglasses, hat, mask, yaw in degrees, is_id)
CONDITIONS = {
    ord("1"): ("plain",      0, 0, 0, 0,   0, 0),
    ord("2"): ("glasses",    1, 0, 0, 0,   0, 0),
    ord("3"): ("sunglasses", 1, 1, 0, 0,   0, 0),   # sunglasses also count as glasses
    ord("4"): ("hat",        0, 0, 1, 0,   0, 0),
    ord("5"): ("mask",       0, 0, 0, 1,   0, 0),
    ord("6"): ("turnL15",    0, 0, 0, 0, -15, 0),
    ord("7"): ("turnR15",    0, 0, 0, 0,  15, 0),
    ord("8"): ("turnL30",    0, 0, 0, 0, -30, 0),
    ord("9"): ("turnR30",    0, 0, 0, 0,  30, 0),
    ord("0"): ("idphoto",    0, 0, 0, 0,   0, 1),
    ord("c"): ("idcard",     0, 0, 0, 0,   0, 1),
}
LIGHTING = ["light1", "light2", "light3"]

CSV_FIELDS = ["file", "camera", "person", "condition", "lighting",
              "glasses", "sunglasses", "hat", "mask", "yaw_deg", "is_id",
              "width", "height", "timestamp"]


# ---------------------------------------------------------------- helpers
def open_camera(index):
    """Open the webcam, request the target resolution and let auto exposure settle."""
    print(f"Opening camera {index}...", flush=True)
    cam = cv2.VideoCapture(index, CAMERA_BACKEND)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera {index}. Run find_cameras.py to see which numbers work.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(WARMUP_FRAMES):
        cam.read()
    print(f"Camera {index} open at {int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
          f"{int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))}", flush=True)
    return cam


def next_index(person, condition, lighting):
    """Count how many frames already exist for this person/condition/lighting, return the next number."""
    prefix = f"{CAMERA_TAG}_p{person:02d}_{condition}_{lighting}_"
    if not os.path.isdir(OUT_DIR):
        return 1
    existing = [f for f in os.listdir(OUT_DIR) if f.startswith(prefix)]
    return len(existing) + 1


def append_label(row):
    """Append one row to labels.csv, writing the header first if the file is new."""
    new_file = not os.path.exists(LABELS_CSV)
    with open(LABELS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def save_frame(frame, person, cond, lighting):
    """Save the raw frame as a lossless PNG and log its labels. Returns the filename."""
    name, g, sg, hat, mask, yaw, is_id = cond
    n = next_index(person, name, lighting)
    fname = f"{CAMERA_TAG}_p{person:02d}_{name}_{lighting}_{n:02d}.png"
    cv2.imwrite(os.path.join(OUT_DIR, fname), frame)
    append_label({
        "file": fname, "camera": CAMERA_TAG, "person": person, "condition": name,
        "lighting": lighting, "glasses": g, "sunglasses": sg, "hat": hat, "mask": mask,
        "yaw_deg": yaw, "is_id": is_id,
        "width": frame.shape[1], "height": frame.shape[0],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    return fname


def draw_guide(frame, person, cond_name, lighting, n_saved, flash):
    """Return a preview copy with the framing guide and status text drawn on it."""
    view = cv2.flip(frame, 1) if MIRROR_PREVIEW else frame.copy()
    h, w = view.shape[:2]
    cx, eye_y = w // 2, int(0.42 * h)
    face_w, face_h = int(0.16 * w), int(0.30 * h)

    # framing guide: centre line, eye line, face oval
    cv2.line(view, (cx, 0), (cx, h), (0, 255, 255), 1)
    cv2.line(view, (cx - face_w, eye_y), (cx + face_w, eye_y), (0, 255, 255), 1)
    cv2.ellipse(view, (cx, eye_y + face_h // 6), (face_w // 2 + 20, face_h // 2 + 20),
                0, 0, 360, (0, 255, 255), 1)

    # status bar
    cv2.rectangle(view, (0, 0), (w, 34), (0, 0, 0), -1)
    text = f"{CAMERA_TAG}  person p{person:02d}  |  {cond_name}  |  {lighting}  |  saved this combo: {n_saved}"
    cv2.putText(view, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if flash:
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), (0, 255, 0), 8)
    return view


# ---------------------------------------------------------------- main loop
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cam = open_camera(CAMERA_INDEX)

    person, cond, light_i = START_PERSON, CONDITIONS[ord("1")], 0
    flash_until, rotation = 0.0, ROTATION
    print(__doc__)

    while True:
        ok, frame = cam.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break
        if ROTATE_CODES[rotation] is not None:            # saved frames are turned too, so faces are upright
            frame = cv2.rotate(frame, ROTATE_CODES[rotation])

        n_saved = next_index(person, cond[0], LIGHTING[light_i]) - 1
        view = draw_guide(frame, person, cond[0], LIGHTING[light_i], n_saved,
                          flash=time.time() < flash_until)
        if view.shape[0] > DISPLAY_MAX_H:
            view = cv2.resize(view, None, fx=DISPLAY_MAX_H / view.shape[0], fy=DISPLAY_MAX_H / view.shape[0])
        cv2.imshow("capture (SPACE save, r rotate, q quit)", view)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            fname = save_frame(frame, person, cond, LIGHTING[light_i])
            print("saved", fname)
            flash_until = time.time() + 0.25
        elif key in CONDITIONS:
            cond = CONDITIONS[key]
        elif key == ord("n"):
            person += 1
        elif key == ord("p"):
            person = max(1, person - 1)
        elif key == ord("l"):
            light_i = (light_i + 1) % len(LIGHTING)
        elif key == ord("r"):
            rotation = (rotation + 90) % 360
            print(f"picture turned {rotation} degrees (set ROTATION = {rotation} to keep it)")

    cam.release()
    cv2.destroyAllWindows()
    print("Frames saved in:", OUT_DIR)
    print("Labels in:", LABELS_CSV)


if __name__ == "__main__":
    main()
