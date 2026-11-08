"""
face_detect.py  -  find the face and both eye centres with the Week 11 detectors

Two detectors, both from the Week 11 face-track demo:
  * YuNet (CNN face detector, cv2.FaceDetectorYN): gives the face box AND five
    landmarks (both eyes, nose tip, mouth corners). This is the main one.
  * Viola-Jones Haar cascades (cv2.CascadeClassifier): face cascade, then the
    eye cascade inside the face box. Used as the comparison and as a fallback.

Setup: make a "models" folder next to this file and copy these three files into
it from the Week 11 zip (AMME5710Week11_facetrack_demo_python.zip):
    face_detection_yunet_2026may.onnx
    haarcascade_frontalface_default.xml
    haarcascade_eye.xml

Other Member 3 code calls find_eyes(img) to get the eye centres for alignment.

Running this file goes through every frame in data/own/labels.csv and:
  * writes data/own/detections.csv  (what each detector found, per frame)
  * writes data/own/eyes.csv        (the eye centres used for alignment:
                                     YuNet first, Haar if YuNet missed,
                                     -1 if both missed)
  * prints the detection rate of each detector per condition
    (plain, glasses, sunglasses, hat, mask, head turns), which is a result
    for the report in its own right.
Rows you fixed by hand with label_eyes.py (method "manual") are kept.
"""

import os
import csv
from collections import defaultdict

import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
YUNET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2026may.onnx")
HAAR_FACE = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
HAAR_EYE = os.path.join(MODEL_DIR, "haarcascade_eye.xml")

RAW_DIR = os.path.join(BASE_DIR, "data", "own", "raw")
LABELS_CSV = os.path.join(BASE_DIR, "data", "own", "labels.csv")
EYES_CSV = os.path.join(BASE_DIR, "data", "own", "eyes.csv")
DETECTIONS_CSV = os.path.join(BASE_DIR, "data", "own", "detections.csv")

YUNET_SCORE = 0.9          # YuNet confidence threshold (OpenCV default, as in the Week 11 demo)
HAAR_SCALE, HAAR_NEIGHBOURS = 1.3, 5   # face cascade settings from the Week 11 demo
EYE_ZONE = 0.6             # Haar eyes must sit in the top 60% of the face box (stops nostril/mouth hits)

EYE_FIELDS = ["file", "method", "left_x", "left_y", "right_x", "right_y", "score"]


# ---------------------------------------------------------------- detectors
def load_yunet():
    """Create the YuNet detector from the Week 11 model file."""
    if not os.path.exists(YUNET_MODEL):
        raise FileNotFoundError(f"YuNet model not found: {YUNET_MODEL}\n"
                                "Copy it from the Week 11 zip into the models folder.")
    return cv2.FaceDetectorYN.create(model=YUNET_MODEL, config="", input_size=(320, 320),
                                     score_threshold=YUNET_SCORE)


def load_haar():
    """Load the face and eye Haar cascades. Returns None if this OpenCV build has no cascades."""
    if not hasattr(cv2, "CascadeClassifier"):
        print("Note: this OpenCV version has no CascadeClassifier, Haar comparison skipped.")
        return None
    face = cv2.CascadeClassifier(HAAR_FACE)
    eye = cv2.CascadeClassifier(HAAR_EYE)
    if face.empty() or eye.empty():
        raise FileNotFoundError("Haar cascade XML files not found in the models folder.")
    return face, eye


def _sorted_eyes(p, q):
    """Return two eye points ordered (image-left, image-right)."""
    return (p, q) if p[0] <= q[0] else (q, p)


def detect_yunet(detector, img, pick="largest"):
    """
    One face found by YuNet, or None.
    pick="largest" takes the biggest face (the person at the rig); pick="center"
    takes the face nearest the image centre (used for LFW, where the labelled
    person is always in the middle).
    Each YuNet row is: x, y, w, h, right eye (x, y), left eye (x, y), nose tip,
    right mouth corner, left mouth corner, score. "Right eye" is the person's right,
    which is on the image left in an unmirrored photo, so the eyes are re-ordered
    by x to give image-left / image-right like the rest of the code uses.
    The raw row is kept in "row" for code that needs all five landmarks.
    """
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    if pick == "center":
        f = min(faces, key=lambda r: (r[0] + r[2] / 2 - w / 2) ** 2 + (r[1] + r[3] / 2 - h / 2) ** 2)
    else:
        f = max(faces, key=lambda r: r[2] * r[3])
    left, right = _sorted_eyes((float(f[4]), float(f[5])), (float(f[6]), float(f[7])))
    return {"box": tuple(int(v) for v in f[:4]), "left_eye": left, "right_eye": right,
            "nose": (float(f[8]), float(f[9])), "score": float(f[14]), "row": f.copy()}


def scale_detection(det, s):
    """The same detection with every coordinate multiplied by s (detect on a small copy, use on the full frame)."""
    row = det["row"].copy()
    row[:14] *= s
    return {"box": tuple(int(round(v * s)) for v in det["box"]),
            "left_eye": (det["left_eye"][0] * s, det["left_eye"][1] * s),
            "right_eye": (det["right_eye"][0] * s, det["right_eye"][1] * s),
            "nose": (det["nose"][0] * s, det["nose"][1] * s), "score": det["score"], "row": row}


def detect_scaled(detector, img, max_w=640, pick="largest"):
    """
    YuNet on a copy shrunk to at most max_w pixels wide, with the result scaled back to
    the full image. The live demo and the ID enrolment both use this, so a face gets the
    same eye positions whether it comes from the webcam or from a saved ID photo.
    """
    s = max(1.0, img.shape[1] / max_w)
    small = img if s == 1.0 else cv2.resize(img, (int(round(img.shape[1] / s)), int(round(img.shape[0] / s))))
    det = detect_yunet(detector, small, pick)
    return None if det is None else scale_detection(det, s)


def detect_photo(detector, img, sizes=(640, 480, 320, 800)):
    """
    detect_scaled for a saved photo: tries 640 px first (same as the live demo), then
    other sizes. YuNet can miss a face that fills most of a close-up photo at one size
    and find it at another, because it only looks for faces within a range of sizes.
    """
    for max_w in sizes:
        det = detect_scaled(detector, img, max_w)
        if det is not None:
            return det
    return None


def detect_haar(cascades, img):
    """
    Largest Haar face, plus one eye from each half of the face box, or None.
    Returns a dict with "box", and "left_eye"/"right_eye" (None if the eyes were not both found).
    """
    if cascades is None:
        return None
    face_cascade, eye_cascade = cascades
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, HAAR_SCALE, HAAR_NEIGHBOURS)
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    eyes = eye_cascade.detectMultiScale(gray[y:y + fh, x:x + fw])

    # keep eye boxes in the upper part of the face, then the biggest one in each half
    halves = {"left": [], "right": []}
    for (ex, ey, ew, eh) in eyes:
        cx, cy = x + ex + ew / 2.0, y + ey + eh / 2.0
        if cy - y > EYE_ZONE * fh:
            continue
        halves["left" if cx < x + fw / 2.0 else "right"].append((ew * eh, (cx, cy)))
    result = {"box": (int(x), int(y), int(fw), int(fh)), "left_eye": None, "right_eye": None}
    if halves["left"] and halves["right"]:
        result["left_eye"] = max(halves["left"])[1]
        result["right_eye"] = max(halves["right"])[1]
    return result


def find_eyes(img, yunet=None, cascades=None):
    """
    Eye centres for alignment: YuNet first, Haar if YuNet finds nothing.
    Returns (left_eye, right_eye, method) or None. Pass in loaded detectors to
    avoid reloading them every call.
    """
    yunet = yunet if yunet is not None else load_yunet()
    d = detect_yunet(yunet, img)
    if d is not None:
        return d["left_eye"], d["right_eye"], "yunet"
    hd = detect_haar(cascades, img)
    if hd is not None and hd["left_eye"] is not None:
        return hd["left_eye"], hd["right_eye"], "haar"
    return None


# ---------------------------------------------------------------- batch over own captures
def read_csv(path):
    """Rows of a CSV as dicts, or [] if the file does not exist."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    """Write a whole CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    labels = read_csv(LABELS_CSV)
    if not labels:
        print("No labels.csv yet: run capture.py first.")
        return
    manual = {r["file"]: r for r in read_csv(EYES_CSV) if r["method"] == "manual"}
    yunet, cascades = load_yunet(), load_haar()

    det_rows, eye_rows = [], []
    stats = defaultdict(lambda: [0, 0, 0, 0])          # condition -> [n, yunet face, haar face, haar both eyes]
    disagreement = []                                    # eye position difference YuNet vs Haar, in eye spacings

    for i, lab in enumerate(labels, 1):
        fname, cond = lab["file"], lab["condition"]
        img = cv2.imread(os.path.join(RAW_DIR, fname))
        if img is None:
            print("missing file, skipped:", fname)
            continue
        y = detect_yunet(yunet, img)
        hd = detect_haar(cascades, img)
        haar_eyes = hd is not None and hd["left_eye"] is not None

        s = stats[cond]
        s[0] += 1
        s[1] += y is not None
        s[2] += hd is not None
        s[3] += haar_eyes
        if y is not None and haar_eyes:
            spacing = np.hypot(*np.subtract(y["right_eye"], y["left_eye"]))
            err = max(np.hypot(*np.subtract(y["left_eye"], hd["left_eye"])),
                      np.hypot(*np.subtract(y["right_eye"], hd["right_eye"])))
            disagreement.append(err / spacing)

        det_rows.append({
            "file": fname, "condition": cond,
            "yunet_found": int(y is not None), "yunet_score": round(y["score"], 3) if y else "",
            "yunet_lx": round(y["left_eye"][0], 1) if y else "", "yunet_ly": round(y["left_eye"][1], 1) if y else "",
            "yunet_rx": round(y["right_eye"][0], 1) if y else "", "yunet_ry": round(y["right_eye"][1], 1) if y else "",
            "haar_face_found": int(hd is not None) if cascades else "",
            "haar_eyes_found": int(haar_eyes) if cascades else "",
            "haar_lx": round(hd["left_eye"][0], 1) if haar_eyes else "", "haar_ly": round(hd["left_eye"][1], 1) if haar_eyes else "",
            "haar_rx": round(hd["right_eye"][0], 1) if haar_eyes else "", "haar_ry": round(hd["right_eye"][1], 1) if haar_eyes else "",
        })

        if fname in manual:
            eye_rows.append(manual[fname])
        elif y is not None:
            eye_rows.append({"file": fname, "method": "yunet",
                             "left_x": round(y["left_eye"][0], 1), "left_y": round(y["left_eye"][1], 1),
                             "right_x": round(y["right_eye"][0], 1), "right_y": round(y["right_eye"][1], 1),
                             "score": round(y["score"], 3)})
        elif haar_eyes:
            eye_rows.append({"file": fname, "method": "haar",
                             "left_x": round(hd["left_eye"][0], 1), "left_y": round(hd["left_eye"][1], 1),
                             "right_x": round(hd["right_eye"][0], 1), "right_y": round(hd["right_eye"][1], 1),
                             "score": ""})
        else:
            eye_rows.append({"file": fname, "method": "none", "left_x": -1, "left_y": -1,
                             "right_x": -1, "right_y": -1, "score": ""})
        if i % 25 == 0:
            print(f"  {i}/{len(labels)} frames done")

    write_csv(DETECTIONS_CSV, list(det_rows[0].keys()), det_rows)
    write_csv(EYES_CSV, EYE_FIELDS, eye_rows)

    # detection rate per condition
    print(f"\n{'condition':12s} {'frames':>6s} {'YuNet face':>11s} {'Haar face':>10s} {'Haar 2 eyes':>12s}")
    for cond in sorted(stats):
        n, yf, hf, he = stats[cond]
        haar_cols = f"{hf / n:10.0%} {he / n:12.0%}" if cascades else f"{'n/a':>10s} {'n/a':>12s}"
        print(f"{cond:12s} {n:6d} {yf / n:11.0%} {haar_cols}")
    if disagreement:
        print(f"\nYuNet vs Haar eye positions, where both found the eyes: median "
              f"{np.median(disagreement):.2f}, max {np.max(disagreement):.2f} eye spacings")
    n_none = sum(r["method"] == "none" for r in eye_rows)
    print(f"\n{len(eye_rows) - n_none} frames have eye positions, {n_none} have none "
          f"(fix those with label_eyes.py if you want them in training)")
    print("Saved:", DETECTIONS_CSV)
    print("Saved:", EYES_CSV)


if __name__ == "__main__":
    main()
