"""
face_detect.py  -  find the face, both eye centres and the nose with YuNet

YuNet (CNN face detector from the Week 11 face-track demo, cv2.FaceDetectorYN)
gives the face box AND five landmarks (both eyes, nose tip, mouth corners).
Every other file that needs a face or the eyes calls detect_yunet,
detect_scaled or detect_photo from here.

Setup: the model file face_detection_yunet_2026may.onnx (from the Week 11 zip,
AMME5710Week11_facetrack_demo_python.zip) goes in the models folder.
"""

import os

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
YUNET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2026may.onnx")
YUNET_SCORE = 0.9          # YuNet confidence threshold (OpenCV default, as in the Week 11 demo)


# ---------------------------------------------------------------- detectors
def load_yunet():
    """Create the YuNet detector from the Week 11 model file."""
    if not os.path.exists(YUNET_MODEL):
        raise FileNotFoundError(f"YuNet model not found: {YUNET_MODEL}\n"
                                "Copy it from the Week 11 zip into the models folder.")
    return cv2.FaceDetectorYN.create(model=YUNET_MODEL, config="", input_size=(320, 320),
                                     score_threshold=YUNET_SCORE)


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
    the full image. The webcam frames and the saved ID photos both go through this, so a face
    gets the same eye positions whether it comes from the webcam or from a saved photo.
    """
    s = max(1.0, img.shape[1] / max_w)
    small = img if s == 1.0 else cv2.resize(img, (int(round(img.shape[1] / s)), int(round(img.shape[0] / s))))
    det = detect_yunet(detector, small, pick)
    return None if det is None else scale_detection(det, s)


def detect_photo(detector, img, sizes=(640, 480, 320, 800)):
    """
    detect_scaled for a saved photo: tries 640 px first (same as the webcam frames), then
    other sizes. YuNet can miss a face that fills most of a close-up photo at one size
    and find it at another, because it only looks for faces within a range of sizes.
    """
    for max_w in sizes:
        det = detect_scaled(detector, img, max_w)
        if det is not None:
            return det
    return None
