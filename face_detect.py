"""
Face, eye and nose detection with YuNet (cv2.FaceDetectorYN, the Week 11 demo model).

Needs models/face_detection_yunet_2026may.onnx.
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
    One YuNet face, or None. pick="largest" takes the biggest face, pick="center" the one
    nearest the image centre (LFW). Eyes are ordered image-left / image-right; the raw YuNet
    row with all five landmarks is kept in "row".
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
    YuNet on a copy at most max_w pixels wide, scaled back to full size, so webcam frames and
    saved photos give consistent landmarks.
    """
    s = max(1.0, img.shape[1] / max_w)
    small = img if s == 1.0 else cv2.resize(img, (int(round(img.shape[1] / s)), int(round(img.shape[0] / s))))
    det = detect_yunet(detector, small, pick)
    return None if det is None else scale_detection(det, s)


def detect_photo(detector, img, sizes=(640, 480, 320, 800)):
    """detect_scaled at several sizes, since YuNet can miss a face that fills a close-up photo at one size."""
    for max_w in sizes:
        det = detect_scaled(detector, img, max_w)
        if det is not None:
            return det
    return None
