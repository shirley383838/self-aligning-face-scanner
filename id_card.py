"""
id_card.py  -  find the mock ID card in a camera frame, flatten it, crop its photo

Same approach as the Connect Four board in Assignment 1:
  grey -> blur -> adaptive threshold (Week 1) -> contours (Week 3)
  -> approxPolyDP keeps 4-corner shapes -> check it looks like our card
  -> homography to flatten it (Weeks 3 to 5) -> crop the photo box.

The card has a dark frame inside a white margin (make_id_cards.py draws it), so
the frame shows up as a dark ring even against dark clothes. A candidate only
counts as the card if its frame band is dark and the strip just inside it is
white, which rules out doors, screens and other dark rectangles.

After flattening, YuNet looks for the face in the photo box. If there is none,
it tries the card turned upside down (YuNet only finds upright faces).

Before a card read is used it has to pass some checks, so a blurry frame from
the camera still focusing is never compared:
  * still: the corners barely move between frames
  * sharp: edge sharpness of the ID number boxes (Sobel gradient, Week 3, divided
    by the black-to-white contrast). The boxes are the same on every card, and the
    measure doesn't depend on how big the card is in the picture.
  * focused: the sharpness has stopped changing (webcam autofocus has settled)
  * no glare: hardly any washed-out white pixels on the face in the card photo
The live face is only compared when it is big enough, facing the camera and in
the middle of the picture. If no face is found the view is turned by 90 degrees
every few frames until one is (camera on its side).

Running this file opens the webcam and does the e-gate order of steps:
  1. hold the printed card up close: once 5 frames in a row pass the checks, the
     face in its photo is read
  2. put the card down and look at the camera: your face is compared with the
     card photo over 10 frames and it says MATCHES or DOES NOT MATCH
    c  read a new card    s  snapshot    r  turn the view 90 degrees    q / Esc  quit
    y / n  after a result: say whether it really is the person in the card photo. Each check is
           saved to results/card_checks.csv and the best threshold for YOUR camera
           is printed, since the LFW threshold does not carry over to front-on photos.
"""

import os
import csv
from collections import deque
from datetime import datetime

import numpy as np
import cv2

import face_detect as fd

# ---------------------------------------------------------------- card layout (millimetres, from the card's top-left)
CARD_W_MM, CARD_H_MM = 85.6, 54.0          # credit-card size
MARGIN_MM = 3.0                             # white margin outside the dark frame
FRAME_MM = 3.0                              # thickness of the dark frame
                                            # then a 1.5 mm plain white strip inside the frame (the ring check looks at it)
PANEL_MM = (7.5, 7.5, 78.1, 46.5)           # printed area inside that strip: x0, y0, x1, y1
HEADER_MM = (7.5, 7.5, 78.1, 14.0)          # coloured title band at the top of the panel
PHOTO_MM = (9.5, 16.0, 31.5, 45.33)         # photo box (22 x 29.3 mm, 3:4 like a passport photo)
NAME_MM = (34.5, 18.6, 22.6)                # x, label baseline, name baseline
DIGITS_MM = (34.5, 27.0, 7.5, 10.0, 1.0)    # first digit box x0, y0, box width, box height, gap
N_DIGITS = 4
GHOST_MM = (69.5, 27.0, 77.0, 37.0)         # small faded copy of the photo, as on real ID cards
DATES_MM = (34.5, 49.5, 40.5, 43.3)         # issued x, expires x, label baseline, value baseline
SIGN_MM = (64.5, 43.3, 77.0)                # signature start x, baseline, end x (same row as the dates)

# the flattened card covers the OUTER edge of the black frame, at 10 pixels per mm
PX_PER_MM = 10.0
WARP_W = int(round((CARD_W_MM - 2 * MARGIN_MM) * PX_PER_MM))   # 796
WARP_H = int(round((CARD_H_MM - 2 * MARGIN_MM) * PX_PER_MM))   # 480

# detection settings
MIN_AREA_FRAC = 0.01        # the card must cover at least 1% of the frame
ASPECT_RANGE = (1.30, 2.10) # allowed width/height of the found shape (allows for tilt)
RING_CONTRAST = 50          # inside-the-ring brightness minus ring brightness, out of 255
HEADER_SURE = 25            # title band this much darker than the bottom strip = sure which way up the card is
CARD_FACE_SCORE = 0.35      # YuNet score needed for the face IN THE CARD PHOTO (we know one is there)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(BASE_DIR, "figures_member3")
CAMERA_INDEX = 1
CAMERA_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
ROTATION = 0
ROTATE_CODES = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def warp_box(box_mm):
    """A box given in card millimetres, as pixels on the flattened card."""
    x0, y0, x1, y1 = box_mm
    f = lambda v: int(round((v - MARGIN_MM) * PX_PER_MM))
    return f(x0), f(y0), f(x1), f(y1)


def digit_boxes():
    """Pixel boxes of the ID number digits on the flattened card."""
    x0, y0, bw, bh, gap = DIGITS_MM
    return [warp_box((x0 + k * (bw + gap), y0, x0 + k * (bw + gap) + bw, y0 + bh)) for k in range(N_DIGITS)]


# ---------------------------------------------------------------- finding the card
def order_corners(pts):
    """Order four corner points as top-left, top-right, bottom-right, bottom-left, long side across the top."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]                              # clockwise in image coordinates (y down)
    s = pts.sum(axis=1)
    pts = np.roll(pts, -int(np.argmin(s)), axis=0)          # start at the corner nearest the top-left
    top = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2
    side = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2
    if side > top:                                          # card held on its side: start one corner later
        pts = np.roll(pts, -1, axis=0)
    return pts


def warp_card(img, corners):
    """Homography from the four frame corners to a flat WARP_W x WARP_H card."""
    dst = np.float32([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]])
    H = cv2.getPerspectiveTransform(np.float32(corners), dst)
    return cv2.warpPerspective(img, H, (WARP_W, WARP_H))


def ring_contrast(warped_grey):
    """How much brighter the white strip just inside the frame is than the black frame itself."""
    ring = int(FRAME_MM * PX_PER_MM)                        # 30 px
    g = warped_grey.astype(np.float32)
    band = lambda a, b: np.concatenate([g[a:b, :].ravel(), g[-b:-a or None, :].ravel(),
                                        g[:, a:b].ravel(), g[:, -b:-a or None].ravel()])
    dark = band(4, ring - 6).mean()                         # inside the black frame
    light = band(ring + 3, ring + 15).mean()                # white strip inside it
    return light - dark


def find_card(img):
    """Corners of the card's black frame (TL, TR, BR, BL), or None."""
    h, w = img.shape[:2]
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    block = max(31, (min(h, w) // 12) | 1)                  # odd neighbourhood size, scales with the frame
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 10)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Every 4-corner shape that passes the checks is a candidate. The one with the strongest
    # dark-frame / white-strip contrast wins: against a dark background the card's outer white
    # edge can also pass, but its "frame" band is part background, part white margin, so it
    # scores much lower than the real frame.
    best, best_score = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_FRAC * w * h:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        pts = order_corners(approx)
        top = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2
        side = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2
        if not ASPECT_RANGE[0] < top / max(side, 1) < ASPECT_RANGE[1]:
            continue
        score = ring_contrast(warp_card(grey, pts))
        if score < RING_CONTRAST or score <= best_score:
            continue
        best, best_score = pts, score
    return best


# ---------------------------------------------------------------- reading the card
def photo_crop(warped):
    """The photo box of a flattened card."""
    x0, y0, x1, y1 = warp_box(PHOTO_MM)
    return warped[y0:y1, x0:x1]


def header_at_bottom(flat):
    """
    True if the flattened card is upside down. The coloured title band is the darkest wide
    strip on the card and it belongs at the top, so compare the band's area with the same
    area turned 180 degrees. Works even when the card photo is too small or blurred for YuNet.
    Returns (upside_down, how sure: brightness difference out of 255).
    """
    g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x0, y0, x1, y1 = warp_box(HEADER_MM)
    top = g[y0:y1, x0:x1].mean()
    bottom = cv2.rotate(g, cv2.ROTATE_180)[y0:y1, x0:x1].mean()
    return bool(bottom < top), float(abs(top - bottom))


def card_face(yunet, photo):
    """
    The face in the card's photo box, or None. We know there is a face there, so YuNet's
    score only needs to pass CARD_FACE_SCORE rather than its usual 0.9: a printed photo seen
    by a webcam held close is often a bit soft, which drops the score to about 0.4 to 0.9.
    Tried at the flattened size first (upscaling a soft photo makes it softer), then 2x.
    """
    try:
        old = yunet.getScoreThreshold()
    except AttributeError:
        old = fd.YUNET_SCORE if hasattr(fd, "YUNET_SCORE") else 0.9
    yunet.setScoreThreshold(CARD_FACE_SCORE)
    try:
        for fx in (1, 2):
            face = fd.detect_yunet(yunet, photo if fx == 1 else cv2.resize(photo, None, fx=fx, fy=fx))
            if face is not None:
                return face if fx == 1 else fd.scale_detection(face, 1.0 / fx)
        return None
    finally:
        yunet.setScoreThreshold(old)


def read_card(img, yunet, corners=None):
    """
    Find, flatten and orient the card, crop the photo and find the face in it.
    Returns a dict (corners, warped, photo, face) or None if no card was found.
    face is a YuNet detection in photo coordinates, or None if no face was found on the card.
    The card is turned the right way up from where its title band is; if that isn't clear,
    from which way round YuNet finds the face in the photo (YuNet only finds upright faces).
    """
    corners = find_card(img) if corners is None else corners
    if corners is None:
        return None
    flat = warp_card(img, corners)
    upside_down, sure = header_at_bottom(flat)
    orders = (upside_down,) if sure > HEADER_SURE else (upside_down, not upside_down)
    for ud in orders:
        warped = cv2.rotate(flat, cv2.ROTATE_180) if ud else flat
        photo = photo_crop(warped)
        face = card_face(yunet, photo)
        if face is not None or ud == orders[-1]:
            out_corners = np.roll(corners, 2, axis=0) if ud else corners
            return {"corners": out_corners, "warped": warped, "photo": photo, "face": face}


def person_face(img, yunet, corners):
    """The biggest face in the frame that is NOT on the card (the person holding it)."""
    h, w = img.shape[:2]
    s = max(1.0, w / 640)
    small = cv2.resize(img, (int(round(w / s)), int(round(h / s)))) if s > 1 else img
    yunet.setInputSize((small.shape[1], small.shape[0]))
    _, faces = yunet.detect(small)
    if faces is None:
        return None
    best = None
    for f in faces:
        cx, cy = (f[0] + f[2] / 2) * s, (f[1] + f[3] / 2) * s
        if corners is not None and cv2.pointPolygonTest(np.float32(corners), (float(cx), float(cy)), False) >= 0:
            continue
        if best is None or f[2] * f[3] > best[2] * best[3]:
            best = f
    if best is None:
        return None
    left, right = fd._sorted_eyes((float(best[4]), float(best[5])), (float(best[6]), float(best[7])))
    det = {"box": tuple(int(v) for v in best[:4]), "left_eye": left, "right_eye": right,
           "nose": (float(best[8]), float(best[9])), "score": float(best[14]), "row": best.copy()}
    return fd.scale_detection(det, s)


# ---------------------------------------------------------------- is the card read good enough to use?
STILL_FRAC = 0.015          # corners may move at most 1.5% of the card width between frames
MIN_SHARPNESS = 1.6         # edge sharpness of the ID number boxes (about 2.8 when crisp, 1.0 when badly blurred)
STEADY = 0.85               # the last few sharpness values must be within 15% of each other (focus settled)
FOCUS_FRAMES = 4            # how many recent frames that steadiness is checked over
GLARE_LEVEL = 250           # grey level counted as washed out by a reflection
GLARE_FRAC = 0.01           # more than 1% washed-out pixels in the middle of the card photo = glare


def card_sharpness(card):
    """
    Edge sharpness of the ID number boxes: the strongest Sobel gradients (Week 3) divided
    by the black-to-white contrast. A crisp edge goes from black to white within about a
    pixel, so its gradient is big compared with the contrast; a blurred edge spreads the
    same change over several pixels. It is measured at the camera's own resolution (the
    flattened card is shrunk back first), so it doesn't depend on how far away the card is.
    """
    g = cv2.cvtColor(card["warped"], cv2.COLOR_BGR2GRAY).astype(np.float32)
    boxes = digit_boxes()
    x0, y0, x1, y1 = boxes[0][0], boxes[0][1], boxes[-1][2], boxes[-1][3]
    region = g[y0:y1, x0:x1]
    scale = min(1.0, card_width(card["corners"]) / WARP_W)    # camera pixels per flattened pixel
    if scale < 1.0:
        region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2).ravel()
    strongest = np.sort(mag)[-max(1, mag.size // 50):].mean()  # top 2% of edge pixels
    contrast = np.percentile(region, 95) - np.percentile(region, 5)
    return float(strongest / (contrast + 1e-6))


def card_glare(card):
    """
    Fraction of washed-out (near 255) pixels in the middle of the card photo, where the face
    is. A printed face never comes out pure white, so these are reflections (or far too much
    light). Uses a fixed area rather than the detected face, because glare often stops the
    face being found at all.
    """
    grey = cv2.cvtColor(card["photo"], cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    middle = grey[int(0.2 * h):int(0.9 * h), int(0.2 * w):int(0.8 * w)]
    return float(np.mean(middle >= GLARE_LEVEL))


def card_width(corners):
    """Length of the card's top edge in pixels."""
    return float(np.linalg.norm(np.float32(corners[1]) - np.float32(corners[0])))


def corner_shift(a, b):
    """Largest movement of any corner between two frames, in pixels."""
    return float(np.max(np.linalg.norm(np.float32(a) - np.float32(b), axis=1)))


# ---------------------------------------------------------------- is the live face good enough to compare?
MIN_EYE_FRAC = 0.03         # eye spacing at least 3% of the picture width (and 30 px)
MAX_TURN = 0.20             # nose offset along the eye line over eye spacing (about 0.2 at a 15 degree turn)
CENTRE_ZONE = (0.25, 0.75)  # face centre must be in the middle half of the picture, left to right


def face_problem(det, frame_w, frame_h):
    """None if the live face is fine to compare, otherwise what to tell the person."""
    le, re, nose = (np.float32(det[k]) for k in ("left_eye", "right_eye", "nose"))
    eye_d = float(np.linalg.norm(re - le))
    if eye_d < max(30.0, MIN_EYE_FRAC * frame_w):
        return "Come a bit closer to the camera"
    u = (re - le) / (eye_d + 1e-6)                        # unit vector along the eye line
    turn = float(np.dot(nose - (le + re) / 2, u)) / eye_d
    if abs(turn) > MAX_TURN:
        return "Look straight at the camera"
    bx, by, bw, bh = det["box"]
    cx = (bx + bw / 2) / frame_w
    if not CENTRE_ZONE[0] < cx < CENTRE_ZONE[1]:
        return "Move to the middle of the picture"
    return None


def pick_method(method=None):
    """The feature type to compare with: MATCH_METHOD in id_match.py (SFace, or HOG if its model file is missing)."""
    import id_match as idm
    return method or idm.current_method()


CARD_PHOTO_MODE = "auto"    # which face the live face is compared with as "the ID photo":
                            #   "auto"    the printed card photo if it came out sharp, else the record photo
                            #             for the number on the card (like an e-gate reading the chip)
                            #   "card"    always the printed photo     "record"  always the record photo
CARD_PHOTO_SHARP = 2.2      # card sharpness needed to trust the printed photo (about 2.8 when crisp)
RECORDS_CSV = os.path.join(BASE_DIR, "data", "cards", "records.csv")
_record_cache = {}
_told = set()


def record_feature_for_card(card, method):
    """
    Feature of the enrolled photo (data/ids) for the ID number printed on this card, or None.
    This is the trusted copy, like the photo on a passport chip: a blurry webcam view of the
    printed photo doesn't matter, only the number has to be read right.
    """
    import csv
    import id_match as idm
    import digits as dg
    number = dg.read_number(card)
    if not os.path.exists(RECORDS_CSV):
        return None
    with open(RECORDS_CSV) as fh:
        name = {r["id_number"]: r["name"] for r in csv.DictReader(fh)}.get(number)
    if name is None:
        return None
    key = (name.lower(), method)
    if key not in _record_cache:
        _record_cache[key] = None
        for f in sorted(os.listdir(idm.IDS_DIR)):
            if idm.person_name(f).lower() == name.lower():
                img = cv2.imread(os.path.join(idm.IDS_DIR, f))
                det = fd.detect_photo(fd.load_yunet(), img) if img is not None else None
                if det is not None:
                    _record_cache[key] = idm.features(img, det, [method])[method]
                    break
    return _record_cache[key]


def card_photo_usable(card):
    """Is the printed photo sharp enough to compare a face with?"""
    return card["face"] is not None and card_sharpness(card) >= CARD_PHOTO_SHARP


def card_feature(card, method):
    """
    The face feature that stands for "the ID photo" (see CARD_PHOTO_MODE). With "auto", a sharp
    card gives its printed photo, so a card with a swapped photo is still caught; a blurry card
    gives the record photo for its number instead (then the swapped-photo check can't be done).
    """
    import id_match as idm
    use_record = CARD_PHOTO_MODE == "record" or (CARD_PHOTO_MODE == "auto" and not card_photo_usable(card))
    if use_record:
        f = record_feature_for_card(card, method)
        if f is not None:
            if "record" not in _told:
                print("Card photo too blurry to compare with: using the record photo for the card's number "
                      "(the swapped-photo check needs a sharp card).", flush=True)
                _told.add("record")
            return f
    return idm.features(card["photo"], card["face"], [method])[method]


def threshold_for(method):
    """
    Match threshold, in this order: measured on our own camera by calibrate_ids.py; THRESHOLDS
    below (from the old y / n checks); the LFW run; OpenCV's published one for SFace.
    """
    import id_match as idm
    for t in (idm.own_threshold(method), THRESHOLDS.get(method)):
        if t is not None:
            return t
    return idm.threshold_of(method)


def log_check(method, d, same_person):
    """Save one labelled card check to results/card_checks.csv and suggest a threshold from all of them."""
    import id_match as idm
    new_file = not os.path.exists(CHECKS_CSV)
    os.makedirs(os.path.dirname(CHECKS_CSV), exist_ok=True)
    with open(CHECKS_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["time", "method", "distance", "same_person"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), method, f"{d:.4f}", int(same_person)])
    with open(CHECKS_CSV) as fh:
        rows = [r for r in csv.DictReader(fh) if r["method"] == method]
    d_all = np.array([float(r["distance"]) for r in rows])
    same = np.array([r["same_person"] == "1" for r in rows])
    print(f"  saved: {'same person' if same_person else 'different person'}, distance {d:.2f}  "
          f"({same.sum()} same, {(~same).sum()} different so far)")
    if same.any() and (~same).any():
        t, acc = idm.best_threshold(d_all, same)
        print(f"  same person: {d_all[same].min():.2f} to {d_all[same].max():.2f}   "
              f"different: {d_all[~same].min():.2f} to {d_all[~same].max():.2f}")
        print(f"  best threshold on your camera so far: {t:.2f} ({acc:.0%} right). Set THRESHOLDS[\"{method}\"] = {t:.2f} in id_card.py to use it.")


# ---------------------------------------------------------------- webcam test
CARD_FRAMES = 5             # good card reads averaged before the card is accepted
CARD_MIN_WIDTH = 0.30       # card must be at least this fraction of the frame width, so its photo is sharp
SMOOTH_FRAMES = 10          # live-face distances averaged before deciding
THRESHOLDS = {"hog": 0.61,     # from the y / n checks; only used if calibrate_ids.py hasn't been run
              "sface": None}   # None = use the LFW one (or OpenCV's published one for SFace)
CHECKS_CSV = os.path.join(BASE_DIR, "results", "card_checks.csv")
DISPLAY_MAX_H = 900         # shrink the window if the view is taller than this (camera on its side)
AUTO_ROTATE = True          # with no face found, try the other orientations and switch if one finds a face
AUTO_EVERY = 5              # how often (in frames) to try them while no face is found
GREEN, RED, YELLOW, GREY = (0, 220, 0), (0, 0, 255), (0, 255, 255), (200, 200, 200)


def text(img, msg, org, colour, scale=0.7, thick=2):
    (tw, th), base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)


def paste(view, small, x, y):
    """Copy a small image into the view with its top-left corner at (x, y), if it fits."""
    h, w = small.shape[:2]
    if y + h <= view.shape[0] and x + w <= view.shape[1] and x >= 0:
        view[y:y + h, x:x + w] = small


def rotate(frame, degrees):
    """Turn the camera picture clockwise by 0, 90, 180 or 270 degrees."""
    code = ROTATE_CODES[degrees]
    return frame if code is None else cv2.rotate(frame, code)


def compared_inset(live_face, card_face, scale=2):
    """The two crops the matcher actually compares (aligned, grey, equalised), side by side."""
    pair = np.hstack([card_face, np.full((card_face.shape[0], 6), 255, np.uint8), live_face])
    pair = cv2.resize(pair, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(pair, cv2.COLOR_GRAY2BGR)


def main():
    import id_match as idm
    yunet = fd.load_yunet()
    can_compare = os.path.exists(idm.THRESH_FILE) and os.path.exists(idm.EIGEN_FILE)
    if can_compare:
        method = pick_method()
        threshold = threshold_for(method)
        print(f"Face check uses {method} features, threshold {threshold:.2f} "
              f"({'measured on our camera' if THRESHOLDS.get(method) is not None else 'from LFW / OpenCV'})")
    else:
        print("Run id_match.py once first to switch on the face check. Card finding still works.")
    print(f"Opening camera {CAMERA_INDEX}...", flush=True)
    cam = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}. Run find_cameras.py to see which numbers work.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)        # some drivers remember autofocus OFF (capture_grid.py turns it off)
    print(__doc__)

    rotation, n_snap, n_frame = ROTATION, 0, 0
    stage, locked, logged = "card", None, False            # stage 1 reads the card, stage 2 checks the face
    good, sharp_hist, prev_corners = [], deque(maxlen=FOCUS_FRAMES), None
    dists = deque(maxlen=SMOOTH_FRAMES)
    while True:
        ok, raw = cam.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break
        n_frame += 1
        frame = rotate(raw, rotation)
        view = frame.copy()
        h, w = frame.shape[:2]
        corners = find_card(frame)
        result = None                                       # (method, averaged distance) once a decision is shown

        if stage == "card":
            card = read_card(frame, yunet, corners) if corners is not None else None
            problem, sharp = None, None
            if card is None:
                problem = "Step 1: hold the ID card up to the camera, flat and close"
                sharp_hist.clear()
                prev_corners = None
            else:
                sharp = card_sharpness(card)
                sharp_hist.append(sharp)
                still = prev_corners is not None and \
                    corner_shift(card["corners"], prev_corners) < STILL_FRAC * card_width(card["corners"])
                if card_width(card["corners"]) < CARD_MIN_WIDTH * w:
                    problem = "Card found: bring it closer"
                elif not still:
                    problem = "Hold the card still"
                elif sharp < MIN_SHARPNESS:
                    problem = "Card is blurry: hold still, or a little further away so the camera can focus"
                elif len(sharp_hist) < FOCUS_FRAMES or min(sharp_hist) < STEADY * max(sharp_hist):
                    problem = "Waiting for the camera to focus..."
                elif card_glare(card) > GLARE_FRAC:
                    problem = "Glare on the card photo: tilt the card a little"
                elif card["face"] is None:
                    problem = "Card found, but no face found in its photo"
                elif not can_compare:
                    problem = "Card read OK (run id_match.py to switch on the face check)"
                prev_corners = card["corners"]
            if problem is not None:
                good = []
            else:
                good.append(card)

            if card is not None:
                cv2.polylines(view, [np.int32(card["corners"])], True, GREEN if problem is None else YELLOW, 3)
                inset = card["warped"].copy()
                if card["face"] is not None:
                    bx, by, bw, bh = card["face"]["box"]
                    x0, y0, _, _ = warp_box(PHOTO_MM)
                    cv2.rectangle(inset, (x0 + bx, y0 + by), (x0 + bx + bw, y0 + by + bh), GREEN, 2)
                inset = cv2.resize(inset, None, fx=0.45, fy=0.45)
                paste(view, inset, w - inset.shape[1] - 10, 10)
                text(view, f"card sharpness {sharp:.2f} (needs {MIN_SHARPNESS:.1f})", (20, 80), GREY, 0.5, 1)
            if problem is not None:
                text(view, problem, (20, 40), YELLOW)
            else:
                text(view, f"Reading card... {len(good)}/{CARD_FRAMES}  keep it still", (20, 40), GREEN)
                if len(good) >= CARD_FRAMES:
                    feats = [card_feature(c, method) for c in good]
                    f = np.mean(feats, axis=0)              # average of the good reads, back to unit length
                    best = max(good, key=card_sharpness)
                    locked = {"feature": f / (np.linalg.norm(f) + 1e-9), "photo": best["photo"].copy(),
                              "crop": idm.face_vector(best["photo"], best["face"])}
                    stage, good, logged = "face", [], False
                    dists.clear()
                    print(f"Card read (sharpness {card_sharpness(best):.2f}). Now look at the camera.")

        else:
            det = person_face(frame, yunet, corners)       # a face still on the card is ignored
            if det is None and AUTO_ROTATE and n_frame % AUTO_EVERY == 0:
                for r in (90, 180, 270):                    # camera on its side: try the other ways round
                    cand = (rotation + r) % 360
                    f2 = rotate(raw, cand)
                    d2 = person_face(f2, yunet, find_card(f2))
                    if d2 is not None:
                        rotation, frame, det = cand, f2, d2
                        view, (h, w) = f2.copy(), f2.shape[:2]
                        print(f"Face found with the view turned {rotation} degrees, switched to that.")
                        break
            photo = cv2.resize(locked["photo"], None, fx=0.6, fy=0.6)
            paste(view, photo, w - photo.shape[1] - 10, 10)
            text(view, "card photo", (w - photo.shape[1] - 10, 10 + photo.shape[0] + 22), (255, 255, 255), 0.5, 1)
            # guide: the middle zone the face should be in
            cv2.line(view, (int(CENTRE_ZONE[0] * w), 0), (int(CENTRE_ZONE[0] * w), h), (90, 90, 90), 1)
            cv2.line(view, (int(CENTRE_ZONE[1] * w), 0), (int(CENTRE_ZONE[1] * w), h), (90, 90, 90), 1)
            if det is None:
                dists.clear()
                text(view, "Step 2: card read. Now look at the camera", (20, 40), YELLOW)
            else:
                bx, by, bw, bh = det["box"]
                for ex, ey in (det["left_eye"], det["right_eye"]):
                    cv2.circle(view, (int(ex), int(ey)), 4, (255, 255, 255), -1)
                problem = face_problem(det, w, h)
                if problem is not None:
                    cv2.rectangle(view, (bx, by), (bx + bw, by + bh), YELLOW, 2)
                    text(view, problem, (20, 40), YELLOW)
                else:
                    live_crop = idm.face_vector(frame, det)
                    d = idm.distance(idm.features(frame, det, [method])[method], locked["feature"])
                    dists.append(d)
                    pair = compared_inset(live_crop, locked["crop"])
                    paste(view, pair, w - pair.shape[1] - 10, h - pair.shape[0] - 40)
                    text(view, "compared: card | you", (w - pair.shape[1] - 10, h - 20),
                         (255, 255, 255), 0.5, 1)
                    if len(dists) < SMOOTH_FRAMES:
                        cv2.rectangle(view, (bx, by), (bx + bw, by + bh), YELLOW, 2)
                        text(view, f"Checking face... {len(dists)}/{SMOOTH_FRAMES}", (20, 40), YELLOW)
                    else:
                        d_avg = float(np.mean(dists))
                        result = (method, d_avg)
                        same = d_avg < threshold
                        cv2.rectangle(view, (bx, by), (bx + bw, by + bh), GREEN if same else RED, 2)
                        text(view, "MATCHES the card photo" if same else "DOES NOT MATCH the card photo",
                             (20, 40), GREEN if same else RED, 0.9)
                        text(view, f"{method} distance {d_avg:.2f}, needs < {threshold:.2f}", (20, 80), GREY, 0.6, 1)
            text(view, "c new card  |  after a result: y same person as card photo, n different person",
                 (20, h - 50), GREY, 0.5, 1)
        text(view, f"view turned {rotation} deg (r to turn)", (20, h - 20), GREY, 0.5, 1)

        if h > DISPLAY_MAX_H:
            view = cv2.resize(view, None, fx=DISPLAY_MAX_H / h, fy=DISPLAY_MAX_H / h)
        cv2.imshow("ID card test (c new card, s snapshot, r rotate, q quit)", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("y"), ord("n")) and result is not None:
            if logged:
                print("  already saved for this card: press c to read the next card")
            else:
                log_check(result[0], result[1], key == ord("y"))
                logged = True
        if key == ord("c"):
            stage, locked, good, prev_corners = "card", None, [], None
            sharp_hist.clear()
            dists.clear()
        if key == ord("r"):
            rotation = (rotation + 90) % 360
            print(f"view turned {rotation} degrees (set ROTATION = {rotation} to keep it)")
        if key == ord("s"):
            os.makedirs(FIGURE_DIR, exist_ok=True)
            n_snap += 1
            path = os.path.join(FIGURE_DIR, f"card_snapshot_{n_snap:02d}.png")
            cv2.imwrite(path, view)
            print("snapshot saved:", path)
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
