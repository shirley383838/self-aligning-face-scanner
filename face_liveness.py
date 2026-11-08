"""
face_liveness.py  -  face following with a LIVE 3D liveness check (real face, or photo / screen?)

Builds on face_follow.py and stereo_face_liveness.py (both in the same folder).
The verdict is worked out continuously while the rig follows the face.

Choosing the camera
-------------------
Set CAMERA_INDEX in the settings below (0, 1, 2 ...) and run the script.

How it works
------------
1. The rig centres the tip of the nose (YuNet landmark) in the middle of the picture.
2. Once it is centred, the rig takes STEREO PAIRS: it moves STEREO_BASELINE mm
   (40 mm, along rig X, left / right, by default) to one side of the centred position, stops, waits
   for vibration to die down and takes a still picture; then it moves to the other
   side and takes another. It keeps going back and forth, and every new picture
   makes a pair with the one before it, so there is a new reading about once a
   second. The rig positions are known, so the baseline of each pair is known.
3. Each pair goes through the stereo method of stereo_face_liveness.py:
     - SIFT features are matched inside the central part of the face,
     - the matches are triangulated into 3D points (mm) using the baseline, the
       focal length and the direction the face moved in the picture,
     - three coplanarity tests:
         A. flatness: distance of the points from the best-fit plane, relative
            to face size (photo ~ noise, real face: nose, eye sockets),
         B. parallax: what is left after the best single homography, in pixels
            (a flat object maps between the two views exactly by a homography),
         C. smooth surface: what is left after the best quadratic surface
            (a photo bent into a curve passes A and B, but has no nose).
     A and B both 3D and C 3D = LIVE, A and B 3D but C smooth = CURVED (bent
     photo / curved screen), A and B both flat = FLAT, otherwise UNSURE.
   D. face vs its surroundings (BG_CHECK): points are also matched in a band AROUND the
      face (hair, background, or the screen / paper the face is printed on). The camera
      only slides sideways, so for any flat surface the shift of a point between the two
      pictures (its disparity, f x baseline / depth) is a plane in the picture:
      d = a + b x + c y. A plane is fitted to the surroundings' shifts and the face is
      compared with it. A face on a phone, tablet or photo lies on the SAME flat surface
      as the screen / paper around it, so it lands on that plane: the pair is FLAT,
      whatever A to C say. A real face is ~1 m in front of the wall behind it (and its
      nose sticks out), so it is many pixels off it.
      On its own this test never makes a pair LIVE, because a small phone held in front
      of a wall is ALSO off the wall's plane. But when the face's own shape is 3D on A and
      C and only B is borderline (a short camera move gives a real face only a couple of
      pixels of parallax), a face clearly off the surroundings (BG_DIFF_PX) settles it: LIVE.
4. A pair only counts if the face moved in the picture the way the rig's own
   movement predicts (a video replayed on a phone, or someone moving, doesn't).
5. The answer is a plain YES or NO:
     YES  as soon as LIVE_NEEDED pairs say LIVE (and they outnumber the rest),
     NO   as soon as FAKE_NEEDED pairs say FLAT / CURVED (and outnumber LIVE),
     NO   if DECISION_PAIRS pairs have been analysed without a YES,
     NO   if there is still no answer DECISION_TIME seconds after measuring started
          (e.g. every pair was skipped). The NO message says why, and every skipped
          pair is printed in the terminal with the reason.
   The answer then stays on screen and the rig stops taking pairs (it keeps
   following the face). Press v to check again; a new face also starts a new check.

The coloured dots on the face are the latest 3D points (red = nearer than the best
plane, blue = further).

Press 1, 2 or 3 to record the latest scores with what is really in front of the
camera (real / print / screen) in liveness_live_log.csv, then run
    python face_liveness.py --analyse
to see the scores per type and pick the thresholds.

Optional, recommended:
    python face_liveness.py --intrinsics "C:\\Users\\...\\Downloads\\calib_poses_YYYYMMDD_HHMMSS"
calibrates the lens from the capture_grid.py images. With it the focal length is
known (so depths are in real mm) and lens distortion, which can give a flat photo
a small fake curvature, is taken out. The calibration belongs to the camera that
took those images: if you switch to another camera, calibrate that one too.

Keys (in the video window):
    SPACE   connect the rig (at the start), afterwards tracking on / off
    c       calibrate the tracking (face in view, keep still)
    x       cancel a calibration
    v       check again (new YES / NO)
    d       stereo pairs (when centred) on / off
    1 2 3   log the latest scores as: 1 real face, 2 printed photo, 3 phone or screen
    t       tracking on / off
    m       move the rig back to the middle
    r       turn the view 90 degrees
    f       mirror the picture on / off (display only)
    h       show / hide the key list
    q / Esc quit (the rig goes back to the start corner)
"""

import os
import csv
import glob
import json
import time
import argparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime

import numpy as np
import cv2

import stereo_face_liveness as sfl           # the stereo method (same folder)
import face_follow as ff                     # also puts MajorProject on the path
from face_follow import (RigMover, Calibration, CalibrationFailed, load_calibration, save_calibration,
                         plan_step, rot, face_roll, face_centre, text, rotate, open_camera, read_key,
                         window_closed, bring_to_front, YELLOW, GREEN, RED, GREY, ORANGE, WHITE)

fd = ff.fd
ff.WINDOW = "face liveness"                  # the face_follow helpers use this window name

# ---------------------------------------------------------------- settings
CAMERA_INDEX = 0          # which camera to use: 0, 1, 2 ... (change this and run again)
STEREO = True             # take stereo pairs once the face is centred
STEREO_BASELINE = 40.0    # mm the rig moves between the two pictures of a pair
STEREO_AXIS = (1.0, 0.0)  # rig direction (X, Y) of that move: (1, 0) = rig X, left / right; (0, 1) = rig Y
STEREO_FEED = 3000        # mm/min for the stereo moves
SETTLE_S = 0.25           # wait this long after the rig stops before taking a picture (vibration)
SETTLE_FRAMES = 5         # and at least this many frames (flushes old frames out of the camera buffer)
MIN_BASELINE_MM = 15.0    # pairs where the rig really moved less than this (at a limit) are skipped
RECENTRE_BAND = 0.10      # while taking pairs, re-centre if the nose is this far off centre (fraction of
                          # the frame), measured as if the rig were back at the centred position, so the
                          # stereo moves themselves don't count
DIRECTION_MIN_COS = 0.7   # face movement must point the way the rig movement predicts
DECISION_PAIRS = 6        # analysed pairs without a YES = NO
DECISION_TIME = 20.0      # s after measuring starts with no answer at all = NO (safety net)
LIVE_NEEDED = 2           # LIVE pairs needed for YES
FAKE_NEEDED = 3           # FLAT / CURVED pairs that give an early NO

FACE_INNER = (0.70, 0.85) # use this central fraction of the face box (width, height), keeps background out
BG_CHECK = True           # D: also compare the face with its surroundings (see the top)
BG_PAD = 0.8              # surroundings = the face box grown by this many face widths on every side ...
BG_GAP = 0.15             # ... minus the face box grown by this much (keeps the face's own edge out)
BG_MIN_POINTS = 25        # matched points needed in the surroundings for the test to count
BG_SAME_PX = 1.5          # face within this many pixels of the surroundings' plane = same flat surface
BG_DIFF_PX = 3.0          # face this far off it = not part of what's around it (helps a borderline B)
CROP_PAD = 0.15           # the matcher works on the face area plus this margin (much faster than the frame)

# decision thresholds of the stereo method, for this rig (tune with --analyse)
FLATNESS_THRESH = 0.03    # A: plane residual RMS / face size
PARALLAX_THRESH_PX = 2.0  # B: 90th percentile homography residual, px
QUAD_RESID_THRESH = 0.018 # C: residual RMS after a quadratic surface / face size
sfl.FLATNESS_THRESH, sfl.PARALLAX_THRESH_PX, sfl.QUAD_RESID_THRESH = \
    FLATNESS_THRESH, PARALLAX_THRESH_PX, QUAD_RESID_THRESH

FALLBACK_HFOV_DEG = 70.0  # horizontal field of view guess, used only without camera_intrinsics.json
BOARD_INNER = None        # checkerboard inner corners (cols, rows), None = find it automatically

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTRINSICS_FILE = os.path.join(BASE_DIR, "camera_intrinsics.json")
LOG_FILE = os.path.join(BASE_DIR, "liveness_live_log.csv")
LABELS = {"1": "real", "2": "print", "3": "screen"}


# ---------------------------------------------------------------- camera in the background
class FrameGrabber:
    """
    Reads the camera in a background thread and keeps only the newest frame. The main
    loop then never works on old frames queued in the camera's buffer (which makes the
    picture trail behind what is really happening), and one slow loop doesn't make the
    next ones slower. read() waits for a frame newer than the last one it returned.
    """

    def __init__(self, cam):
        self.cam = cam
        try:
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # not every camera driver supports this
        except Exception:
            pass
        self.frame, self.id, self.ok, self.running = None, 0, True, True
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, f = self.cam.read()
            with self.cond:
                if ok and f is not None:
                    self.frame, self.id = f, self.id + 1
                else:
                    self.ok = False
                self.cond.notify_all()
            if not ok:
                break

    def read(self, last_id, timeout=2.0):
        """(ok, frame, id) with a frame newer than last_id."""
        with self.cond:
            self.cond.wait_for(lambda: self.id != last_id or not self.ok, timeout)
            if self.id == last_id:
                return False, None, last_id
            return True, self.frame, self.id

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cam.release()


def bg_text(r):
    """One line about test D for printing."""
    bg = r.get("bg")
    if bg is None:
        return "surroundings: not enough points" if BG_CHECK else ""
    return (f"face {'ON' if bg['same_plane'] else 'off'} the surroundings' plane by {bg['face_off_px']:.1f}px "
            f"(same surface below {BG_SAME_PX}px, {bg['n_around']} pts around)")


def timed_analyse(*args):
    """analyse_pair, run in the background, with its run time in ms added."""
    t = time.perf_counter()
    r = analyse_pair(*args)
    r["ms"] = 1000 * (time.perf_counter() - t)
    return r


# ---------------------------------------------------------------- lens: undistortion and focal length
def view_to_raw(pts, rotation, w, h):
    """Points in the rotated view -> points in the raw camera picture (w x h)."""
    u, v = pts[:, 0], pts[:, 1]
    if rotation == 90:
        return np.stack([v, h - 1 - u], 1)
    if rotation == 180:
        return np.stack([w - 1 - u, h - 1 - v], 1)
    if rotation == 270:
        return np.stack([w - 1 - v, u], 1)
    return pts.copy()


def raw_to_view(pts, rotation, w, h):
    """Points in the raw camera picture (w x h) -> points in the rotated view."""
    x, y = pts[:, 0], pts[:, 1]
    if rotation == 90:
        return np.stack([h - 1 - y, x], 1)
    if rotation == 180:
        return np.stack([w - 1 - x, h - 1 - y], 1)
    if rotation == 270:
        return np.stack([y, w - 1 - x], 1)
    return pts.copy()


class Lens:
    """Camera intrinsics from camera_intrinsics.json, scaled to the live resolution, if there are any."""

    def __init__(self, path=INTRINSICS_FILE):
        self.K0 = self.dist = self.size0 = None
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            self.K0, self.dist, self.size0 = np.array(d["K"]), np.array(d["dist"]), tuple(d["image_size"])
            self.note = f"Camera intrinsics loaded ({self.size0[0]} x {self.size0[1]}, RMS {d.get('rms', 0):.2f} px)"
        else:
            self.note = ("No camera_intrinsics.json: points are not undistorted and the focal length is "
                         "guessed, so depths in mm are approximate (run --intrinsics to fix this)")
        self._cache = {}

    def K(self, w, h):
        """K for a w x h raw picture, or None if there is no calibration that fits that shape."""
        if self.K0 is None:
            return None
        if (w, h) not in self._cache:
            W0, H0 = self.size0
            if abs(w / h - W0 / H0) > 0.01:
                print(f"Intrinsics are for {W0} x {H0}, camera gives {w} x {h} (different shape): not used")
                self._cache[(w, h)] = None
            else:
                K = self.K0.copy()
                K[:2] *= w / W0                   # same sensor, scaled picture
                self._cache[(w, h)] = K
        return self._cache[(w, h)]

    def active(self, w, h):
        return self.K(w, h) is not None

    def focal(self, w, h):
        """Focal length in pixels for a w x h raw picture."""
        K = self.K(w, h)
        if K is not None:
            return float((K[0, 0] + K[1, 1]) / 2)
        return (w / 2) / np.tan(np.radians(FALLBACK_HFOV_DEG) / 2)

    def principal_view(self, rotation, w, h):
        """Principal point (cx, cy) in the rotated view of a w x h raw picture."""
        K = self.K(w, h)
        if K is not None:
            return raw_to_view(np.array([[K[0, 2], K[1, 2]]]), rotation, w, h)[0]
        vw, vh = (h, w) if rotation in (90, 270) else (w, h)
        return np.array([vw / 2.0, vh / 2.0])

    def undistort_view(self, pts, rotation, w, h):
        """Undistort points given in the rotated view (the lens model lives in the raw picture)."""
        K = self.K(w, h)
        if K is None:
            return pts
        raw = view_to_raw(pts, rotation, w, h).reshape(-1, 1, 2)
        und = cv2.undistortPoints(raw, K, self.dist, P=K).reshape(-1, 2)
        return raw_to_view(und, rotation, w, h)


# ---------------------------------------------------------------- where the rig is, even mid-move
class Motion:
    """
    Wraps RigMover so the rig's position can be estimated while it is moving
    (RigMover only knows where a move ended), and remembers the (clipped) target.
    """

    def __init__(self, mover):
        self.mover = mover
        self.start = self.target = mover.pos.copy()
        self.t0, self.speed = 0.0, 1.0

    def move(self, target, feed=ff.TRACK_FEED):
        target = np.clip(np.asarray(target, float), ff.LO, ff.HI)
        self.start, self.target = self.now(), target
        self.t0, self.speed = time.time(), feed / 60.0
        self.mover.move(target, feed=feed)

    def now(self):
        """Estimated rig position (mm) right now."""
        if self.mover.idle.is_set():
            return self.mover.pos.copy()
        d = self.target - self.start
        L = np.linalg.norm(d)
        if L < 1e-6:
            return self.target.copy()
        frac = min(1.0, (time.time() - self.t0) * self.speed / L)
        return self.start + frac * d


# ---------------------------------------------------------------- stereo pairs from the rig
class StereoCapture:
    """
    Once the face is centred, moves the rig to anchor -/+ STEREO_BASELINE / 2 along
    STEREO_AXIS, back and forth, and takes a still picture at each end once the rig
    has stopped and settled. step() returns (previous picture, new picture) each time
    a new picture is taken, after the first.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.anchor, self.last, self.side = None, None, -1
        self.waiting, self.idle_since, self.idle_frames = False, None, 0

    @property
    def active(self):
        return self.anchor is not None

    @property
    def phase(self):
        if self.anchor is None:
            return "waiting for the rig to stop"
        return "moving / settling" if self.waiting else "next move"

    def step(self, mover, motion, grey, det):
        if self.anchor is None:
            if not mover.idle.is_set():
                return None
            self.anchor = mover.pos.copy()
        if not self.waiting:
            if not mover.idle.is_set():
                return None
            axis = np.asarray(STEREO_AXIS, float)
            axis /= np.linalg.norm(axis)
            motion.move(self.anchor + self.side * STEREO_BASELINE / 2 * axis, feed=STEREO_FEED)
            self.waiting, self.idle_since, self.idle_frames = True, None, 0
            return None
        # wait for the rig to arrive and settle
        arrived = mover.idle.is_set() and np.linalg.norm(mover.pos - motion.target) < 0.5
        if not arrived:
            self.idle_since, self.idle_frames = None, 0
            return None
        if self.idle_since is None:
            self.idle_since = time.time()
        self.idle_frames += 1
        if self.idle_frames < SETTLE_FRAMES or time.time() - self.idle_since < SETTLE_S or det is None:
            return None
        cap = {"grey": grey, "box": tuple(float(v) for v in det["box"]),
               "pos": mover.pos.copy(), "centre": face_centre(det)}
        prev, self.last = self.last, cap
        self.side, self.waiting = -self.side, False
        return None if prev is None else (prev, cap)


def inner_mask(shape, box, offset):
    """Mask of the central FACE_INNER part of the face box, in a crop that starts at offset."""
    x, y, w, h = box
    fw, fh = FACE_INNER
    cx, cy = x + w / 2 - offset[0], y + h / 2 - offset[1]
    mask = np.zeros(shape, np.uint8)
    x0, x1 = int(max(0, cx - fw * w / 2)), int(min(shape[1], cx + fw * w / 2))
    y0, y1 = int(max(0, cy - fh * h / 2)), int(min(shape[0], cy + fh * h / 2))
    mask[y0:y1, x0:x1] = 255
    return mask


def analyse_pair(c0, c1, lens, rotation, raw_w, raw_h, J):
    """
    Runs the stereo method on two captures. Returns the stereo_face_liveness result
    dict plus 'draw_pts' (view pixels in the second picture), 'height' (mm, + = nearer
    than the plane) and 'baseline' (mm). Raises sfl.StereoError if the pair can't be used.
    """
    d_rig = c1["pos"] - c0["pos"]
    B = float(np.linalg.norm(d_rig))
    if B < MIN_BASELINE_MM:
        raise sfl.StereoError(f"rig only moved {B:.1f} mm between the pictures (at a limit?)")

    # match only in the face area: much faster than the whole frame
    H, W = c0["grey"].shape
    bx = np.array([c0["box"], c1["box"]], float)
    x0, y0 = bx[:, 0].min(), bx[:, 1].min()
    x1, y1 = (bx[:, 0] + bx[:, 2]).max(), (bx[:, 1] + bx[:, 3]).max()
    px, py = CROP_PAD * (x1 - x0), CROP_PAD * (y1 - y0)
    x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    x1, y1 = int(min(W, x1 + px)), int(min(H, y1 + py))
    g0 = np.ascontiguousarray(c0["grey"][y0:y1, x0:x1])
    g1 = np.ascontiguousarray(c1["grey"][y0:y1, x0:x1])
    m0, m1 = inner_mask(g0.shape, c0["box"], (x0, y0)), inner_mask(g1.shape, c1["box"], (x0, y0))
    p0, p1, *_ = sfl.extract_and_match(g0, g1, m0, m1)
    if len(p0) < 15:
        raise sfl.StereoError(f"only {len(p0)} matches")
    p0 += (x0, y0)
    p1 += (x0, y0)

    # did the face move the way the rig's movement predicts?
    flow = np.median(p1 - p0, axis=0)
    if J is not None:
        expected = J @ d_rig
        u = expected / (np.linalg.norm(expected) + 1e-9)
        if float(flow @ u) / (np.linalg.norm(flow) + 1e-9) < DIRECTION_MIN_COS:
            raise sfl.StereoError("face didn't move with the rig (someone moving, or a video?)")
    else:
        u = flow / (np.linalg.norm(flow) + 1e-9)

    q0 = lens.undistort_view(p0, rotation, raw_w, raw_h)
    q1 = lens.undistort_view(p1, rotation, raw_w, raw_h)
    f = lens.focal(raw_w, raw_h)
    cx, cy = lens.principal_view(rotation, raw_w, raw_h)
    r = sfl.analyse(q0, q1, f, cx, cy, B, "rectified", verbose=False, direction=u)

    n = r["plane"]["normal"]
    n = -n if n[2] > 0 else n                    # point the normal at the camera
    r["height"] = (r["points"] - r["plane"]["centroid"]) @ n
    r["draw_pts"] = p1[r["keep"]]
    r["face_centre"] = c1["centre"]
    r["baseline"] = B

    # D. face vs its surroundings: same flat surface = a picture of a face
    r["bg"] = background_test(c0, c1, u) if BG_CHECK else None
    r["code_face"] = r["code"]
    bg, pl = r["bg"], r["plane"]
    shape_3d = pl["flatness"] > sfl.FLATNESS_THRESH and pl["quad_flatness"] > sfl.QUAD_RESID_THRESH   # A and C
    if bg is not None and bg["same_plane"]:
        r["code"] = "FLAT"                       # printed on the same surface as its surroundings
    elif r["code"] == "UNSURE" and shape_3d and bg is not None and bg["face_off_px"] >= BG_DIFF_PX:
        r["code"] = "LIVE"                       # 3D face, not part of the surface around it
    return r


def fit_disparity_plane(xy, d, iters=300, tol=1.0, rng=np.random.default_rng(0)):
    """
    Robust (RANSAC) fit of d = a + b x + c y. Under a sideways camera move every flat surface,
    turned any way, gives disparities that fit this exactly. Returns (coef, inlier mask) or None.
    """
    n = len(d)
    if n < 3:
        return None
    X = np.column_stack([np.ones(n), xy])
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            coef = np.linalg.solve(X[idx], d[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(X @ coef - d) < tol
        if best is None or inl.sum() > best[1].sum():
            best = (coef, inl)
    if best is None or best[1].sum() < 3:
        return None
    inl = best[1]
    coef, *_ = np.linalg.lstsq(X[inl], d[inl], rcond=None)           # refit on the inliers
    return coef, np.abs(X @ coef - d) < tol


def background_test(c0, c1, u):
    """
    D: is the face on the same flat surface as what is around it?
    Matches SIFT points in the face box grown by BG_PAD face widths, splits them into face
    (central part of the face box) and surroundings (outside the face box grown by BG_GAP),
    takes each point's shift along the camera move (disparity), fits the surroundings' plane
    (the biggest flat surface around the face: a phone screen / the paper of a photo, or for a
    real person the wall or their shoulders) and measures how far the face is off it.
    Returns None if there aren't enough points around the face to tell (plain background).
    """
    H, W = c0["grey"].shape
    x, y, w, h = c0["box"]
    pad = BG_PAD * w
    x0, y0 = int(max(0, x - pad)), int(max(0, y - pad))
    x1, y1 = int(min(W, x + w + pad)), int(min(H, y + h + pad))
    g0 = np.ascontiguousarray(c0["grey"][y0:y1, x0:x1])
    g1 = np.ascontiguousarray(c1["grey"][y0:y1, x0:x1])
    mask = np.full(g0.shape, 255, np.uint8)
    try:
        p0, p1, *_ = sfl.extract_and_match(g0, g1, mask, mask)
    except sfl.StereoError:
        return None
    if len(p0) < BG_MIN_POINTS:
        return None
    p0 = p0 + (x0, y0)
    p1 = p1 + (x0, y0)
    d = (p1 - p0) @ u                                   # disparity along the move
    fw, fh = FACE_INNER
    cx, cy = x + w / 2, y + h / 2
    in_face = (np.abs(p0[:, 0] - cx) < fw * w / 2) & (np.abs(p0[:, 1] - cy) < fh * h / 2)
    gx, gy = BG_GAP * w, BG_GAP * h
    around = (p0[:, 0] < x - gx) | (p0[:, 0] > x + w + gx) | (p0[:, 1] < y - gy) | (p0[:, 1] > y + h + gy)
    if around.sum() < BG_MIN_POINTS or in_face.sum() < 10:
        return None
    fit = fit_disparity_plane(p0[around], d[around])
    if fit is None or fit[1].sum() < BG_MIN_POINTS:
        return None
    coef, inl = fit
    pred_face = np.column_stack([np.ones(in_face.sum()), p0[in_face]]) @ coef
    off = float(np.median(np.abs(d[in_face] - pred_face)))
    return {"face_off_px": off, "same_plane": off < BG_SAME_PX, "n_around": int(inl.sum()),
            "n_face": int(in_face.sum()), "face_disp": float(np.median(d[in_face])),
            "around_disp": float(np.median(d[around][inl]))}


class Decision:
    """
    Turns the pair verdicts into one YES / NO. The answer comes from the number of
    pairs analysed (DECISION_PAIRS), not from a short clock, because each pair takes
    over a second of rig movement. DECISION_TIME is only a safety net for when every
    pair gets skipped. Skipped pairs are counted with their reason so a NO says why.
    state: WAITING (not measuring yet), CHECKING, YES or NO.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.t0, self.codes, self.state, self.reason = None, [], "WAITING", ""
        self.skipped = Counter()
        self.expected_px = None                  # parallax a real face gives at this distance

    def start(self):
        if self.t0 is None:
            self.t0, self.state = time.time(), "CHECKING"

    @property
    def elapsed(self):
        return 0.0 if self.t0 is None else time.time() - self.t0

    @property
    def done(self):
        return self.state in ("YES", "NO")

    def skip(self, reason):
        if not self.done:
            self.skipped[re.sub(r"\d+(\.\d+)?", "#", reason)] += 1

    def _hints(self):
        out = []
        if self.skipped:
            why, n = self.skipped.most_common(1)[0]
            out.append(f"{sum(self.skipped.values())} pairs skipped, mostly: {why} ({n}x)")
        if self.expected_px is not None and self.expected_px < 2 * PARALLAX_THRESH_PX:
            out.append(f"too far away: a real face only gives ~{self.expected_px:.1f} px of parallax here, move closer")
        return ("; " + "; ".join(out)) if out else ""

    def add(self, code, expected_px=None):
        if self.done:
            return
        self.codes.append(code)
        if expected_px is not None:
            self.expected_px = expected_px
        n = Counter(self.codes)
        live, fake = n["LIVE"], n["FLAT"] + n["CURVED"]
        if live >= LIVE_NEEDED and live > len(self.codes) - live:
            self.state, self.reason = "YES", f"3D face confirmed ({live} of {len(self.codes)} pairs)"
        elif fake >= FAKE_NEEDED and fake > live:
            kind = "bent photo or curved screen" if n["CURVED"] > n["FLAT"] else "flat photo or screen"
            self.state, self.reason = "NO", f"{kind} ({' '.join(self.codes)}){self._hints()}"
        elif len(self.codes) >= DECISION_PAIRS:
            self.state, self.reason = "NO", f"only {live} of {len(self.codes)} pairs LIVE ({' '.join(self.codes)}){self._hints()}"

    def tick(self):
        if self.state == "CHECKING" and self.elapsed > DECISION_TIME:
            self.state = "NO"
            self.reason = (f"no answer in {DECISION_TIME:.0f} s, {len(self.codes)} pairs analysed "
                           f"({' '.join(self.codes) or 'none'}){self._hints()}")


# ---------------------------------------------------------------- log of labelled scores
LOG_FIELDS = ["time", "label", "verdict", "pair_verdict", "flatness", "parallax_p90", "quad_flatness",
              "relief_mm", "depth_mm", "baseline_mm", "points", "lens_corrected"]
METRICS = [("flatness", FLATNESS_THRESH, "FLATNESS_THRESH"),
           ("parallax_p90", PARALLAX_THRESH_PX, "PARALLAX_THRESH_PX"),
           ("quad_flatness", QUAD_RESID_THRESH, "QUAD_RESID_THRESH")]


def log_scores(label, verdict, r, lens_ok):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": datetime.now().isoformat(timespec="seconds"), "label": label,
                    "verdict": verdict, "pair_verdict": r["code"],
                    "flatness": r["plane"]["flatness"], "parallax_p90": r["homog"]["p90_px"],
                    "quad_flatness": r["plane"]["quad_flatness"], "relief_mm": r["plane"]["relief"],
                    "depth_mm": r["median_depth"], "baseline_mm": r["baseline"],
                    "points": r["plane"]["n_points"], "lens_corrected": int(lens_ok)})


def offline_report(path=LOG_FILE):
    if not os.path.exists(path):
        print(f"No log yet ({path}). Press 1, 2 or 3 during a live run to add rows.")
        return
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    rows = [r for r in rows if r.get("flatness") and r.get("verdict") in ("YES", "NO", "CHECKING")]
    if not rows:
        print("No rows from the stereo method in the log yet.")
        return
    num = lambda v: float(v) if v not in ("", "None", None) else None
    print(f"{len(rows)} logged readings\n")
    for label in ("real", "print", "screen"):
        rs = [r for r in rows if r["label"] == label]
        if not rs:
            continue
        print(f"{label:7s} {len(rs):3d} readings")
        for key, _, _ in METRICS + [("relief_mm", None, None)]:
            vals = [num(r[key]) for r in rs if num(r[key]) is not None]
            if vals:
                print(f"        {key:14s} min {min(vals):8.4f}  mean {np.mean(vals):8.4f}  max {max(vals):8.4f}")
        verdicts = {v: sum(r["verdict"] == v for r in rs) for v in ("YES", "NO", "CHECKING")}
        print(f"        verdicts  {verdicts}")
    real = [r for r in rows if r["label"] == "real"]
    attacks = [r for r in rows if r["label"] in ("print", "screen")]
    if attacks:
        print(f"\nAPCER (attacks answered YES): {np.mean([r['verdict'] == 'YES' for r in attacks]):.1%} of {len(attacks)}")
    if real:
        print(f"BPCER (real faces not YES):    {np.mean([r['verdict'] != 'YES' for r in real]):.1%} of {len(real)}")
    if real and attacks:
        print()
        for key, now, name in METRICS:
            rv = [num(r[key]) for r in real if num(r[key]) is not None]
            av = [num(r[key]) for r in attacks if num(r[key]) is not None]
            if rv and av:
                gap = "separated" if max(av) < min(rv) else "OVERLAP"
                print(f"{key:14s} highest attack {max(av):.4f}, lowest real {min(rv):.4f} ({gap}): "
                      f"put {name} between them (now {now}).")


# ---------------------------------------------------------------- camera intrinsics from the capture_grid images
def find_board(gray_small):
    """Inner-corner count of the checkerboard in a picture, tried from the biggest down."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    sizes = sorted({(c, r) for c in range(4, 15) for r in range(3, 12) if c > r}, key=lambda s: -s[0] * s[1])
    for size in sizes:
        if cv2.findChessboardCorners(gray_small, size, flags=flags)[0]:
            return size
    return None


def calibrate_intrinsics(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
    if not files:
        print(f"No images in {folder}")
        return
    print(f"{len(files)} images in {folder}")
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)

    def small_grey(path):
        img = cv2.imread(path)
        if img is None:
            return None, None, 1.0
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = min(1.0, 960 / grey.shape[1])
        return grey, cv2.resize(grey, None, fx=s, fy=s) if s < 1 else grey, s

    board = tuple(BOARD_INNER) if BOARD_INNER else None
    if board is None:
        for p in files[:8]:
            _, small, _ = small_grey(p)
            if small is not None:
                board = find_board(small)
                if board:
                    break
        if board is None:
            print("Couldn't find a checkerboard. Set BOARD_INNER = (cols, rows) of INNER corners and retry.")
            return
    print(f"Checkerboard with {board[0]} x {board[1]} inner corners")

    obj = np.zeros((board[0] * board[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)   # square size doesn't matter for K
    obj_pts, img_pts, size = [], [], None
    for i, p in enumerate(files, 1):
        grey, small, s = small_grey(p)
        if grey is None:
            continue
        size = (grey.shape[1], grey.shape[0])
        ok, corners = cv2.findChessboardCorners(small, board, flags=flags)
        if not ok:
            continue
        corners = cv2.cornerSubPix(grey, (corners / s).astype(np.float32), (11, 11), (-1, -1), criteria)
        obj_pts.append(obj)
        img_pts.append(corners)
        if i % 20 == 0:
            print(f"  {i}/{len(files)} images, board found in {len(img_pts)}")
    if len(img_pts) < 8:
        print(f"Board found in only {len(img_pts)} images, need at least 8.")
        return
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    with open(INTRINSICS_FILE, "w") as f:
        json.dump({"K": K.tolist(), "dist": dist.ravel().tolist(), "image_size": list(size),
                   "rms": rms, "n_images": len(img_pts), "board_inner": list(board), "folder": folder}, f, indent=2)
    hfov = np.degrees(2 * np.arctan(size[0] / 2 / K[0, 0]))
    print(f"Calibrated from {len(img_pts)} images: RMS reprojection error {rms:.3f} px "
          f"({'good' if rms < 1 else 'high, check the images'})")
    print(f"fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}  (horizontal view {hfov:.0f} deg)")
    print("Saved:", INTRINSICS_FILE)


# ---------------------------------------------------------------- face centring
def tracking_step(det, A, roll, mover, motion, centre, band):
    """One centring step. Returns (status, colour, settled)."""
    err = face_centre(det) - centre
    err[np.abs(err) < band] = 0
    if not err.any():
        return "Centred", GREEN, True
    if mover.ready():
        J_inv = np.linalg.inv(rot(roll) @ A)
        target, limits = plan_step(J_inv, err, mover.pos)
        if np.abs(target - mover.pos).max() >= ff.MIN_MOVE:
            motion.move(target)
        if limits:
            return f"AT LIMIT ({', '.join(limits)}): can't centre the face", RED, True
    return "Centring", ORANGE, False


def help_lines(connected, busy):
    if not connected:
        return ["SPACE  connect the rig (carriage at bottom left first)", "r      turn the view 90 deg",
                "f      mirror on / off", "h      hide this list", "q      quit"]
    if busy:
        return ["x      cancel", "q      quit"]
    return ["t / SPACE  tracking on / off", "v      check again (new YES / NO)", "d      stereo pairs on / off", "c      calibrate tracking (keep still)",
            "1 2 3  log scores as: real / print / screen", "m      rig to the middle",
            "r      turn the view 90 deg", "f      mirror on / off", "h      hide this list",
            "q      quit (rig goes back to the corner)"]


VERDICT_TEXT = {"YES": ("YES - REAL FACE", GREEN), "NO": ("NO - NOT A REAL FACE", RED),
                "CHECKING": ("CHECKING...", YELLOW), "WAITING": ("WAITING TO MEASURE", GREY)}


def depth_colour(dz, span=15.0):
    """Red = in front of the face plane, blue = behind it, green = on it (dz and span in mm)."""
    t = float(np.clip((dz + span) / (2 * span), 0, 1))
    return (int(255 * (1 - t)), int(255 * (1 - abs(2 * t - 1))), int(255 * t))


def score_bar(img, org, value, threshold, width=220):
    """Bar from 0 to 2 x threshold with the threshold marked."""
    x, y = org
    cv2.rectangle(img, (x, y), (x + width, y + 10), (60, 60, 60), -1)
    if value is not None:
        fill = int(width * min(1.0, value / (2 * threshold)))
        cv2.rectangle(img, (x, y), (x + fill, y + 10), GREEN if value > threshold else RED, -1)
    cv2.line(img, (x + width // 2, y - 3), (x + width // 2, y + 13), WHITE, 2)


# ---------------------------------------------------------------- main loop
def main(camera_index=CAMERA_INDEX):
    ff.CAMERA_INDEX = camera_index           # so any face_follow helper that reads it uses the same camera
    lens = Lens()
    print(lens.note)
    yunet = fd.load_yunet()
    print(f"Opening camera {camera_index}")
    cam = FrameGrabber(open_camera(camera_index))
    cv2.namedWindow(ff.WINDOW, cv2.WINDOW_AUTOSIZE)

    rotation, mirror = ff.ROTATION, ff.MIRROR
    A = load_calibration()
    print("Tracking calibration:", "loaded" if A is not None else "none yet, press c once a face is in view")
    print(__doc__)
    print(">>> Click on the video window, then use the keys there. <<<", flush=True)

    rig = mover = motion = cal = None
    tracking, stereo_on, show_help, first_frame = False, STEREO, True, True
    stereo, decision = StereoCapture(), Decision()
    roll_vec, n_frame, lost, last_centre = None, 0, 0, None
    last, why, pairs = None, "", 0                  # latest pair result, why the last pair failed, pairs used
    note = {"text": "", "until": 0.0, "colour": WHITE}
    fps, t_prev = 0.0, time.time()
    frame_id, det_ms = 0, 0.0
    pool = ThreadPoolExecutor(max_workers=1)    # the stereo analysis runs here, off the video loop
    job, gen = None, 0                          # (running analysis, gen it belongs to), bumped on every reset

    def say(msg, colour=WHITE):
        print(msg, flush=True)
        note.update(text=msg, until=time.time() + ff.MESSAGE_TIME, colour=colour)

    def reset_live():
        nonlocal last, why, pairs, gen
        gen += 1                                 # an analysis still running for the old check is ignored
        stereo.reset()
        decision.reset()
        last, why, pairs = None, "", 0

    try:
        while True:
            ok, raw, frame_id = cam.read(frame_id)
            if not ok:
                print("Frame grab failed, stopping.")
                break
            n_frame += 1
            raw_h, raw_w = raw.shape[:2]
            frame = rotate(raw, rotation)
            t_det = time.perf_counter()
            det = fd.detect_scaled(yunet, frame, max_w=ff.DETECT_W)
            det_ms = 0.9 * det_ms + 0.1 * 1000 * (time.perf_counter() - t_det)

            # camera turned so far the face is sideways: try the other 90 degree turns
            if det is None and ff.AUTO_ROTATE and cal is None and n_frame % ff.AUTO_EVERY == 0:
                for r in (90, 270, 180):
                    cand = (rotation + r) % 360
                    f2 = rotate(raw, cand)
                    d2 = fd.detect_scaled(yunet, f2, max_w=ff.DETECT_W)
                    if d2 is not None:
                        rotation, frame, det, roll_vec = cand, f2, d2, None
                        reset_live()
                        say(f"Face found with the view turned {rotation} degrees, switched to that", YELLOW)
                        break

            h, w = frame.shape[:2]
            centre = np.array([w / 2.0, h / 2.0])
            band = ff.DEADBAND * np.array([w, h])

            if det is not None:
                lost = 0
                v = np.array([np.cos(face_roll(det)), np.sin(face_roll(det))])
                roll_vec = v if roll_vec is None else (1 - ff.ROLL_SMOOTH) * roll_vec + ff.ROLL_SMOOTH * v
                fc = face_centre(det)
                # a jump much bigger than a stereo move: a different face, start again
                if last_centre is not None and np.linalg.norm(fc - last_centre) > 0.5 * det["box"][2]:
                    reset_live()
                last_centre = fc
            else:
                lost += 1
                if lost == 10:
                    reset_live()
                    last_centre = None
            roll = None if roll_vec is None else float(np.arctan2(roll_vec[1], roll_vec[0]))
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if det is not None else None

            # ---- rig (centring, then stereo pairs)
            pair = None
            if mover is None:
                status, colour = "Rig not connected: carriage to the bottom left, then press SPACE", YELLOW
            elif mover.error is not None:
                tracking, cal = False, None
                status, colour = f"Rig error: {mover.error}  (press q and check the rig)", RED
            elif cal is not None:
                status, colour = f"Calibrating: {cal.label}. Keep still (x cancels)", ORANGE
                try:
                    result = cal.step(det)
                    if result is not None:
                        cal = None
                        A, msg = result
                        save_calibration(A)
                        reset_live()
                        say("Calibrated. " + msg + "   Press t to start", GREEN)
                except CalibrationFailed as e:
                    cal = None
                    say(f"Calibration failed: {e}. Press c to try again", RED)
            elif det is None:
                status, colour = "No face found: holding", YELLOW
            elif A is None:
                status, colour = "Not calibrated: press c (keep still)", YELLOW
            elif not tracking:
                status, colour = "Tracking off: press t (the rig has to move to measure depth)", GREY
            else:
                if stereo.active and mover.idle.is_set() and roll is not None:
                    # where the nose would be with the rig back at the centred position
                    at_anchor = face_centre(det) - rot(roll) @ A @ (mover.pos - stereo.anchor)
                    if np.any(np.abs(at_anchor - centre) > RECENTRE_BAND * np.array([w, h])):
                        stereo.reset()               # you moved: centre on the nose again, then carry on
                measuring = stereo_on and not decision.done
                if measuring and stereo.active:
                    if job is None:                  # wait for the last pair's analysis before the next move
                        pair = stereo.step(mover, motion, grey, det)
                    decision.start()                 # the clock runs from the first stereo move
                    status, colour = f"Centred, stereo pairs ({stereo.phase})", GREEN
                else:
                    status, colour, settled = tracking_step(det, A, roll, mover, motion, centre, band)
                    if settled and measuring:
                        pair = stereo.step(mover, motion, grey, det)   # anchors once the rig is still
                    elif decision.done and settled:
                        status = "Centred, answer given (v to check again)"

            # ---- liveness: each stereo pair is analysed in the background, the video keeps running
            if pair is not None:
                if job is None:
                    J = rot(roll) @ A if (A is not None and roll is not None) else None
                    job = (pool.submit(timed_analyse, pair[0], pair[1], lens, rotation, raw_w, raw_h, J), gen)
                else:
                    why = "previous pair still being analysed, this one skipped"
                    decision.skip(why)
            if job is not None and job[0].done():
                fut, job_gen = job
                job = None
                if job_gen == gen:                   # not from before a reset
                    try:
                        last = fut.result()
                        decision.add(last["code"], last.get("expected_face_parallax_px"))
                        pairs += 1
                        why = ""
                        print(f"pair {pairs}: {last['code']:6s} flatness {last['plane']['flatness']:.4f}  "
                              f"parallax p90 {last['homog']['p90_px']:.2f}px  "
                              f"smooth {last['plane']['quad_flatness']:.4f}  relief {last['plane']['relief']:.1f}mm  "
                              f"{last['plane']['n_points']} pts  B {last['baseline']:.1f}mm  "
                              f"Z {last['median_depth']:.0f}mm  real face would give ~{last['expected_face_parallax_px']:.1f}px  "
                              f"{bg_text(last)}  "
                              f"({last['ms']:.0f} ms)", flush=True)
                    except sfl.StereoError as e:
                        why = str(e)
                        decision.skip(why)
                        print(f"pair skipped: {why}", flush=True)
                    except Exception as e:           # a bug in the analysis shouldn't stop the video
                        why = f"analysis error: {e}"
                        decision.skip("analysis error")
                        print("Analysis error:", repr(e), flush=True)
            was_done = decision.done
            decision.tick()
            if decision.done and not was_done:
                stereo.reset()                       # stop the stereo moves, keep following the face
                say(f"{decision.state}: {decision.reason}", GREEN if decision.state == "YES" else RED)
            shown = decision.state

            # ---- draw
            view = frame.copy()
            cx, cy = int(centre[0]), int(centre[1])
            bx, by = int(band[0]), int(band[1])
            cv2.rectangle(view, (cx - bx, cy - by), (cx + bx, cy + by), (255, 255, 0), 1)
            cv2.drawMarker(view, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 30, 1)
            if det is not None:
                if last is not None:
                    shift = face_centre(det) - last["face_centre"]       # follow the face since the picture
                    inl = last["plane"]["inliers"]
                    for (px, py), dz, s in zip(last["draw_pts"] + shift, last["height"], inl):
                        cv2.circle(view, (int(px), int(py)), 2 if s else 1,
                                   depth_colour(dz) if s else (150, 150, 150), -1)
                x, y, fw, fh = (int(v) for v in det["box"])
                cv2.rectangle(view, (x, y), (x + fw, y + fh), VERDICT_TEXT[shown][1], 2)
                nose = tuple(int(v) for v in face_centre(det))           # the point the rig centres on
                cv2.line(view, (cx, cy), nose, (255, 255, 0), 1)
                cv2.circle(view, nose, 5, (255, 0, 255), -1)
            if mirror:
                view = cv2.flip(view, 1)

            text(view, status, (20, 40), colour)
            y0 = 76
            if time.time() < note["until"]:
                text(view, note["text"], (20, y0), note["colour"], 0.6, 1)
                y0 += 30
            if det is not None:
                title, vcol = VERDICT_TEXT[shown]
                y0 += 14
                text(view, title, (20, y0), vcol, 1.0, 2)
                y0 += 30
                if decision.state == "CHECKING":
                    n_skip = sum(decision.skipped.values())
                    text(view, f"pair {len(decision.codes)}/{DECISION_PAIRS}: {' '.join(decision.codes) or '-'}"
                               f"{f'   ({n_skip} skipped)' if n_skip else ''}", (20, y0), WHITE, 0.5, 1)
                    y0 += 24
                elif decision.done:
                    text(view, decision.reason + "   (v to check again)", (20, y0), vcol, 0.55, 1)
                    y0 += 26
                if last is not None:
                    pl, hg = last["plane"], last["homog"]
                    for label, value, thr in (("A flatness", pl["flatness"], FLATNESS_THRESH),
                                              ("B parallax", hg["p90_px"], PARALLAX_THRESH_PX),
                                              ("C not smooth", pl["quad_flatness"], QUAD_RESID_THRESH)):
                        text(view, f"{label:12s} {value:.4f}  (needs {thr})", (20, y0), WHITE, 0.5, 1)
                        score_bar(view, (330, y0 - 10), value, thr)
                        y0 += 24
                    extra = (f"{pl['n_points']} points, relief {pl['relief']:.0f} mm at {last['median_depth']:.0f} mm, "
                             f"baseline {last['baseline']:.0f} mm, {pairs} pairs")
                    exp_px = last.get("expected_face_parallax_px")
                    if exp_px is not None:
                        far = exp_px < 2 * PARALLAX_THRESH_PX
                        text(view, f"a real face should give ~{exp_px:.1f} px parallax here"
                                   f"{'  -> TOO FAR, move closer' if far else ''}", (20, y0), ORANGE if far else GREY, 0.45, 1)
                        y0 += 22
                else:
                    extra = f"{pairs} pairs"
                if why:
                    extra += f"   last pair skipped: {why}"
                text(view, extra, (20, y0), GREY, 0.45, 1)
                y0 += 24
            if show_help:
                y0 += 10
                for line in help_lines(mover is not None, cal is not None):
                    text(view, line, (20, y0), WHITE, 0.5, 1)
                    y0 += 24

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t_prev)
            t_prev = now
            rig_text = "rig not connected" if mover is None else \
                f"rig X {mover.pos[0]:6.1f}  Y {mover.pos[1]:6.1f} mm"
            ana = f"   analysis {last['ms']:.0f} ms" if last is not None and "ms" in last else ""
            text(view, f"{fps:4.1f} fps   detect {det_ms:.0f} ms{ana}   camera {camera_index}   {rig_text}   tracking {'ON' if tracking else 'off'}   "
                       f"stereo {'on' if stereo_on else 'off'} ({STEREO_BASELINE:.0f} mm)   view {rotation} deg   "
                       f"{'lens corrected' if lens.active(raw_w, raw_h) else 'no lens correction'}",
                 (20, h - 20), GREY, 0.5, 1)
            if h > ff.DISPLAY_MAX_H:
                view = cv2.resize(view, None, fx=ff.DISPLAY_MAX_H / h, fy=ff.DISPLAY_MAX_H / h)
            cv2.imshow(ff.WINDOW, view)
            if first_frame:
                bring_to_front()
                first_frame = False

            # ---- keys
            k = read_key()
            if window_closed() or k in ("q", "\x1b"):
                break
            if k is None:
                continue
            if k == "h":
                show_help = not show_help
                continue
            if k == "f":
                mirror = not mirror
                say("Picture mirrored" if mirror else "Picture not mirrored", WHITE)
                continue
            if k in LABELS:
                if last is None:
                    say("No scores yet to log (wait for a stereo pair)", YELLOW)
                else:
                    log_scores(LABELS[k], shown, last, lens.active(raw_w, raw_h))
                    say(f"Logged as '{LABELS[k]}': pair {last['code']}, answer {shown}", GREEN)
                continue
            if k == "v":
                reset_live()
                say("Checking again", WHITE)
                continue
            if k == "d":
                stereo_on = not stereo_on
                stereo.reset()
                say(f"Stereo pairs {'on' if stereo_on else 'off'}", WHITE)
                continue
            if k == "x":
                if cal is not None:
                    cal.cancel()
                    cal = None
                    say("Calibration cancelled", YELLOW)
                continue
            if cal is not None:
                say("Calibrating: wait, or press x to cancel", YELLOW)
                continue
            if k == "r":
                rotation = (rotation + 90) % 360
                roll_vec = None
                reset_live()
                say(f"View turned {rotation} degrees", YELLOW)
                continue

            if mover is None:
                if k in (" ", "\r"):
                    connecting = view.copy()
                    text(connecting, f"Connecting to the rig on {ff.PORT}...", (20, 110), ORANGE)
                    cv2.imshow(ff.WINDOW, connecting)
                    cv2.waitKey(1)
                    try:
                        rig = ff.Rig()
                        mover = RigMover(rig)
                        motion = Motion(mover)
                        motion.move(ff.MIDDLE, feed=ff.START_FEED)
                        say("Rig connected, moving to the middle of the travel", GREEN)
                    except Exception as e:
                        rig = None
                        say(f"Could not connect on {ff.PORT}: {e}. Close UGS / Arduino IDE, then SPACE again", RED)
                elif k in ("c", "t", "m"):
                    say("Connect the rig first: press SPACE", YELLOW)
                continue

            if mover.error is not None:
                say("The rig reported an error: press q, check it, then restart", RED)
                continue

            if k == "c":
                if det is None:
                    say("No face in view: get in frame, then press c", YELLOW)
                else:
                    tracking = False
                    stereo.reset()
                    cal = Calibration(mover)
                    say("Calibrating: keep still for a few seconds", ORANGE)
            elif k in ("t", " "):
                if A is None:
                    say("Calibrate first: press c with a face in view", YELLOW)
                else:
                    tracking = not tracking
                    stereo.reset()
                    say("Tracking ON" if tracking else "Tracking off", GREEN if tracking else GREY)
            elif k == "m":
                tracking = False
                stereo.reset()
                motion.move(ff.MIDDLE)
                say("Moving to the middle, tracking off", YELLOW)

    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        if mover is not None:
            mover.stop()
            if mover.error is None:
                print("Going back to the start corner...")
                try:
                    rig.home_corner()
                except Exception as e:
                    print(f"Could not go back to the corner: {e}")
            else:
                print("Not moving the rig after an error. Check it, then move the carriage back by hand.")
        if rig is not None:
            rig.close()
        pool.shutdown(wait=False)
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Face following with a live 3D liveness check (stereo pairs)")
    ap.add_argument("--intrinsics", metavar="FOLDER", help="calibrate the camera from capture_grid.py images")
    ap.add_argument("--analyse", action="store_true", help="report on the logged scores")
    args = ap.parse_args()
    if args.intrinsics:
        calibrate_intrinsics(args.intrinsics)
    elif args.analyse:
        offline_report()
    else:
        main()