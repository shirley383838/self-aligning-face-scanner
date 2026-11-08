"""
Moves the camera rig so the detected face stays centred.

Whenever the rig is still, the face position from YuNet is compared with the image
centre and the rig steps towards it through a calibrated image-to-rig mapping
(face_follow_cal.json, measured with the c key). The mapping is turned by the eye-line
roll every frame, so it still holds when the camera is rotated on its mount. Targets
are clamped to the safe area in rig_grid.py, and moves run on a background thread.

Start with the carriage at the bottom left (opening the port makes it (0, 0)).
Keys (video window): SPACE connect / tracking on-off, c calibrate, x cancel,
t tracking, m go to the middle, r rotate view, f mirror, h help, q quit.
"""

import os
import sys
import json
import time
import queue
import threading

import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "MajorProject"))

import face_detect as fd                                        # noqa: E402
from rig_grid import Rig, PORT, FEED, MARGIN, X_MAX, Y_MAX      # noqa: E402

# ---------------------------------------------------------------- settings
CAMERA_INDEX = 2          # the number find_cameras.py showed for the rig webcam
FRAME_W, FRAME_H = 1280, 720
DETECT_W = 640            # YuNet runs on a copy this wide for speed
ROTATION = 0              # 0, 90, 180 or 270: turn the camera picture clockwise (camera on its side)
MIRROR = True             # mirror the DISPLAY like a selfie camera (control always uses the unmirrored frame)
DISPLAY_MAX_H = 900       # shrink the window if the view is taller than this
AUTO_ROTATE = True        # with no face found, try the other 90 degree view turns and keep one that finds a face
AUTO_EVERY = 5            # how often (in frames) to try the other turns while no face is found
ROLL_SMOOTH = 0.3         # how quickly the measured camera roll follows the eyes (1 = no smoothing)

DEADBAND = 0.025          # nose counts as centred within this fraction of the frame size of the middle
GAIN = 0.6                # fraction of the error corrected per step (below 1 so it doesn't overshoot)
MAX_STEP = 20.0           # mm, biggest single tracking move
MIN_MOVE = 0.2            # mm, smaller moves than this are not sent
TRACK_FEED = FEED         # mm/min for tracking moves
START_FEED = 600          # mm/min for the first move from the corner to the middle
SETTLE_TIME = 0.25        # s to wait after a move before trusting the picture again

CAL_STEP = 25.0           # mm moved along each axis during calibration (bigger = less affected by small head movements)
CAL_FRAMES = 5            # face detections averaged at each calibration point
CAL_SKIP = 3              # frames thrown away after each calibration move (old buffered frames)
CAL_MAX_MISSES = 15       # frames in a row without a face before calibration gives up
CAL_FILE = os.path.join(BASE_DIR, "face_follow_cal.json")
CAL_VERSION = 2           # version 1 files (one fixed view angle) can't be converted, recalibrate once
CAL_MIN_ANGLE = 60.0      # deg: X and Y moves must push the face at least this far apart in direction, else redo
CAL_MAX_RATIO = 1.6       # and one axis may move the face at most this many times further per mm than the other

WINDOW = "face follow"
MESSAGE_TIME = 3.0        # s an on-screen message stays up after a key press

CAMERA_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
ROTATE_CODES = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

YELLOW, GREEN, RED, GREY, ORANGE, WHITE = (0, 255, 255), (0, 220, 0), (0, 0, 255), (200, 200, 200), (0, 200, 255), (255, 255, 255)

# safe area from rig_grid.py, the same box Rig.goto() checks
LO = np.array([MARGIN, MARGIN])
HI = np.array([X_MAX - MARGIN, Y_MAX - MARGIN])
MIDDLE = (LO + HI) / 2


# ---------------------------------------------------------------- rig on a background thread
class RigMover:
    """Runs rig moves on a background thread so the video doesn't freeze while the rig moves."""

    def __init__(self, rig, start=(0.0, 0.0)):
        self.rig = rig
        self.pos = np.array(start, float)    # where the last finished move left the rig, in mm
        self.done_at = time.time()           # when the last move finished
        self.error = None                    # set if GRBL complained, moves stop
        self.jobs = queue.Queue()
        self.idle = threading.Event()
        self.idle.set()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            x, y, feed = job
            try:
                self.rig.goto(x, y, feed=feed)
                self.pos = np.array([x, y])
            except Exception as e:           # GRBL error/alarm, timeout, or outside the safe area
                self.error = e
            self.done_at = time.time()
            if self.jobs.empty():
                self.idle.set()

    def move(self, target, feed=TRACK_FEED):
        """
        Start a move to target (x, y) in mm, clamped to the safe area. Returns at once.
        A move that is waiting to start is replaced, so the newest request always wins.
        """
        x, y = np.clip(target, LO, HI)
        while True:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                break
        self.idle.clear()
        self.jobs.put((float(x), float(y), feed))

    def ready(self):
        """True when the rig is still and has had time to settle."""
        return self.idle.is_set() and time.time() - self.done_at > SETTLE_TIME

    def stop(self):
        self.jobs.put(None)
        self.thread.join(timeout=60)


# ---------------------------------------------------------------- helpers
def open_camera(index):
    """Open the webcam and request the target resolution."""
    print(f"Opening camera {index}...", flush=True)
    cam = cv2.VideoCapture(index, CAMERA_BACKEND)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera {index}. Run MajorProject/find_cameras.py to see which numbers work.")
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # most webcams only reach 30 fps at 720p in MJPG
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)                              # keep the picture as fresh as possible
    for _ in range(10):                      # first frames are often black
        cam.read()
    print(f"Camera {index} open at {int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
          f"{int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))}", flush=True)
    return cam


def rotate(frame, degrees):
    """Turn the camera picture clockwise by 0, 90, 180 or 270 degrees."""
    code = ROTATE_CODES[degrees]
    return frame if code is None else cv2.rotate(frame, code)


def text(img, msg, org, colour, scale=0.7, thick=2):
    """Text with a dark box behind it. Shrinks the text if it would run off the right edge."""
    (tw, th), base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    room = img.shape[1] - x - 10
    if tw > room > 0:
        scale *= room / tw
        (tw, th), base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)


def face_centre(det):
    """
    Nose tip (YuNet landmark) in pixels, the point the rig centres on; falls back to the face box centre.
    """
    x, y, w, h = det["box"]
    row = det.get("row")
    if row is not None and len(row) >= 10 and row[2] > 0 and row[3] > 0:
        fx, fy = (row[8] - row[0]) / row[2], (row[9] - row[1]) / row[3]
        if -0.2 < fx < 1.2 and -0.2 < fy < 1.2:
            return np.array([x + fx * w, y + fy * h], float)
    return np.array([x + w / 2.0, y + h / 2.0])


def face_roll(det):
    """
    Eye-line angle in radians (0 = upright), from YuNet's raw landmarks so it works with the face sideways.
    """
    row = det["row"]
    return float(np.arctan2(row[7] - row[5], row[6] - row[4]))


def rot(theta):
    """2x2 rotation in picture coordinates: upright-face vector -> picture vector."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def mean_angle(angles):
    """Average of angles, done with unit vectors so -179 and +179 average to 180, not 0."""
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def read_key():
    """The key pressed in the video window as a lowercase character, or None."""
    key = cv2.waitKey(1)
    if key == -1:
        return None
    key &= 0xFF
    return chr(key).lower() if key < 128 else None


def window_closed():
    """True if the window was closed with its X button."""
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def bring_to_front():
    """Raise the video window above the terminal so it gets the key presses."""
    try:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 0)
    except (cv2.error, AttributeError):      # older OpenCV builds don't have this
        pass


# ---------------------------------------------------------------- calibration
def axis_angle(A):
    """Angle in degrees between the picture directions of the X and Y rig moves (ideally 90)."""
    a, b = A[:, 0], A[:, 1]
    cos = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, 0, 1))))


def square_up(A):
    """
    Make the two measured axis directions in A exactly perpendicular, as the rig axes are, keeping
    each axis's px/mm. A near-singular A would send the rig the wrong way.
    """
    U, _, Vt = np.linalg.svd(A)
    Q = U @ Vt                                    # nearest pure rotation / reflection
    return Q * np.linalg.norm(A, axis=0)          # each column back to its measured length


def load_calibration():
    """
    Saved calibration A (2x2, face pixels per mm in the upright-face frame), or None.
    At camera roll phi the image motion is rot(phi) @ A.
    """
    if not os.path.exists(CAL_FILE):
        return None
    with open(CAL_FILE) as f:
        cal = json.load(f)
    if cal.get("version") != CAL_VERSION:
        print("Old calibration file (made for one fixed view angle): press c to calibrate again.")
        return None
    A = np.array(cal["A"])
    if axis_angle(A) < CAL_MIN_ANGLE:
        print(f"The saved calibration looks wrong (X and Y move the face only {axis_angle(A):.0f} deg apart, "
              f"should be about 90). Using a squared-up version, but press c to calibrate again, keeping still.")
    return square_up(A)


def save_calibration(A):
    with open(CAL_FILE, "w") as f:
        json.dump({"version": CAL_VERSION, "A": A.tolist()}, f, indent=2)


class CalibrationFailed(Exception):
    pass


class Calibration:
    """
    Steps the rig along X then Y and measures the face motion, one detection per frame through
    step(). The measured J is turned back by the eye-line angle to give A, which does not depend
    on camera roll.
    """

    def __init__(self, mover):
        self.mover = mover
        self.home = None
        self.rolls = []
        self._gen = self._steps()
        self.label = next(self._gen)

    def step(self, det):
        """Feed one frame's detection. Returns None while running, (J, message) when done."""
        try:
            self.label = self._gen.send(det)
            return None
        except StopIteration as done:
            return done.value
        except CalibrationFailed:
            self.cancel()
            raise

    def cancel(self):
        """Send the rig back to where calibration started."""
        if self.home is not None and self.mover.error is None:
            self.mover.move(self.home)

    def _wait_for_rig(self, label):
        while not self.mover.ready():
            if self.mover.error is not None:
                raise CalibrationFailed(f"rig error: {self.mover.error}")
            yield label
        if self.mover.error is not None:
            raise CalibrationFailed(f"rig error: {self.mover.error}")

    def _measure(self, label):
        for _ in range(CAL_SKIP):            # frames from before the rig stopped
            yield label
        points, misses = [], 0
        while len(points) < CAL_FRAMES:
            det = yield label
            if det is None:
                misses += 1
                if misses > CAL_MAX_MISSES:
                    raise CalibrationFailed("lost the face, stay in view and keep still")
            else:
                misses = 0
                points.append(face_centre(det))
                self.rolls.append(face_roll(det))
        return np.mean(points, axis=0)

    def _steps(self):
        yield from self._wait_for_rig("waiting for the rig to stop")
        self.home = self.mover.pos.copy()
        steps = np.where(self.home < MIDDLE, CAL_STEP, -CAL_STEP)   # step towards the middle so it always fits

        J = np.zeros((2, 2))
        for axis, name in enumerate("XY"):
            # measure the start position again before each axis, so a slow head drift
            # during calibration doesn't end up in the result
            p0 = yield from self._measure(f"measuring start position ({name})")
            target = self.home.copy()
            target[axis] += steps[axis]
            self.mover.move(target)
            yield from self._wait_for_rig(f"stepping in {name}")
            moved = self.mover.pos[axis] - self.home[axis]          # the real step (the target may be clipped)
            if abs(moved) < 5.0:
                raise CalibrationFailed(f"the rig could only move {abs(moved):.1f} mm in {name} (at a limit?), press m first")
            p = yield from self._measure(f"measuring after the {name} step")
            self.mover.move(self.home)
            yield from self._wait_for_rig("moving back")
            J[:, axis] = (p - p0) / moved

        c1, c2 = np.linalg.norm(J[:, 0]), np.linalg.norm(J[:, 1])
        if min(c1, c2) < 0.1:
            raise CalibrationFailed("the face hardly moved (too far away, or did it move?)")
        angle = axis_angle(J)
        if angle < CAL_MIN_ANGLE:
            raise CalibrationFailed(f"X and Y moved the face only {angle:.0f} deg apart (should be ~90): "
                                    "keep your head completely still")
        if max(c1, c2) / min(c1, c2) > CAL_MAX_RATIO:
            raise CalibrationFailed(f"X moved the face {c1:.2f} px/mm but Y {c2:.2f} px/mm (should be similar): "
                                    "keep your head completely still")
        roll = mean_angle(self.rolls)
        A = square_up(rot(-roll) @ J)
        return A, (f"X {c1:.2f} px/mm, Y {c2:.2f} px/mm, {angle:.0f} deg apart (squared up to 90), "
                   f"camera roll {np.degrees(roll):+.0f} deg")


# ---------------------------------------------------------------- control
def plan_step(J_inv, err_px, pos):
    """
    Rig target that should centre the face. Returns (target, blocked), where blocked lists the
    limits in the way.
    """
    step = -GAIN * (J_inv @ err_px)
    n = np.linalg.norm(step)
    if n > MAX_STEP:
        step *= MAX_STEP / n
    wanted = pos + step
    target = np.clip(wanted, LO, HI)
    blocked = []
    for axis, name in enumerate("XY"):
        if wanted[axis] != target[axis] and abs(target[axis] - pos[axis]) < MIN_MOVE:
            blocked.append(f"{name} {'max' if wanted[axis] > HI[axis] else 'min'}")
    return target, blocked


def help_lines(connected, calibrating):
    """The key list shown on screen, matched to what the keys do right now."""
    if not connected:
        return ["SPACE  connect the rig (carriage at bottom left first)",
                "r      turn the view 90 deg", "f      mirror on / off", "h      hide this list", "q      quit"]
    if calibrating:
        return ["x      cancel calibration", "q      quit"]
    return ["c      calibrate (face in view, keep still)", "t / SPACE  tracking on / off",
            "m      rig to the middle", "r      turn the view 90 deg", "f      mirror on / off", "h      hide this list",
            "q      quit (rig goes back to the corner)"]


# ---------------------------------------------------------------- main loop
def main():
    yunet = fd.load_yunet()
    cam = open_camera(CAMERA_INDEX)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    rotation = ROTATION
    A = load_calibration()
    print("Loaded calibration from", CAL_FILE if A is not None else "nowhere yet: press c once a face is in view")
    roll_vec = None                          # smoothed (cos, sin) of the eye-line angle
    n_frame = 0
    print(__doc__)
    print(">>> Click on the video window, then use the keys there. <<<", flush=True)

    rig, mover, cal = None, None, None
    mirror = MIRROR
    tracking, limits, show_help, first_frame = False, [], True, True
    note = {"text": "", "until": 0.0, "colour": WHITE}

    def say(msg, colour=WHITE):
        """Show a message on screen for a few seconds, and print it in the terminal too."""
        print(msg, flush=True)
        note.update(text=msg, until=time.time() + MESSAGE_TIME, colour=colour)

    try:
        while True:
            ok, raw = cam.read()
            if not ok:
                print("Frame grab failed, stopping.")
                break
            n_frame += 1
            frame = rotate(raw, rotation)
            det = fd.detect_scaled(yunet, frame, max_w=DETECT_W)

            # camera turned so far the face is sideways: try the other 90 degree turns
            # (not during calibration, the picture has to stay the same way round for that)
            if det is None and AUTO_ROTATE and cal is None and n_frame % AUTO_EVERY == 0:
                for r in (90, 270, 180):
                    cand = (rotation + r) % 360
                    f2 = rotate(raw, cand)
                    d2 = fd.detect_scaled(yunet, f2, max_w=DETECT_W)
                    if d2 is not None:
                        rotation, frame, det = cand, f2, d2
                        roll_vec = None
                        say(f"Face found with the view turned {rotation} degrees, switched to that", YELLOW)
                        break

            # camera roll from the eye line, smoothed so one noisy frame can't jerk the rig
            if det is not None:
                v = np.array([np.cos(face_roll(det)), np.sin(face_roll(det))])
                roll_vec = v if roll_vec is None else (1 - ROLL_SMOOTH) * roll_vec + ROLL_SMOOTH * v
            roll = None if roll_vec is None else float(np.arctan2(roll_vec[1], roll_vec[0]))
            h, w = frame.shape[:2]
            centre = np.array([w / 2.0, h / 2.0])
            band = DEADBAND * np.array([w, h])

            # ---- decide what the rig should do
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
                        say("Calibrated. " + msg + "   Press t to start tracking", GREEN)
                except CalibrationFailed as e:
                    cal = None
                    say(f"Calibration failed: {e}. Press c to try again", RED)
            elif det is None:
                status, colour, limits = "No face found: holding", YELLOW, []
            elif A is None:
                status, colour = "Not calibrated: press c (keep still)", YELLOW
            elif not tracking:
                status, colour = "Tracking off: press t", GREY
            else:
                err = face_centre(det) - centre
                err[np.abs(err) < band] = 0           # close enough on that axis
                if not err.any():
                    status, colour, limits = "Centred", GREEN, []
                else:
                    if mover.ready():
                        J_inv = np.linalg.inv(rot(roll) @ A)   # calibration turned to the current camera roll
                        target, limits = plan_step(J_inv, err, mover.pos)
                        if np.abs(target - mover.pos).max() >= MIN_MOVE:
                            mover.move(target)
                    if limits:
                        status, colour = f"AT LIMIT ({', '.join(limits)}): can't centre the face", RED
                    else:
                        status, colour = "Moving", ORANGE

            # ---- draw
            view = frame.copy()
            cx, cy = int(centre[0]), int(centre[1])
            bx, by = int(band[0]), int(band[1])
            cv2.rectangle(view, (cx - bx, cy - by), (cx + bx, cy + by), (255, 255, 0), 1)
            cv2.drawMarker(view, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 30, 1)
            if det is not None:
                x, y, fw, fh = det["box"]
                fc = face_centre(det).astype(int)
                cv2.rectangle(view, (x, y), (x + fw, y + fh), colour, 2)
                cv2.circle(view, tuple(int(v) for v in fc), 4, colour, -1)
                cv2.line(view, (cx, cy), tuple(int(v) for v in fc), colour, 1)
            if mirror:
                view = cv2.flip(view, 1)

            text(view, status, (20, 40), colour)
            y0 = 76
            if time.time() < note["until"]:
                text(view, note["text"], (20, y0), note["colour"], 0.6, 1)
                y0 += 30
            if show_help:
                y0 += 10
                for line in help_lines(mover is not None, cal is not None):
                    text(view, line, (20, y0), WHITE, 0.5, 1)
                    y0 += 24

            if mover is None:
                rig_text = "rig not connected"
            else:
                moving = "" if mover.idle.is_set() else "  (moving)"
                rig_text = f"rig X {mover.pos[0]:6.1f}  Y {mover.pos[1]:6.1f} mm{moving}"
            text(view, f"{rig_text}   tracking {'ON' if tracking else 'off'}   "
                       f"{'calibrated' if A is not None else 'not calibrated'}   view {rotation} deg   "
                       f"camera roll {'?' if roll is None else f'{np.degrees(roll):+.0f}'} deg   "
                       f"{'mirrored' if mirror else 'not mirrored'}",
                 (20, h - 20), GREY, 0.5, 1)
            if h > DISPLAY_MAX_H:
                view = cv2.resize(view, None, fx=DISPLAY_MAX_H / h, fy=DISPLAY_MAX_H / h)
            cv2.imshow(WINDOW, view)
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
                say("Picture mirrored (like a selfie camera)" if mirror else "Picture not mirrored (raw camera view)", WHITE)
                continue

            if k == "r":
                if cal is not None:
                    say("Can't turn the view while calibrating (x cancels)", YELLOW)
                    continue
                rotation = (rotation + 90) % 360
                roll_vec = None
                say(f"View turned {rotation} degrees (calibration still holds)", YELLOW)
                continue

            if mover is None:
                if k in (" ", "\r"):
                    connecting = view.copy()
                    text(connecting, f"Connecting to the rig on {PORT}...", (20, 110), ORANGE)
                    cv2.imshow(WINDOW, connecting)
                    cv2.waitKey(1)
                    try:
                        rig = Rig()
                        mover = RigMover(rig)
                        mover.move(MIDDLE, feed=START_FEED)
                        say("Rig connected, moving to the middle of the travel", GREEN)
                    except Exception as e:
                        rig = None
                        say(f"Could not connect on {PORT}: {e}. Close UGS / Arduino IDE, then SPACE again", RED)
                elif k in ("c", "t", "m", "x"):
                    say("Connect the rig first: press SPACE", YELLOW)
                continue

            if mover.error is not None:
                say("The rig reported an error: press q, check it, then restart", RED)
                continue

            if cal is not None:
                if k == "x":
                    cal.cancel()
                    cal = None
                    say("Calibration cancelled", YELLOW)
                else:
                    say("Calibrating: wait, or press x to cancel", YELLOW)
                continue

            if k == "c":
                if det is None:
                    say("No face in view: get in frame, then press c", YELLOW)
                else:
                    tracking = False
                    cal = Calibration(mover)
                    say("Calibrating: keep still for a few seconds", ORANGE)
            elif k in ("t", " "):
                if A is None:
                    say("Calibrate first: press c with a face in view", YELLOW)
                else:
                    tracking = not tracking
                    say("Tracking ON" if tracking else "Tracking off", GREEN if tracking else GREY)
            elif k == "m":
                tracking = False
                mover.move(MIDDLE)
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
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()