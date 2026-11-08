"""
main.py  -  the whole e-gate on the rig: ID card, positioning, glasses / hat, 3D live check, face match

Airport e-gates check a traveller's identity without staff. This is the small-scale
version on the XY plotter: one webcam on the carriage, the traveller on the foot markers.
The plotter only moves the camera; every decision comes from images.

The steps, in order (the screen and the voice tell the traveller what to do):
  0. Start        the operator puts the carriage at the bottom left and presses SPACE.
                  The rig connects and moves the camera to the middle of its travel.
  1. Scan ID      "Hold your ID card up to the camera". The card doesn't have to be held
                  still: on every usable frame the card is found and flattened and its
                  4-digit number read; once the same number has come up CARD_VOTES times
                  (and the card photo was seen) the card is accepted. The face in the card
                  photo is stored. Nothing is confirmed yet.
  2. Position     the rig moves the camera to the traveller's height and centres the nose.
                  If the rig can't reach (at the end of its travel), a big arrow and the
                  voice ask the traveller to move left / right / up / down.
  3. Glasses, hat if the glasses or hat classifier sees one, the traveller is asked to
                  take it off, and the gate waits until the face is clear.
  4. Live check   ON THE RIG: the person holds still and the CAMERA moves: the rig slides it
                  LIVE_BASELINE_MM left and right (LIVE_AXIS) and takes still pictures at each
                  end. face_liveness.py compares each pair: SIFT points on the face are
                  triangulated and tested for 3D shape (nose, eye sockets), and points around
                  the face (the phone screen / paper, or the wall behind a real person) are
                  compared with the face: a face lying on the same flat surface as what's around
                  it is a picture. One automatic retry.
                  WITHOUT THE RIG (run_no_rig.py) the camera can't move, so instead:
                  a) head turn: "turn your head to your left ... now to your
                     right", in a random order. The eyes and mouth corners of a flat face (phone,
                     tablet, photo) lie in one plane, so those 4 points fix exactly how that plane
                     looks from any angle (a homography), and the nose must then sit exactly where
                     the flat face puts it, however the screen is turned, tilted or moved. A real
                     nose sticks out ~25 mm, so when a real head turns it lands away from that
                     prediction. Measured in eye spacings, along the eye line.
                  The gate never says IDENTITY CONFIRMED unless a live check has passed.
  5. Face match   "Look straight into the camera". The live face is compared with the card
                  photo AND with the enrolled record photo for that ID number, and the
                  card photo with the record photo (catches a swapped card photo).
  6. Photo        if everything matches, a photo is taken.
  7. Result       IDENTITY CONFIRMED, or why not (e.g. NOT THE SAME PERSON AS THE ID).
                  "Thank you, scanning is finished", plus "you can put your glasses back
                  on" if glasses were taken off. SPACE (or RESULT_HOLD_S seconds) = next traveller.

Session database (wiped every time main.py starts): session_db/
    session.db        SQLite, tables "travellers" (everyone who passed) and "attempts" (every run)
    travellers.csv    the same as the travellers table, to open in Excel
    attempts.csv
    images/           live photo, passport-style face crop, ID card face, flattened ID card,
                      record photo, for every traveller who passed (and the live photo of failed runs)
Match scores are percentages: 100 % = identical, 50 % = exactly at the match threshold,
0 % = twice the threshold or worse. Above 50 % counts as a match.

Voice: pip install pyttsx3 (falls back to Windows' built-in speech through PowerShell).
Every instruction shown on screen is also read out (NARRATE), so a traveller who can't
read the screen (e.g. glasses off) hears every step; v turns the voice off.

Keys (click the video window first):
    SPACE   start (connect the rig) / next traveller after a result
    n       start again with a new traveller (back to the ID scan)
    c       calibrate the tracking (face in view, keep still)    x  cancel it
    d       show / hide the details for the operator
    v       voice on / off
    r       turn the view 90 degrees         f  mirror the screen on / off
    w       full screen on / off             k  the webcam mirrors the picture: yes / no (normally found
                                                from the ID card by itself, and remembered)
    q / Esc quit (the rig goes back to the start corner)

Put this file next to face_follow.py and face_liveness.py (MajorProject folder beside them).
"""

import os
import csv
import sys
import time
import queue
import shutil
import sqlite3
import json
import random
import textwrap
from collections import Counter
import threading
import subprocess
from collections import deque
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
import face_liveness as fl            # also imports face_follow, which puts MajorProject on the path
import face_follow as ff
# face_follow puts the MajorProject folder FIRST on the search path. If there are copies of
# id_match.py, id_card.py ... next to main.py too, use those (the ones you run and build models with).
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)
import stereo_face_liveness as sfl
import face_detect as fd
import accessories as acc
import id_match as idm
import id_card as card
import digits as dg
import make_id_cards as mic
from face_align import align_face
import id_alert                       # (added 1 Nov, Shirley) red alert screen + who is really at the gate

# ---------------------------------------------------------------- settings
CAMERA_INDEX = fl.CAMERA_INDEX   # the rig webcam (same as face_liveness.py)
MIRROR = True                    # show the picture like a mirror (easier for the traveller)

CAMERA_MIRRORED = "auto"         # does the webcam itself mirror the picture? "auto" = work it out from the
                                 # ID card (its number only reads correctly the right way round) and
                                 # remember it in egate_camera.json; True / False = force it
MIRROR_DURING_SCAN = False       # mirror the screen while the ID card is scanned? False = the card is shown
                                 # the right way round, so you can read it on screen
FULLSCREEN = True                # use the whole monitor (w toggles); a camera on its side gets a side panel
CARD_GUIDE = 0.6                 # the on-screen box showing where to hold the card: this fraction of the
                                 # picture width (the whole card has to be in the picture to be found)
AUTOFOCUS = True                 # switch the webcam's autofocus on (capture_grid.py switches it off)
CARD_WINDOW_S = 3.0              # ID scan: number reads from the last few seconds are voted on
CARD_VOTES = 3                   # the same number read this many times = card accepted (no need to hold still)
CARD_MIN_WIDTH = 0.25            # card at least this fraction of the picture width (digits big enough to read)
CARD_MIN_SHARP = 0.8 * card.MIN_SHARPNESS   # frames blurrier than this are skipped (not a reset)

CHECK_ACCESSORIES = True         # ask to take off glasses / hat
ACC_THRESHOLDS = {"glasses": 0.0, "hat": 0.0}   # classifier score needed to say "wearing it"
ACC_FRAMES = 10                  # scores averaged over this many frames; clear for this many frames to go on

CHECK_HEAD_TURN = "auto"         # head-turn live check: "auto" = only when the rig's 3D check is off
                                 # (run_no_rig.py); True = as well as the rig's check; False = never
TURN_MIN = 0.10                  # nose off the flat-face prediction, in eye spacings, that counts as 3D
                                 # (real head turned 20 deg: ~0.13; flat face turned / tilted any way: noise)
TURN_FRAMES = 3                  # ... in this many frames in a row (each a median of the last 3 frames)
TURN_TIMEOUT_S = 8.0             # seconds to make each turn
TURN_MIN_EYE_PX = 70             # eye spacing needed for the turn check (person on the foot markers: ~110 px).
                                 # A small face, e.g. a phone held further back, has landmarks too noisy to judge
CHECK_LIVENESS = True            # 3D live check with the rig moving the camera (run_no_rig.py turns it off)
LIVE_AXIS = (1.0, 0.0)           # rig direction of that move: (1, 0) = rig X (left / right), (0, 1) = rig Y
LIVE_BASELINE_MM = 40.0          # how far the camera moves between the two pictures of a pair
LIVENESS_TRIES = 2               # a NO gets one automatic retry before the gate says no

CENTRED_HOLD_S = 0.8             # nose centred this long before going on
MATCH_FRAMES = 10                # live-face distances averaged before deciding
MATCH_THRESHOLD = None           # None = id_card.py's THRESHOLD (from your own camera checks)
PHOTO_DELAY_S = 1.5              # "taking your photo" -> this long -> photo
RESULT_HOLD_S = 20.0             # result stays up this long, then the next traveller (SPACE = sooner)
LOST_FRAMES = 15                 # face missing this many frames in a row = traveller has left the spot

VOICE = "auto"                   # "auto" = Windows' own speech engine (SAPI), else PowerShell speech, else
                                 # pyttsx3; or force one: "sapi", "system", "pyttsx3"; "off" = no voice
VOICE_RATE = 165                 # words per minute (pyttsx3)
SAPI_RATE = 0                    # Windows speech speed, -10 (slow) to 10 (fast), 0 = normal
REPEAT_S = 8.0                   # repeat an unchanged instruction this often
MIN_GAP_S = 3.0                  # at most one new (non-urgent) instruction this often, so it doesn't babble
NARRATE = True                   # read out every instruction shown on screen (not just the start of each step)
NARRATE_HOLD_S = 0.6             # ... once it has been on screen this long (a one-frame flicker isn't read out)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "session_db")
CAMERA_FILE = os.path.join(BASE_DIR, "egate_camera.json")
FLIP_X = np.diag([-1.0, 1.0])            # a left-right flip of the picture, for the tracking calibration
RECORDS_CSV = os.path.join(card.BASE_DIR, "data", "cards", "records.csv")
IDS_DIR = os.path.join(card.BASE_DIR, "data", "ids")
WINDOW = "e-gate"
ff.WINDOW = WINDOW                   # face_follow's window_closed() looks at this name

GREEN, RED, YELLOW, WHITE, GREY = (0, 200, 0), (0, 0, 230), (0, 220, 255), (255, 255, 255), (170, 170, 170)

# stages
CONNECT, HOMING, SCAN, POSITION, CLEAR, LIVE, MATCH, PHOTO, RESULT, TURN = range(10)
STEPS = ["Scan ID", "Position", "Glasses / hat", "Live check", "Face match", "Done"]
STEP_OF = {SCAN: 0, POSITION: 1, CLEAR: 2, TURN: 3, LIVE: 3, MATCH: 4, PHOTO: 4, RESULT: 5}

RESULTS = {   # kind -> (screen title, colour, what the voice says)
    "pass": ("IDENTITY CONFIRMED", GREEN, "Identity confirmed. Welcome, {name}."),
    "not_same_person": ("NOT THE SAME PERSON AS THE ID", RED, "Sorry, you are not the same person as the ID card."),
    "photo_swapped": ("ID PHOTO DOES NOT MATCH OUR RECORDS", RED,
                      "Sorry, the photo on this ID card does not match our records."),
    "unknown_number": ("ID NUMBER NOT IN OUR RECORDS", RED, "Sorry, this ID number is not in our records."),
    "no_record_photo": ("NO RECORD PHOTO FOR THIS ID", RED, "Sorry, we have no record photo for this ID."),
    "not_live": ("LIVE FACE CHECK FAILED", RED, "Sorry, we could not confirm a live face."),
    "no_live_check": ("NOT CONFIRMED: NO LIVE CHECK", RED,
                      "Sorry, your identity can't be confirmed because no live face check was done."),
}


# ---------------------------------------------------------------- voice
class Speaker:
    """
    Text to speech on a background thread, so the video never waits for the voice.
    Only the newest message is kept: if the traveller has already moved on, an old
    instruction is dropped instead of being read out late.
    """

    def __init__(self, mode=VOICE):
        self.mode, self.on = mode, mode != "off"
        self.q = queue.Queue()
        self.last, self.last_t, self.any_t = None, 0.0, 0.0
        if self.on:
            threading.Thread(target=self._run, daemon=True).start()

    def say(self, msg, urgent=False, repeat=REPEAT_S):
        """urgent = a new step (always spoken at once); otherwise rate-limited and not repeated too often."""
        now = time.time()
        if msg == self.last and now - self.last_t < repeat:
            return
        if not urgent and msg != self.last and now - self.any_t < MIN_GAP_S:
            return
        self.last, self.last_t, self.any_t = msg, now, now
        print("[voice]", msg, flush=True)
        if not self.on:
            return
        while True:                               # drop anything not spoken yet
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.q.put(msg)

    @staticmethod
    def _sapi():
        """
        Windows' own speech engine (SAPI.SpVoice), created in this thread. Unlike pyttsx3 (which on
        Windows often speaks the first sentence from a background thread and then goes silent or
        hangs), it speaks every sentence. Uses comtypes (installed with pyttsx3) or pywin32.
        """
        if os.name != "nt":
            raise RuntimeError("SAPI is Windows only")
        try:
            import comtypes
            import comtypes.client
            comtypes.CoInitialize()
            voice = comtypes.client.CreateObject("SAPI.SpVoice")
        except ImportError:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            voice = win32com.client.Dispatch("SAPI.SpVoice")
        voice.Rate = SAPI_RATE
        return lambda t: voice.Speak(t)                     # waits until the sentence has been spoken

    @staticmethod
    def _pyttsx3():
        try:                                                # COM has to be started in this thread on Windows
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
        import pyttsx3

        def speak(t):                                       # a fresh engine each time: a reused one goes quiet
            engine = pyttsx3.init()
            engine.setProperty("rate", VOICE_RATE)
            engine.say(t)
            engine.runAndWait()
            engine.stop()
            del engine
        return speak

    @staticmethod
    def _system():
        if os.name == "nt":
            def speak(t):
                t = t.replace("'", "''")
                subprocess.run(["powershell", "-NoProfile", "-Command",
                                "Add-Type -AssemblyName System.Speech; "
                                f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{t}')"],
                               creationflags=0x08000000, check=False, timeout=30)   # no console window
            return speak
        if sys.platform == "darwin":
            return lambda t: subprocess.run(["say", t], check=False, timeout=30)
        if shutil.which("espeak"):
            return lambda t: subprocess.run(["espeak", t], check=False, timeout=30)
        return None

    def _backends(self):
        order = {"auto": ["sapi", "system", "pyttsx3"], "sapi": ["sapi"], "system": ["system"],
                 "pyttsx3": ["pyttsx3"]}.get(self.mode, [])
        return [(name, {"sapi": self._sapi, "system": self._system, "pyttsx3": self._pyttsx3}[name]) for name in order]

    def _run(self):
        """Speak queued sentences. If a voice engine fails, the next one is tried (and the sentence kept)."""
        backends, speak, name = self._backends(), None, None
        msg = None
        while True:
            if speak is None:
                while backends and speak is None:
                    name, make = backends.pop(0)
                    try:
                        speak = make()
                        if speak is not None:
                            print(f"Voice: using {name}", flush=True)
                    except Exception as e:
                        print(f"Voice: {name} not available ({e})", flush=True)
                        speak = None
                if speak is None:
                    print("No text-to-speech available: instructions are shown on screen only.", flush=True)
                    self.on = False
                    return
            if msg is None:
                msg = self.q.get()
            try:
                speak(msg)
                msg = None
            except Exception as e:                           # this engine broke: try the next one with the same words
                print(f"Voice: {name} failed ({e}), trying the next one", flush=True)
                speak = None


# ---------------------------------------------------------------- temporary database
class SessionDB:
    """SQLite + CSV + images in session_db/, deleted and made fresh every time main.py starts."""

    TRAVELLER_FIELDS = ["id", "time", "name", "id_number", "face_vs_id_photo_pct", "face_vs_record_pct",
                        "id_photo_vs_record_pct", "distance_id_photo", "distance_record", "distance_id_vs_record",
                        "threshold", "method", "liveness", "removed", "live_photo", "face_photo",
                        "id_face", "id_card_image", "record_photo"]
    ATTEMPT_FIELDS = ["id", "time", "id_number", "name", "result", "reason", "face_vs_id_photo_pct",
                      "face_vs_record_pct", "liveness", "live_photo"]

    def __init__(self, folder=DB_DIR):
        try:
            if os.path.isdir(folder):
                shutil.rmtree(folder)
        except OSError as e:                      # e.g. the old database is open in another program
            folder = f"{folder}_{datetime.now():%H%M%S}"
            print(f"Could not wipe the old session database ({e}); using {folder} instead")
        self.folder, self.img_dir = folder, os.path.join(folder, "images")
        os.makedirs(self.img_dir)
        self.con = sqlite3.connect(os.path.join(folder, "session.db"))
        self.con.execute("CREATE TABLE travellers (id INTEGER PRIMARY KEY, time TEXT, name TEXT, id_number TEXT, "
                         "face_vs_id_photo_pct REAL, face_vs_record_pct REAL, id_photo_vs_record_pct REAL, "
                         "distance_id_photo REAL, distance_record REAL, distance_id_vs_record REAL, threshold REAL, "
                         "method TEXT, liveness TEXT, removed TEXT, live_photo TEXT, face_photo TEXT, id_face TEXT, "
                         "id_card_image TEXT, record_photo TEXT)")
        self.con.execute("CREATE TABLE attempts (id INTEGER PRIMARY KEY, time TEXT, id_number TEXT, name TEXT, "
                         "result TEXT, reason TEXT, face_vs_id_photo_pct REAL, face_vs_record_pct REAL, "
                         "liveness TEXT, live_photo TEXT)")
        self.con.commit()
        self.n = 0
        print("Fresh session database:", self.folder)

    def next_id(self):
        self.n += 1
        return self.n

    def image(self, n, tag, img):
        """Save one image, return its path relative to the database folder (or '' if there is none)."""
        if img is None:
            return ""
        rel = os.path.join("images", f"{n:03d}_{tag}.png")
        cv2.imwrite(os.path.join(self.folder, rel), img)
        return rel

    def _add(self, table, fields, row):
        self.con.execute(f"INSERT INTO {table} ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})",
                         [row.get(f) for f in fields])
        self.con.commit()
        path = os.path.join(self.folder, f"{table}.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    def add_traveller(self, row):
        self._add("travellers", self.TRAVELLER_FIELDS, row)

    def add_attempt(self, row):
        self._add("attempts", self.ATTEMPT_FIELDS, row)


# ---------------------------------------------------------------- helpers
ROBUST_RING_CONTRAST = 45       # a bit looser than id_card.py's 50, only used when its own finder missed


def find_card_robust(img):
    """
    id_card.find_card first. If that misses (fingers over the card's frame, a soft or dim
    picture), try again on a smaller copy with three different thresholds, and take the
    convex hull of each shape (fills in the notches fingers make) before looking for 4 corners,
    or the smallest rectangle around it. The card's dark-frame / white-strip ring check from
    id_card.py still has to pass, so random rectangles are not taken for the card.
    """
    found = card.find_card(img)
    if found is not None:
        return found
    h, w = img.shape[:2]
    s = min(1.0, 800.0 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    sh, sw = small.shape[:2]
    grey_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    masks = []
    for block in ((min(sh, sw) // 8) | 1, (min(sh, sw) // 20) | 1):
        masks.append(cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                           max(15, block), 8))
    masks.append(cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
    masks.append(cv2.dilate(cv2.Canny(blur, 30, 90), np.ones((3, 3), np.uint8), iterations=2))
    best, best_score = None, ROBUST_RING_CONTRAST
    for m in masks:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < card.MIN_AREA_FRAC * sw * sh:
                continue
            hull = cv2.convexHull(c)
            cands = []
            approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            if len(approx) == 4:
                cands.append(approx)
            rect = cv2.boxPoints(cv2.minAreaRect(hull))
            if cv2.contourArea(hull) > 0.85 * cv2.contourArea(rect.astype(np.float32)):
                cands.append(rect)
            for q in cands:
                pts = card.order_corners(np.float32(q).reshape(4, 2) / s)
                top = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2
                side = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2
                if not card.ASPECT_RANGE[0] < top / max(side, 1) < card.ASPECT_RANGE[1]:
                    continue
                score = card.ring_contrast(card.warp_card(grey_full, pts))
                if score > best_score:
                    best, best_score = pts, score
    return best


def load_records():
    """ID number -> name, from data/cards/records.csv (make_id_cards.py)."""
    if not os.path.exists(RECORDS_CSV):
        return {}
    with open(RECORDS_CSV, newline="") as fh:
        return {r["id_number"]: r["name"] for r in csv.DictReader(fh)}


def record_photo(yunet, name, method):
    """(feature, path) of the enrolled photo for that name in data/ids, or (None, None)."""
    if not os.path.isdir(IDS_DIR):
        return None, None
    for f in sorted(os.listdir(IDS_DIR)):
        if idm.person_name(f).lower() == name.lower():
            path = os.path.join(IDS_DIR, f)
            img = cv2.imread(path)
            det = fd.detect_photo(yunet, img) if img is not None else None
            if det is not None:
                return idm.features(img, det, [method])[method], path
    return None, None


def match_pct(d, threshold):
    """Distance as a percentage: 100 identical, 50 at the threshold, 0 at twice the threshold or more."""
    if d is None:
        return None
    return round(100.0 * float(np.clip(1.0 - d / (2.0 * threshold), 0.0, 1.0)), 1)


def say_name(name):
    return (name or "").replace("_", " ")


def move_hint(err, mirror):
    """
    What to tell the traveller when the rig can't reach (err = nose minus picture centre,
    in the unmirrored picture). Returns (message, arrow direction on the SCREEN).
    The camera faces the traveller, so the traveller's right is the left of the picture.
    """
    if abs(err[0]) >= abs(err[1]):
        go = -np.sign(err[0])                          # which way the face has to move in the picture
        msg = f"Please move a little to your {'right' if go < 0 else 'left'}"
        return msg, (-go if mirror else go, 0)
    go = -np.sign(err[1])
    msg = "Please stand up straight" if go < 0 else "Please lower your head a little"
    return msg, (0, go)


def passport_crop(frame, det):
    p = mic.PASSPORT
    return align_face(frame, det["left_eye"], det["right_eye"], canvas=p["canvas"], eye_mid=p["eye_mid"],
                      d0=p["d0"], border=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- drawing (big and simple)
FONT = cv2.FONT_HERSHEY_DUPLEX


def put(img, msg, org, scale, colour, thick):
    """Text with a black outline so it stays readable on any background."""
    x, y = org
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
        cv2.putText(img, msg, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, msg, org, FONT, scale, colour, thick, cv2.LINE_AA)


def wrap(msg, scale, thick, max_w):
    lines = []
    for chunk in textwrap.wrap(msg, 60) or [""]:
        words, line = chunk.split(), ""
        for wd in words:
            test = (line + " " + wd).strip()
            if cv2.getTextSize(test, FONT, scale, thick)[0][0] > max_w and line:
                lines.append(line)
                line = wd
            else:
                line = test
        lines.append(line)
    return lines


def draw_banner(view, msg, colour, progress=None, sub=None):
    """The one instruction, big, in a dark band across the top."""
    h, w = view.shape[:2]
    scale = 1.25 * w / 1280
    thick = max(2, int(round(2 * w / 1280)))
    lines = wrap(msg, scale, thick, w - 60)
    line_h = int(48 * w / 1280)
    band = 24 + line_h * len(lines) + (int(34 * w / 1280) if sub else 0) + (16 if progress is not None else 0)
    dark = view[:band].copy()
    dark[:] = (25, 25, 25)
    view[:band] = cv2.addWeighted(view[:band], 0.25, dark, 0.75, 0)
    y = 14 + line_h
    for line in lines:
        tw = cv2.getTextSize(line, FONT, scale, thick)[0][0]
        put(view, line, ((w - tw) // 2, y - 10), scale, colour, thick)
        y += line_h
    if sub:
        s2 = 0.6 * w / 1280
        tw = cv2.getTextSize(sub, FONT, s2, 1)[0][0]
        put(view, sub, ((w - tw) // 2, y - 14), s2, WHITE, 1)
        y += int(34 * w / 1280)
    if progress is not None:
        x0, x1 = w // 4, 3 * w // 4
        cv2.rectangle(view, (x0, y - 18), (x1, y - 8), (80, 80, 80), -1)
        cv2.rectangle(view, (x0, y - 18), (x0 + int((x1 - x0) * min(1.0, progress)), y - 8), colour, -1)


def draw_steps(view, current):
    """The steps along the bottom: done in green, current in yellow, the rest grey."""
    h, w = view.shape[:2]
    scale = 0.7 * w / 1280
    seg = w / len(STEPS)
    cv2.rectangle(view, (0, h - int(56 * w / 1280)), (w, h), (25, 25, 25), -1)
    for k, name in enumerate(STEPS):
        col = GREEN if k < current else (YELLOW if k == current else (110, 110, 110))
        label = ("> " if k == current else "") + name
        tw = cv2.getTextSize(label, FONT, scale, 2)[0][0]
        cv2.putText(view, label, (int(seg * k + (seg - tw) / 2), h - int(20 * w / 1280)), FONT, scale, col, 2,
                    cv2.LINE_AA)


def draw_arrow(view, direction, centre=None, length=None):
    """Big arrow (in the middle of the screen by default) pointing the way the traveller should move."""
    h, w = view.shape[:2]
    d = np.array(direction, float)
    c = np.array(centre if centre is not None else (w / 2, h / 2), float)
    length = length or 0.18 * min(w, h)
    p0, p1 = (c - d * length).astype(int), (c + d * length).astype(int)
    cv2.arrowedLine(view, tuple(p0), tuple(p1), (0, 0, 0), 34, cv2.LINE_AA, tipLength=0.45)
    cv2.arrowedLine(view, tuple(p0), tuple(p1), YELLOW, 22, cv2.LINE_AA, tipLength=0.45)


def paste(view, img, x, y, h_target):
    if img is None:
        return
    s = h_target / img.shape[0]
    small = cv2.resize(img, None, fx=s, fy=s)
    hh, ww = small.shape[:2]
    if y + hh <= view.shape[0] and x + ww <= view.shape[1] and x >= 0:
        view[y:y + hh, x:x + ww] = small
        cv2.rectangle(view, (x, y), (x + ww, y + hh), WHITE, 2)


# ---------------------------------------------------------------- the gate
class Gate:
    def __init__(self, yunet, voice, db, records, method, threshold, lens):
        self.yunet, self.voice, self.db, self.records = yunet, voice, db, records
        self.method, self.threshold, self.lens = method, threshold, lens
        # Is the webcam's own picture mirrored? If so every frame is flipped back before use.
        self.cam_flip = bool(CAMERA_MIRRORED) if CAMERA_MIRRORED != "auto" else self.saved_mirror()
        # face_follow_cal.json is shared with face_follow.py / face_liveness.py, which use the
        # camera picture as it comes, so a flipped picture needs the calibration flipped too.
        A = ff.load_calibration()
        self.A = FLIP_X @ A if (A is not None and self.cam_flip) else A
        self.rig = self.mover = self.motion = self.cal = None
        self.pool = ThreadPoolExecutor(max_workers=1)        # stereo analysis off the video loop
        self.stereo, self.decision = fl.StereoCapture(), fl.Decision()
        self.job, self.gen = None, 0
        self.roll_vec, self.limits = None, []
        self.stage, self.stage_t0 = CONNECT, time.time()
        self.debug = []
        self.new_traveller(speak=False)

    # ------------------------------------------------ bookkeeping
    def new_traveller(self, speak=True):
        self.gen += 1                                          # ignore any analysis still running
        self.t = {"number": None, "name": None, "card_feature": None, "card_photo": None, "card_warped": None,
                  "db_feature": None, "db_path": None, "liveness": "not checked"}
        self.reads = deque()                                   # (time, number, card read or None, feature, sharpness)
        self.card_seen_t = 0.0
        self.acc_hist = {item: deque(maxlen=ACC_FRAMES) for item in acc.ITEMS}
        self.clear_count, self.seen = 0, set()
        self.dists, self.dists_db, self.best = [], [], None
        self.centred_since, self.live_tries, self.lost, self.last_nose = None, 0, 0, None
        self.turn, self.turn_tries, self.turn_ok, self.stereo_ok = None, 0, False, False
        self.last_pair = None
        self.photo, self.result, self.result_t = None, None, 0.0
        self.stereo.reset()
        self.decision.reset()
        if self.mover is not None and self.mover.error is None:
            self.motion.move(ff.MIDDLE)
            self.go(SCAN, "Welcome. Please hold your ID card up to the camera." if speak else None)

    def go(self, stage, speech=None):
        self.stage, self.stage_t0 = stage, time.time()
        self.centred_since = None
        if speech:
            self.voice.say(speech, urgent=True)

    @staticmethod
    def saved_mirror():
        try:
            with open(CAMERA_FILE) as fh:
                return bool(json.load(fh)["camera_mirrored"])
        except (OSError, ValueError, KeyError):
            return False

    def set_mirror(self, flip):
        """The webcam turned out to mirror (or not): flip every frame from now on, and remember it."""
        if flip == self.cam_flip:
            return
        self.cam_flip = flip
        if self.A is not None:
            self.A = FLIP_X @ self.A                       # X moves now push the face the other way in the picture
        self.roll_vec = None
        if CAMERA_MIRRORED == "auto":
            with open(CAMERA_FILE, "w") as fh:
                json.dump({"camera_mirrored": flip}, fh)
        print(f"The webcam {'mirrors' if flip else 'does not mirror'} the picture: "
              f"{'flipping' if flip else 'not flipping'} every frame from now on (saved in {CAMERA_FILE})", flush=True)

    def rotated(self):
        """The view was turned: forget anything measured in the old view."""
        self.roll_vec, self.lost, self.last_nose, self.centred_since = None, 0, None, None
        self.dists, self.dists_db, self.reads = [], [], deque()

    def connect(self):
        self.rig = ff.Rig()
        self.mover = ff.RigMover(self.rig)
        self.motion = fl.Motion(self.mover)
        self.motion.move(ff.MIDDLE, feed=ff.START_FEED)
        self.go(HOMING, "Starting up. Please wait.")

    # ------------------------------------------------ rig: centre the nose
    def track(self, det, roll, centre, band):
        """One centring step. Returns (centred, err) where err = nose minus picture centre."""
        err = ff.face_centre(det) - centre
        e = err.copy()
        e[np.abs(e) < band] = 0
        if not e.any():
            self.limits = []
            return True, err
        if self.mover.ready() and roll is not None:
            target, self.limits = ff.plan_step(np.linalg.inv(ff.rot(roll) @ self.A), e, self.mover.pos)
            if np.abs(target - self.mover.pos).max() >= ff.MIN_MOVE:
                self.motion.move(target)
        return False, err

    # ------------------------------------------------ one frame
    def update(self, frame, raw_w, raw_h, rotation):
        """Run the current step on one frame, and read out the instruction on screen."""
        ui = self.step(frame, raw_w, raw_h, rotation)
        if NARRATE:
            self.narrate(ui)
        return ui

    def narrate(self, ui):
        """
        Voice-over: whatever instruction is on screen is spoken once it has stayed there for
        NARRATE_HOLD_S. The Speaker drops a repeat of the same words within REPEAT_S, and keeps
        new ones at least MIN_GAP_S apart, so it reads along without talking over itself.
        The start of each step and the result are already spoken (go()), so they are not doubled.
        """
        msg = (ui.get("text") or "").strip()
        if self.stage in (CONNECT, RESULT) or not msg or msg.startswith(("Operator", "Rig error")):
            self._shown = None
            return
        if msg.startswith("Calibrating the camera"):          # the calibration label changes every second
            msg = "Calibrating the camera. Please keep still."
        msg = msg.replace("...", ".").rstrip(".") + "."
        now = time.time()
        if msg != getattr(self, "_shown", None):
            self._shown, self._shown_t = msg, now
            return
        if now - self._shown_t >= NARRATE_HOLD_S:
            self.voice.say(msg)

    def step(self, frame, raw_w, raw_h, rotation):
        """Run the current step on one frame. Returns what to show: dict(text, colour, progress, arrow, ...)."""
        h, w = frame.shape[:2]
        centre, band = np.array([w / 2.0, h / 2.0]), ff.DEADBAND * np.array([w, h])
        ui = {"text": "", "colour": WHITE, "progress": None, "arrow": None, "det": None, "corners": None, "sub": None,
              "card": None, "guide": False}
        self.debug = []

        if self.stage == CONNECT:
            ui.update(text="Operator: put the carriage at the bottom left, then press SPACE to start", colour=YELLOW)
            return ui
        if self.mover.error is not None:
            ui.update(text=f"Rig error: {self.mover.error}. Press q and check the rig.", colour=RED)
            return ui
        if self.stage == HOMING:
            ui.update(text="Starting up, please wait...", colour=WHITE)
            if self.mover.idle.is_set():
                self.go(SCAN, "Welcome. Please hold your ID card up to the camera.")
            return ui
        if self.stage == SCAN and self.cal is None:
            return self.scan_card(frame, ui)
        if self.stage == RESULT:
            return self.show_result(ui)

        # every step from here on needs the traveller's face
        det = fd.detect_scaled(self.yunet, frame, max_w=ff.DETECT_W)
        ui["det"] = det
        if det is not None:
            v = np.array([np.cos(ff.face_roll(det)), np.sin(ff.face_roll(det))])
            self.roll_vec = v if self.roll_vec is None else (1 - ff.ROLL_SMOOTH) * self.roll_vec + ff.ROLL_SMOOTH * v
        roll = None if self.roll_vec is None else float(np.arctan2(self.roll_vec[1], self.roll_vec[0]))

        if self.cal is not None:                                 # operator calibration (c), any step
            ui.update(text=f"Calibrating the camera: {self.cal.label}. Please keep still.", colour=YELLOW)
            try:
                res = self.cal.step(det)
                if res is not None:
                    self.cal = None
                    self.A, msg = res
                    ff.save_calibration(FLIP_X @ self.A if self.cam_flip else self.A)
                    print("Calibrated:", msg)
            except ff.CalibrationFailed as e:
                self.cal = None
                print("Calibration failed:", e)
            return ui
        if self.A is None:
            ui.update(text="Operator: press C to calibrate the camera (face in view, keep still)", colour=YELLOW)
            return ui

        if det is None:
            return self.no_face(ui, centre)
        self.lost, self.last_nose = 0, ff.face_centre(det)

        if self.stage == POSITION:
            return self.position(det, roll, centre, band, ui)
        if self.stage == CLEAR:
            return self.clear_face(frame, det, roll, centre, band, ui)
        if self.stage == TURN:
            return self.turn_check(det, ui)
        if self.stage == LIVE:
            return self.live_check(frame, det, roll, centre, band, raw_w, raw_h, rotation, ui)
        if self.stage == MATCH:
            return self.face_match(frame, det, roll, centre, band, ui)
        if self.stage == PHOTO:
            return self.take_photo(frame, det, roll, centre, band, ui)
        return ui

    # ------------------------------------------------ 1. ID card
    def read_card_any(self, frame):
        """
        Read the card and its number. If the number isn't on record, the picture flipped
        left-right is tried too: when THAT gives a number on record, the webcam mirrors the
        picture, and from now on every frame is flipped back (so the faces are the right way
        round for the face match as well). Returns (card read, number, flipped) or (None, None, False).
        """
        first = None
        for flip in ((False, True) if CAMERA_MIRRORED == "auto" else (False,)):
            img = cv2.flip(frame, 1) if flip else frame
            corners = find_card_robust(img)
            if corners is None:
                continue
            c = card.read_card(img, self.yunet, corners)
            number = dg.read_number(c)
            if number in self.records:
                if flip:
                    self.set_mirror(not self.cam_flip)
                    self.reads.clear()                     # reads so far were of the mirrored card
                return c, number, flip
            if first is None:
                first = (c, number, flip)
        return first if first is not None else (None, None, False)

    def scan_card(self, frame, ui):
        """
        The card does NOT have to be held still: every frame where the card is found, big
        enough and not too blurred, its 4-digit number is read. Once the same number has been
        read CARD_VOTES times in the last CARD_WINDOW_S seconds (and the card photo has been
        seen at least once), the card is accepted. A bad frame is just skipped.
        """
        h, w = frame.shape[:2]
        now = time.time()
        while self.reads and now - self.reads[0][0] > CARD_WINDOW_S:
            self.reads.popleft()
        c, number, flipped = self.read_card_any(frame)
        problem = None
        if c is None:
            ui["guide"] = True
            if now - max(self.stage_t0, self.card_seen_t) > 4.0:
                problem = "Please hold your ID card inside the box on the screen, about 20 centimetres from the camera"
            else:
                problem = "Please hold your ID card up to the camera, photo side towards it"
            self.debug.append("card: outline not found (whole card in the picture? fingers off the dark border?)")
        else:
            corners = np.float32(c["corners"]).copy()
            if flipped:                                        # back to the unflipped picture for drawing
                corners[:, 0] = w - 1 - corners[:, 0]
            ui["corners"], ui["card"] = corners, c["warped"]
            self.card_seen_t = now
            sharp = card.card_sharpness(c)
            self.debug.append(f"card: number {number}{' (mirrored)' if flipped else ''}, sharpness {sharp:.2f} "
                              f"(needs {CARD_MIN_SHARP:.2f}), width {card.card_width(c['corners']) / w:.0%}")
            if card.card_width(c["corners"]) < CARD_MIN_WIDTH * w:
                problem = "Please bring your card a little closer"
            elif sharp < CARD_MIN_SHARP:
                problem = "Please hold the card a little steadier"
            else:
                photo_ok = c["face"] is not None and card.card_glare(c) <= card.GLARE_FRAC
                feat = card.card_feature(c, self.method) if photo_ok else None
                self.reads.append((now, number, c if photo_ok else None, feat, sharp))

        votes = Counter(r[1] for r in self.reads)
        top, n_top = votes.most_common(1)[0] if votes else (None, 0)
        with_photo = [r for r in self.reads if r[1] == top and r[2] is not None]
        if n_top >= CARD_VOTES and n_top >= 0.6 * len(self.reads):
            if with_photo:
                return self.lock_card(top, with_photo)
            problem = "Please keep the photo on the card uncovered, and tilt it away from the light"
        if n_top:
            ui.update(text=problem or "Reading your ID card...", colour=YELLOW if problem else GREEN,
                      progress=min(n_top, CARD_VOTES) / CARD_VOTES, sub=f"ID number {top}")
            return ui
        self.voice.say(problem, repeat=12.0)
        ui.update(text=problem, colour=YELLOW)
        return ui

    def lock_card(self, number, reads):
        f = np.mean([r[3] for r in reads], axis=0)
        best = max(reads, key=lambda r: r[4])[2]
        name = self.records.get(number)
        db_feature, db_path = record_photo(self.yunet, name, self.method) if name else (None, None)
        self.t.update(number=number, name=name, card_feature=f / (np.linalg.norm(f) + 1e-9),
                      card_photo=best["photo"].copy(), card_warped=best["warped"].copy(),
                      db_feature=db_feature, db_path=db_path)
        print(f"Card read: number {number}, on record as {name or 'NOBODY'}"
              f"{'' if db_path else ' (no record photo)'}  (not confirmed yet)", flush=True)
        self.reads.clear()
        self.go(POSITION, "ID card scanned. Thank you. Please put your card down, stand on the foot markers, "
                          "and look at the camera.")
        return {"text": "ID card scanned", "colour": GREEN, "progress": None, "arrow": None, "det": None,
                "corners": None, "sub": f"ID number {number}"}

    # ------------------------------------------------ face lost
    def no_face(self, ui, centre):
        self.lost += 1
        if self.stage == LIVE and self.lost > LOST_FRAMES:      # stop the stereo moves, start the check again later
            self.gen += 1
            self.stereo.reset()
            self.decision.reset()
            self.go(POSITION, "I've lost your face. Please stand on the foot markers and look at the camera.")
        if self.stage == MATCH:
            self.dists, self.dists_db = [], []
        if self.stage == TURN and self.turn is not None and self.lost > LOST_FRAMES:
            self.turn.update(phase="front", base=[], hits=0, i=0, t0=time.time())   # start the turn check over
        if self.lost < 4:                                        # one missed detection: carry on quietly
            ui.update(text="Please look at the camera", colour=YELLOW)
            return ui
        if self.last_nose is not None and self.lost < 3 * LOST_FRAMES:
            err = self.last_nose - centre
            size = 2 * centre
            if np.max(np.abs(err) / size) > 0.3:                 # left the picture over an edge
                msg, arrow = move_hint(err, self.mirror)
                self.voice.say(msg)
                ui.update(text=msg, colour=YELLOW, arrow=arrow)
                return ui
        msg = "Please stand on the foot markers and look at the camera"
        self.voice.say(msg)
        ui.update(text=msg, colour=YELLOW)
        return ui

    # ------------------------------------------------ 2. position
    def position(self, det, roll, centre, band, ui):
        centred, err = self.track(det, roll, centre, band)
        self.debug.append(f"nose off centre {err[0]:+.0f}, {err[1]:+.0f} px   limits {self.limits or '-'}")
        if self.limits:
            msg, arrow = move_hint(err, self.mirror)
            self.voice.say(msg)
            ui.update(text=msg, colour=YELLOW, arrow=arrow)
            self.centred_since = None
            return ui
        if not centred:
            self.centred_since = None
            ui.update(text="Please stand still while the camera finds your face", colour=WHITE)
            return ui
        if self.centred_since is None:
            self.centred_since = time.time()
        ui.update(text="Please stand still", colour=GREEN,
                  progress=(time.time() - self.centred_since) / CENTRED_HOLD_S)
        if time.time() - self.centred_since > CENTRED_HOLD_S and self.mover.ready():
            if CHECK_ACCESSORIES:
                self.go(CLEAR, "Thank you. Now checking for glasses and hats.")
            else:
                self.next_live_step()
        return ui

    # ------------------------------------------------ 3. glasses / hat
    def wearing(self, frame, det):
        scores = acc.accessory_scores(frame, det["left_eye"], det["right_eye"])
        for item in acc.ITEMS:
            self.acc_hist[item].append(scores[item])
        avg = {item: float(np.mean(hh)) for item, hh in self.acc_hist.items()}
        self.debug.append("   ".join(f"{i} {avg[i]:+.2f} (> {ACC_THRESHOLDS[i]:+.1f} = on)" for i in acc.ITEMS))
        on = [i for i in acc.ITEMS if len(self.acc_hist[i]) >= 3 and avg[i] > ACC_THRESHOLDS[i]]
        self.seen.update(on)
        return on

    def clear_face(self, frame, det, roll, centre, band, ui):
        self.track(det, roll, centre, band)
        on = self.wearing(frame, det)
        if on:
            self.clear_count = 0
            msg = "Please remove your " + " and ".join(on)
            self.voice.say(msg, urgent=self.voice.last != msg)
            ui.update(text=msg, colour=RED)
            return ui
        self.clear_count += 1
        ui.update(text="Checking for glasses and hats...", colour=WHITE, progress=self.clear_count / ACC_FRAMES)
        if self.clear_count >= ACC_FRAMES:
            self.dists, self.dists_db = [], []
            self.next_live_step("Thank you." if self.seen else None)
        return ui

    # ------------------------------------------------ 4. live checks
    @staticmethod
    def head_turn_on():
        return CHECK_HEAD_TURN is True or (CHECK_HEAD_TURN == "auto" and not CHECK_LIVENESS)

    def live_verified(self):
        """At least one live check passed, and none that is switched on is still to do."""
        todo = (self.head_turn_on() and not self.turn_ok) or (CHECK_LIVENESS and not self.stereo_ok)
        return (self.turn_ok or self.stereo_ok) and not todo

    def next_live_step(self, before=None):
        """After the glasses / hat check: the live checks still to do, then the face match."""
        if self.head_turn_on() and not self.turn_ok:
            self.start_turn(before)
        elif CHECK_LIVENESS and not self.stereo_ok:
            self.start_live(before)
        else:
            self.go(MATCH, ((before + " ") if before else "") + "Please look straight into the camera.")

    def add_liveness(self, text):
        old = self.t["liveness"]
        self.t["liveness"] = text if not old.startswith("LIVE") else old + "; " + text[len("LIVE: "):]

    # ---- 4a. head turn (no rig needed)
    @staticmethod
    def landmarks(det):
        """YuNet's five points by the person's side: right eye, left eye, nose, right / left mouth corner."""
        r = np.asarray(det["row"], np.float64)
        return {"re": r[4:6], "le": r[6:8], "nose": r[8:10], "mr": r[10:12], "ml": r[12:14]}

    @staticmethod
    def turn_index(det):
        """Plain nose offset along the eye line (eye spacings): only used to see the person is facing ahead."""
        le, re, nose = (np.float32(det[k]) for k in ("left_eye", "right_eye", "nose"))
        d = float(np.linalg.norm(re - le)) + 1e-6
        return float((nose - (le + re) / 2) @ ((re - le) / d)) / d

    @staticmethod
    def off_plane(ref, cur):
        """
        How far the nose is from where a FLAT face would put it, in eye spacings, along the eye
        line (positive = towards the right of the picture = the person's left).
        The homography that takes the eyes and mouth corners from the straight-ahead frame to this
        frame is exactly how a flat face (and everything printed on it) would have moved, from any
        angle, so a phone / photo nose lands on the prediction; a real nose doesn't.
        """
        keys = ("re", "le", "mr", "ml")
        try:
            H = cv2.getPerspectiveTransform(np.float32([ref[k] for k in keys]), np.float32([cur[k] for k in keys]))
        except cv2.error:
            return 0.0
        p = H @ np.r_[ref["nose"], 1.0]
        if abs(p[2]) < 1e-9:
            return 0.0
        e = cur["le"] - cur["re"]                           # person's right eye -> left eye: towards picture right
        d = float(np.linalg.norm(e)) + 1e-6
        return float((cur["nose"] - p[:2] / p[2]) @ (e / d)) / d

    def start_turn(self, before=None):
        order = [1, -1]                                     # +1 = the traveller's left, -1 = right
        random.shuffle(order)                               # random order: a pre-recorded video can't know it
        self.turn = {"phase": "front", "order": order, "i": 0, "base": [], "hits": 0, "t0": time.time(),
                     "best": [0.0, 0.0], "recent": deque(maxlen=3), "ref": None}
        self.go(TURN, ((before + " ") if before else "") + "Now a quick live check. Please look straight at the camera.")

    def turn_prompt(self):
        d = self.turn["order"][self.turn["i"]]
        return f"Please turn your head slowly to your {'left' if d > 0 else 'right'}"

    def turn_check(self, det, ui):
        T, now = self.turn, time.time()
        if T["phase"] == "front":                           # straight-ahead reference: median of a few frames
            if abs(self.turn_index(det)) < 0.25:
                T["base"].append(self.landmarks(det))
            if len(T["base"]) >= 6 or (now - T["t0"] > 6.0 and T["base"]):
                T["ref"] = {k: np.median([b[k] for b in T["base"]], axis=0) for k in T["base"][0]}
                T.update(phase="turn", t0=now, hits=0)
                T["recent"].clear()
                self.voice.say(self.turn_prompt(), urgent=True)
            ui.update(text="Please look straight at the camera", colour=WHITE)
            return ui

        L = self.landmarks(det)
        if np.linalg.norm(L["le"] - L["re"]) < TURN_MIN_EYE_PX:  # too small to measure the nose reliably
            T["hits"] = 0
            T["recent"].clear()
            self.voice.say("Please come a little closer")
            ui.update(text="Please come a little closer", colour=YELLOW)
            return ui
        T["recent"].append(self.off_plane(T["ref"], L))
        s = float(np.median(T["recent"]))                    # median of the last 3 frames: one noisy frame can't do it
        if T["phase"] == "back":                            # back to the front before the second turn
            if abs(s) < TURN_MIN / 2 or now - T["t0"] > 5.0:
                T.update(phase="turn", t0=now, hits=0)
                T["recent"].clear()
                self.voice.say(self.turn_prompt(), urgent=True)
            ui.update(text="Now look back at the camera", colour=WHITE)
            return ui

        d = T["order"][T["i"]]
        k = 0 if d > 0 else 1
        T["best"][k] = max(T["best"][k], s * d)
        T["hits"] = T["hits"] + 1 if s * d >= TURN_MIN else 0
        self.debug.append(f"nose off the flat-face position: {s * d:+.3f} (needs {TURN_MIN})   "
                          f"best left {T['best'][0]:.2f}, right {T['best'][1]:.2f}")
        if T["hits"] >= TURN_FRAMES:
            T["i"] += 1
            if T["i"] == len(T["order"]):                   # both turns seen: a real, three-dimensional head
                self.turn_ok = True
                self.add_liveness(f"LIVE: head turn, nose off the flat-face position by {T['best'][0]:.2f} (left) "
                                  f"and {T['best'][1]:.2f} (right), needs {TURN_MIN}")
                print("Live check (head turn):", self.t["liveness"], flush=True)
                self.next_live_step("Thank you.")
                return ui
            T.update(phase="back", t0=now, hits=0)
            T["recent"].clear()
            self.voice.say("Thank you. Now look back at the camera.", urgent=True)
            return ui
        if now - T["t0"] > TURN_TIMEOUT_S:                  # no 3D turn: a photo or a screen
            self.turn_tries += 1
            why = (f"no 3D head turn: the nose stayed where a flat face puts it (at most {max(T['best']):.2f} "
                   f"eye spacings off, needs {TURN_MIN}), so a photo or a screen")
            print("Live check (head turn) failed:", why, flush=True)
            if self.turn_tries < LIVENESS_TRIES:
                self.start_turn("Let's try that again.")
                return ui
            self.t["liveness"] = "NOT LIVE: " + why
            self.finish("not_live", why)
            return self.show_result(ui)
        ui.update(text=self.turn_prompt(), colour=YELLOW, arrow=((-d if self.mirror else d), 0),
                  progress=float(np.clip(s * d / TURN_MIN, 0, 1)))
        return ui

    # ---- 4b. 3D stereo pairs (needs the rig)
    def start_live(self, before=None):
        self.gen += 1
        self.stereo.reset()
        self.decision.reset()
        fl.STEREO_AXIS, fl.STEREO_BASELINE = LIVE_AXIS, LIVE_BASELINE_MM
        self.go(LIVE, ((before + " ") if before else "") +
                "Please hold still. The camera will move from side to side to check your face in 3D.")

    def live_check(self, frame, det, roll, centre, band, raw_w, raw_h, rotation, ui):
        h, w = frame.shape[:2]
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pair = None
        if self.stereo.active and self.mover.idle.is_set() and roll is not None:
            at_anchor = ff.face_centre(det) - ff.rot(roll) @ self.A @ (self.mover.pos - self.stereo.anchor)
            if np.any(np.abs(at_anchor - centre) > fl.RECENTRE_BAND * np.array([w, h])):
                self.stereo.reset()                              # traveller moved: centre again first
        if self.stereo.active:
            if self.job is None:                                 # wait for the last pair's analysis before moving on
                pair = self.stereo.step(self.mover, self.motion, grey, det)
            self.decision.start()
        else:
            centred, _ = self.track(det, roll, centre, band)
            if centred:
                pair = self.stereo.step(self.mover, self.motion, grey, det)

        if pair is not None and self.job is None:
            J = ff.rot(roll) @ self.A if roll is not None else None
            self.job = (self.pool.submit(fl.timed_analyse, pair[0], pair[1], self.lens, rotation, raw_w, raw_h, J),
                        self.gen)
        elif pair is not None:
            self.decision.skip("previous pair still being analysed")
        if self.job is not None and self.job[0].done():
            fut, g = self.job
            self.job = None
            if g == self.gen:
                try:
                    r = fut.result()
                    self.decision.add(r["code"], r.get("expected_face_parallax_px"))
                    self.last_pair = r
                    print(f"  3D pair: {r['code']:6s} (face shape {r.get('code_face', r['code'])}: flatness "
                          f"{r['plane']['flatness']:.4f}, parallax {r['homog']['p90_px']:.2f}px, smooth "
                          f"{r['plane']['quad_flatness']:.4f})  {fl.bg_text(r)}", flush=True)
                except sfl.StereoError as e:
                    self.decision.skip(str(e))
                    print("  3D pair skipped:", e, flush=True)
                except Exception as e:
                    self.decision.skip("analysis error")
                    print("  3D analysis error:", repr(e), flush=True)
        self.decision.tick()
        self.debug.append(f"3D pairs: {' '.join(self.decision.codes) or '-'}   "
                          f"skipped {sum(self.decision.skipped.values())}")
        if getattr(self, "last_pair", None) is not None:
            self.debug.append("last pair: " + fl.bg_text(self.last_pair))

        if self.decision.state == "YES":
            self.stereo_ok = True
            self.add_liveness("LIVE: 3D " + self.decision.reason)
            self.stereo.reset()
            self.next_live_step("Thank you.")
        elif self.decision.state == "NO":
            self.live_tries += 1
            print("Live check NO:", self.decision.reason, flush=True)
            if self.live_tries < LIVENESS_TRIES:
                self.start_live("Let's try that again.")
            else:
                self.t["liveness"] = "NOT LIVE: " + self.decision.reason
                self.stereo.reset()
                self.finish("not_live", self.decision.reason)
                return self.show_result(ui)
        ui.update(text="Please hold still: the camera is checking your face in 3D", colour=WHITE,
                  progress=len(self.decision.codes) / fl.DECISION_PAIRS)
        return ui

    # ------------------------------------------------ 5. face match
    def face_match(self, frame, det, roll, centre, band, ui):
        h, w = frame.shape[:2]
        self.track(det, roll, centre, band)
        if CHECK_ACCESSORIES:
            on = self.wearing(frame, det)
            if on:                                               # put them back on: go back a step
                self.dists, self.dists_db = [], []
                self.clear_count = 0
                self.go(CLEAR, "Please take off your " + " and ".join(on) + " again.")
                return ui
        problem = card.face_problem(det, w, h)
        if problem == "Look straight at the camera":
            problem = "Please look straight into the camera"
        elif problem == "Come a bit closer to the camera":
            problem = "Please come a little closer"
        elif problem is not None:
            problem = "Please stand still while the camera lines up"
        if problem is not None:
            self.voice.say(problem)
            ui.update(text=problem, colour=YELLOW)
            return ui

        f_live = idm.features(frame, det, [self.method])[self.method]
        d_card = idm.distance(f_live, self.t["card_feature"])
        d_db = idm.distance(f_live, self.t["db_feature"]) if self.t["db_feature"] is not None else None
        self.dists.append(d_card)
        if d_db is not None:
            self.dists_db.append(d_db)
        score = d_card + (d_db if d_db is not None else 0)
        if self.best is None or score < self.best[0]:
            self.best = (score, frame.copy(), det)
        self.debug.append(f"distance to ID photo {np.mean(self.dists):.2f}"
                          + (f", to record photo {np.mean(self.dists_db):.2f}" if self.dists_db else "")
                          + f"  (match < {self.threshold:.2f})")
        ui.update(text="Checking your face against your ID...", colour=WHITE,
                  progress=len(self.dists) / MATCH_FRAMES)
        if len(self.dists) >= MATCH_FRAMES:
            self.decide()
            if self.stage == RESULT:
                return self.show_result(ui)
        return ui

    def decide(self):
        t, thr = self.t, self.threshold
        t["d_card"] = float(np.mean(self.dists))
        t["d_db"] = float(np.mean(self.dists_db)) if self.dists_db else None
        t["d_card_db"] = idm.distance(t["card_feature"], t["db_feature"]) if t["db_feature"] is not None else None
        if t["name"] is None:
            kind = "unknown_number"
        elif t["db_feature"] is None:
            kind = "no_record_photo"
        elif t["d_card"] < thr and t["d_db"] < thr:
            kind = "pass"
        elif t["d_card"] < thr:
            kind = "photo_swapped"
        else:
            kind = "not_same_person"
        print(f"Face match: {kind}  (ID photo {t['d_card']:.2f}"
              + (f", record photo {t['d_db']:.2f}" if t["d_db"] is not None else "")
              + f", needs < {thr:.2f})", flush=True)
        if kind == "pass" and not self.live_verified():         # e.g. every live check switched off
            kind = "no_live_check"
        if kind == "pass":
            self.go(PHOTO, "Everything matches. Taking your photo now, please hold still.")
        else:
            self.finish(kind)

    # ------------------------------------------------ 6. photo
    def take_photo(self, frame, det, roll, centre, band, ui):
        h, w = frame.shape[:2]
        self.track(det, roll, centre, band)
        left = PHOTO_DELAY_S - (time.time() - self.stage_t0)
        if left > 0:
            ui.update(text="Taking your photo... hold still", colour=WHITE, progress=1 - left / PHOTO_DELAY_S)
            return ui
        timed_out = time.time() - self.stage_t0 > PHOTO_DELAY_S + 5
        if card.face_problem(det, w, h) is None or timed_out:
            if card.face_problem(det, w, h) is None:
                self.photo = (frame.copy(), det)
            else:                                                # never front-on: use the best match frame
                self.photo = (self.best[1], self.best[2])
            self.finish("pass")
            return self.show_result(ui)
        self.voice.say("Please look straight into the camera")
        ui.update(text="Please look straight into the camera", colour=YELLOW)
        return ui

    # ------------------------------------------------ 7. result
    def finish(self, kind, reason=""):
        t, thr = self.t, self.threshold
        title, colour, speech = RESULTS[kind]
        n = self.db.next_id()
        now = datetime.now().isoformat(timespec="seconds")
        pct_card, pct_db = match_pct(t.get("d_card"), thr), match_pct(t.get("d_db"), thr)
        live_img = self.photo[0] if self.photo else (self.best[1] if self.best else None)
        live_path = self.db.image(n, "live_photo", live_img)
        if kind == "pass":
            frame, det = self.photo
            self.db.add_traveller({
                "id": n, "time": now, "name": t["name"], "id_number": t["number"],
                "face_vs_id_photo_pct": pct_card, "face_vs_record_pct": pct_db,
                "id_photo_vs_record_pct": match_pct(t["d_card_db"], thr),
                "distance_id_photo": round(t["d_card"], 4), "distance_record": round(t["d_db"], 4),
                "distance_id_vs_record": round(t["d_card_db"], 4), "threshold": thr, "method": self.method,
                "liveness": t["liveness"], "removed": ", ".join(sorted(self.seen)) or "nothing",
                "live_photo": live_path, "face_photo": self.db.image(n, "face", passport_crop(frame, det)),
                "id_face": self.db.image(n, "id_face", t["card_photo"]),
                "id_card_image": self.db.image(n, "id_card", t["card_warped"]),
                "record_photo": self.db.image(n, "record_photo", cv2.imread(t["db_path"]))})
            print(f"Saved traveller {n}: {t['name']} ({t['number']}) in {self.db.folder}", flush=True)
        self.db.add_attempt({"id": n, "time": now, "id_number": t["number"], "name": t["name"], "result": kind,
                             "reason": reason or title, "face_vs_id_photo_pct": pct_card,
                             "face_vs_record_pct": pct_db, "liveness": t["liveness"], "live_photo": live_path})
        self.result = {"kind": kind, "title": title, "colour": colour, "pct_card": pct_card, "pct_db": pct_db,
                       "face": passport_crop(*self.photo) if self.photo else None}
        self.result_t = time.time()

        words = [speech.format(name=say_name(t["name"])), "Thank you, scanning is finished."]
        if "glasses" in self.seen and "hat" in self.seen:
            words.append("You can put your glasses and hat back on now.")
        elif "glasses" in self.seen:
            words.append("You can put your glasses back on now.")
        elif "hat" in self.seen:
            words.append("You can put your hat back on now.")
        self.result["speech"] = " ".join(words)
        if kind in id_alert.ALERT_KINDS and self.best is not None:      # (added 1 Nov) wrong person: alert
            self.result["alert"] = id_alert.make_alert(self, kind)
            self.result["speech"] = self.result["alert"]["speech"]
        self.motion.move(ff.MIDDLE)
        self.go(RESULT, self.result["speech"])

    def show_result(self, ui):
        r = self.result
        sub = "Thank you, scanning is finished."
        if "glasses" in self.seen:
            sub += " You can put your glasses back on."
        if r["kind"] == "pass":
            sub = f"Welcome, {say_name(self.t['name'])}.  " + sub
        ui.update(text=r["title"], colour=r["colour"], sub=sub, progress=None, arrow=None)
        if r["pct_card"] is not None:
            self.debug.append(f"match with ID photo {r['pct_card']}%"
                              + (f", with record photo {r['pct_db']}%" if r["pct_db"] is not None else "")
                              + "  (50% = threshold)")
        if time.time() - self.result_t > RESULT_HOLD_S and not r.get("alert"):   # an alert waits for SPACE
            self.new_traveller()
        return ui

    def close(self):
        self.pool.shutdown(wait=False)
        if self.mover is not None:
            self.mover.stop()
            if self.mover.error is None:
                print("Going back to the start corner...")
                try:
                    self.rig.home_corner()
                except Exception as e:
                    print(f"Could not go back to the corner: {e}")
        if self.rig is not None:
            self.rig.close()


def camera_layer(gate, ui, frame):
    """The camera picture with the card outline / face box, mirrored for the traveller."""
    view = frame.copy()
    if ui["corners"] is not None:
        cv2.polylines(view, [np.int32(ui["corners"])], True, GREEN, max(3, frame.shape[1] // 250))
    if ui["det"] is not None:
        x, y, bw, bh = (int(v) for v in ui["det"]["box"])
        col = RED if ui["colour"] == RED else (GREEN if ui["colour"] == GREEN else WHITE)
        cv2.rectangle(view, (x, y), (x + bw, y + bh), col, max(3, frame.shape[1] // 250))
    if ui.get("guide"):                                    # where to hold the card
        h, w = view.shape[:2]
        gw = CARD_GUIDE * w
        gh = gw * card.CARD_H_MM / card.CARD_W_MM
        x0, y0, x1, y1 = int((w - gw) / 2), int((h - gh) / 2), int((w + gw) / 2), int((h + gh) / 2)
        L, t = int(0.18 * gw), max(4, w // 160)
        for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1)):
            cv2.line(view, (cx, cy), (cx + dx * L, cy), WHITE, t, cv2.LINE_AA)
            cv2.line(view, (cx, cy), (cx, cy + dy * L), WHITE, t, cv2.LINE_AA)
    if gate.mirror and (gate.stage != SCAN or MIRROR_DURING_SCAN):
        view = cv2.flip(view, 1)
    if ui.get("guide"):
        h, w = view.shape[:2]
        sc = 0.9 * w / 720
        tw = cv2.getTextSize("ID card here", FONT, sc, 2)[0][0]
        put(view, "ID card here", ((w - tw) // 2, int(h / 2 + CARD_GUIDE * w * 0.32) + int(50 * sc)), sc, WHITE, 2)
    return view


def render(gate, ui, frame, info=None, size=(1280, 720)):
    """The whole screen, size = (width, height). Landscape camera: full picture with text on top.
    Camera on its side (portrait): picture on the left at full height, big instructions on the right."""
    if gate.stage == RESULT and gate.result is not None and gate.result.get("alert"):   # (added 1 Nov)
        return id_alert.render(gate, size)
    getattr(id_alert, "stop_alarm", lambda: None)()                               # (added 1 Nov)
    cam = camera_layer(gate, ui, frame)
    h, w = cam.shape[:2]
    SW, SH = size
    if w >= h:
        return render_overlay(gate, ui, cam, info, size)
    return render_side(gate, ui, cam, info, size)


def shown_id_photo(gate):
    """
    (Display only, added 1 Nov.) The picture shown on the result screen: the record photo from
    data/ids for the card's number, passport-style, instead of the webcam's (often blurry) view
    of the printed card photo. Falls back to the card photo if there is no record photo.
    Nothing here changes any decision.
    """
    t = gate.t
    if t.get("db_path"):
        if t.get("record_view") is None:
            img = cv2.imread(t["db_path"])
            det = fd.detect_photo(gate.yunet, img) if img is not None else None
            t["record_view"] = passport_crop(img, det) if det is not None else img
        return t["record_view"], "record photo"
    return t["card_photo"], "ID photo"


def render_overlay(gate, ui, cam, info, size):
    SW, SH = size
    h0, w0 = cam.shape[:2]
    s = min(SW / w0, SH / h0)
    view = cv2.resize(cam, (int(w0 * s), int(h0 * s)))
    h, w = view.shape[:2]
    if ui["arrow"] is not None:
        draw_arrow(view, ui["arrow"])
    if gate.stage == RESULT and gate.result is not None:
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), gate.result["colour"], 16)
        ph, y = int(0.32 * h), h - int(0.32 * h) - int(80 * w / 1280)
        photo, label = shown_id_photo(gate)
        paste(view, photo, 30, y, ph)
        put(view, label, (30, y - 10), 0.6 * w / 1280, WHITE, 1)
        if gate.result["face"] is not None:
            x = w - 30 - int(ph * 0.75)
            paste(view, gate.result["face"], x, y, ph)
            put(view, "your photo", (x, y - 10), 0.6 * w / 1280, WHITE, 1)
    if ui.get("card") is not None:                           # the flattened card, the right way round
        ch = int(0.22 * h)
        x = w - 20 - int(ch * ui["card"].shape[1] / ui["card"].shape[0])
        paste(view, ui["card"], x, int(0.2 * h), ch)
    draw_banner(view, ui["text"], ui["colour"], ui["progress"], ui.get("sub"))
    if gate.stage in STEP_OF:
        draw_steps(view, STEP_OF[gate.stage])
    if info:
        y0 = h - int(70 * w / 1280) - 24 * len(info)
        for line in info:
            put(view, line, (14, y0), 0.5, WHITE, 1)
            y0 += 24
    canvas = np.zeros((SH, SW, 3), np.uint8)
    y, x = (SH - h) // 2, (SW - w) // 2
    canvas[y:y + h, x:x + w] = view
    return canvas


def text_block(img, msg, x0, x1, y, scale, colour, thick):
    """Wrapped text centred between x0 and x1, first baseline at y. Returns the y below it."""
    line_h = int(cv2.getTextSize("Ag", FONT, scale, thick)[0][1] * 1.7)
    for line in wrap(msg, scale, thick, x1 - x0 - 40):
        tw = cv2.getTextSize(line, FONT, scale, thick)[0][0]
        put(img, line, (x0 + (x1 - x0 - tw) // 2, y), scale, colour, thick)
        y += line_h
    return y


def render_side(gate, ui, cam, info, size):
    SW, SH = size
    h0, w0 = cam.shape[:2]
    s = min(SH / h0, 0.55 * SW / w0)                       # camera at full height, at most 55 % of the width
    camv = cv2.resize(cam, (int(w0 * s), int(h0 * s)))
    ch, cw = camv.shape[:2]
    canvas = np.full((SH, SW, 3), 30, np.uint8)
    cy = (SH - ch) // 2
    canvas[cy:cy + ch, :cw] = camv
    if ui["arrow"] is not None:                            # also on the picture itself
        draw_arrow(canvas, ui["arrow"], centre=(cw / 2, SH / 2), length=0.3 * cw)

    x0, x1 = cw, SW                                        # the panel
    pw = x1 - x0
    k = pw / 900                                           # text scale for this panel width
    big, thick = 1.5 * k, max(2, int(round(3 * k)))
    y = int(0.12 * SH)
    y = text_block(canvas, ui["text"], x0, x1, y, big, ui["colour"], thick)
    if ui.get("sub"):
        y = text_block(canvas, ui["sub"], x0, x1, y + int(10 * k), 0.8 * k, WHITE, max(1, int(round(1.6 * k))))
    if ui["progress"] is not None:
        bx0, bx1 = x0 + pw // 6, x1 - pw // 6
        cv2.rectangle(canvas, (bx0, y), (bx1, y + int(16 * k)), (80, 80, 80), -1)
        cv2.rectangle(canvas, (bx0, y), (bx0 + int((bx1 - bx0) * min(1.0, ui["progress"])), y + int(16 * k)),
                      ui["colour"], -1)
        y += int(40 * k)

    # middle of the panel: arrow, card, or the result photos
    mid_top, mid_bot = y + int(30 * k), SH - int(170 * k) - (24 * len(info) if info else 0)
    mid_h = max(60, mid_bot - mid_top)
    if ui["arrow"] is not None:
        draw_arrow(canvas, ui["arrow"], centre=(x0 + pw / 2, mid_top + mid_h / 2), length=0.3 * min(pw, mid_h))
    elif ui.get("card") is not None:
        ph = int(min(mid_h, 0.6 * pw / 1.66))
        paste(canvas, ui["card"], x0 + (pw - int(ph * 1.66)) // 2, mid_top, ph)
    elif gate.stage == RESULT and gate.result is not None:
        ph = int(min(mid_h - 40 * k, 0.4 * pw / 0.75))
        xa, xb = x0 + pw // 4 - int(ph * 0.375), x0 + 3 * pw // 4 - int(ph * 0.375)
        photo, label = shown_id_photo(gate)
        paste(canvas, photo, xa, mid_top + int(40 * k), ph)
        put(canvas, label, (xa, mid_top + int(25 * k)), 0.7 * k, WHITE, max(1, int(round(1.5 * k))))
        if gate.result["face"] is not None:
            paste(canvas, gate.result["face"], xb, mid_top + int(40 * k), ph)
            put(canvas, "your photo", (xb, mid_top + int(25 * k)), 0.7 * k, WHITE, max(1, int(round(1.5 * k))))

    # steps, one per line in two columns at the bottom of the panel
    if gate.stage in STEP_OF:
        cur = STEP_OF[gate.stage]
        sc, th = 0.75 * k, max(1, int(round(2 * k)))
        for i, name in enumerate(STEPS):
            col = GREEN if i < cur else (YELLOW if i == cur else (110, 110, 110))
            cx = x0 + int(0.08 * pw) + (i // 3) * int(0.46 * pw)
            cyy = SH - int(130 * k) + (i % 3) * int(40 * k)
            label = f"{i + 1}. {name}" + ("  <" if i == cur else "")
            cv2.putText(canvas, label, (cx, cyy), FONT, sc, col, th, cv2.LINE_AA)
    if info:
        yy = SH - int(170 * k) - 24 * len(info) + 10
        for line in info:
            put(canvas, line, (x0 + 14, yy), 0.5, WHITE, 1)
            yy += 24
    if gate.stage == RESULT and gate.result is not None:
        cv2.rectangle(canvas, (0, 0), (SW - 1, SH - 1), gate.result["colour"], 18)
    return canvas


def screen_size():
    """Size of the main monitor in pixels (Windows), or 1920 x 1080 if it can't be found."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()                    # real pixels, not scaled ones
        except Exception:
            pass
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


# ---------------------------------------------------------------- main loop
def main():
    print(__doc__)
    print("Using these files:")
    for m in (ff, fl, fd, idm, card, acc, dg):
        print(f"    {m.__name__:22s} {m.__file__}")
    missing = [p for p in (idm.THRESH_FILE, idm.EIGEN_FILE, fd.YUNET_MODEL) if not os.path.exists(p)]
    if missing:
        print("\nThese model files are missing:")
        for p in missing:
            print("    ", p)
        print(f"Run {os.path.join(os.path.dirname(idm.__file__), 'id_match.py')} once (that builds the first two),\n"
              "or check there is only one copy of id_match.py / face_detect.py and its models folder.")
        return
    records = load_records()
    if not records:
        print(f"No ID records found: {RECORDS_CSV} is missing or empty. Run make_id_cards.py first.")
        return
    global CHECK_ACCESSORIES
    try:
        acc._load_models()
    except FileNotFoundError:
        print("No glasses / hat models (run accessories.py): that step is skipped.")
        CHECK_ACCESSORIES = False
    dg.load()
    method = card.pick_method()
    threshold = MATCH_THRESHOLD if MATCH_THRESHOLD is not None else card.threshold_for(method)
    print(f"{len(records)} ID numbers on record. Face match: {method} features, threshold {threshold:.2f}")

    voice = Speaker()
    db = SessionDB()
    lens = fl.Lens()
    print(lens.note)
    yunet = fd.load_yunet()
    ff.CAMERA_INDEX = CAMERA_INDEX
    raw_cam = ff.open_camera(CAMERA_INDEX)
    if AUTOFOCUS:
        raw_cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)          # capture_grid.py turns it off, and some drivers remember that
    cam = fl.FrameGrabber(raw_cam)
    screen = screen_size()
    fullscreen = FULLSCREEN

    def set_window():
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, int(0.75 * screen[0]), int(0.75 * screen[1]))
    set_window()

    gate = Gate(yunet, voice, db, records, method, threshold, lens)
    gate.mirror = MIRROR
    rotation, show_debug, frame_id, first, n_frame = ff.ROTATION, False, 0, True, 0
    fps, t_prev = 0.0, time.time()
    try:
        while True:
            ok, raw, frame_id = cam.read(frame_id)
            if not ok:
                print("Frame grab failed, stopping.")
                break
            raw_h, raw_w = raw.shape[:2]
            frame = ff.rotate(raw, rotation)
            if gate.cam_flip:                              # undo the webcam's own mirroring
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            ui = gate.update(frame, raw_w, raw_h, rotation)
            n_frame += 1

            # camera turned so far the face is sideways: try the other 90 degree turns (as in face_liveness.py).
            # Not while calibrating or during the 3D check, which need the picture to stay the same way round.
            if ff.AUTO_ROTATE and gate.cal is None and gate.stage in (SCAN, POSITION, CLEAR, MATCH) \
                    and n_frame % ff.AUTO_EVERY == 0:
                no_face = ui["det"] is None if gate.stage != SCAN else \
                    (ui["corners"] is None and fd.detect_scaled(yunet, frame, max_w=ff.DETECT_W) is None)
                if no_face:
                    for r in (90, 270, 180):
                        cand = (rotation + r) % 360
                        if fd.detect_scaled(yunet, ff.rotate(raw, cand), max_w=ff.DETECT_W) is not None:
                            rotation = cand
                            gate.rotated()
                            print(f"Face found with the view turned {rotation} degrees, switched to that", flush=True)
                            break                      # the next frame is processed the new way round

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t_prev)
            t_prev = now
            info = None
            if show_debug:
                rig = "rig not connected" if gate.mover is None else \
                    f"rig X {gate.mover.pos[0]:.0f} Y {gate.mover.pos[1]:.0f} mm"
                info = [f"{fps:.0f} fps   {rig}   camera {CAMERA_INDEX}   view {rotation} deg   "
                        f"webcam mirror {'flipped back' if gate.cam_flip else 'no'}   voice {'on' if voice.on else 'off'}   tracking {'calibrated' if gate.A is not None else 'NOT calibrated'}",
                        f"ID {gate.t['number'] or '-'} -> {gate.t['name'] or '-'}   liveness: {gate.t['liveness']}"]
                info += gate.debug
            size = screen if fullscreen else (int(0.75 * screen[0]), int(0.75 * screen[1]))
            cv2.imshow(WINDOW, render(gate, ui, frame, info, size))
            if first:
                ff.bring_to_front()
                first = False

            # ---- keys (operator)
            k = ff.read_key()
            if ff.window_closed() or k in ("q", "\x1b"):
                break
            if k is None:
                continue
            if k == " ":
                if gate.stage == CONNECT:
                    try:
                        gate.connect()
                        print("Rig connected, moving to the middle", flush=True)
                    except Exception as e:
                        print(f"Could not connect on {ff.PORT}: {e}. Close UGS / Arduino IDE, then SPACE again")
                elif gate.stage == RESULT:
                    if not getattr(id_alert, "next_stage", lambda g: False)(gate):   # (added 1 Nov) alert: STOP -> officer view
                        gate.new_traveller()
            elif k == "n" and gate.mover is not None:
                gate.new_traveller()
            elif k == "c" and gate.mover is not None and gate.cal is None:
                gate.gen += 1
                gate.stereo.reset()
                gate.cal = ff.Calibration(gate.mover)
                print("Calibrating the tracking: keep still")
            elif k == "x" and gate.cal is not None:
                gate.cal.cancel()
                gate.cal = None
            elif k == "d":
                show_debug = not show_debug
            elif k == "v":
                voice.on = not voice.on and voice.mode != "off"
                print("Voice", "on" if voice.on else "off")
            elif k == "r":
                rotation = (rotation + 90) % 360
                gate.rotated()
                print(f"View turned {rotation} degrees (set ROTATION in face_follow.py to keep it)")
            elif k == "f":
                gate.mirror = not gate.mirror
            elif k == "w":
                fullscreen = not fullscreen
                set_window()
            elif k == "k":                                 # operator: the webcam mirrors / doesn't, by hand
                gate.set_mirror(not gate.cam_flip)

    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        gate.close()
        cam.release()
        cv2.destroyAllWindows()
        print("Session database:", db.folder)


if __name__ == "__main__":
    main()