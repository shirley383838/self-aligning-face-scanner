"""
Alert shown when the face does not belong to the ID.

  1. Looks up who the ID belongs to (record photo and name).
  2. Compares the live face with every record photo (one-to-many search):
     identified, weak best guess, or not registered.
  3. Shows a flashing STOP screen with a chime and no personal details; SPACE opens
     the officer view with the ID owner, and SPACE again moves on.
  4. Logs the alert to session_db/alerts.csv.

main.py calls make_alert(), render(), stop_alarm() and next_stage().
"""

import os
import csv
import time
import wave

import numpy as np
import cv2

import face_detect as fd
import id_match as idm
import make_id_cards as mic
from face_align import align_face

ALERT_KINDS = ("not_same_person", "photo_swapped", "unknown_number")
GUESS_MARGIN = 1.15         # closest record within threshold x this = "best guess (weak match)"
FLASH_S = 0.5               # red flash period
ALARM = True                # soft chime while the STOP screen is up (Windows; silent elsewhere)
ALARM_VOLUME = 0.25         # 0 to 1, on top of the computer's own volume
OFFICER_SHOWS_PERSON = False   # officer view: also show who the person at the gate was identified as

_gallery = {}


def passport(img, det):
    p = mic.PASSPORT
    return align_face(img, det["left_eye"], det["right_eye"], canvas=p["canvas"], eye_mid=p["eye_mid"],
                      d0=p["d0"], border=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- the soft alarm
_alarm_on = False


def _alarm_file():
    """A gentle two-note chime (ding-dong, then a pause) written once as a WAV file."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "alert_chime.wav")
    if not os.path.exists(path):
        rate = 22050

        def note(freq, secs):
            t = np.arange(int(rate * secs)) / rate
            env = np.minimum(1.0, t / 0.02) * np.exp(-3.0 * t)                # soft attack, ringing decay
            return np.sin(2 * np.pi * freq * t) * env + 0.3 * np.sin(4 * np.pi * freq * t) * env * 0.5
        sound = np.concatenate([note(880, 0.45), note(660, 0.7), np.zeros(int(rate * 0.9))])
        data = (np.clip(sound / np.abs(sound).max() * ALARM_VOLUME, -1, 1) * 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(data.tobytes())
    return path


def start_alarm():
    global _alarm_on
    if not ALARM or _alarm_on:
        return
    try:
        import winsound
        winsound.PlaySound(_alarm_file(), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
        _alarm_on = True
    except Exception as e:                       # not Windows, or no sound device
        print(f"(no alarm sound: {e})", flush=True)


def stop_alarm():
    global _alarm_on
    if not _alarm_on:
        return
    try:
        import winsound
        winsound.PlaySound(None, 0)
    except Exception:
        pass
    _alarm_on = False


def next_stage(gate):
    """
    SPACE on a result. Returns True if the alert used it (STOP screen -> officer view);
    False means main.py should go on to the next traveller as usual.
    """
    a = gate.result.get("alert") if gate.result else None
    if a is None or a["stage"] != "stop":
        stop_alarm()
        return False
    a["stage"] = "officer"
    stop_alarm()
    print("Officer view: SPACE again for the next traveller.", flush=True)
    return True


def gallery(yunet, method):
    """Every enrolled record photo: list of (name, feature, passport-style crop)."""
    if method not in _gallery:
        out = []
        for f in sorted(os.listdir(idm.IDS_DIR)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = cv2.imread(os.path.join(idm.IDS_DIR, f))
            det = fd.detect_photo(yunet, img) if img is not None else None
            if det is not None:
                out.append((idm.person_name(f), idm.features(img, det, [method])[method], passport(img, det)))
        _gallery[method] = out
    return _gallery[method]


def identify(yunet, method, frame, det, threshold):
    """1-to-N: the live face against every enrolled record. Returns a dict for the screen."""
    f = idm.features(frame, det, [method])[method]
    scored = sorted(((idm.distance(f, g), name, crop) for name, g, crop in gallery(yunet, method)),
                    key=lambda x: x[0])
    if not scored:
        return {"status": "none", "name": None, "d": None, "crop": None}
    d, name, crop = scored[0]
    if d < threshold:
        status = "identified"
    elif d < threshold * GUESS_MARGIN:
        status = "guess"
    else:
        status = "unknown"
    return {"status": status, "name": name, "d": d, "crop": crop}


def make_alert(gate, kind):
    """Called by main.py's finish() for a failed face match: who the ID is, who is really there."""
    t = gate.t
    _, frame, det = gate.best
    who = identify(gate.yunet, gate.method, frame, det, gate.threshold)
    owner_crop = None
    if t.get("db_path"):
        img = cv2.imread(t["db_path"])
        d0 = fd.detect_photo(gate.yunet, img) if img is not None else None
        owner_crop = passport(img, d0) if d0 is not None else img
    owner = t.get("name")
    same_as_owner = owner is not None and who["name"] is not None and who["name"].lower() == owner.lower() \
        and who["status"] in ("identified", "guess")
    a = {"kind": kind, "number": t.get("number"), "owner": owner, "owner_crop": owner_crop,
         "live_crop": passport(frame, det), "who": who, "same_as_owner": same_as_owner, "t0": time.time(),
         "stage": "stop"}

    if same_as_owner:
        line = (f"ALERT: ID {a['number']} ({owner}). The face is closest to the ID owner's own record "
                f"(distance {who['d']:.2f}), so this may be a false alarm: officer to check.")
    elif who["status"] == "identified":
        line = (f"ALERT: ID {a['number']} belongs to {owner or 'nobody on record'}, but the person at the gate "
                f"is {who['name']} (distance {who['d']:.2f}).")
    elif who["status"] == "guess":
        line = (f"ALERT: ID {a['number']} belongs to {owner or 'nobody on record'}. Person at the gate not "
                f"confidently identified, best guess {who['name']} (distance {who['d']:.2f}).")
    else:
        line = (f"ALERT: ID {a['number']} belongs to {owner or 'nobody on record'}. Person at the gate is "
                f"not registered in our database.")
    a["line"] = line
    a["speech"] = "Stop. Please stay where you are. An officer has been called."
    print("\n" + "*" * 80 + "\n" + line + "\n" + "*" * 80, flush=True)
    start_alarm()

    folder = gate.db.folder
    n = gate.db.n
    path = os.path.join(folder, "alerts.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["attempt", "time", "result", "id_number", "id_owner", "person_at_gate", "status",
                        "distance", "threshold", "live_photo", "owner_photo"])
        w.writerow([n, time.strftime("%Y-%m-%dT%H:%M:%S"), kind, a["number"], owner or "",
                    who["name"] or "", "same as owner" if same_as_owner else who["status"],
                    f"{who['d']:.4f}" if who["d"] is not None else "", f"{gate.threshold:.4f}",
                    gate.db.image(n, "alert_live", a["live_crop"]), gate.db.image(n, "alert_owner", owner_crop)])
    return a


# ---------------------------------------------------------------- the red screen
def _text(img, msg, centre_x, y, scale, colour, thick):
    font = cv2.FONT_HERSHEY_DUPLEX
    tw = cv2.getTextSize(msg, font, scale, thick)[0][0]
    cv2.putText(img, msg, (int(centre_x - tw / 2), int(y)), font, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, msg, (int(centre_x - tw / 2), int(y)), font, scale, colour, thick, cv2.LINE_AA)


def _fit(msg, max_w, scale, thick):
    """Shrink the text scale until the message fits max_w pixels."""
    while scale > 0.3 and cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, scale, thick)[0][0] > max_w:
        scale *= 0.92
    return scale


def _panel(canvas, cx, top, ph, crop, title, lines):
    pw = int(ph * 0.75)
    k = canvas.shape[1] / 1280
    _text(canvas, title, cx, top - 18 * k, _fit(title, 0.45 * canvas.shape[1], 0.9 * k, 2), (255, 255, 255), 2)
    x0 = int(cx - pw / 2)
    if crop is not None:
        canvas[top:top + ph, x0:x0 + pw] = cv2.resize(crop, (pw, ph))
    else:
        cv2.rectangle(canvas, (x0, top), (x0 + pw, top + ph), (40, 40, 40), -1)
        _text(canvas, "no photo", cx, top + ph / 2, 0.8 * k, (200, 200, 200), 2)
    cv2.rectangle(canvas, (x0, top), (x0 + pw, top + ph), (255, 255, 255), 3)
    y = top + ph + 45 * k
    for msg, colour, big in lines:
        sc = _fit(msg, 0.46 * canvas.shape[1], (1.0 if big else 0.7) * k, 2)
        _text(canvas, msg, cx, y, sc, colour, 2)
        y += (48 if big else 36) * k


def _who_lines(gate, who, same_as_owner):
    if same_as_owner:
        return [(f"closest record: {who['name'].replace('_', ' ')}", (0, 255, 255), True),
                ("the ID owner themselves: officer to check", (230, 230, 230), False)]
    if who["status"] == "identified":
        return [(who["name"].replace("_", " ").upper(), (0, 255, 255), True),
                (f"identified from our records (distance {who['d']:.2f})", (230, 230, 230), False)]
    if who["status"] == "guess":
        return [(f"best guess: {who['name'].replace('_', ' ')}", (0, 255, 255), True),
                (f"weak match (distance {who['d']:.2f}, needs < {gate.threshold:.2f})", (230, 230, 230), False)]
    return [("NOT REGISTERED", (0, 255, 255), True), ("not in our database", (230, 230, 230), False)]


def render(gate, size):
    """The alert screen: STOP (flashing red, nothing personal) or, after SPACE, the officer view."""
    a = gate.result["alert"]
    SW, SH = size
    k = SW / 1280
    if a["stage"] == "stop":
        bright = int((time.time() - a["t0"]) / FLASH_S) % 2 == 0
        canvas = np.full((SH, SW, 3), (0, 0, 200) if bright else (0, 0, 110), np.uint8)
        _text(canvas, "STOP", SW / 2, SH * 0.42, _fit("STOP", 0.8 * SW, 6.0 * k, 12), (255, 255, 255), 12)
        msg = "Please stay where you are. An officer has been called."
        _text(canvas, msg, SW / 2, SH * 0.58, _fit(msg, 0.9 * SW, 1.1 * k, 2), (255, 255, 255), 2)
        _text(canvas, "Officer: press SPACE", SW / 2, SH - 25 * k, 0.6 * k, (255, 255, 255), 1)
        return canvas

    canvas = np.full((SH, SW, 3), (30, 20, 70), np.uint8)              # steady dark red: officer view
    _text(canvas, "OFFICER VIEW - IDENTITY DOES NOT MATCH", SW / 2, 60 * k,
          _fit("OFFICER VIEW - IDENTITY DOES NOT MATCH", 0.95 * SW, 1.3 * k, 3), (255, 255, 255), 3)
    reason = {"not_same_person": "the person at the gate is not the holder of this ID",
              "photo_swapped": "the photo on this card is not the holder's",
              "unknown_number": "this ID number is not in our records"}[a["kind"]]
    _text(canvas, reason, SW / 2, 105 * k, _fit(reason, 0.9 * SW, 0.8 * k, 2), (0, 255, 255), 2)
    ph, top = int(0.48 * SH), int(170 * k)
    owner = a["owner"]
    left = [(owner.replace("_", " ").upper() if owner else "NOT ON RECORD", (255, 255, 255), True),
            (f"ID number {a['number']}", (230, 230, 230), False)]
    if OFFICER_SHOWS_PERSON:
        _panel(canvas, SW * 0.27, top, ph, a["owner_crop"], "THIS ID BELONGS TO", left)
        _panel(canvas, SW * 0.73, top, ph, a["live_crop"], "PERSON AT THE GATE",
               _who_lines(gate, a["who"], a["same_as_owner"]))
    else:
        _panel(canvas, SW * 0.5, top, ph, a["owner_crop"], "THIS ID BELONGS TO", left)
    _text(canvas, "SPACE: next traveller   (full details in the terminal and session_db/alerts.csv)",
          SW / 2, SH - 25 * k, 0.6 * k, (255, 255, 255), 1)
    return canvas
