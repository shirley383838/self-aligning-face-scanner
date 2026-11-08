"""
gate_demo.py  -  the whole Member 3 part as one e-gate run

  1. Scan your card   hold the printed ID card up close. It is found, flattened and
                      checked (still, sharp, focused, no glare) like in id_card.py,
                      then the face in its photo and the 4-digit ID number are read
                      (digits.py, HOG + SVM). The number is looked up in
                      data/cards/records.csv to get the name and the enrolled photo.
  2. Clear your face  look at the camera. The glasses and hat classifiers
                      (accessories.py) check the aligned face; if either is on, it
                      asks you to take it off and waits.
  3. Face check       once the face is clear, big enough, front-on and centred, it is
                      compared with the card photo over 10 frames (id_match.py features).
  4. Result           your face has to match BOTH the card photo and the enrolled photo
                      for that ID number:
                        both match           -> GATE OPEN, welcome <name>
                        card photo only      -> CARD DOES NOT MATCH OUR RECORDS (photo swapped)
                        neither / card only  -> DOES NOT MATCH THE CARD
                        number not on record -> CARD NUMBER NOT IN OUR RECORDS
                      The result stays on screen until you press c for the next person.

Everything is reused from id_card.py, accessories.py, id_match.py and face_detect.py,
so id_card.py and live_demo.py still run on their own as a backup.

Keys (click the window first):
    c       next person (start again at step 1)
    y / n   after a result: was it really the person on the card? (saved to
            results/card_checks.csv, same as id_card.py, and the best threshold is printed)
    - / =   lower / raise the hat threshold
    r       turn the view 90 degrees        b   show / hide the aligned-face inset
    s       snapshot to figures_member3/    q / Esc  quit
"""

import os
import csv
from collections import deque

import numpy as np
import cv2

import face_detect as fd
import accessories as acc
import id_match as idm
import id_card as card
import digits as dg
from face_align import align_face, draw_regions

# ---------------------------------------------------------------- settings
CAMERA_INDEX = 1          # the number find_cameras.py showed for your webcam
THRESHOLD = None          # None = id_card.py's threshold for the feature type in use (THRESHOLDS)
GLASSES_THRESHOLD = 0.0   # accessory score needed to say glasses (0 = the model's own boundary)
HAT_THRESHOLD = 0.0       # same for the hat; change live with - and =
THRESHOLD_STEP = 0.1
SMOOTH_FRAMES = 10        # accessory scores and face distances are averaged over this many frames
ROTATION = 0
AUTO_ROTATE = True        # no face found: try the other orientations every AUTO_EVERY frames
AUTO_EVERY = 5
DISPLAY_MAX_H = 900

GREEN, RED, YELLOW, GREY, WHITE = (0, 220, 0), (0, 0, 255), (0, 255, 255), (200, 200, 200), (255, 255, 255)
STEP_NAMES = ["1 Scan card", "2 Clear face", "3 Face check", "4 Result"]
RECORDS_CSV = os.path.join(card.BASE_DIR, "data", "cards", "records.csv")
IDS_DIR = os.path.join(card.BASE_DIR, "data", "ids")


def load_records():
    """ID number -> name, from the records make_id_cards.py wrote."""
    if not os.path.exists(RECORDS_CSV):
        return {}
    with open(RECORDS_CSV) as fh:
        return {r["id_number"]: r["name"] for r in csv.DictReader(fh)}


def enrolled_feature(yunet, name, method):
    """Feature of that person's enrolled photo in data/ids (the trusted copy, like a passport chip)."""
    for f in sorted(os.listdir(IDS_DIR)):
        if idm.person_name(f).lower() == name.lower():
            img = cv2.imread(os.path.join(IDS_DIR, f))
            det = fd.detect_photo(yunet, img) if img is not None else None
            if det is not None:
                return idm.features(img, det, [method])[method]
    return None


RESULT_TEXT = {"open": "GATE OPEN",
               "swapped": "CARD DOES NOT MATCH OUR RECORDS",
               "mismatch": "DOES NOT MATCH THE CARD",
               "unknown": "CARD NUMBER NOT IN OUR RECORDS",
               "no_photo": "NO ENROLLED PHOTO FOR THIS CARD"}


def steps_bar(view, current):
    """The four steps along the bottom, with the current one highlighted."""
    h = view.shape[0]
    x = 20
    for k, name in enumerate(STEP_NAMES):
        colour = GREEN if k < current else (YELLOW if k == current else (110, 110, 110))
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(view, (x - 6, h - 72), (x + tw + 6, h - 48), (0, 0, 0), -1)
        cv2.putText(view, name, (x, h - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)
        x += tw + 24


def main():
    yunet = fd.load_yunet()
    if not (os.path.exists(idm.THRESH_FILE) and os.path.exists(idm.EIGEN_FILE)):
        print("Run id_match.py once first: the face check needs models/id_thresholds.json and models/eigenfaces.npz.")
        return
    method = card.pick_method()
    records = load_records()
    dg.load()
    print(f"{len(records)} ID numbers on record")
    threshold = THRESHOLD if THRESHOLD is not None else card.threshold_for(method)
    acc_thresholds = {"glasses": GLASSES_THRESHOLD, "hat": HAT_THRESHOLD}
    print(f"Face check: {method} features, threshold {threshold:.2f}")
    print(f"Opening camera {CAMERA_INDEX}...", flush=True)
    cam = cv2.VideoCapture(CAMERA_INDEX, card.CAMERA_BACKEND)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}. Run find_cameras.py to see which numbers work.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)        # some drivers remember autofocus OFF (capture_grid.py turns it off)
    print(__doc__)

    rotation, n_frame, n_snap, show_inset = ROTATION, 0, 0, True
    stage, locked, result, logged = 0, None, None, False   # stage = index into STEP_NAMES
    good, sharp_hist, prev_corners = [], deque(maxlen=card.FOCUS_FRAMES), None
    acc_hist = {item: deque(maxlen=SMOOTH_FRAMES) for item in acc.ITEMS}
    dists = deque(maxlen=SMOOTH_FRAMES)
    dists_db = deque(maxlen=SMOOTH_FRAMES)

    def reset():
        nonlocal stage, locked, result, logged, good, prev_corners
        stage, locked, result, logged, good, prev_corners = 0, None, None, False, [], None
        sharp_hist.clear()
        dists.clear()
        dists_db.clear()
        for hist in acc_hist.values():
            hist.clear()

    while True:
        ok, raw = cam.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break
        n_frame += 1
        frame = card.rotate(raw, rotation)
        view = frame.copy()
        h, w = frame.shape[:2]
        corners = card.find_card(frame)

        # ------------------------------------------------ step 1: scan the card
        if stage == 0:
            c = card.read_card(frame, yunet, corners) if corners is not None else None
            problem, sharp = None, None
            if c is None:
                problem = "Hold your ID card up to the camera, flat and close"
                sharp_hist.clear()
                prev_corners = None
            else:
                sharp = card.card_sharpness(c)
                sharp_hist.append(sharp)
                still = prev_corners is not None and \
                    card.corner_shift(c["corners"], prev_corners) < card.STILL_FRAC * card.card_width(c["corners"])
                if card.card_width(c["corners"]) < card.CARD_MIN_WIDTH * w:
                    problem = "Card found: bring it closer"
                elif not still:
                    problem = "Hold the card still"
                elif sharp < card.MIN_SHARPNESS:
                    problem = "Card is blurry: hold still, or a little further away so the camera can focus"
                elif len(sharp_hist) < card.FOCUS_FRAMES or min(sharp_hist) < card.STEADY * max(sharp_hist):
                    problem = "Waiting for the camera to focus..."
                elif card.card_glare(c) > card.GLARE_FRAC:
                    problem = "Glare on the card photo: tilt the card a little"
                elif c["face"] is None:
                    problem = "Card found, but no face found in its photo"
                prev_corners = c["corners"]
                cv2.polylines(view, [np.int32(c["corners"])], True, GREEN if problem is None else YELLOW, 3)
                inset = cv2.resize(c["warped"], None, fx=0.45, fy=0.45)
                card.paste(view, inset, w - inset.shape[1] - 10, 10)
                card.text(view, f"card sharpness {sharp:.2f} (needs {card.MIN_SHARPNESS:.1f})", (20, 80), GREY, 0.5, 1)
            if problem is not None:
                good = []
                card.text(view, problem, (20, 40), YELLOW)
            else:
                good.append(c)
                card.text(view, f"Reading card... {len(good)}/{card.CARD_FRAMES}  keep it still", (20, 40), GREEN)
                if len(good) >= card.CARD_FRAMES:
                    f = np.mean([card.card_feature(g, method) for g in good], axis=0)
                    best = max(good, key=card.card_sharpness)
                    number = dg.vote([dg.read_number(g) for g in good])
                    name = records.get(number)
                    locked = {"feature": f / (np.linalg.norm(f) + 1e-9), "photo": best["photo"].copy(),
                              "crop": idm.face_vector(best["photo"], best["face"]), "number": number,
                              "name": name, "db_feature": enrolled_feature(yunet, name, method) if name else None}
                    good = []
                    if name is None:
                        result, stage = {"kind": "unknown", "d": None}, 3
                        print(f"Card number {number} is not in our records.")
                    elif locked["db_feature"] is None:
                        result, stage = {"kind": "no_photo", "d": None}, 3
                        print(f"Card number {number} ({name}) has no usable enrolled photo in data/ids.")
                    else:
                        stage = 1
                        print(f"Card read: {number} = {name}. Now look at the camera.")

        # ------------------------------------------------ steps 2 and 3: clear face, then compare
        elif stage in (1, 2):
            det = card.person_face(frame, yunet, corners)          # a face still on the card is ignored
            if det is None and AUTO_ROTATE and n_frame % AUTO_EVERY == 0:
                for r in (90, 180, 270):
                    cand = (rotation + r) % 360
                    f2 = card.rotate(raw, cand)
                    d2 = card.person_face(f2, yunet, card.find_card(f2))
                    if d2 is not None:
                        rotation, frame, det = cand, f2, d2
                        view, (h, w) = f2.copy(), f2.shape[:2]
                        print(f"Face found with the view turned {rotation} degrees, switched to that.")
                        break
            cv2.line(view, (int(card.CENTRE_ZONE[0] * w), 0), (int(card.CENTRE_ZONE[0] * w), h), (90, 90, 90), 1)
            cv2.line(view, (int(card.CENTRE_ZONE[1] * w), 0), (int(card.CENTRE_ZONE[1] * w), h), (90, 90, 90), 1)
            if det is None:
                dists.clear()
                dists_db.clear()
                for hist in acc_hist.values():
                    hist.clear()
                stage = 1
                card.text(view, "Card read. Now look at the camera", (20, 40), YELLOW)
            else:
                le, re = det["left_eye"], det["right_eye"]
                bx, by, bw, bh = det["box"]
                for ex, ey in (le, re):
                    cv2.circle(view, (int(ex), int(ey)), 4, WHITE, -1)
                scores = acc.accessory_scores(frame, le, re)
                for item in acc.ITEMS:
                    acc_hist[item].append(scores[item])
                avg = {item: float(np.mean(hh)) for item, hh in acc_hist.items()}
                wearing = [item for item in acc.ITEMS
                           if len(acc_hist[item]) >= 3 and avg[item] > acc_thresholds[item]]
                problem = card.face_problem(det, w, h)
                if wearing:
                    stage = 1
                    dists.clear()
                    dists_db.clear()
                    cv2.rectangle(view, (bx, by), (bx + bw, by + bh), RED, 2)
                    card.text(view, "Please remove your " + " and ".join(wearing), (20, 40), RED, 0.9)
                elif problem is not None:
                    stage = 1
                    cv2.rectangle(view, (bx, by), (bx + bw, by + bh), YELLOW, 2)
                    card.text(view, problem, (20, 40), YELLOW)
                else:
                    stage = 2
                    f_live = idm.features(frame, det, [method])[method]
                    dists.append(idm.distance(f_live, locked["feature"]))
                    dists_db.append(idm.distance(f_live, locked["db_feature"]))
                    cv2.rectangle(view, (bx, by), (bx + bw, by + bh), YELLOW, 2)
                    card.text(view, f"Checking face... {len(dists)}/{SMOOTH_FRAMES}", (20, 40), YELLOW)
                    pair = card.compared_inset(idm.face_vector(frame, det), locked["crop"])
                    card.paste(view, pair, w - pair.shape[1] - 10, h - pair.shape[0] - 90)
                    if len(dists) >= SMOOTH_FRAMES:
                        d_avg, d_db = float(np.mean(dists)), float(np.mean(dists_db))
                        same_card, same_db = d_avg < threshold, d_db < threshold
                        kind = "open" if same_card and same_db else ("swapped" if same_card else "mismatch")
                        result = {"kind": kind, "d": d_avg, "d_db": d_db}
                        stage = 3
                        print(f"Result: {RESULT_TEXT[kind]} (card photo {d_avg:.2f}, enrolled photo {d_db:.2f}, "
                              f"needs < {threshold:.2f})")
                acc_line = "   ".join(f"{item} {avg[item]:+.2f} (>{acc_thresholds[item]:+.1f} = on)" for item in acc.ITEMS)
                card.text(view, acc_line, (20, 80), GREY, 0.5, 1)
                if show_inset:
                    inset = cv2.resize(draw_regions(align_face(frame, le, re)), None, fx=1.2, fy=1.2)
                    card.paste(view, inset, 10, 100)
            photo = cv2.resize(locked["photo"], None, fx=0.6, fy=0.6)
            card.paste(view, photo, w - photo.shape[1] - 10, 10)
            card.text(view, f"card {locked['number']}: {locked['name']}",
                      (w - photo.shape[1] - 10, 10 + photo.shape[0] + 22), WHITE, 0.5, 1)

        # ------------------------------------------------ step 4: result (held until c)
        else:
            kind = result["kind"]
            colour = GREEN if kind == "open" else RED
            cv2.rectangle(view, (0, 0), (w - 1, h - 1), colour, 12)
            msg = RESULT_TEXT[kind] + (f", welcome {locked['name']}" if kind == "open" else "")
            card.text(view, msg, (20, 50), colour, 1.1, 3)
            if result["d"] is not None:
                card.text(view, f"card {locked['number']} ({locked['name']})   {method} distance to card photo "
                          f"{result['d']:.2f}, to enrolled photo {result['d_db']:.2f}, needs < {threshold:.2f}",
                          (20, 95), GREY, 0.55, 1)
            else:
                card.text(view, f"number read off the card: {locked['number']}", (20, 95), GREY, 0.6, 1)
            card.text(view, "c next person   |   y same person as card photo, n different (to log the check)",
                      (20, 130), GREY, 0.5, 1)
            photo = cv2.resize(locked["photo"], None, fx=0.6, fy=0.6)
            card.paste(view, photo, w - photo.shape[1] - 10, 10)
            card.text(view, "card photo", (w - photo.shape[1] - 10, 10 + photo.shape[0] + 22), WHITE, 0.5, 1)

        steps_bar(view, stage)
        card.text(view, f"view turned {rotation} deg (r)   hat needs > {acc_thresholds['hat']:+.1f} (- / =)",
                  (20, h - 20), GREY, 0.5, 1)
        if h > DISPLAY_MAX_H:
            view = cv2.resize(view, None, fx=DISPLAY_MAX_H / h, fy=DISPLAY_MAX_H / h)
        cv2.imshow("E-gate demo (c next person, q quit)", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            reset()
        if key in (ord("y"), ord("n")) and result is not None and result["d"] is not None:
            if logged:
                print("  already saved for this person: press c for the next one")
            else:
                card.log_check(method, result["d"], key == ord("y"))
                logged = True
        if key == ord("-"):
            acc_thresholds["hat"] -= THRESHOLD_STEP
            print(f"hat threshold {acc_thresholds['hat']:+.1f} (set HAT_THRESHOLD to keep it)")
        if key == ord("="):
            acc_thresholds["hat"] += THRESHOLD_STEP
            print(f"hat threshold {acc_thresholds['hat']:+.1f} (set HAT_THRESHOLD to keep it)")
        if key == ord("r"):
            rotation = (rotation + 90) % 360
            print(f"view turned {rotation} degrees (set ROTATION = {rotation} to keep it)")
        if key == ord("b"):
            show_inset = not show_inset
        if key == ord("s"):
            os.makedirs(card.FIGURE_DIR, exist_ok=True)
            n_snap += 1
            path = os.path.join(card.FIGURE_DIR, f"gate_snapshot_{n_snap:02d}.png")
            cv2.imwrite(path, view)
            print("snapshot saved:", path)
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
