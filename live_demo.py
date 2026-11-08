"""
live_demo.py  -  live webcam demo of the Member 3 part

Each frame:  YuNet finds the face and eyes (Week 11)  ->  the face is aligned on
the eyes  ->  the glasses and hat classifiers check the eye band and the hat
region  ->  the screen tells the person what to remove  ->  once the face is
clear, it is matched against the enrolled ID photos (id_match.py).

The small picture in the corner is the aligned face with the crop boxes, which
is what the classifiers actually look at (useful for the presentation).

Camera on its side: set ROTATION below (0, 90, 180 or 270), press r to turn the
view by 90 degrees, or leave AUTO_ROTATE on and it tries the other three
orientations by itself whenever it can't find a face. Small tilts don't need
any of this: the alignment step already levels the eyes.

Hat confidence: each model gives a score (above 0 = yes at the model's own
boundary, further from 0 = more sure). The hat says YES only when its score,
averaged over the last few frames, is above HAT_THRESHOLD. Press - to lower it
(says hat more easily) or = to raise it (needs to be more sure). The current
value prints in the terminal, so you can copy it into HAT_THRESHOLD.

If accessories.py has not been run yet, this first trains a quick DEMO model.
ID matching switches on once id_match.py has been run.

Keys (click on the window first):
    -  /  =  lower / raise the hat threshold
    r       turn the view 90 degrees
    e       enrol the person in front of the camera (type their name in the terminal)
    s       save a snapshot to figures_member3/
    b       show / hide the aligned-face inset
    q / Esc quit
"""

import os
import time
import pickle
from collections import deque

import numpy as np
import cv2

import face_detect as fd
import accessories as acc
import id_match as idm
from face_align import align_face, draw_regions

# ---------------------------------------------------------------- settings
CAMERA_INDEX = 0         # the number find_cameras.py showed for your webcam
FRAME_W, FRAME_H = 1280, 720
DETECT_W = 640            # YuNet runs on a copy this wide for speed (as in the Week 11 demo)
SMOOTH_FRAMES = 10        # scores are averaged over the last N frames, so the answer does not flicker
MIRROR = True             # mirror the display like a selfie camera (processing uses the unmirrored frame)
INSET_SCALE = 1.5         # size of the aligned-face inset
DISPLAY_MAX_H = 900       # shrink the window if the view is taller than this (camera on its side)

ROTATION = 0              # 0, 90, 180 or 270: how far to turn the camera picture clockwise
AUTO_ROTATE = True        # with no face found, try the other orientations and switch if one finds a face
AUTO_EVERY = 5            # how often (in frames) to try the other orientations while no face is found

GLASSES_THRESHOLD = 0.0   # score needed to say "glasses: YES" (0 = the model's own boundary)
HAT_THRESHOLD = 0.0       # score needed to say "hat: YES"; change live with - and =
THRESHOLD_STEP = 0.1

# On Windows, the DirectShow backend opens USB webcams much faster than the default
CAMERA_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

NAMES = {"glasses": "glasses", "hat": "hat"}
ROTATE_CODES = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


# ---------------------------------------------------------------- demo model (only if no trained model yet)
def train_demo_model(item):
    """Fixed-setting SVM on HOG + HSV, trained on the whole CelebA dev set. No tuning, demo only."""
    region, size, _ = acc.ITEMS[item]
    ids, y, split = acc.load_set(item)
    dev = split == "dev"
    crops = acc.load_crops([i for i, d in zip(ids, dev) if d], region)
    X = np.array([acc.extract(c, size, "HOG+HSV") for c in crops])
    Xs, (mu, sd) = acc.standardise(X)
    params = {"C": 10, "gamma": "scale"}
    clf = acc.make_classifier("SVM", params).fit(Xs, y[dev])
    model = {"item": item, "region": region, "size": size, "feature_set": "HOG+HSV", "classifier": "SVM",
             "params": params, "model": clf, "mu": mu, "sd": sd, "demo": True,
             "hog": (acc.HOG_CELL, acc.HOG_BLOCK, acc.HOG_STRIDE, acc.HOG_BINS),
             "hsv": (acc.HSV_BINS, acc.HSV_GRID)}
    os.makedirs(acc.MODEL_DIR, exist_ok=True)
    with open(os.path.join(acc.MODEL_DIR, f"accessory_{item}.pkl"), "wb") as f:
        pickle.dump(model, f)


def ensure_models():
    """Train a quick demo model for any item that has no saved model yet."""
    for item in acc.ITEMS:
        if not os.path.exists(os.path.join(acc.MODEL_DIR, f"accessory_{item}.pkl")):
            print(f"No trained {item} model yet: training a quick demo model (about a minute)...")
            t0 = time.time()
            train_demo_model(item)
            print(f"  done in {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- helpers
def text(img, msg, org, colour, scale=0.7, thick=2):
    """Text with a dark box behind it so it stays readable on any background."""
    (tw, th), base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)


def rotate(frame, degrees):
    """Turn the camera picture clockwise by 0, 90, 180 or 270 degrees."""
    code = ROTATE_CODES[degrees]
    return frame if code is None else cv2.rotate(frame, code)


def id_ready():
    """True once id_match.py has made the eigenfaces, thresholds and database."""
    return all(os.path.exists(f) for f in (idm.DB_FILE, idm.THRESH_FILE, idm.EIGEN_FILE))


def enrol_from_frame(frame, yunet):
    """Save the current (rotated) frame as an ID photo under a typed name, then rebuild the database."""
    name = input("\nName for this ID photo (letters, no spaces, Enter to cancel): ").strip().replace(" ", "_")
    if not name:
        print("cancelled")
        return
    os.makedirs(idm.IDS_DIR, exist_ok=True)
    path, n = os.path.join(idm.IDS_DIR, f"{name}.png"), 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(idm.IDS_DIR, f"{name}_{n}.png")
    cv2.imwrite(path, frame)
    print("saved", path)
    idm.enrol_folder(yunet, idm.methods())
    idm.reload_database()


# ---------------------------------------------------------------- main loop
def main():
    ensure_models()
    yunet = fd.load_yunet()
    print(f"Opening camera {CAMERA_INDEX}...", flush=True)
    cam = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}. Run find_cameras.py to see which numbers work.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    print(f"Camera {CAMERA_INDEX} open at {int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
          f"{int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))}. The window should appear now.", flush=True)

    thresholds = {"glasses": GLASSES_THRESHOLD, "hat": HAT_THRESHOLD}
    rotation = ROTATION
    history = {item: deque(maxlen=SMOOTH_FRAMES) for item in acc.ITEMS}
    id_history = deque(maxlen=SMOOTH_FRAMES)
    use_id = id_ready()
    print("ID matching on." if use_id else "ID matching off: run id_match.py first to switch it on.")
    show_inset, n_snap, t_prev, fps, n_frame = True, 0, time.time(), 0.0, 0
    print(__doc__)

    while True:
        ok, raw = cam.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break
        n_frame += 1
        frame = rotate(raw, rotation)
        det = fd.detect_scaled(yunet, frame, max_w=DETECT_W)   # YuNet on a smaller copy, scaled back

        # camera on its side: if no face, try the other orientations every few frames
        if det is None and AUTO_ROTATE and n_frame % AUTO_EVERY == 0:
            for r in (90, 180, 270):
                cand = (rotation + r) % 360
                f2 = rotate(raw, cand)
                d2 = fd.detect_scaled(yunet, f2, max_w=DETECT_W)
                if d2 is not None:
                    rotation, frame, det = cand, f2, d2
                    print(f"Face found with the view turned {rotation} degrees, switched to that.")
                    break

        h, w = frame.shape[:2]
        view = frame.copy()
        inset, id_result, stable, avg = None, None, None, None
        if det is not None:
            le, re = det["left_eye"], det["right_eye"]
            scores = acc.accessory_scores(frame, le, re)
            for item in acc.ITEMS:
                history[item].append(scores[item])
            avg = {item: float(np.mean(hist)) for item, hist in history.items()}
            stable = {item: avg[item] > thresholds[item] for item in acc.ITEMS}
            if use_id:
                name, d, is_match = idm.match_id(frame, det)
                id_history.append(name if is_match else None)
                votes = [n for n in id_history if n is not None]
                top = max(set(votes), key=votes.count) if votes else None
                id_result = (top if top is not None and votes.count(top) > len(id_history) / 2 else None,
                             name, d)
            colour = (0, 0, 255) if any(stable.values()) else (0, 200, 0)
            bx, by, bw, bh = det["box"]
            cv2.rectangle(view, (bx, by), (bx + bw, by + bh), colour, 2)
            for (ex, ey) in (le, re):
                cv2.circle(view, (int(ex), int(ey)), 4, (255, 255, 255), -1)
            if show_inset:
                inset = draw_regions(align_face(frame, le, re))
                inset = cv2.resize(inset, None, fx=INSET_SCALE, fy=INSET_SCALE)
        else:
            for hist in history.values():
                hist.clear()
            id_history.clear()

        if MIRROR:
            view = cv2.flip(view, 1)

        # status text (drawn after the flip so it reads the right way round)
        if stable is None:
            text(view, "No face found", (20, 40), (0, 255, 255))
            text(view, "step into the frame and face the camera", (20, 76), (0, 255, 255), 0.6, 1)
        else:
            y0 = 40
            for item, present in stable.items():
                text(view, f"{NAMES[item]}: {'YES' if present else 'no'}", (20, y0),
                     (0, 0, 255) if present else (0, 220, 0))
                text(view, f"score {avg[item]:+.2f}  needs > {thresholds[item]:+.2f}", (190, y0),
                     (200, 200, 200), 0.5, 1)
                y0 += 36
            if id_result is not None:
                confirmed, closest, d = id_result
                if closest is None:
                    text(view, "ID: nobody enrolled yet (press e)", (20, y0), (0, 255, 255))
                elif confirmed:
                    text(view, f"ID: {confirmed}  (distance {d:.2f})", (20, y0), (0, 220, 0))
                else:
                    text(view, f"ID: no match  (closest {closest}, {d:.2f})", (20, y0), (0, 0, 255))
                y0 += 36
            wearing = [NAMES[i] for i, p in stable.items() if p]
            if wearing:
                text(view, "Please remove your " + " and ".join(wearing), (20, y0 + 10), (0, 0, 255), 0.9)
            elif id_result is not None and id_result[0] is not None:
                text(view, f"Identity confirmed: {id_result[0]}", (20, y0 + 10), (0, 220, 0), 0.9)
            elif id_result is not None:
                text(view, "Face clear, but not recognised", (20, y0 + 10), (0, 0, 255), 0.9)
            else:
                text(view, "Face clear: OK to scan", (20, y0 + 10), (0, 220, 0), 0.9)

        if inset is not None and inset.shape[0] + 40 < h and inset.shape[1] + 20 < w:
            ih, iw = inset.shape[:2]
            view[10:10 + ih, w - iw - 10:w - 10] = inset
            text(view, "what the classifiers see", (w - iw - 10, 10 + ih + 22), (255, 255, 255), 0.5, 1)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(1e-3, now - t_prev)
        t_prev = now
        text(view, f"{fps:4.1f} fps   view turned {rotation} deg   hat needs > {thresholds['hat']:+.1f}",
             (20, h - 20), (200, 200, 200), 0.5, 1)

        if h > DISPLAY_MAX_H:                       # tall view (camera on its side): shrink to fit the screen
            view = cv2.resize(view, None, fx=DISPLAY_MAX_H / h, fy=DISPLAY_MAX_H / h)
        cv2.imshow("Member 3 live demo (- = hat threshold, r rotate, e enrol, s snapshot, q quit)", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("-"):
            thresholds["hat"] = round(thresholds["hat"] - THRESHOLD_STEP, 2)
            print(f"hat threshold now {thresholds['hat']:+.2f} (says hat more easily)")
        if key in (ord("="), ord("+")):
            thresholds["hat"] = round(thresholds["hat"] + THRESHOLD_STEP, 2)
            print(f"hat threshold now {thresholds['hat']:+.2f} (needs to be more sure)")
        if key == ord("r"):
            rotation = (rotation + 90) % 360
            for hist in history.values():
                hist.clear()
            print(f"view turned {rotation} degrees")
        if key == ord("b"):
            show_inset = not show_inset
        if key == ord("e"):
            if id_ready():
                enrol_from_frame(frame, yunet)
                use_id = True
                id_history.clear()
            else:
                print("Run id_match.py once first (it builds the eigenfaces and LFW thresholds), then enrol.")
        if key == ord("s"):
            os.makedirs(acc.FIGURE_DIR, exist_ok=True)
            n_snap += 1
            path = os.path.join(acc.FIGURE_DIR, f"demo_snapshot_{n_snap:02d}.png")
            cv2.imwrite(path, view)
            print("snapshot saved:", path)

    cam.release()
    cv2.destroyAllWindows()
    print(f"Last hat threshold: {thresholds['hat']:+.2f}, last view rotation: {rotation} degrees "
          "(copy these into HAT_THRESHOLD and ROTATION at the top to keep them)")


if __name__ == "__main__":
    main()
