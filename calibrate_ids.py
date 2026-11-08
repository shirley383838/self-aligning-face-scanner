"""
Sets the face match threshold on the gate camera.

The LFW threshold does not carry over to front-on, evenly lit photos, so it is
measured here instead:
  1. Each enrolled person types their name and looks at the camera; 15 good frames
     are saved to data/calib/<Name>_NN.jpg.
  2. Every frame is compared with every person's record photo (data/ids) and card
     photo (data/cards), giving same-person and different-person distances.
  3. The threshold with the best balanced accuracy is saved to
     models/id_thresholds.json under "own_camera".

RECOMPUTE_ONLY = True skips the camera and reuses data/calib.
Run: python calibrate_ids.py (empty name + Enter to finish). Keys: r rotate, q skip person.
"""

import os
import glob
import json

import numpy as np
import cv2

import face_detect as fd
import id_match as idm
import id_card as card
import make_id_cards as mic

CAMERA_INDEX = 1            # your webcam (find_cameras.py)
ROTATION = 0                # 0, 90, 180 or 270 if the camera is on its side (or press r)
FRAMES_PER_PERSON = 15
EVERY = 4                   # take every 4th good frame, so the 15 aren't all the same
RECOMPUTE_ONLY = False      # True = don't open the camera, just recompute from data/calib

CALIB_DIR = os.path.join(card.BASE_DIR, "data", "calib")
CARDS_DIR = os.path.join(card.BASE_DIR, "data", "cards")
ROTATE_CODES = card.ROTATE_CODES


# ---------------------------------------------------------------- the enrolled photos
def enrolled_names():
    return sorted({idm.person_name(f) for f in os.listdir(idm.IDS_DIR)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))})


def card_photo(name):
    """The photo box of that person's (genuine) card image in data/cards, as printed."""
    for path in glob.glob(os.path.join(CARDS_DIR, "*.png")):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.lower().endswith("_fake") or stem.rsplit("_", 1)[0].lower() != name.lower():
            continue
        img = cv2.imread(path)
        x0, y0, x1, y1 = card.PHOTO_MM
        return img[mic.mm(y0):mic.mm(y1), mic.mm(x0):mic.mm(x1)]
    return None


def templates(yunet, method):
    """name -> list of feature vectors (record photo, card photo)."""
    out = {}
    for name in enrolled_names():
        feats = []
        for f in os.listdir(idm.IDS_DIR):
            if idm.person_name(f).lower() == name.lower():
                img = cv2.imread(os.path.join(idm.IDS_DIR, f))
                det = fd.detect_photo(yunet, img) if img is not None else None
                if det is not None:
                    feats.append(idm.features(img, det, [method])[method])
        photo = card_photo(name)
        if photo is not None:
            det = fd.detect_photo(yunet, cv2.resize(photo, None, fx=2, fy=2))
            if det is not None:
                feats.append(idm.features(photo, fd.scale_detection(det, 0.5), [method])[method])
        if feats:
            out[name] = feats
    return out


# ---------------------------------------------------------------- step 1: frames from the camera
def capture(yunet):
    os.makedirs(CALIB_DIR, exist_ok=True)
    names = {n.lower(): n for n in enrolled_names()}
    print("Enrolled:", ", ".join(names.values()))
    cam = cv2.VideoCapture(CAMERA_INDEX, card.CAMERA_BACKEND)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}. Run find_cameras.py to see which numbers work.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)        # some drivers remember autofocus OFF (capture_grid.py turns it off)
    rotation = ROTATION
    while True:
        typed = input("\nName (as on their ID), or just Enter to finish: ").strip()
        if not typed:
            break
        if typed.lower() not in names:
            print(f"  '{typed}' has no photo in data/ids. Enrolled: {', '.join(names.values())}")
            continue
        name = names[typed.lower()]
        old = glob.glob(os.path.join(CALIB_DIR, f"{name}_*.jpg"))
        start = len(old)
        print(f"  {name}: look at the camera (glasses and hat off). {FRAMES_PER_PERSON} frames are taken by themselves.")
        got, good = 0, 0
        while got < FRAMES_PER_PERSON:
            ok, raw = cam.read()
            if not ok:
                break
            frame = raw if ROTATE_CODES[rotation] is None else cv2.rotate(raw, ROTATE_CODES[rotation])
            h, w = frame.shape[:2]
            det = fd.detect_scaled(yunet, frame)
            view = frame.copy()
            if det is None:
                msg, colour = "No face found (r turns the view)", card.YELLOW
            else:
                problem = card.face_problem(det, w, h)
                bx, by, bw, bh = det["box"]
                if problem is None or problem == "Move to the middle of the picture":
                    good += 1
                    if good % EVERY == 0:
                        cv2.imwrite(os.path.join(CALIB_DIR, f"{name}_{start + got + 1:02d}.jpg"), frame)
                        got += 1
                    msg, colour = f"{name}: {got}/{FRAMES_PER_PERSON}", card.GREEN
                else:
                    msg, colour = problem, card.YELLOW
                cv2.rectangle(view, (bx, by), (bx + bw, by + bh), colour, 2)
            card.text(view, msg, (20, 40), colour)
            if h > card.DISPLAY_MAX_H:
                view = cv2.resize(view, None, fx=card.DISPLAY_MAX_H / h, fy=card.DISPLAY_MAX_H / h)
            cv2.imshow("calibrate (r rotate, q skip)", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                rotation = (rotation + 90) % 360
            if key in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
        print(f"  saved {got} frames of {name}")
    cam.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------- steps 2 and 3: distances and threshold
def balanced_threshold(same_d, diff_d):
    """
    Threshold with the best balanced accuracy (same- and different-person pairs weighted
    equally). Returns (threshold, balanced accuracy).
    """
    order = np.sort(np.concatenate([same_d, diff_d]))
    cands = np.concatenate([[order[0] - 1e-6], (order[:-1] + order[1:]) / 2, [order[-1] + 1e-6]])
    scores = [(np.mean(same_d < c) + np.mean(diff_d >= c)) / 2 for c in cands]
    i = int(np.argmax(scores))
    return float(cands[i]), float(scores[i])


def compute(yunet, method):
    temps = templates(yunet, method)
    frames = sorted(f for f in glob.glob(os.path.join(CALIB_DIR, "*"))
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    same_d, diff_d, people = [], [], set()
    for path in frames:
        who = idm.person_name(os.path.basename(path))
        if who not in temps:
            continue
        img = cv2.imread(path)
        det = fd.detect_scaled(yunet, img)
        if det is None:
            continue
        f = idm.features(img, det, [method])[method]
        people.add(who)
        for name, feats in temps.items():
            d = [idm.distance(f, t) for t in feats]
            (same_d if name == who else diff_d).extend(d)
    if not same_d or not diff_d:
        print("Not enough frames: need at least one enrolled person in data/calib (and two people enrolled).")
        return None
    same_d, diff_d = np.array(same_d), np.array(diff_d)
    d_all = np.concatenate([same_d, diff_d])
    t, acc = balanced_threshold(same_d, diff_d)
    false_reject = int(np.sum(same_d >= t))
    false_accept = int(np.sum(diff_d < t))
    print(f"\n{method}: {len(frames)} frames of {len(people)} people, compared with {len(temps)} people's photos")
    print(f"  same person:      {len(same_d):5d} comparisons, distance {same_d.min():.2f} to {same_d.max():.2f} "
          f"(median {np.median(same_d):.2f})")
    print(f"  different people: {len(diff_d):5d} comparisons, distance {diff_d.min():.2f} to {diff_d.max():.2f} "
          f"(median {np.median(diff_d):.2f})")
    print(f"  best threshold {t:.3f}: {acc:.1%} right, "
          f"{false_reject}/{len(same_d)} same-person wrongly rejected, "
          f"{false_accept}/{len(diff_d)} different-person wrongly accepted")
    if same_d.max() < diff_d.min():
        print("  the two groups don't overlap at all")

    fig, ax = idm.plt.subplots(figsize=(6, 3))
    bins = np.linspace(d_all.min(), d_all.max(), 40)
    ax.hist(same_d, bins, alpha=0.6, label="same person")
    ax.hist(diff_d, bins, alpha=0.6, label="different people")
    ax.axvline(t, color="k", ls="--", lw=1, label="threshold")
    ax.set_xlabel("distance")
    ax.set_title(f"{method} on our camera: {acc:.1%} right at {t:.2f}", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    idm.save_fig(fig, f"08_own_camera_distances_{method}.png")

    with open(idm.THRESH_FILE) as fh:
        info = json.load(fh)
    info.setdefault("own_camera", {})[method] = t
    info.setdefault("own_camera_stats", {})[method] = {
        "frames": len(frames), "people": len(people), "accuracy": acc,
        "same": len(same_d), "different": len(diff_d),
        "false_reject": false_reject, "false_accept": false_accept}
    with open(idm.THRESH_FILE, "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"Saved as the {method} threshold for our camera in {idm.THRESH_FILE}")
    return t


def main():
    yunet = fd.load_yunet()
    method = card.pick_method()
    print(f"Feature type in use: {method}")
    if not RECOMPUTE_ONLY:
        capture(yunet)
    compute(yunet, method)
    if method != "hog":                                    # HOG too, for the report comparison
        compute(yunet, "hog")


if __name__ == "__main__":
    main()
