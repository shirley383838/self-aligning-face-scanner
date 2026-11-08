"""
label_eyes.py  -  hand-label the eyes on frames the detectors MISSED (optional)

Run face_detect.py first. It fills data/own/eyes.csv using the Week 11 detectors
(YuNet, then Haar). Frames where both missed are marked "none". This script only
shows you those frames, so you can click the eyes if you want to keep them for
training. Clicked rows are saved as method "manual" and face_detect.py keeps them
if you re-run it.

Keep in mind that the live system cannot classify a frame its detector missed,
so report test results on detector-found frames, and say how many were missed
per condition (face_detect.py prints this).

For each image:
    click the eye on the LEFT of the image, then the eye on the RIGHT
    ENTER or SPACE   accept (needs both clicks)
    r                redo the clicks
    s                skip (leave as "none")
    q                quit (progress is saved after every image)
"""

import os
import csv

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "own", "raw")
EYES_CSV = os.path.join(BASE_DIR, "data", "own", "eyes.csv")
MAX_VIEW_W = 1280          # images wider than this are shown scaled down (clicks are scaled back)
EYE_FIELDS = ["file", "method", "left_x", "left_y", "right_x", "right_y", "score"]
WINDOW = "label eyes: left-of-image eye first, ENTER accept, r redo, s skip, q quit"


def read_eyes():
    """All rows of eyes.csv."""
    if not os.path.exists(EYES_CSV):
        return []
    with open(EYES_CSV, newline="") as f:
        return list(csv.DictReader(f))


def write_eyes(rows):
    """Rewrite eyes.csv (small file, so it is simply rewritten after each image)."""
    with open(EYES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EYE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def order_eyes(p1, p2):
    """Return (left, right) by image x position, so a click in the wrong order cannot flip the face."""
    return (p1, p2) if p1[0] <= p2[0] else (p2, p1)


def label_one(img):
    """Show one image and collect two clicks. Returns ((xl, yl), (xr, yr)), 'skip' or 'quit'."""
    scale = min(1.0, MAX_VIEW_W / img.shape[1])
    view = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img.copy()
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        shown = view.copy()
        for (x, y) in clicks:
            cv2.circle(shown, (x, y), 4, (0, 255, 0), -1)
            cv2.circle(shown, (x, y), 12, (0, 255, 0), 1)
        if len(clicks) == 2:
            cv2.line(shown, clicks[0], clicks[1], (0, 255, 0), 1)
        cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10, ord(" ")) and len(clicks) == 2:
            (l, r) = order_eyes(*clicks)
            return ((l[0] / scale, l[1] / scale), (r[0] / scale, r[1] / scale))
        if key == ord("r"):
            clicks.clear()
        if key == ord("s"):
            return "skip"
        if key in (ord("q"), 27):
            return "quit"


def main():
    rows = read_eyes()
    if not rows:
        print("No eyes.csv yet: run face_detect.py first.")
        return
    todo = [i for i, r in enumerate(rows) if r["method"] == "none"]
    print(f"{len(todo)} frames where the detectors found no eyes")

    for n, i in enumerate(todo, 1):
        fname = rows[i]["file"]
        img = cv2.imread(os.path.join(RAW_DIR, fname))
        if img is None:
            print("missing file, skipped:", fname)
            continue
        print(f"[{n}/{len(todo)}] {fname}")
        result = label_one(img)
        if result == "quit":
            break
        if result == "skip":
            continue
        (xl, yl), (xr, yr) = result
        rows[i] = {"file": fname, "method": "manual", "left_x": round(xl, 1), "left_y": round(yl, 1),
                   "right_x": round(xr, 1), "right_y": round(yr, 1), "score": ""}
        write_eyes(rows)

    cv2.destroyAllWindows()
    print("Eye labels saved in:", EYES_CSV)


if __name__ == "__main__":
    main()
