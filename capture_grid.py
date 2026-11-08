"""
Calibration capture with several board angles.

1. Shows a snapshot from each camera it finds.
2. You type the camera number in the terminal.
3. For each board angle (pose):
     - it asks you to set the board angle and press Enter
     - it moves the rig to IMAGES_PER_POSE positions and saves an image at each
4. Every image is taken from a different rig position. Each pose's positions
   are spread over the whole rig area, so every angle is seen from all over
   the frame.

Total images = IMAGES_PER_POSE x N_POSES (10 x 10 = 100 by default).

Setup (once):  pip install opencv-python pyserial
Put this file in the same folder as rig_grid.py.
Close UGS and the Arduino IDE before running.
The carriage must be at the bottom left before you press Enter to start.
Type q at a pose prompt to finish early, or press Ctrl + C to stop at any time.
"""

import time
from datetime import datetime
from pathlib import Path

import cv2

from rig_grid import Rig, FEED, MARGIN, X_MAX, Y_MAX

# ---------------- Settings ----------------
IMAGES_PER_POSE = 10      # images per board angle (also the number of grid columns)
N_POSES = 10              # number of board angles (also the number of grid rows)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
SETTLE_TIME = 0.4         # seconds to wait after each move before capturing
SAVE_ROTATED = False      # False = save raw images (recommended for calibration)
MAX_CAMERAS_TO_CHECK = 6
# ------------------------------------------


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap


def show_all_cameras():
    """Open a window for each camera showing one snapshot. Returns the list of indices found."""
    found = []
    for i in range(MAX_CAMERAS_TO_CHECK):
        cap = open_camera(i)
        if cap is None:
            continue
        frame = None
        for _ in range(10):                  # first frames are often black
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is None:
            continue

        found.append(i)
        preview = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h, w = preview.shape[:2]
        scale = min(1.0, 500 / h)
        preview = cv2.resize(preview, (int(w * scale), int(h * scale)))
        cv2.imshow(f"Camera {i}", preview)
        cv2.waitKey(300)                     # gives the window time to draw
        print(f"Camera {i} found")
    return found


def grab_fresh(cap):
    """Throw away old buffered frames so the image is from after the move."""
    for _ in range(5):
        cap.grab()
    ok, frame = cap.read()
    return frame if ok else None


def pose_positions():
    """
    Build an IMAGES_PER_POSE x N_POSES grid and split it into one group per pose.

    Pose p gets the points (column i, row (i + p) mod N_POSES), one in every
    column, on a wrapped diagonal. That spreads each pose over the whole area,
    and every grid point is used exactly once across all poses.
    """
    nx, ny = IMAGES_PER_POSE, N_POSES
    xs = [MARGIN + i * (X_MAX - 2 * MARGIN) / (nx - 1) for i in range(nx)]
    ys = [MARGIN + j * (Y_MAX - 2 * MARGIN) / (ny - 1) for j in range(ny)]

    poses = []
    for p in range(ny):
        points = [(xs[i], ys[(i + p) % ny]) for i in range(nx)]
        if p % 2 == 1:
            points.reverse()                 # alternate direction to shorten moves
        poses.append(points)
    return poses


def main():
    # 1. Preview cameras and choose one
    print("Looking for cameras...")
    found = show_all_cameras()
    if not found:
        print("No cameras found.")
        return

    while True:
        choice = input(f"Type the camera number to use {found}: ").strip()
        if choice.isdigit() and int(choice) in found:
            index = int(choice)
            break
        print("Not a valid camera number, try again.")
    cv2.destroyAllWindows()

    cap = open_camera(index)
    if cap is None:
        print(f"Could not open camera {index}.")
        return
    print(f"Using camera {index} at "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # 2. Make the output folder
    out_dir = Path.home() / "Downloads" / f"calib_poses_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    poses = pose_positions()
    total = sum(len(p) for p in poses)
    print(f"{len(poses)} board angles x {IMAGES_PER_POSE} images = {total} images")

    input("Put the carriage at the bottom left, then press Enter to connect the rig...")

    # 3. For each pose: wait for the board angle, then capture at its positions
    rig = Rig()
    saved = 0
    try:
        for p, points in enumerate(poses, start=1):
            print(f"\n=== Board angle {p} of {len(poses)} ===")
            reply = input("Set the board to a new angle, then press Enter (q = finish): ")
            if reply.strip().lower() == "q":
                break

            for k, (x, y) in enumerate(points, start=1):
                rig.goto(x, y, feed=FEED)
                time.sleep(SETTLE_TIME)

                frame = grab_fresh(cap)
                if frame is None:
                    print(f"  Image {k}: no frame, skipping")
                    continue
                if SAVE_ROTATED:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                name = f"pose{p:02d}_img{k:02d}_x{x:06.1f}_y{y:06.1f}_{stamp}.png"
                cv2.imwrite(str(out_dir / name), frame)
                saved += 1
                print(f"  [{k}/{len(points)}] saved {name}")

    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        rig.home_corner()
        rig.close()
        cap.release()
        print(f"\n{saved} images saved in {out_dir}")


if __name__ == "__main__":
    main()