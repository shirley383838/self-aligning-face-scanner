"""
find_cameras.py  -  which camera number is which?

Tries camera numbers 0 to 4, grabs one picture from each camera that opens, and
saves it to figures_member3/camera_<number>.png. Open those pictures to see
which number is the laptop camera and which is the USB webcam, then put that
number in CAMERA_INDEX at the top of live_demo.py and capture.py.

Close any other app that might be using a camera (Zoom, Teams, the Windows
Camera app, another Python window) before running this.
"""

import os
import time

import cv2

MAX_INDEX = 4
CAMERA_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY   # DirectShow is faster on Windows
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(BASE_DIR, "figures_member3")


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    found = []
    for i in range(MAX_INDEX + 1):
        print(f"camera {i}: ", end="", flush=True)
        t0 = time.time()
        cam = cv2.VideoCapture(i, CAMERA_BACKEND)
        if not cam.isOpened():
            print("nothing there")
            continue
        ok, frame = False, None
        for _ in range(15):                  # first few frames are often black while the camera starts
            ok, frame = cam.read()
        cam.release()
        if not ok or frame is None:
            print(f"opened but gave no picture ({time.time() - t0:.1f} s)")
            continue
        path = os.path.join(FIGURE_DIR, f"camera_{i}.png")
        cv2.imwrite(path, frame)
        found.append(i)
        print(f"OK, {frame.shape[1]} x {frame.shape[0]}, opened in {time.time() - t0:.1f} s, picture saved: {path}")
    if found:
        print(f"\nWorking cameras: {found}. Open the saved pictures to see which is the USB webcam.")
    else:
        print("\nNo cameras worked. Check the webcam is plugged in, nothing else is using it, and Windows "
              "Settings > Privacy > Camera allows desktop apps to use the camera.")


if __name__ == "__main__":
    main()
