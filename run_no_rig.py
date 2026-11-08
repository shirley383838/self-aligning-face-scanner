"""
run_no_rig.py  -  run main.py (the whole e-gate) with just a webcam, no XY plotter

For testing at home. Nothing in main.py or the rig files is changed: this file swaps
the rig for a stand-in before main.py starts, then runs it as normal.

What's different without the rig:
  * the stand-in "rig" accepts every move but nothing moves, so in the Position step
    the camera can't centre on you. After a moment the screen shows an arrow asking
    you to move instead: just shuffle until your face is in the middle.
  * the stereo 3D check needs the rig to move the camera for the stereo pairs, so it is
    switched off (CHECK_LIVENESS = False). The head-turn live check needs no rig and
    stays ON, so a photo, phone or tablet is still turned away. Everything else runs
    exactly as on the rig: ID card, number lookup, glasses / hat, head-turn live check,
    face match, photo, result, session database.

Run it from this folder:  python run_no_rig.py
Keys are main.py's: SPACE start / next traveller, n new traveller, d details, v voice,
r turn the view, f mirror, w full screen, q quit.
"""

import os
import sys
import time
import types

CAMERA_INDEX = 0          # your webcam (find_cameras.py); the rig webcam number in main.py is not used
LIVENESS = False          # the stereo 3D check can't work without the rig moving the camera
                          # (the head-turn live check, main.CHECK_HEAD_TURN, still runs)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:                                              # rig_grid.py needs pyserial even though nothing is sent
    import serial                                 # noqa: F401
except ImportError:
    sys.modules["serial"] = types.SimpleNamespace(Serial=None)

import rig_grid                                   # noqa: E402


class NoRig:
    """Stands in for rig_grid.Rig: same methods, no port, no movement."""

    def __init__(self, *args, **kwargs):
        self.x = self.y = 0.0
        print("No rig: using a stand-in, the camera will not move.", flush=True)

    def send(self, line, timeout=10.0):
        return None

    def status(self):
        return "Idle", self.x, self.y

    def wait_until_idle(self):
        return self.x, self.y

    def goto(self, x, y, feed=rig_grid.FEED):
        time.sleep(0.05)
        self.x, self.y = x, y
        return x, y

    def home_corner(self):
        self.x = self.y = 0.0

    def close(self):
        pass


rig_grid.Rig = NoRig
import face_follow as ff                          # noqa: E402
ff.Rig = NoRig
import main                                       # noqa: E402

main.CAMERA_INDEX = CAMERA_INDEX
main.CHECK_LIVENESS = LIVENESS

if __name__ == "__main__":
    main.main()
