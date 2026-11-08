"""
Runs main.py with only a webcam, no XY plotter.

Swaps the rig for a stand-in that accepts every move without moving, and turns off the
stereo check (it needs the rig). The head-turn live check runs instead; everything else
is the same as on the rig. Run: python run_no_rig.py (keys as in main.py).
"""

import os
import sys
import time
import types

CAMERA_INDEX = 0          # your webcam (find_cameras.py); the rig webcam number in main.py is not used
LIVENESS = False          # the stereo check needs the rig; the head-turn check still runs

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
