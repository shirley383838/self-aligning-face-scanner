"""
Serial control of the camera rig (GRBL 1.1 on an EleksMaker Mana SE).

Set PORT, X_MAX and Y_MAX (travel measured in UGS), close UGS and the Arduino IDE, and
put the carriage at the bottom left before connecting: opening the port resets GRBL
and makes that position (0, 0).
"""

import time
import serial

PORT = "COM5"
BAUD = 115200

# Usable travel measured from the start corner, in mm
X_MAX = 323
Y_MAX = 247.850
MARGIN = 5.0          # stay this far inside the measured boundary

FEED = 3000           # mm/min for normal moves


class Rig:
    def __init__(self, port=PORT, baud=BAUD):
        self.ser = serial.Serial(port, baud, timeout=0.5)
        time.sleep(2)                     # the Nano resets when the port opens
        self.ser.reset_input_buffer()
        self.send("$X")                   # clear the alarm if there is one
        self.send("G21 G90")              # millimetres, absolute positions

    def send(self, line, timeout=10.0):
        """Send one G-code line and wait for GRBL to answer 'ok'."""
        self.ser.write((line + "\n").encode())
        start = time.time()
        while time.time() - start < timeout:
            resp = self.ser.readline().decode(errors="ignore").strip()
            if resp == "ok":
                return
            if resp.startswith("error") or resp.startswith("ALARM"):
                raise RuntimeError(f"GRBL replied '{resp}' to '{line}'")
        raise TimeoutError(f"No reply from GRBL to '{line}'")

    def status(self):
        """Return (state, x, y) from a GRBL status report."""
        self.ser.write(b"?")
        start = time.time()
        while time.time() - start < 2.0:
            resp = self.ser.readline().decode(errors="ignore").strip()
            if resp.startswith("<") and resp.endswith(">"):
                fields = resp[1:-1].split("|")
                state = fields[0]
                for f in fields[1:]:
                    if f.startswith("MPos:") or f.startswith("WPos:"):
                        x, y = map(float, f[5:].split(",")[:2])
                        return state, x, y
                return state, None, None
        raise TimeoutError("No status report from GRBL")

    def wait_until_idle(self):
        time.sleep(0.05)
        while True:
            state, x, y = self.status()
            if state.startswith("Idle"):
                return x, y
            if state.startswith("Alarm"):
                raise RuntimeError("GRBL is in alarm. Check the rig, then restart.")
            time.sleep(0.05)

    def goto(self, x, y, feed=FEED):
        """Move to (x, y) in mm and block until the move has finished."""
        lo_x, hi_x = MARGIN, X_MAX - MARGIN
        lo_y, hi_y = MARGIN, Y_MAX - MARGIN
        if not (lo_x <= x <= hi_x and lo_y <= y <= hi_y):
            raise ValueError(
                f"({x:.1f}, {y:.1f}) is outside the safe area "
                f"X {lo_x}-{hi_x}, Y {lo_y}-{hi_y}"
            )
        self.send(f"G1 X{x:.3f} Y{y:.3f} F{feed}")
        return self.wait_until_idle()

    def home_corner(self):
        """Go back to the start corner (0, 0), e.g. before closing."""
        self.send(f"G1 X0 Y0 F{FEED}")
        self.wait_until_idle()

    def close(self):
        self.ser.close()


def grid_points(nx, ny):
    """nx by ny points over the safe area, in snake order to avoid long jumps."""
    xs = [MARGIN + i * (X_MAX - 2 * MARGIN) / (nx - 1) for i in range(nx)]
    ys = [MARGIN + j * (Y_MAX - 2 * MARGIN) / (ny - 1) for j in range(ny)]
    points = []
    for j, y in enumerate(ys):
        row = xs if j % 2 == 0 else list(reversed(xs))
        for x in row:
            points.append((x, y))
    return points


if __name__ == "__main__":
    rig = Rig()
    try:
        input("Carriage should be in the start corner. Press Enter to begin...")

        # 1. Visit the four corners of the safe area slowly, as a check
        corners = [
            (MARGIN, MARGIN),
            (X_MAX - MARGIN, MARGIN),
            (X_MAX - MARGIN, Y_MAX - MARGIN),
            (MARGIN, Y_MAX - MARGIN),
        ]
        for x, y in corners:
            print(f"Corner -> ({x:.1f}, {y:.1f})")
            rig.goto(x, y, feed=600)
            time.sleep(1.0)

        # 2. Visit every point of a 10 x 10 grid
        for k, (x, y) in enumerate(grid_points(20, 20)):
            pos = rig.goto(x, y)
            print(f"Point {k:3d}: target ({x:6.1f}, {y:6.1f})  reported {pos}")
            time.sleep(0.2)        # this is where you would grab a camera frame

        rig.home_corner()
    finally:
        rig.close()