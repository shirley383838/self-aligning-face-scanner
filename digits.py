"""
digits.py  -  read the 4-digit ID number off the flattened card (HOG + SVM, Week 7)

After id_card.py flattens the card, every digit sits in a known box, so reading
the number is just classifying 4 small crops as 0 to 9.

Training data: we don't have photos of hundreds of digits, but we know exactly
how make_id_cards.py draws them (same font, size and box). So the digits are
drawn the same way and then made to look like a webcam picture of a printed
card: shrunk to camera size, blurred, noise and JPEG added, brightness and
contrast changed, slightly rotated and shifted, then blown back up to the size
the flattened card has. Each crop -> grey -> HOG -> SVM (same tools as accessories.py).

Running this file trains the model (about 20 s), saves models/digits.pkl, and
tests it on fresh rendered digits it has not seen.

Every 10th read is saved to results/digit_reads/ (the flattened card, the digit boxes
in red and what was read), so wrong reads on the real camera can be looked at.
"""

import os
import pickle

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.svm import SVC

import id_card as card
import make_id_cards as mic

MODEL_FILE = os.path.join(card.BASE_DIR, "models", "digits.pkl")
CROP = (32, 40)                 # each digit crop is resized to this (width, height) before HOG
N_PER_DIGIT = 600               # training crops per digit
SEED = 5710

_hog = cv2.HOGDescriptor(_winSize=CROP, _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)


# ---------------------------------------------------------------- features
def digit_feature(crop_bgr):
    """Grey, stretched to full contrast, resized, HOG."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.resize(g, CROP, interpolation=cv2.INTER_AREA)
    return _hog.compute(g).ravel()


def digit_crops(warped):
    """The 4 digit boxes of a flattened card, as images."""
    return [warped[y0:y1, x0:x1] for x0, y0, x1, y1 in card.digit_boxes()]


# ---------------------------------------------------------------- training data
def render_box(ch, margin_mm=1.5):
    """One digit box drawn exactly like make_id_cards.py does, at 300 dpi, with some card around it."""
    _, _, bw, bh, _ = card.DIGITS_MM
    m = mic.mm(margin_mm)
    W, H = mic.mm(bw) + 2 * m, mic.mm(bh) + 2 * m
    img = np.full((H, W, 3), mic.TINT, np.uint8)
    cv2.rectangle(img, (m, m), (m + mic.mm(bw), m + mic.mm(bh)), (255, 255, 255), -1)
    cv2.rectangle(img, (m, m), (m + mic.mm(bw), m + mic.mm(bh)), mic.BOX_EDGE, 2)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(mic.FONT_BOLD, int(round(mic.mm(bh * 0.62) / mic.CAP)))
    ImageDraw.Draw(pil).text((m + mic.mm(bw) / 2, m + mic.mm(bh) / 2), ch, font=font,
                             fill=mic.INK[::-1], anchor="mm")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), m


def camera_look(img, m, rng):
    """Make a clean rendered box look like it came off a webcam and through the card flattening."""
    _, _, bw, bh, _ = card.DIGITS_MM
    box_px_300 = mic.mm(bw)
    cam_w = rng.uniform(14, 90)                                  # box width in camera pixels (14 = card ~150 px wide)
    s = cam_w / box_px_300
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # small rotation and shear, as the flattening is never perfect
    h, w = small.shape[:2]
    A = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3, 3), rng.uniform(0.95, 1.05))
    A[0, 1] += rng.uniform(-0.04, 0.04)
    small = cv2.warpAffine(small, A, (w, h), borderMode=cv2.BORDER_REPLICATE)
    small = cv2.GaussianBlur(small, (0, 0), rng.uniform(0.3, 1.4))
    small = small.astype(np.float32) * rng.uniform(0.6, 1.1) + rng.uniform(-25, 15)
    small = np.clip(small + rng.normal(0, rng.uniform(1, 6), small.shape), 0, 255).astype(np.uint8)
    small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(55, 95))])[1], 1)
    # back up to the flattened card's scale (10 px per mm), like warpPerspective does
    up = card.PX_PER_MM / (300 / 25.4)
    big = cv2.resize(small, (int(round(img.shape[1] * up)), int(round(img.shape[0] * up))),
                     interpolation=cv2.INTER_LINEAR)
    # crop the box with a small random shift, as the found corners are off by a pixel or two
    mb = int(round(m * up))
    bwp, bhp = int(round(bw * card.PX_PER_MM)), int(round(bh * card.PX_PER_MM))
    dx, dy = rng.integers(-5, 6, 2)
    x0, y0 = max(0, mb + dx), max(0, mb + dy)
    return big[y0:y0 + bhp, x0:x0 + bwp]


def make_set(n_per_digit, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for d in range(10):
        img, m = render_box(str(d))
        for _ in range(n_per_digit):
            X.append(digit_feature(camera_look(img, m, rng)))
            y.append(d)
    return np.array(X, np.float32), np.array(y)


# ---------------------------------------------------------------- model
def train():
    X, y = make_set(N_PER_DIGIT, SEED)
    model = SVC(kernel="rbf", C=10, gamma="scale").fit(X, y)
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    Xt, yt = make_set(100, SEED + 1)                             # fresh crops it has not seen
    acc = float(np.mean(model.predict(Xt) == yt))
    conf = np.zeros((10, 10), int)
    for t, p in zip(yt, model.predict(Xt)):
        conf[t, p] += 1
    print(f"Digit model trained on {len(y)} crops, test accuracy on {len(yt)} new crops: {acc:.3f}")
    print("confusion (rows = true digit 0 to 9, columns = read as):")
    print(conf)
    return model


_model = None


def load():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_FILE):
            print("No digit model yet, training one (about 20 s)...")
            _model = train()
        else:
            with open(MODEL_FILE, "rb") as f:
                _model = pickle.load(f)
    return _model


SAVE_READS = True               # keep a rolling set of flattened cards with what was read, for checking
SAVE_DIR = os.path.join(card.BASE_DIR, "results", "digit_reads")
SAVE_EVERY, SAVE_KEEP = 10, 40
_n_reads = 0


def read_number(c):
    """The 4-digit ID number on a read card (from id_card.read_card), as a string."""
    global _n_reads
    X = np.array([digit_feature(crop) for crop in digit_crops(c["warped"])], np.float32)
    number = "".join(str(d) for d in load().predict(X))
    _n_reads += 1
    if SAVE_READS and _n_reads % SAVE_EVERY == 0:
        img = c["warped"].copy()
        for x0, y0, x1, y1 in card.digit_boxes():
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.putText(img, f"read {number}", (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        os.makedirs(SAVE_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(SAVE_DIR, f"read_{(_n_reads // SAVE_EVERY) % SAVE_KEEP:02d}.jpg"), img)
    return number


def vote(numbers):
    """Most common digit in each position over several reads, so one bad frame doesn't matter."""
    out = ""
    for k in range(card.N_DIGITS):
        ds = [n[k] for n in numbers]
        out += max(set(ds), key=ds.count)
    return out


if __name__ == "__main__":
    train()
