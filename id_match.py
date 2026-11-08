"""
Face features and matching for the ID check.

Features, computed on the eye-aligned face:
  "hog"    HOG of the histogram-equalised grey face crop
  "eigen"  Eigenfaces from the SVD of FFHQ training faces
  "sface"  OpenCV's pre-trained SFace network (outside the unit; used if its model exists)

Running this file builds the Eigenfaces (RUN_EIGEN_TRAINING) and runs the 10-fold LFW
benchmark (RUN_LFW), saving models/eigenfaces.npz, models/id_thresholds.json and figures.
Other code calls features(img, det, [method]), distance(a, b) and threshold_of(method).
"""

import os
import re
import json
import time

import numpy as np
import cv2

import face_detect as fd
from face_align import align_face, crop_region

# ---------------------------------------------------------------- settings
RUN_EIGEN_TRAINING = False    # set False to reuse models/eigenfaces.npz
RUN_LFW = False               # set False to skip the LFW benchmark
MATCH_METHOD = "sface"        # "sface" (pre-trained CNN, outside the unit; "auto" if missing), "auto" = best in-unit, "hog", "eigen"
SFACE_THRESHOLD = 1.128       # OpenCV's published SFace threshold, used until one is measured

FACE_SIZE = (64, 80)          # face crop is resized to this (width, height) before features
N_COMPONENTS = 100            # eigenfaces kept
LFW_FALLBACK_SCORE = 0.6      # YuNet threshold for a second try on LFW photos it missed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFHQ_DIR = os.path.join(BASE_DIR, "data", "ffhq_thumbs")
LFW_HOME = os.path.join(BASE_DIR, "data", "lfw", "lfw_home")
IDS_DIR = os.path.join(BASE_DIR, "data", "ids")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIGURE_DIR = os.path.join(BASE_DIR, "figures_member3")
EIGEN_FILE = os.path.join(MODEL_DIR, "eigenfaces.npz")
THRESH_FILE = os.path.join(MODEL_DIR, "id_thresholds.json")
SFACE_MODEL = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")

SHOW_FIGURES = True

FIGURE_DPI = 150
if not SHOW_FIGURES:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- face crop and features
def face_vector(img, det):
    """Aligned, grey, histogram-equalised whole-face crop resized to FACE_SIZE (uint8)."""
    aligned = align_face(img, det["left_eye"], det["right_eye"])
    grey = cv2.cvtColor(crop_region(aligned, "face"), cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(cv2.resize(grey, FACE_SIZE))


_hog = None


def feat_hog(face):
    """HOG of the face crop (8 px cells, 2x2-cell blocks, 9 bins), scaled to unit length."""
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor(_winSize=FACE_SIZE, _blockSize=(16, 16), _blockStride=(8, 8),
                                 _cellSize=(8, 8), _nbins=9)
    v = _hog.compute(face).ravel().astype(np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


_eigen = None


def load_eigen():
    """Mean face and eigenfaces from models/eigenfaces.npz."""
    global _eigen
    if _eigen is None:
        d = np.load(EIGEN_FILE)
        _eigen = (d["mean"], d["components"])
    return _eigen


def feat_eigen(face):
    """Eigenface coefficients of the face crop, scaled to unit length."""
    mean, comps = load_eigen()
    v = comps @ (face.astype(np.float64).ravel() / 255.0 - mean)
    return v / (np.linalg.norm(v) + 1e-9)


_sface = None


def sface_available():
    return os.path.exists(SFACE_MODEL) and hasattr(cv2, "FaceRecognizerSF")


def feat_sface(img, det):
    """SFace feature (benchmark only, outside the unit), scaled to unit length."""
    global _sface
    if _sface is None:
        _sface = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
    crop = _sface.alignCrop(img, det["row"])
    v = _sface.feature(crop).ravel().astype(np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


def methods():
    """Feature types to run: always hog and eigen, plus sface if its model file is there."""
    return ["hog", "eigen"] + (["sface"] if sface_available() else [])


def features(img, det, which):
    """{method: feature vector} for one detected face."""
    face = face_vector(img, det)
    out = {}
    if "hog" in which:
        out["hog"] = feat_hog(face)
    if "eigen" in which:
        out["eigen"] = feat_eigen(face)
    if "sface" in which:
        out["sface"] = feat_sface(img, det)
    return out


def distance(a, b):
    """Euclidean distance between two unit-length feature vectors (0 = identical, 2 = opposite)."""
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------- figures
def save_fig(fig, name):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, name)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print("  figure saved:", path)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- 1. eigenfaces
def train_eigenfaces(yunet):
    """
    Eigenfaces from FFHQ: SVD of the mean-centred face crops, X - mean = U S V^T.
    The rows of V^T are the eigenfaces.
    """
    files = sorted(f for f in os.listdir(FFHQ_DIR) if f.lower().endswith((".png", ".jpg")))
    rows, missed = [], 0
    for f in files:
        img = cv2.imread(os.path.join(FFHQ_DIR, f))
        det = fd.detect_yunet(yunet, img) if img is not None else None
        if det is None:
            missed += 1
            continue
        rows.append(face_vector(img, det).astype(np.float64).ravel() / 255.0)
    X = np.array(rows)
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    k = min(N_COMPONENTS, len(S))
    explained = (S[:k] ** 2).sum() / (S ** 2).sum()
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(EIGEN_FILE, mean=mean, components=Vt[:k], singular_values=S[:k])
    print(f"Eigenfaces: {len(X)} FFHQ faces used ({missed} with no face found), "
          f"{k} eigenfaces keep {explained:.1%} of the variation")

    fig, axes = plt.subplots(1, 9, figsize=(11, 1.9))
    w, h = FACE_SIZE
    axes[0].imshow(mean.reshape(h, w), cmap="gray")
    axes[0].set_title("mean face", fontsize=7)
    for i, ax in enumerate(axes[1:]):
        ax.imshow(Vt[i].reshape(h, w), cmap="gray")
        ax.set_title(f"eigenface {i + 1}", fontsize=7)
    for ax in axes:
        ax.axis("off")
    save_fig(fig, "07_eigenfaces.png")
    global _eigen
    _eigen = None


# ---------------------------------------------------------------- 2. LFW benchmark
def read_pairs():
    """LFW pairs.txt -> list of (fold, image path 1, image path 2, same person 1/0)."""
    with open(os.path.join(LFW_HOME, "pairs.txt")) as f:
        lines = [l.split() for l in f if l.strip()]
    n_folds, n_per = int(lines[0][0]), int(lines[0][1])
    img_dir = os.path.join(LFW_HOME, "lfw_funneled")

    def path(name, n):
        return os.path.join(img_dir, name, f"{name}_{int(n):04d}.jpg")

    pairs, k = [], 1
    for fold in range(n_folds):
        for _ in range(n_per):
            name, a, b = lines[k]
            pairs.append((fold, path(name, a), path(name, b), 1))
            k += 1
        for _ in range(n_per):
            n1, a, n2, b = lines[k]
            pairs.append((fold, path(n1, a), path(n2, b), 0))
            k += 1
    return pairs


def best_threshold(d, same):
    """Threshold on distance that gives the highest accuracy (match if distance < threshold)."""
    order = np.sort(d)
    cands = np.concatenate([[order[0] - 1e-6], (order[:-1] + order[1:]) / 2, [order[-1] + 1e-6]])
    accs = [np.mean((d < t) == same) for t in cands]
    i = int(np.argmax(accs))
    return float(cands[i]), float(accs[i])


def lfw_benchmark(yunet, which):
    """Standard LFW 10-fold verification. Returns {method: overall threshold} and prints the table."""
    pairs = read_pairs()
    paths = sorted({p for _, a, b, _ in pairs for p in (a, b)})
    print(f"LFW: {len(pairs)} pairs, {len(paths)} different photos")
    fallback = cv2.FaceDetectorYN.create(model=fd.YUNET_MODEL, config="", input_size=(250, 250),
                                         score_threshold=LFW_FALLBACK_SCORE)
    feats, missed, t0 = {}, [], time.time()
    for i, p in enumerate(paths, 1):
        img = cv2.imread(p)
        if img is None:
            missed.append(p)
            continue
        det = fd.detect_yunet(yunet, img, pick="center") or fd.detect_yunet(fallback, img, pick="center")
        if det is None:
            missed.append(p)
            continue
        feats[p] = features(img, det, which)
        if i % 1000 == 0:
            print(f"  {i}/{len(paths)} photos ({time.time() - t0:.0f} s)")
    print(f"  no face found in {len(missed)} photos; pairs using them are left out")

    kept = [(f, a, b, s) for f, a, b, s in pairs if a in feats and b in feats]
    folds = np.array([f for f, _, _, _ in kept])
    same = np.array([s for _, _, _, s in kept], dtype=bool)
    results, thresholds = {}, {}
    for m in which:
        d = np.array([distance(feats[a][m], feats[b][m]) for _, a, b, _ in kept])
        accs = []
        for k in range(folds.max() + 1):
            t, _ = best_threshold(d[folds != k], same[folds != k])
            accs.append(np.mean((d[folds == k] < t) == same[folds == k]))
        t_all, _ = best_threshold(d, same)
        results[m] = (np.mean(accs), np.std(accs), d)
        thresholds[m] = t_all

    print(f"\n{'method':8s} {'LFW accuracy (10-fold)':>24s} {'threshold':>10s}")
    for m in which:
        mean_acc, sd_acc, _ = results[m]
        print(f"{m:8s} {mean_acc:17.3f} +/- {sd_acc:.3f} {thresholds[m]:10.3f}")
    print(f"  ({len(kept)} of {len(pairs)} pairs used; chance would be 0.500)")

    fig, axes = plt.subplots(1, len(which), figsize=(4 * len(which), 2.8), squeeze=False)
    for ax, m in zip(axes[0], which):
        d = results[m][2]
        bins = np.linspace(d.min(), d.max(), 40)
        ax.hist(d[same], bins, alpha=0.6, label="same person")
        ax.hist(d[~same], bins, alpha=0.6, label="different people")
        ax.axvline(thresholds[m], color="k", ls="--", lw=1, label="threshold")
        ax.set_title(f"{m}: LFW accuracy {results[m][0]:.3f}", fontsize=9)
        ax.set_xlabel("distance", fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "06_lfw_distances.png")

    in_unit = [m for m in which if m != "sface"]
    best = max(in_unit, key=lambda m: results[m][0])
    info = {"thresholds": thresholds, "best_in_unit": best,
            "lfw_accuracy": {m: [results[m][0], results[m][1]] for m in which}}
    if os.path.exists(THRESH_FILE):                          # keep the own-camera thresholds from calibrate_ids.py
        with open(THRESH_FILE) as f:
            old = json.load(f)
        for k in ("own_camera", "own_camera_stats"):
            if k in old:
                info[k] = old[k]
    with open(THRESH_FILE, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Best in-unit method on LFW: {best}. Thresholds saved: {THRESH_FILE}")
    return info


# ---------------------------------------------------------------- 3. using the features
def person_name(filename):
    """Shirley_Wong_2.jpg -> Shirley_Wong (a trailing _number allows several photos per person)."""
    stem = os.path.splitext(filename)[0]
    return re.sub(r"_\d+$", "", stem)


def current_method():
    """The feature type in use: MATCH_METHOD, with "sface" falling back to "auto" if the model file is missing."""
    if MATCH_METHOD == "sface" and not sface_available():
        print("SFace model not found in models/, using the best in-unit method instead.")
    if MATCH_METHOD == "auto" or (MATCH_METHOD == "sface" and not sface_available()):
        with open(THRESH_FILE) as fh:
            return json.load(fh)["best_in_unit"]
    return MATCH_METHOD


def own_threshold(method):
    """Threshold measured on our own camera by calibrate_ids.py, or None if it hasn't been run for this method."""
    with open(THRESH_FILE) as fh:
        return json.load(fh).get("own_camera", {}).get(method)


def threshold_of(method):
    """Match threshold for a feature type: our own camera (calibrate_ids.py) if measured, else the LFW run,
    else OpenCV's published one for SFace."""
    own = own_threshold(method)
    if own is not None:
        return own
    with open(THRESH_FILE) as fh:
        t = json.load(fh)["thresholds"]
    if method in t:
        return t[method]
    if method == "sface":
        return SFACE_THRESHOLD
    raise KeyError(f"No threshold for {method}: run id_match.py with RUN_LFW = True")


def main():
    t0 = time.time()
    yunet = fd.load_yunet()
    which = methods()
    print("Methods:", ", ".join(which), "" if "sface" in which else "(SFace model not found, skipped)")
    if RUN_EIGEN_TRAINING or not os.path.exists(EIGEN_FILE):
        train_eigenfaces(yunet)
    if RUN_LFW or not os.path.exists(THRESH_FILE):
        lfw_benchmark(yunet, which)
    print(f"Total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
