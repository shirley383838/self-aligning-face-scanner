"""
Glasses and hat classifiers.

Training (run after prep_celeba.py): crops the eye band and hat region from the
aligned CelebA faces, extracts HOG and HSV histogram features, and compares SVM,
AdaBoost and KNN by stratified 5-fold cross-validated grid search. The best model
per item is retrained on the dev set, tested once on the held-out set and saved to
models/accessory_<item>.pkl. QUICK = True runs a small grid for testing.

Live use: accessory_scores(frame, left_eye, right_eye).
"""

import os
import csv
import time
import pickle
import itertools

import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold

from face_align import align_face, crop_region

# ---------------------------------------------------------------- settings
QUICK = False
SEED = 5710

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETS_DIR = os.path.join(BASE_DIR, "data", "sets")
ALIGNED_DIR = os.path.join(BASE_DIR, "data", "aligned", "celeba")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(BASE_DIR, "figures_member3")

# item -> (region in face_align.REGIONS, size the crop is resized to for HOG)
ITEMS = {
    "glasses": ("eyes", (128, 48)),
    "hat":     ("hat",  (128, 64)),
}

# HOG: 16 px cells, 2x2-cell blocks, 9 orientation bins (coarse cells suit small crops)
HOG_CELL, HOG_BLOCK, HOG_STRIDE, HOG_BINS = 16, 32, 16, 9
# HSV histogram: 8 bins per channel, on a 2 x 2 grid of the crop
HSV_BINS, HSV_GRID = 8, 2

FEATURE_SETS = ["HOG", "HSV", "HOG+HSV"]

if QUICK:
    N_FOLDS = 3
    GRIDS = {
        "SVM":      {"C": [1, 10], "gamma": ["scale"]},
        "AdaBoost": {"n_estimators": [100], "learning_rate": [1.0]},
        "KNN":      {"n_neighbors": [5, 15]},
    }
else:
    N_FOLDS = 5
    GRIDS = {
        "SVM":      {"C": [0.1, 1, 10, 100], "gamma": ["scale", 1e-3, 1e-4]},
        "AdaBoost": {"n_estimators": [100, 200], "learning_rate": [1.0]},   # AdaBoost is the slow one
        "KNN":      {"n_neighbors": [3, 7, 15, 31]},
    }


def make_classifier(name, params):
    """Build one classifier with the given settings."""
    if name == "SVM":
        return SVC(kernel="rbf", **params)
    if name == "AdaBoost":   # default weak learner is a depth-1 decision tree (a stump), as in Viola-Jones
        return AdaBoostClassifier(random_state=SEED, **params)
    if name == "KNN":
        return KNeighborsClassifier(**params)
    raise ValueError(name)


SHOW_FIGURES = False
FIGURE_DPI = 150
if not SHOW_FIGURES:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- features
_hog_cache = {}


def hog_feature(crop_bgr, size):
    """HOG of the grayscale crop resized to `size` (Week 7 tutorial, cv2.HOGDescriptor)."""
    if size not in _hog_cache:
        _hog_cache[size] = cv2.HOGDescriptor(_winSize=size, _blockSize=(HOG_BLOCK, HOG_BLOCK),
                                             _blockStride=(HOG_STRIDE, HOG_STRIDE),
                                             _cellSize=(HOG_CELL, HOG_CELL), _nbins=HOG_BINS)
    gray = cv2.cvtColor(cv2.resize(crop_bgr, size), cv2.COLOR_BGR2GRAY)
    return _hog_cache[size].compute(gray).ravel()


def hsv_feature(crop_bgr):
    """8-bin H, S and V histograms on a 2 x 2 grid, each normalised to sum to 1 (Week 2 colour spaces)."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    feats = []
    for gy in range(HSV_GRID):
        for gx in range(HSV_GRID):
            cell = hsv[gy * h // HSV_GRID:(gy + 1) * h // HSV_GRID, gx * w // HSV_GRID:(gx + 1) * w // HSV_GRID]
            for ch, top in zip(range(3), (180, 256, 256)):      # OpenCV hue runs 0-179
                hist, _ = np.histogram(cell[:, :, ch], bins=HSV_BINS, range=(0, top))
                feats.append(hist / max(1, hist.sum()))
    return np.concatenate(feats).astype(np.float32)


def extract(crop_bgr, size, feature_set):
    """Feature vector for one crop and one feature set name."""
    parts = []
    if "HOG" in feature_set:
        parts.append(hog_feature(crop_bgr, size))
    if "HSV" in feature_set:
        parts.append(hsv_feature(crop_bgr))
    return np.concatenate(parts)


# ---------------------------------------------------------------- metrics (numpy, as in the Week 7 tutorial)
def metrics(y_true, y_pred):
    """Confusion matrix [[TN, FP], [FN, TP]], accuracy, precision, recall and F1 for a yes/no task."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"confusion": np.array([[tn, fp], [fn, tp]]), "accuracy": (tp + tn) / len(y_true),
            "precision": precision, "recall": recall, "f1": f1}


def standardise(train, *others):
    """Scale every feature to zero mean, unit spread using the TRAINING data only."""
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd[sd == 0] = 1.0
    return [(train - mu) / sd] + [(o - mu) / sd for o in others] + [(mu, sd)]


def cross_validate(X, y, name, params, folds):
    """Mean and spread of F1 and accuracy over the CV folds for one classifier setting."""
    f1s, accs = [], []
    for tr, va in folds:
        Xtr, Xva, _ = standardise(X[tr], X[va])
        clf = make_classifier(name, params).fit(Xtr, y[tr])
        m = metrics(y[va], clf.predict(Xva))
        f1s.append(m["f1"])
        accs.append(m["accuracy"])
    return np.mean(f1s), np.std(f1s), np.mean(accs)


def grid(params):
    """Every combination of a settings grid, as a list of dicts."""
    keys = list(params)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(params[k] for k in keys))]


# ---------------------------------------------------------------- data loading
def load_set(item):
    """Image ids, labels and split ('dev' or 'test') for one item from prep_celeba.py."""
    path = os.path.join(SETS_DIR, f"celeba_{item}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run prep_celeba.py first.")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return ([r["image_id"] for r in rows], np.array([int(r["label"]) for r in rows]),
            np.array([r["split"] for r in rows]))


def load_crops(ids, region):
    """Crop one region out of each aligned CelebA face. Stops with a clear message if faces are missing."""
    crops, missing = [], []
    for i in ids:
        img = cv2.imread(os.path.join(ALIGNED_DIR, i.replace(".jpg", ".png")))
        if img is None:
            missing.append(i)
            continue
        crops.append(crop_region(img, region))
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(ids)} aligned faces are missing (first: {missing[0]}). "
                                "Run prep_celeba.py again and check it finishes without errors.")
    return crops


# ---------------------------------------------------------------- figures
def save_fig(fig, name):
    """Save a figure into figures_member3 and close it."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, name)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print("  figure saved:", path)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def cv_figure(item, best):
    """Bar chart of cross-validated F1 for every feature set and classifier."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    names = list(GRIDS)
    width = 0.8 / len(FEATURE_SETS)
    for k, fs in enumerate(FEATURE_SETS):
        means = [best[(fs, n)]["f1"] for n in names]
        sds = [best[(fs, n)]["f1_sd"] for n in names]
        ax.bar(np.arange(len(names)) + (k - 1) * width, means, width, yerr=sds, capsize=3, label=fs)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel(f"F1 ({N_FOLDS}-fold CV)")
    lo = min(best[k]["f1"] for k in best)
    ax.set_ylim(max(0.0, lo - 0.1), 1.0)
    ax.legend(fontsize=8)
    ax.set_title(f"{item}: cross-validated F1 by feature set and classifier", fontsize=9)
    save_fig(fig, f"03_{item}_cv.png")


def test_figure(item, m, crops, y, pred, title, fname, n_show=8):
    """Confusion matrix plus up to n_show misclassified crops."""
    wrong = np.where(pred != y)[0][:n_show]
    fig = plt.figure(figsize=(2.6 + 1.3 * max(1, len(wrong)), 2.6))
    ax = fig.add_axes([0.02, 0.1, 0.22, 0.75])
    ax.imshow(m["confusion"], cmap="Blues")
    for (r, c), v in np.ndenumerate(m["confusion"]):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no", "yes"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no", "yes"])
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("true", fontsize=8)
    ax.set_title(f"F1 {m['f1']:.3f}", fontsize=8)
    for k, idx in enumerate(wrong):
        a = fig.add_axes([0.28 + k * (0.7 / max(1, len(wrong))), 0.15, 0.68 / max(1, len(wrong)), 0.6])
        a.imshow(cv2.cvtColor(crops[idx], cv2.COLOR_BGR2RGB))
        a.set_title(f"true {'yes' if y[idx] else 'no'}", fontsize=7)
        a.axis("off")
    fig.suptitle(title, fontsize=9)
    save_fig(fig, fname)


# ---------------------------------------------------------------- training
def train_item(item, summary):
    """Cross-validate every feature set and classifier, then test the best on the held-out set."""
    region, size = ITEMS[item]
    ids, y, split = load_set(item)
    print(f"\n=== {item} ({region} region): {np.sum(split == 'dev')} dev, {np.sum(split == 'test')} test")
    crops = load_crops(ids, region)
    feats = {fs: np.array([extract(c, size, fs) for c in crops]) for fs in FEATURE_SETS}
    for fs in FEATURE_SETS:
        print(f"  {fs:8s} features: {feats[fs].shape[1]} numbers per image")

    dev, test = np.where(split == "dev")[0], np.where(split == "test")[0]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(dev, y[dev]))

    best = {}
    for fs in FEATURE_SETS:
        X = feats[fs][dev]
        for name, params_grid in GRIDS.items():
            t0 = time.time()
            for params in grid(params_grid):
                f1, f1_sd, acc = cross_validate(X, y[dev], name, params, folds)
                summary.append([item, fs, name, str(params), round(f1, 4), round(f1_sd, 4), round(acc, 4)])
                if (fs, name) not in best or f1 > best[(fs, name)]["f1"]:
                    best[(fs, name)] = {"f1": f1, "f1_sd": f1_sd, "acc": acc, "params": params}
            b = best[(fs, name)]
            print(f"  {fs:8s} {name:9s} best {str(b['params']):42s} F1 {b['f1']:.3f} +/- {b['f1_sd']:.3f}"
                  f"  acc {b['acc']:.3f}  ({time.time() - t0:.0f} s)")
    cv_figure(item, best)

    # retrain the overall winner on all dev data, test once on the held-out set
    (fs, name), b = max(best.items(), key=lambda kv: kv[1]["f1"])
    Xdev, Xtest, (mu, sd) = standardise(feats[fs][dev], feats[fs][test])
    clf = make_classifier(name, b["params"]).fit(Xdev, y[dev])
    pred = clf.predict(Xtest)
    m = metrics(y[test], pred)
    print(f"  WINNER {fs} + {name} {b['params']}")
    print(f"  held-out CelebA test: accuracy {m['accuracy']:.3f}, precision {m['precision']:.3f}, "
          f"recall {m['recall']:.3f}, F1 {m['f1']:.3f}")
    print(f"  confusion [[TN FP] [FN TP]] = {m['confusion'].tolist()}")
    test_figure(item, m, [crops[i] for i in test], y[test], pred,
                f"{item}: held-out CelebA test ({fs} + {name}), with misclassified examples",
                f"04_{item}_celeba_test.png")

    model = {"item": item, "region": region, "size": size, "feature_set": fs, "classifier": name,
             "params": b["params"], "model": clf, "mu": mu, "sd": sd,
             "hog": (HOG_CELL, HOG_BLOCK, HOG_STRIDE, HOG_BINS), "hsv": (HSV_BINS, HSV_GRID)}
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f"accessory_{item}.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"  model saved: models/accessory_{item}.pkl")
    return m


def main():
    t_start = time.time()
    print("QUICK run (small grid, 3 folds)" if QUICK else "FULL run (full grid, 5 folds)")
    summary = []
    for item in ITEMS:
        train_item(item, summary)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "accessory_cv_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "features", "classifier", "params", "cv_f1", "cv_f1_sd", "cv_accuracy"])
        w.writerows(summary)
    print(f"\nEvery setting tried is in {path}")
    print(f"Total time {(time.time() - t_start) / 60:.1f} min")


# ---------------------------------------------------------------- live use
_models = None


def _load_models():
    """Load the saved accessory models once."""
    global _models
    if _models is None:
        _models = {}
        for item in ITEMS:
            with open(os.path.join(MODEL_DIR, f"accessory_{item}.pkl"), "rb") as f:
                _models[item] = pickle.load(f)
    return _models


def accessory_scores(frame_bgr, left_eye, right_eye):
    """Signed score per item, {"glasses": s, "hat": s}: above 0 means present, larger means more confident."""
    aligned = align_face(frame_bgr, left_eye, right_eye)
    out = {}
    for item, m in _load_models().items():
        x = (extract(crop_region(aligned, m["region"]), m["size"], m["feature_set"]) - m["mu"]) / m["sd"]
        clf = m["model"]
        if hasattr(clf, "decision_function"):
            out[item] = float(clf.decision_function(x[None, :])[0])
        else:                                   # KNN: fraction of neighbours voting yes, shifted so 0 is the boundary
            out[item] = float(clf.predict_proba(x[None, :])[0, 1] - 0.5)
    return out


if __name__ == "__main__":
    main()
