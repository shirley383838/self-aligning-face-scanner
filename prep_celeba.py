"""
Builds the glasses and hat training sets from CelebA (data/celeba, Kaggle CSV or
official TXT version).

Picks balanced dev and held-out test sets per item from CelebA's own partitions (fixed
seed), finds the eyes with YuNet (CelebA landmarks as fallback), aligns every face to
data/aligned/celeba/, measures YuNet's eye error against CelebA's labels and saves an
example figure. Writes data/sets/celeba_<item>.csv for accessories.py.
"""

import os
import csv

import numpy as np
import cv2

from face_detect import load_yunet, detect_yunet
from face_align import align_face, draw_regions

# ---------------------------------------------------------------- settings
N_DEV = 1500        # per class, for training + cross-validation
N_TEST = 300        # per class, held out until the end
SEED = 5710
ITEMS = {"glasses": "Eyeglasses", "hat": "Wearing_Hat"}   # our name -> CelebA attribute
REDO_ALIGNED = False   # True re-aligns images that were already saved
N_CELEBA = 202599      # images in the full aligned CelebA set

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELEBA_DIR = os.path.join(BASE_DIR, "data", "celeba")
ALIGNED_DIR = os.path.join(BASE_DIR, "data", "aligned", "celeba")
SETS_DIR = os.path.join(BASE_DIR, "data", "sets")
FIGURE_DIR = os.path.join(BASE_DIR, "figures_member3")

SHOW_FIGURES = False
FIGURE_DPI = 150
if not SHOW_FIGURES:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- finding and reading CelebA
def find_file(root, names):
    """First file under root whose name is in names (searches nested folders)."""
    for folder, _, files in os.walk(root):
        for n in names:
            if n in files:
                return os.path.join(folder, n)
    return None


def find_image_dir(root):
    """Folder under root that contains 000001.jpg (Kaggle nests it two levels deep)."""
    for folder, _, files in os.walk(root):
        if "000001.jpg" in files:
            return folder
    return None


def read_table(path):
    """Read a CelebA annotation file (Kaggle CSV or official TXT) into {image_id: {column: value}}."""
    table = {}
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                table[row["image_id"]] = row
        return table

    with open(path) as f:
        lines = [l.split() for l in f if l.strip()]
    if len(lines[0]) == 1 and lines[0][0].isdigit():      # count line, then header
        names, rows = lines[1], lines[2:]
    else:                                               # partition file, no header
        names, rows = ["partition"], lines
    for r in rows:
        table[r[0]] = dict(zip(names, r[1:]), image_id=r[0])
    return table


def load_celeba():
    """Find and read the attributes, landmarks and partition files."""
    attr_path = find_file(CELEBA_DIR, ["list_attr_celeba.csv", "list_attr_celeba.txt"])
    lmk_path = find_file(CELEBA_DIR, ["list_landmarks_align_celeba.csv", "list_landmarks_align_celeba.txt"])
    part_path = find_file(CELEBA_DIR, ["list_eval_partition.csv", "list_eval_partition.txt"])
    img_dir = find_image_dir(CELEBA_DIR)
    missing = [n for n, p in [("attributes", attr_path), ("landmarks", lmk_path),
                              ("partition", part_path), ("images", img_dir)] if p is None]
    if missing:
        raise FileNotFoundError(f"Could not find CelebA {', '.join(missing)} under {CELEBA_DIR}")
    print("attributes:", attr_path)
    print("landmarks: ", lmk_path)
    print("partition: ", part_path)
    print("images:    ", img_dir)
    return read_table(attr_path), read_table(lmk_path), read_table(part_path), img_dir


# ---------------------------------------------------------------- sampling
def pick_sets(attrs, parts, rng):
    """Balanced dev and test sets for each item. Returns {item: [(image_id, label, split), ...]}."""
    dev_ids = sorted(i for i in attrs if parts.get(i, {}).get("partition") in ("0", "1"))
    test_ids = sorted(i for i in attrs if parts.get(i, {}).get("partition") == "2")
    sets = {}
    for item, attr in ITEMS.items():
        rows = []
        for split, pool, n in [("dev", dev_ids, N_DEV), ("test", test_ids, N_TEST)]:
            pos = [i for i in pool if int(attrs[i][attr]) == 1]
            neg = [i for i in pool if int(attrs[i][attr]) == -1]
            n_use = min(n, len(pos), len(neg))
            if n_use < n:
                print(f"  note: only {n_use} per class available for {item} {split}")
            rows += [(i, 1, split) for i in rng.choice(pos, n_use, replace=False)]
            rows += [(i, 0, split) for i in rng.choice(neg, n_use, replace=False)]
        sets[item] = rows
        print(f"{item}: {sum(r[2] == 'dev' for r in rows)} dev, {sum(r[2] == 'test' for r in rows)} test images")
    return sets


def celeba_eyes(lmk):
    """CelebA's labelled eye centres for one image, ordered image-left then image-right."""
    a = (float(lmk["lefteye_x"]), float(lmk["lefteye_y"]))
    b = (float(lmk["righteye_x"]), float(lmk["righteye_y"]))
    return (a, b) if a[0] <= b[0] else (b, a)


# ---------------------------------------------------------------- main
def check_images(img_dir):
    """Stop early if the image folder is not the full CelebA set (e.g. unzipping was interrupted)."""
    n = sum(1 for f in os.listdir(img_dir) if f.endswith(".jpg"))
    print(f"found {n} of {N_CELEBA} CelebA images")
    if n < N_CELEBA:
        raise RuntimeError(
            f"Only {n} of {N_CELEBA} images are in {img_dir}.\n"
            "The unzip probably did not finish (or the folder was moved while it was still extracting).\n"
            "Re-extract img_align_celeba from the Kaggle zip, wait until it is completely done, then run this again.")


def write_sets(sets, bad):
    """Write one CSV per item, leaving out any image that could not be read or saved."""
    for item, rows in sets.items():
        with open(os.path.join(SETS_DIR, f"celeba_{item}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "label", "split"])
            w.writerows(r for r in rows if r[0] not in bad)


def main():
    attrs, lmks, parts, img_dir = load_celeba()
    check_images(img_dir)
    rng = np.random.default_rng(SEED)
    sets = pick_sets(attrs, parts, rng)

    os.makedirs(ALIGNED_DIR, exist_ok=True)
    os.makedirs(SETS_DIR, exist_ok=True)

    yunet = load_yunet()
    all_ids = sorted({r[0] for rows in sets.values() for r in rows})
    eye_rows, errors, n_yunet, bad = [], [], 0, set()
    for k, image_id in enumerate(all_ids, 1):
        img = cv2.imread(os.path.join(img_dir, image_id))
        if img is None:
            bad.add(image_id)
            continue
        ref_l, ref_r = celeba_eyes(lmks[image_id])
        det = detect_yunet(yunet, img)
        if det is not None:
            n_yunet += 1
            left, right, method = det["left_eye"], det["right_eye"], "yunet"
            spacing = np.hypot(ref_r[0] - ref_l[0], ref_r[1] - ref_l[1])
            err = max(np.hypot(left[0] - ref_l[0], left[1] - ref_l[1]),
                      np.hypot(right[0] - ref_r[0], right[1] - ref_r[1])) / spacing
            errors.append(err)
        else:
            left, right, method, err = ref_l, ref_r, "celeba", float("nan")
        eye_rows.append([image_id, method, round(err, 4)])

        out_path = os.path.join(ALIGNED_DIR, image_id.replace(".jpg", ".png"))
        if REDO_ALIGNED or not os.path.exists(out_path):
            if not cv2.imwrite(out_path, align_face(img, left, right)):
                bad.add(image_id)
                continue
        if k % 500 == 0:
            print(f"  {k}/{len(all_ids)} images aligned")

    write_sets(sets, bad)
    if bad:
        print(f"  {len(bad)} images could not be read or saved and were left out of the sets")

    with open(os.path.join(SETS_DIR, "celeba_eyes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "method", "eye_error_in_eye_spacings"])
        w.writerows(eye_rows)

    errors = np.array(errors)
    print(f"\nYuNet found the face in {n_yunet}/{len(eye_rows)} images "
          f"({n_yunet / max(1, len(eye_rows)):.1%}); CelebA landmarks used for the rest")
    if len(errors):
        print(f"YuNet eye error vs CelebA labels (fraction of eye spacing): "
              f"median {np.median(errors):.3f}, 95th percentile {np.percentile(errors, 95):.3f}, "
              f"max {errors.max():.3f}")
    example_figure(sets)
    print("Sets saved in:", SETS_DIR)
    print("Aligned faces saved in:", ALIGNED_DIR)


def example_figure(sets, per_row=6):
    """One row each of: glasses, no glasses, hat, no hat (aligned, with the crop boxes)."""
    rng = np.random.default_rng(1)
    rows = [("glasses", 1, "glasses"), ("glasses", 0, "no glasses"), ("hat", 1, "hat"), ("hat", 0, "no hat")]
    fig, axes = plt.subplots(len(rows), per_row, figsize=(1.6 * per_row, 2.2 * len(rows)), squeeze=False)
    for r, (item, label, title) in enumerate(rows):
        ids = [i for i, l, s in sets[item] if l == label and s == "dev"]
        picks = rng.choice(ids, min(per_row, len(ids)), replace=False)
        for c, ax in enumerate(axes[r]):
            ax.axis("off")
            if c < len(picks):
                img = cv2.imread(os.path.join(ALIGNED_DIR, picks[c].replace(".jpg", ".png")))
                if img is not None:
                    ax.imshow(cv2.cvtColor(draw_regions(img), cv2.COLOR_BGR2RGB))
            if c == 0:
                ax.set_title(title, fontsize=8, loc="left")
    fig.suptitle("CelebA after YuNet alignment (green: eye band for glasses, blue: hat region)", fontsize=9)
    fig.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, "02_celeba_examples.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print("Example figure saved:", path)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
