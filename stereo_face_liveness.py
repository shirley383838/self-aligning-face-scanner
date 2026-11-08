#!/usr/bin/env python3
"""
Real 3D face or flat picture, from two images taken a known baseline apart.

Detects the face, matches SIFT features (ORB fallback), triangulates them into a sparse
3D point cloud and runs three tests:
  A. plane fit: relief of the points about their best-fit plane, relative to face size
  B. homography parallax: residual after the best single homography between the views
  C. smooth surface: residual after the best quadratic surface (catches bent photos)

Usage:
  python stereo_face_liveness.py bottom.jpg top.jpg --baseline 50 --focal-px 1400 --out result
  python stereo_face_liveness.py --selftest
--mode rectified (default) assumes the camera only moved vertically; --mode pose
estimates the rotation from the essential matrix (unstable for flat scenes). Without
--focal-px the focal length is taken as 0.9 x the image width.
"""

import argparse
import sys

import numpy as np
import cv2

# ---------------------------------------------------------------- decision thresholds (tune on your own data)
FLATNESS_THRESH = 0.03      # plane-residual RMS / face width; photo ~0.01, face ~0.05-0.1
PARALLAX_THRESH_PX = 2.0    # 90th pct homography residual in px; photo ~<1, face > 3
QUAD_RESID_THRESH = 0.018  # residual RMS after a quadratic / face width; bent photo ~0.01, face ~0.03
TYPICAL_FACE_RELIEF_MM = 25.0  # used only to warn when the baseline is too small


class StereoError(Exception):
    """Raised when a pair can't be analysed (too few features / matches)."""


# ---------------------------------------------------------------- 1. Face detection and feature matching
def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read image: {path}")
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_face(gray):
    """Return the largest face box (x, y, w, h) or None."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(60, 60))
    if len(faces) == 0:
        return None
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def face_mask(shape, box, pad=0.10):
    mask = np.zeros(shape[:2], np.uint8)
    if box is None:
        mask[:] = 255
        return mask
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    mask[max(0, y - py):y + h + py, max(0, x - px):x + w + px] = 255
    return mask


def make_detector():
    if hasattr(cv2, "SIFT_create"):
        # Low contrast threshold: skin is weakly textured, we want many points.
        return cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.01,
                               edgeThreshold=15), cv2.NORM_L2, "SIFT"
    return cv2.ORB_create(nfeatures=6000, fastThreshold=5), cv2.NORM_HAMMING, "ORB"


def extract_and_match(g1, g2, m1, m2, ratio=0.75):
    """Detect features inside the face masks and return matched pixel coords."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g1, g2 = clahe.apply(g1), clahe.apply(g2)
    det, norm, name = make_detector()
    k1, d1 = det.detectAndCompute(g1, m1)
    k2, d2 = det.detectAndCompute(g2, m2)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        raise StereoError("too few features found on the face")

    bf = cv2.BFMatcher(norm)
    fwd = bf.knnMatch(d1, d2, k=2)
    bwd = {m.queryIdx: m.trainIdx for m in bf.match(d2, d1)}  # for mutual check

    good = []
    for pair in fwd:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance and bwd.get(m.trainIdx) == m.queryIdx:
            good.append(m)

    p1 = np.float64([k1[m.queryIdx].pt for m in good])
    p2 = np.float64([k2[m.trainIdx].pt for m in good])
    return p1, p2, good, k1, k2, name


# ---------------------------------------------------------------- 2. Stereo triangulation
def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-9


def triangulate_vertical(p1, p2, f, cx, cy, baseline, direction=(0.0, 1.0)):
    """
    Translating stereo rig: camera moved by `baseline` along `direction` in the image (default
    straight up / down). Z = f * B / d; points are in camera-1 coordinates.
    """
    u = np.asarray(direction, float)
    u = u / (np.linalg.norm(u) + 1e-12)
    flow = p2 - p1
    dy = flow @ u                               # disparity along the baseline
    dx = flow @ np.array([-u[1], u[0]])         # across it: should be ~constant
    flipped = np.median(dy) < 0          # image order (top/bottom) doesn't matter
    if flipped:
        dy = -dy

    # Outlier rejection: horizontal disparity should be ~constant (ideally 0),
    # vertical disparity positive and not wildly off the median.
    keep = (np.abs(dx - np.median(dx)) < max(2.0, 4 * mad(dx))) & (dy > 0.5)
    if keep.sum() < 15:
        raise StereoError(f"only {int(keep.sum())} consistent matches")
    keep &= np.abs(dy - np.median(dy[keep])) < 6 * mad(dy[keep])

    Z = f * baseline / dy[keep]
    X = (p1[keep, 0] - cx) * Z / f
    Y = (p1[keep, 1] - cy) * Z / f
    info = {"median_dx_px": float(np.median(dx)), "median_dy_px": float(np.median(dy)),
            "image_order_flipped": bool(flipped)}
    return np.column_stack([X, Y, Z]), keep, info


def triangulate_pose(p1, p2, K, baseline):
    """General two-view triangulation; translation scaled to the known baseline."""
    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3):
        raise StereoError("essential matrix estimation failed (scene may be planar)")
    _, R, t, mask = cv2.recoverPose(E, p1, p2, K, mask=mask)
    keep = mask.ravel() > 0
    t = t.ravel() / np.linalg.norm(t) * baseline
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    Xh = cv2.triangulatePoints(P1, P2, p1[keep].T, p2[keep].T)
    P = (Xh[:3] / Xh[3]).T
    front = P[:, 2] > 0
    idx = np.flatnonzero(keep)
    keep[:] = False
    keep[idx[front]] = True
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return P[front], keep, {"rotation_deg": float(angle), "t_direction": t / baseline}


# ---------------------------------------------------------------- 3. Coplanarity tests
def fit_plane(P, iters=3):
    """Least-squares plane with iterative outlier trimming. Returns (n, c, inliers)."""
    inl = np.ones(len(P), bool)
    for _ in range(iters):
        c = P[inl].mean(0)
        _, _, Vt = np.linalg.svd(P[inl] - c)
        n = Vt[2]
        d = (P - c) @ n
        inl = np.abs(d) < max(3 * mad(d[inl]), 1e-6) * 2.5   # generous: keep the nose!
    return n, c, inl, Vt


def plane_test(P):
    n, c, inl, Vt = fit_plane(P)
    Q = (P[inl] - c) @ Vt.T                 # coords in the plane frame (u, v, height)
    h = Q[:, 2]
    width = np.percentile(Q[:, 0], 95) - np.percentile(Q[:, 0], 5)
    height_extent = np.percentile(Q[:, 1], 95) - np.percentile(Q[:, 1], 5)
    size = max(width, height_extent, 1e-9)
    rms = float(np.sqrt(np.mean(h ** 2)))
    relief = float(np.percentile(h, 97) - np.percentile(h, 3))

    # Informational: how much of the relief a smooth quadratic (bent paper,
    # curved screen) explains. A real face keeps residual structure (nose).
    u, v = Q[:, 0] / size, Q[:, 1] / size
    A = np.column_stack([u * u, v * v, u * v, u, v, np.ones_like(u)])
    coef, *_ = np.linalg.lstsq(A, h, rcond=None)
    rms_quad = float(np.sqrt(np.mean((h - A @ coef) ** 2)))

    return {"normal": n, "centroid": c, "inliers": inl, "plane_frame": Vt,
            "rms": rms, "relief": relief, "face_size": float(size),
            "flatness": rms / size, "rms_after_quadratic": rms_quad,
            "quad_flatness": rms_quad / size,
            "n_points": int(inl.sum())}


def homography_test(p1, p2):
    """Residual parallax after the best single homography (LMedS)."""
    H, _ = cv2.findHomography(p1, p2, cv2.LMEDS)
    if H is None:
        return {"p90_px": float("inf"), "median_px": float("inf")}
    proj = cv2.perspectiveTransform(p1.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - p2, axis=1)
    return {"H": H, "errors": err, "median_px": float(np.median(err)),
            "p90_px": float(np.percentile(err, 90))}


VERDICTS = {"LIVE": "REAL 3D FACE",
            "CURVED": "CURVED 2D (bent photo / curved screen: smooth surface, no facial relief)",
            "FLAT": "FLAT (2D image of a face)",
            "UNSURE": "UNCERTAIN (tests disagree)"}


def classify(plane, homog):
    """Short verdict code: LIVE, CURVED, FLAT or UNSURE."""
    a = plane["flatness"] > FLATNESS_THRESH
    b = homog["p90_px"] > PARALLAX_THRESH_PX
    c = plane["quad_flatness"] > QUAD_RESID_THRESH
    if a and b:
        return "LIVE" if c else "CURVED"
    if not a and not b:
        return "FLAT"
    return "UNSURE"


def decide(plane, homog):
    return VERDICTS[classify(plane, homog)]


def analyse(p1, p2, f, cx, cy, baseline, mode="rectified", verbose=True, direction=(0.0, 1.0)):
    if len(p1) < 15:
        raise StereoError(f"only {len(p1)} matches - not enough to analyse")
    if mode == "rectified":
        P, keep, info = triangulate_vertical(p1, p2, f, cx, cy, baseline, direction)
    else:
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
        P, keep, info = triangulate_pose(p1, p2, K, baseline)

    plane = plane_test(P)
    homog = homography_test(p1[keep], p2[keep])
    code = classify(plane, homog)
    verdict = VERDICTS[code]

    # Can this baseline resolve face relief at all?
    Zm = float(np.median(P[:, 2]))
    exp_parallax = f * baseline * TYPICAL_FACE_RELIEF_MM / (Zm * (Zm + TYPICAL_FACE_RELIEF_MM))
    depth_noise = Zm ** 2 / (f * baseline) * 0.5   # for ~0.5 px matching noise

    res = dict(points=P, keep=keep, info=info, plane=plane, homog=homog,
               verdict=verdict, code=code, median_depth=Zm,
               expected_face_parallax_px=float(exp_parallax),
               depth_noise=float(depth_noise))
    if verbose:
        report(res, mode)
    return res


def report(r, mode):
    pl, hg = r["plane"], r["homog"]
    print(f"\n=== Stereo face analysis ({mode}) ===")
    for k, v in r["info"].items():
        print(f"  {k}: {np.round(v, 3) if isinstance(v, (float, np.ndarray)) else v}")
    print(f"  3D points used           : {pl['n_points']}")
    print(f"  median depth             : {r['median_depth']:.1f}")
    print(f"  approx. depth noise (1σ) : {r['depth_noise']:.2f}  (same unit as baseline)")
    print(f"  face size in plane       : {pl['face_size']:.1f}")
    print(f"  plane residual RMS       : {pl['rms']:.2f}")
    print(f"  relief (3-97 pct)        : {pl['relief']:.2f}")
    print(f"  [A] flatness (RMS/size)  : {pl['flatness']:.4f}  (3D if > {FLATNESS_THRESH})")
    print(f"  [B] homography parallax  : median {hg['median_px']:.2f} px, "
          f"p90 {hg['p90_px']:.2f} px  (3D if p90 > {PARALLAX_THRESH_PX})")
    print(f"  [C] RMS after quadratic  : {pl['rms_after_quadratic']:.2f}  "
          f"(/size = {pl['quad_flatness']:.4f}; 3D if > {QUAD_RESID_THRESH})")
    print(f"  expected parallax of a real face at this range: ~{r['expected_face_parallax_px']:.1f} px")
    if r["expected_face_parallax_px"] < 2 * PARALLAX_THRESH_PX:
        print("  WARNING: baseline is small for this distance - real faces may look flat. "
              "Increase the baseline or move closer.")
    if r["depth_noise"] > TYPICAL_FACE_RELIEF_MM / 4:
        print("  WARNING: depth noise is a large fraction of typical face relief "
              "(assuming mm units).")
    print(f"\n  VERDICT: {r['verdict']}\n")


# ---------------------------------------------------------------- 4. Coarse surface + outputs
def coarse_surface(P, grid=60):
    """Interpolate the sparse cloud to a regular Z(X, Y) grid."""
    try:
        from scipy.interpolate import griddata
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    # drop points whose depth disagrees with their neighbours (mismatches)
    _, nn = cKDTree(P[:, :2]).query(P[:, :2], k=min(9, len(P)))
    local = np.median(P[nn, 2], axis=1)
    dev = P[:, 2] - local
    P = P[np.abs(dev) < 4 * mad(dev)]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    gx, gy = np.meshgrid(np.linspace(np.percentile(x, 2), np.percentile(x, 98), grid),
                         np.linspace(np.percentile(y, 2), np.percentile(y, 98), grid))
    gz = griddata((x, y), z, (gx, gy), method="linear")
    return gx, gy, gz


def save_ply(path, P):
    with open(path, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(P)}\n"
                 "property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(fh, P, fmt="%.4f")


def save_plots(prefix, r, title=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return
    P = r["points"][r["plane"]["inliers"]]
    Q = (P - r["plane"]["centroid"]) @ r["plane"]["plane_frame"].T
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(131, projection="3d")
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=Q[:, 2], cmap="coolwarm", s=6)
    ax.set_title("Triangulated points\n(colour = distance from best plane)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.invert_zaxis(); ax.invert_yaxis()
    fig.colorbar(sc, ax=ax, shrink=0.6)

    ax = fig.add_subplot(132, projection="3d")
    surf = coarse_surface(P)
    if surf is not None:
        gx, gy, gz = surf
        ax.plot_surface(gx, gy, gz, cmap="viridis", linewidth=0, antialiased=True)
        ax.invert_zaxis(); ax.invert_yaxis()
        ax.set_title("Coarse surface Z(X, Y)")
    else:
        ax.set_title("install scipy for the surface")

    ax = fig.add_subplot(133)
    ax.hist(Q[:, 2], bins=40, color="steelblue")
    ax.set_title(f"Height above plane\nflatness={r['plane']['flatness']:.3f}, "
                 f"parallax p90={r['homog']['p90_px']:.1f}px")
    fig.suptitle(f"{title}  →  {r['verdict']}")
    fig.tight_layout()
    fig.savefig(f"{prefix}_surface.png", dpi=110)
    plt.close(fig)
    print(f"saved {prefix}_surface.png")


# ---------------------------------------------------------------- synthetic self-test
def synthetic_case(kind, f=1400.0, cx=960.0, cy=540.0, B=50.0, Z0=450.0,
                   noise_px=0.4, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    # sample points over an elliptical face region (mm)
    X = rng.uniform(-70, 70, n * 2); Y = rng.uniform(-95, 95, n * 2)
    m = (X / 70) ** 2 + (Y / 95) ** 2 < 1
    X, Y = X[m][:n], Y[m][:n]
    if kind == "face":
        Z = (Z0 + 0.006 * X ** 2 + 0.002 * Y ** 2                       # head curvature
             - 28 * np.exp(-(X / 14) ** 2 - ((Y - 5) / 28) ** 2)        # nose
             + 8 * np.exp(-((np.abs(X) - 32) / 13) ** 2 - ((Y + 25) / 10) ** 2)  # eye sockets
             - 6 * np.exp(-(X / 30) ** 2 - ((Y + 45) / 8) ** 2))       # brow
    elif kind == "photo":                       # flat print, tilted 25° about y
        Z = Z0 + np.tan(np.radians(25)) * X
    elif kind == "bent_photo":                  # photo curled around a cylinder
        Z = Z0 + 0.006 * X ** 2
    else:
        raise ValueError(kind)
    p1 = np.column_stack([f * X / Z + cx, f * Y / Z + cy])
    p2 = np.column_stack([f * X / Z + cx, f * (Y + B) / Z + cy])   # camera 2 above
    p1 += rng.normal(0, noise_px, p1.shape)
    p2 += rng.normal(0, noise_px, p2.shape)
    # a few gross mismatches, like a real matcher produces
    k = n // 30
    p2[:k] += rng.uniform(-40, 40, (k, 2))
    return p1, p2, f, cx, cy, B


def selftest(out=None):
    results = {}
    for kind in ["face", "photo", "bent_photo"]:
        print(f"\n######## synthetic: {kind} ########")
        p1, p2, f, cx, cy, B = synthetic_case(kind)
        r = analyse(p1, p2, f, cx, cy, B, "rectified")
        results[kind] = r["verdict"]
        if out:
            save_plots(f"{out}_{kind}", r, title=f"synthetic {kind}")
    print("Summary:", results)
    return results


# ---------------------------------------------------------------- command line
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image1", nargs="?")
    ap.add_argument("image2", nargs="?")
    ap.add_argument("--baseline", type=float, help="vertical distance between shots (e.g. mm)")
    ap.add_argument("--focal-px", type=float, help="focal length in pixels")
    ap.add_argument("--cx", type=float); ap.add_argument("--cy", type=float)
    ap.add_argument("--mode", choices=["rectified", "pose"], default="rectified")
    ap.add_argument("--out", help="output prefix for plots / matches / .ply")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest(a.out)
        return
    if not (a.image1 and a.image2 and a.baseline):
        ap.error("image1, image2 and --baseline are required (or use --selftest)")

    img1, g1 = load_gray(a.image1)
    img2, g2 = load_gray(a.image2)
    if g1.shape != g2.shape:
        print("WARNING: images differ in size; intrinsics are assumed identical.")
    h, w = g1.shape
    f = a.focal_px or 0.9 * w
    cx = a.cx if a.cx is not None else w / 2
    cy = a.cy if a.cy is not None else h / 2
    if not a.focal_px:
        print(f"No --focal-px given; assuming f = {f:.0f}px (metric scale will be approximate).")

    b1, b2 = detect_face(g1), detect_face(g2)
    if b1 is None or b2 is None:
        print("WARNING: face not detected in one or both images; using the whole image.")
    try:
        p1, p2, matches, k1, k2, det = extract_and_match(
            g1, g2, face_mask(g1.shape, b1), face_mask(g2.shape, b2))
        print(f"{det}: {len(p1)} mutual ratio-test matches on the face")
        r = analyse(p1, p2, f, cx, cy, a.baseline, a.mode)
    except StereoError as e:
        sys.exit(f"Can't analyse these images: {e}")

    if a.out:
        kept = [m for m, k in zip(matches, r["keep"]) if k]
        vis = cv2.drawMatches(img1, k1, img2, k2, kept[:300], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f"{a.out}_matches.jpg", vis)
        save_ply(f"{a.out}_points.ply", r["points"])
        save_plots(a.out, r, title=f"{a.image1} / {a.image2}")
        print(f"saved {a.out}_matches.jpg, {a.out}_points.ply")


if __name__ == "__main__":
    main()
