# Self-Aligning Face Scanner with 3D Liveness Check and ID Matching

AMME5710 Computer Vision and Image Processing major project, University of Sydney, 2026.
Team F: Shun Ying Wong, Michael Delano and Devagya Budhiraja.

A small airport e-gate built around one USB webcam on the carriage of an upright XY plotter.
The plotter only moves the camera, and every decision comes from images:

| Step | What happens | Code |
|---|---|---|
| 1. Scan ID | find the printed card, flatten it with a homography, read the 4-digit number (HOG + SVM), look up the record | `id_card.py`, `digits.py` |
| 2. Position | move the camera until the traveller's nose is centred (YuNet landmarks, calibrated image-to-rig mapping) | `face_follow.py`, `rig_grid.py` |
| 3. Glasses / hat | eye alignment, HOG + HSV features, SVM; ask the traveller to take them off | `face_align.py`, `accessories.py` |
| 4. Live check | camera stops 20 mm either side of centre; SIFT matches, depth from disparity, tests on the face and its surroundings | `face_liveness.py`, `stereo_face_liveness.py` |
| 5. Face match | live face vs card photo and vs record photo, threshold set on the gate camera | `id_match.py` |
| Alert | STOP screen, chime, officer view, one-to-many search | `id_alert.py` |

`main.py` runs all of these in order.

## Files

| File | Purpose |
|---|---|
| `main.py` | the full gate on the rig |
| `run_no_rig.py` | the full gate on a plain webcam (head-turn live check instead of stereo) |
| `rig_grid.py` | GRBL serial control of the plotter |
| `face_follow.py` | rig calibration and centring on the nose |
| `face_liveness.py` | stereo pairs on the rig and the live / not-live decision |
| `stereo_face_liveness.py` | 3D points and shape tests for one stereo pair |
| `face_detect.py` | YuNet face and landmark detection |
| `face_align.py` | eye alignment and crop regions |
| `accessories.py` | glasses and hat classifiers (training and live use) |
| `prep_celeba.py` | glasses / hat training sets from CelebA |
| `id_card.py` | card detection, flattening, sharpness, orientation and a standalone card check |
| `digits.py` | digit reader (training and live use) |
| `make_id_cards.py` | printable mock ID cards and the records file |
| `id_match.py` | face features, Eigenfaces and the LFW benchmark |
| `calibrate_ids.py` | sets the face match threshold on the gate camera |
| `id_alert.py` | STOP screen, officer view and one-to-many search |
| `capture_grid.py` | checkerboard images at rig positions for lens calibration |
| `find_cameras.py` | shows which camera number is which |
| `get_lfw.py` | downloads the LFW pairs |

`models/` holds the trained models and the OpenCV Zoo YuNet and SFace models, `results/` the
cross-validation table and card check log, and `figures_member3/` the benchmark plots.

## Setup

```bash
pip install -r requirements.txt
```

Set the serial port in `rig_grid.py` (`PORT`) and the rig webcam number in `face_liveness.py`
(`CAMERA_INDEX`). `python find_cameras.py` shows which number is which.

Before the first run:

1. Put one front-on photo per person in `data/card_photos/` as `Name_1234.jpeg`, then run
   `python make_id_cards.py`. Print `data/cards/print_sheet_1.pdf` at 100% scale.
2. Optional but recommended: `python calibrate_ids.py` sets the match threshold on your camera,
   since a threshold from LFW does not carry over to front-on photos.

## Running

```bash
python main.py        # on the rig, carriage in the bottom-left corner
python run_no_rig.py  # webcam only
```

Keys: SPACE start / next traveller, `c` calibrate the tracking, `n` new traveller, `d` operator
details, `v` voice, `r` turn the view, `f` mirror, `w` full screen, `q` quit.

## Retraining

| Model | Command |
|---|---|
| Glasses / hat | CelebA in `data/celeba/`, then `python prep_celeba.py` and `python accessories.py` |
| Digits | `python digits.py` (rendered from the card font, no download) |
| Eigenfaces and LFW benchmark | `python get_lfw.py`, set `RUN_EIGEN_TRAINING` / `RUN_LFW` in `id_match.py`, then `python id_match.py` |

## Privacy

No photos of people are included. Card photos, record photos, calibration frames and the
session database (`session_db/`) stay on your own computer and are ignored by git
(see `data/README.md`).
