"""
make_id_cards.py  -  make printable mock ID cards for the demo

Put one photo per person in data/card_photos/, named  Name_1234.jpg
(Name = what goes on the card, 1234 = their 4-digit ID number). Any clear,
front-on photo works, e.g. a phone selfie against a plain wall. For a doctored "fake" card, add a photo of SOMEONE ELSE named
with the real person's name and number plus _fake, e.g. Shirley_1234_fake.jpg.

Running this file:
  * finds the face in each photo (YuNet), lines it up on the eyes and crops a
    passport-style photo, so every card photo is framed the same way
  * draws each card at true size (85.6 x 54 mm): white margin, thick black
    frame, photo on the left, name, ID number in boxes, "not a real ID" note
  * saves each card to data/cards/<file>.png and every card on A4 sheets,
    data/cards/print_sheet_1.pdf (print at 100% / "actual size", not "fit to page")
  * writes data/cards/records.csv (ID number, name, photo) from the genuine
    photos only: this is "our data" the system looks people up in
  * copies each genuine photo to data/ids/Name.jpg: these are the record photos

Print on matte paper if you can (less glare), cut along the light grey line,
and glue onto cardboard so the card stays flat.
"""

import os
import re
import csv
import shutil
from datetime import date

import numpy as np
import cv2
import matplotlib
from PIL import Image, ImageDraw, ImageFont

import face_detect as fd
from face_align import align_face
import id_card as card

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(BASE_DIR, "data", "card_photos")
OUT_DIR = os.path.join(BASE_DIR, "data", "cards")
IDS_DIR = os.path.join(BASE_DIR, "data", "ids")

DPI = 300
PX_MM = DPI / 25.4                       # pixels per millimetre when printed
PASSPORT = {"canvas": (240, 320), "eye_mid": (120.0, 140.0), "d0": 70.0}   # 3:4 crop, eyes 44% down
NAME_RE = re.compile(r"^(?P<name>.+?)_(?P<id>\d{4})(?P<fake>_fake)?$")
FONT = cv2.FONT_HERSHEY_SIMPLEX          # only used for the note on the A4 sheet
VALID_YEARS = 5

# card colours (BGR)
NAVY = (82, 42, 14)
TEAL = (140, 112, 0)
INK = (60, 35, 20)
LABEL = (140, 120, 105)
TINT = (250, 244, 236)
WAVE = (232, 218, 200)
BOX_EDGE = (190, 165, 140)

# fonts: the DejaVu fonts that come with matplotlib, so the card looks the same on every computer
_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
FONT_BOLD = os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")
FONT_REG = os.path.join(_FONT_DIR, "DejaVuSans.ttf")
FONT_ITALIC = os.path.join(_FONT_DIR, "DejaVuSerif-Italic.ttf")
CAP = 0.73                               # capital letter height as a fraction of the font size


def mm(v):
    return int(round(v * PX_MM))


def passport_photo(img, yunet):
    """Face found, levelled and framed passport-style (240 x 320), or None if no face."""
    det = fd.detect_photo(yunet, img)
    if det is None:
        return None
    return align_face(img, det["left_eye"], det["right_eye"], canvas=PASSPORT["canvas"],
                      eye_mid=PASSPORT["eye_mid"], d0=PASSPORT["d0"], border=cv2.BORDER_REPLICATE)


def rounded_mask(w, h, r):
    """White rectangle with rounded corners of radius r on black, w x h pixels."""
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def paste_box(img, small, box_mm, mask=None):
    """Resize small into a box given in mm and copy it in (through a mask if given)."""
    x0, y0, x1, y1 = (mm(v) for v in box_mm)
    small = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    if mask is None:
        img[y0:y1, x0:x1] = small
    else:
        roi = img[y0:y1, x0:x1]
        roi[mask > 0] = small[mask > 0]


def guilloche(img, box_mm):
    """Fine wavy lines across the panel, like the background print on real cards."""
    x0, y0, x1, y1 = (mm(v) for v in box_mm)
    xs = np.arange(x0, x1)
    for k in range(26):
        for amp, period, phase in ((mm(2.2), mm(18), 0.0), (mm(1.6), mm(11), 1.3)):
            ys = y0 + mm(1.0) + k * mm(1.6) + amp * np.sin(2 * np.pi * (xs - x0) / period + phase + 0.45 * k)
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            pts = pts[(pts[:, 1] >= y0) & (pts[:, 1] < y1)]
            if len(pts) > 1:
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, WAVE, 1, cv2.LINE_AA)


def emblem(img, cx_mm, cy_mm, r_mm):
    """Simple lens-shaped badge for the header (rings and a crosshair)."""
    c, r = (mm(cx_mm), mm(cy_mm)), mm(r_mm)
    cv2.circle(img, c, r, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(img, c, int(r * 0.62), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, c, int(r * 0.25), (255, 255, 255), -1, cv2.LINE_AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img, (c[0] + int(dx * r * 0.62), c[1] + int(dy * r * 0.62)),
                 (c[0] + int(dx * r * 1.25), c[1] + int(dy * r * 1.25)), (255, 255, 255), 2, cv2.LINE_AA)


class Writer:
    """Puts TrueType text on the card (OpenCV's own fonts look too plain for this)."""

    def __init__(self, img_bgr):
        self.im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.im)

    def text(self, txt, x_mm, y_mm, cap_mm, path=FONT_REG, colour=INK, max_w_mm=None, anchor="ls"):
        size = mm(cap_mm) / CAP
        font = ImageFont.truetype(path, int(round(size)))
        if max_w_mm is not None and font.getlength(txt) > mm(max_w_mm):
            font = ImageFont.truetype(path, int(size * mm(max_w_mm) / font.getlength(txt)))
        self.draw.text((mm(x_mm), mm(y_mm)), txt, font=font, fill=colour[::-1], anchor=anchor)

    def result(self):
        return cv2.cvtColor(np.array(self.im), cv2.COLOR_RGB2BGR)


def draw_card(photo, name, id_number, issued=None):
    """The card at 300 dpi (1011 x 638 px), laid out with the sizes in id_card.py."""
    issued = issued or date.today()
    expires = issued.replace(year=issued.year + VALID_YEARS)
    W, H = mm(card.CARD_W_MM), mm(card.CARD_H_MM)
    img = np.full((H, W, 3), 255, np.uint8)

    # dark frame with a plain white strip inside it: this is what id_card.py finds
    m, f = card.MARGIN_MM, card.FRAME_MM
    cv2.rectangle(img, (mm(m), mm(m)), (mm(card.CARD_W_MM - m), mm(card.CARD_H_MM - m)), NAVY, -1)
    cv2.rectangle(img, (mm(m + f), mm(m + f)), (mm(card.CARD_W_MM - m - f), mm(card.CARD_H_MM - m - f)),
                  (255, 255, 255), -1)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (200, 200, 200), 1)          # light grey cutting line

    # tinted panel with wavy background lines
    px0, py0, px1, py1 = card.PANEL_MM
    cv2.rectangle(img, (mm(px0), mm(py0)), (mm(px1), mm(py1)), TINT, -1)
    guilloche(img, card.PANEL_MM)

    # header band: navy fading to teal, with the badge
    hx0, hy0, hx1, hy1 = (mm(v) for v in card.HEADER_MM)
    t = np.linspace(0, 1, hx1 - hx0)[None, :, None]
    img[hy0:hy1, hx0:hx1] = (np.array(NAVY) * (1 - t) + np.array(TEAL) * t).astype(np.uint8)
    emblem(img, px0 + 3.6, (card.HEADER_MM[1] + card.HEADER_MM[3]) / 2, 1.9)

    # photo with rounded corners and a thin border, and the faded "ghost" copy
    x0, y0, x1, y1 = card.PHOTO_MM
    pw, ph = mm(x1) - mm(x0), mm(y1) - mm(y0)
    cv2.rectangle(img, (mm(x0) - 3, mm(y0) - 3), (mm(x1) + 2, mm(y1) + 2), BOX_EDGE, -1)
    cv2.rectangle(img, (mm(x0) - 1, mm(y0) - 1), (mm(x1), mm(y1)), TINT, -1)
    paste_box(img, photo, card.PHOTO_MM, rounded_mask(pw, ph, mm(1.0)))
    grey = cv2.cvtColor(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32)
    gx0, gy0, gx1, gy1 = (mm(v) for v in card.GHOST_MM)
    under = img[gy0:gy1, gx0:gx1].astype(np.float32)
    ghost = cv2.resize(grey, (gx1 - gx0, gy1 - gy0), interpolation=cv2.INTER_AREA)
    img[gy0:gy1, gx0:gx1] = (under * (0.65 + 0.35 * ghost / 255)).astype(np.uint8)

    # ID number boxes (white, so the digits stand out for reading them later)
    dx, dy, bw, bh, gap = card.DIGITS_MM
    for k in range(card.N_DIGITS):
        bx = dx + k * (bw + gap)
        cv2.rectangle(img, (mm(bx), mm(dy)), (mm(bx + bw), mm(dy + bh)), (255, 255, 255), -1)
        cv2.rectangle(img, (mm(bx), mm(dy)), (mm(bx + bw), mm(dy + bh)), BOX_EDGE, 2)

    # all the text
    wr = Writer(img)
    wr.text("E-GATE TRAVEL ID", px0 + 7.2, card.HEADER_MM[1] + 3.5, 2.2, FONT_BOLD, (255, 255, 255))
    wr.text("AMME5710 COMPUTER VISION DEMO  •  NOT A REAL ID", px0 + 7.2, card.HEADER_MM[1] + 5.6, 1.05,
            FONT_REG, (255, 255, 255))
    nx, ny_label, ny = card.NAME_MM
    wr.text("NAME", nx, ny_label, 1.05, FONT_REG, LABEL)
    wr.text(name.replace("_", " ").upper(), nx, ny, 2.6, FONT_BOLD, INK, max_w_mm=card.PANEL_MM[2] - nx - 1.5)
    wr.text("ID NUMBER", nx, dy - 1.0, 1.05, FONT_REG, LABEL)
    for k, ch in enumerate(id_number):
        bx = dx + k * (bw + gap)
        wr.text(ch, bx + bw / 2, dy + bh / 2, bh * 0.62, FONT_BOLD, INK, anchor="mm")
    ix, ex, ly, vy = card.DATES_MM
    wr.text("ISSUED", ix, ly, 1.05, FONT_REG, LABEL)
    wr.text(issued.strftime("%d.%m.%Y"), ix, vy, 1.6, FONT_BOLD, INK)
    wr.text("EXPIRES", ex, ly, 1.05, FONT_REG, LABEL)
    wr.text(expires.strftime("%d.%m.%Y"), ex, vy, 1.6, FONT_BOLD, INK)
    sx, sy, sx1 = card.SIGN_MM
    wr.text("SIGNATURE", sx, ly, 1.05, FONT_REG, LABEL)
    wr.text(name.replace("_", " ").title(), sx, sy, 2.2, FONT_ITALIC, (120, 60, 20), max_w_mm=sx1 - sx)
    return wr.result()


def print_sheets(card_files):
    """Lay the cards out at true size on A4 pages (2 across, 5 down) and save as PDF."""
    page_w, page_h = mm(210), mm(297)
    cw, ch = mm(card.CARD_W_MM), mm(card.CARD_H_MM)
    gap, left, top = mm(8), mm(15), mm(15)
    per_page, pages = 10, []
    for start in range(0, len(card_files), per_page):
        page = np.full((page_h, page_w, 3), 255, np.uint8)
        for k, path in enumerate(card_files[start:start + per_page]):
            r, c = divmod(k, 2)
            x, y = left + c * (cw + gap), top + r * (ch + gap)
            page[y:y + ch, x:x + cw] = cv2.imread(path)
        cv2.putText(page, "Print at 100% (actual size). Each card should measure 85.6 x 54 mm.",
                    (left, page_h - mm(8)), FONT, 1.2, (120, 120, 120), 2, cv2.LINE_AA)
        pages.append(Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB)))
    paths = []
    for i, pg in enumerate(pages, 1):
        p = os.path.join(OUT_DIR, f"print_sheet_{i}.pdf")
        pg.save(p, resolution=DPI)
        paths.append(p)
    return paths


def main():
    os.makedirs(PHOTO_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IDS_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(PHOTO_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        print(f"No photos yet. Put them in {PHOTO_DIR} named like Shirley_1234.jpg and run this again.")
        return
    yunet = fd.load_yunet()
    records, made = [], []
    for f in files:
        m = NAME_RE.match(os.path.splitext(f)[0])
        if m is None:
            print(f"  skipped {f}: name it like Shirley_1234.jpg (or Shirley_1234_fake.jpg)")
            continue
        img = cv2.imread(os.path.join(PHOTO_DIR, f))
        photo = passport_photo(img, yunet) if img is not None else None
        if photo is None:
            print(f"  skipped {f}: no face found")
            continue
        name, id_number, fake = m["name"], m["id"], bool(m["fake"])
        out = os.path.join(OUT_DIR, os.path.splitext(f)[0] + ".png")
        cv2.imwrite(out, draw_card(photo, name, id_number))
        made.append(out)
        if not fake:
            records.append({"id_number": id_number, "name": name, "photo": f})
            shutil.copy(os.path.join(PHOTO_DIR, f), os.path.join(IDS_DIR, name + os.path.splitext(f)[1].lower()))
        print(f"  card made: {os.path.basename(out)}{'  (FAKE: photo of someone else)' if fake else ''}")

    ids = [r["id_number"] for r in records]
    if len(ids) != len(set(ids)):
        print("  WARNING: two genuine photos share an ID number")
    with open(os.path.join(OUT_DIR, "records.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id_number", "name", "photo"])
        w.writeheader()
        w.writerows(records)
    sheets = print_sheets(made)
    print(f"\n{len(made)} cards made ({len(records)} genuine). Records: {os.path.join(OUT_DIR, 'records.csv')}")
    for p in sheets:
        print("Print this at 100% scale:", p)
    print("Genuine photos copied to data/ids/ (the record photos).")


if __name__ == "__main__":
    main()
