# data

Nothing in this folder is pushed, because it holds photos of people and large datasets.
The code expects:

| Folder | What goes in it | Made by |
|---|---|---|
| `card_photos/` | one front-on photo per person, `Name_1234.jpeg` (add `_fake` for a doctored card) | you |
| `cards/` | card images, print sheet and `records.csv` | `make_id_cards.py` |
| `ids/` | record photos, `Name.jpeg` | `make_id_cards.py` |
| `calib/` | gate-camera frames for the match threshold, `Name_01.jpg` ... | `calibrate_ids.py` |
| `celeba/` | CelebA images and attribute files | download |
| `sets/`, `aligned/` | CelebA training sets and aligned faces | `prep_celeba.py` |
| `ffhq_thumbs/` | FFHQ thumbnails for the Eigenfaces | download |
| `lfw/` | LFW pairs | `get_lfw.py` |
