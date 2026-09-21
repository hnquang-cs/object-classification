# Object Classification + Spec Lookup

Photograph a part, find out which known part it is, and get back the exact
dimensions (with tolerances) it's supposed to have — ready to compare
against a real measurement.

```
new photo -> which part is this? -> here is its spec sheet (nominal ± tolerance per dimension)
```

Classification is based on the **silhouette shape** of the part (area,
aspect ratio, holes, outline), not color or surface finish — so it works
even from a simple photo on a plain background. `UNKNOWN` is a valid,
intentional answer: if a photo doesn't clearly match one known part, the
tool refuses to guess rather than returning a wrong spec.

---

## 1. Setup

```bash
cd object-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check it installed correctly:

```bash
python -c "import cv2; print(cv2.__version__)"
```

Python 3.9+ works; Python 3.11+ is recommended.

---

## 2. Try it right now

The repo already ships with 6 real parts (photos + specs pulled from their
inspection sheets), so you can run the whole thing immediately:

```bash
python main.py
```

This will:
1. Load every part's reference photos, `spec.json`, and technical drawing
   from `assets/references/`.
2. Build a shape "fingerprint" for each part.
3. Classify every photo in `assets/tests/`.
4. Print an accuracy report and save detailed results to `outputs/`.

Then classify one specific photo and see its spec printed directly:

```bash
python predict.py assets/tests/98003P/images/98003P_1_rot23.png
```

That's the core loop: **`main.py` to build/evaluate the whole database,
`predict.py` to identify one new photo.**

---

## 3. Folder layout

```
object-classification/
├── assets/
│   ├── references/              <- the "known parts" database
│   │   └── <part number>/
│   │       ├── images/*.png     sample photos (used to build the fingerprint)
│   │       ├── spec.json        dimensions + tolerances (see below)
│   │       └── drawing.png      technical drawing, for humans only
│   └── tests/
│       └── <part number>/
│           └── images/*.png     photos to classify / evaluate against
├── tools/
│   ├── build_specs.py           XLS inspection sheet -> spec.json + drawing.png
│   └── crop_samples.py          one-off: how the current sample photos were cropped
├── config.py                    every tunable setting, with comments
├── classifier.py                the actual pipeline (segmentation, matching, spec lookup)
├── main.py                      build the database + evaluate + save reports
├── predict.py                   classify one photo, print its spec
└── outputs/                     everything main.py generates (see section 6)
```

**The part number is the folder name.** It is also the `sku` field inside
that part's `spec.json`, and it's exactly what `predict.py` prints as the
classification result.

---

## 4. Adding a new part

Do these in order. Every step is independent, so if a later one goes wrong
the earlier ones are still good.

### 4.1 Get the spec into the database

If you have the manufacturer's inspection-report spreadsheet (`.XLS`, the
same GALLA/UNICOM format used for the 6 parts already here):

```bash
# put the file at ../target-objects/<PARTNUMBER>.XLS, then add one line to
# PART_TO_CLASS in tools/build_specs.py:
#     "<PARTNUMBER>.XLS": "<PARTNUMBER>",
python tools/build_specs.py
```

This writes `assets/references/<part>/spec.json` (nominal + tolerance for
every dimension it could parse) and `assets/references/<part>/drawing.png`
(the sheet's own technical drawing, pulled out of the spreadsheet file).

**Open the generated `spec.json` next to `drawing.png` and check it by
eye.** The sheets aren't perfectly uniform (merged cells, mixed tolerance
notation), so `build_specs.py` does its best but is not guaranteed correct
— that's why every generated spec starts with `"review_status": "draft"`.
Fix anything wrong, then change that field to `"reviewed"`.

If you don't have a spreadsheet, write `spec.json` by hand using an
existing one (e.g. `assets/references/98661BBS-1/spec.json`) as a template.

### 4.2 Add reference photos

Take several photos of the physical part — ideally backlit (bright
background, dark object), on a plain background, at different rotations.
Crop each photo down to just the object (a tight box around it, small
margin) and save them as:

```
assets/references/<part>/images/<part>_1.png
assets/references/<part>/images/<part>_2.png
...
```

`tools/crop_samples.py` is *not* a generic tool — it's kept only to show
exactly how the current sample images were cut out of the original phone
photos in `target-objects/` (multiple identical parts on one carrier strip
get split into separate crops, see its `PHOTO_PLAN`). For a new part, crop
manually or adapt that script's approach.

More reference photos = a more reliable fingerprint. One photo will "work"
but the self-check in `main.py`'s report will tell you it isn't trustworthy.

### 4.3 (Optional but recommended) Add test photos

Take a few *more* photos — physically separate from the reference photos —
and drop them in `assets/tests/<part>/images/`. `main.py` uses these to
report real accuracy instead of just replaying the reference photos.

### 4.4 Rebuild and check

```bash
python main.py
```

Read the console output: it tells you how many classes were loaded, flags
any class with a spec but no photos yet, and — most importantly — prints
two **self-check** numbers (see section 7) that tell you whether the new
part is actually distinguishable from the others.

---

## 5. Classifying one new photo

```bash
python predict.py path/to/photo.jpg
```

Output:

```
class      : 98003P
accepted   : True
best score : 0.0586  (margin over runner-up: 0.9006)

Spec (tools/build_specs.py output -- verify review_status before trusting it):
{
  "sku": "98003P",
  "unit": "mm",
  "dimensions": [
    {"label": "D", "nominal": 4.0, "min": 3.95, "max": 4.0, ...},
    ...
  ]
}
```

`accepted: false` (class `UNKNOWN`) means the photo didn't clearly match
any known part — check the photo (framing, lighting, focus) or add more
reference photos for the part it should have matched.

This is the integration point for a future measurement step: call
`classifier.predict(image, library, contours, thresholds)` directly from
Python and read `result["spec"]["dimensions"]` to compare a real
measurement against `min`/`max` for each labeled dimension.

---

## 6. What main.py writes to outputs/

| file | what it is |
|---|---|
| `predictions.json` | one entry per test photo: predicted class + **full spec attached** — the main deliverable |
| `evaluation.csv` | one row per test photo: predicted vs. true class, scores, latency |
| `summary.json` | accuracy, thresholds used, self-check results, dataset warnings |
| `reference_library.json` | everything learned about each class (stats, spec, drawing path) |
| `reference_contours.npz` | the raw reference contours (NumPy archive; large, not human-readable) |
| `confusion_matrix.png` | true class (rows) vs. predicted class (columns); off-diagonal = wrong |
| `visualizations/*.png` | six-panel debug figure per test photo (mask, contour, fingerprint, decision) |

---

## 7. Reading the self-checks (important)

With few photos per part, plain accuracy is misleading — it can look
perfect just because two photos of the same part happen to be near-
identical. `main.py` runs two checks that don't have this problem:

- **Rotation robustness**: rotates each reference photo and re-classifies
  it. The part didn't change, so the answer shouldn't either. Reports the
  worst score a *correct* match got.
- **UNKNOWN reject path**: removes one class entirely and classifies a
  photo of it. The only correct answer is `UNKNOWN`. A `LEAK` line means a
  part was confidently (and wrongly) called something else — silhouettes
  that are too similar to separate reliably yet.

If the worst score for a *correct* match is worse than the best score for
a *wrong* match, the two groups overlap and no threshold can safely tell
them apart — `main.py` says so explicitly instead of picking a number that
would quietly misclassify parts in production.

---

## 8. Known limitations

1. **Silhouette-only.** Two parts that differ only in color, engraving, or
   a hidden internal feature look identical to this method — it needs the
   outline to differ.
2. **One view per class, by default.** The matching assumes every
   reference photo of a class is the object at the same camera angle, just
   rotated in-plane (spinning on the table). If a part genuinely needs two
   different-looking views (e.g. `UH-004`'s front/back), mixing both into
   one class's fingerprint can hurt matching more than help — see the note
   in that class's photos. Prefer one consistent camera angle per class,
   and treat other views as reference material for a human, not classifier
   input, until multi-view support is added.
3. **Front-lit demo photos.** The current sample photos are phone photos on
   plain paper, not true backlit captures. The pipeline still works, but
   specular reflections on metal parts can distort the hole count. Real
   backlight hardware removes this.
4. **Specs are drafts until reviewed.** `tools/build_specs.py` is a
   best-effort parser of an inconsistent spreadsheet format. Always check
   `review_status` in a class's `spec.json` before trusting its numbers.

Run `python main.py` and read its printed warnings — it will tell you
directly if a class has too few photos, duplicate-looking data, or
overlapping self-check scores.

---

## 9. Tuning

Everything tunable lives in `config.py`, with a comment on each setting.
The two that matter most:

| setting | effect |
|---|---|
| `MIN_MARGIN` | Raise for fewer wrong classifications (more `UNKNOWN`s instead). Lower for the opposite. |
| `AUTO_CALIBRATE` | `True` (default) measures thresholds from your reference photos instead of guessing. Only set `False` if you want to pin `MATCH_THRESHOLD`/`MIN_MARGIN` yourself. |

For inspection work, prefer more `UNKNOWN` results over a confident wrong
answer.
