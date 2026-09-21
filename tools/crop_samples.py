"""
One-off data-prep tool: crop individual objects out of the raw target-objects
photos and drop them into assets/references/<class>/images/ (plus synthetic
rotated copies into assets/tests/<class>/images/ so the pipeline has
something to classify end-to-end before more real photos exist).

    python tools/crop_samples.py            # writes into assets/
    python tools/crop_samples.py --preview  # only writes debug overlays to
                                             # /tmp for a sanity check

Each source photo is front-lit (object on light grid paper), not backlit,
but the same "object is the dark side of the histogram" assumption from
classifier.segment_backlight still holds well enough for a clean crop.
A photo that shows several identical parts on one carrier strip (e.g. a
5-up terminal strip) is split into one crop per part -- more reference
images for that class instead of one big crop of the whole strip.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
TARGET_OBJECTS_DIR = PROJECT_DIR.parent / "target-objects"
REFERENCES_DIR = PROJECT_DIR / "assets" / "references"
TESTS_DIR = PROJECT_DIR / "assets" / "tests"

MIN_AREA_FRACTION = 0.00015   # drop specks (paper texture, watermark bleed)
PAD_FRACTION = 0.12           # crop margin around each detected object
ROTATION_ANGLES = (23, 71, 149, 233)  # odd angles -> real new pixels, not 90-degree flips

# photo -> (class, how many separate objects to expect, None = "take the biggest")
PHOTO_PLAN = {
    "Image_20260803_063435_917.webp": ("98661BBS-1", 5),
    "Image_20260803_063435_959.webp": ("98661BBS-1", 5),
    "Image_20260803_063436_003.webp": ("UH-004", 2),
    "Image_20260803_063436_056.webp": ("98475", 2),
    "Image_20260803_063436_112.webp": ("UH-8715P", 1),
    "Image_20260803_063436_590.webp": ("98003P", 1),
}


def split_gang(mask, box, expected_n):
    """
    Several identical contacts stamped on one carrier strip are physically
    joined at the base, so connectedComponents sees the whole strip as a
    single blob. Split it into `expected_n` vertical slices instead, using
    the column profile over the UPPER part of the blob -- that's where
    individual contacts are visibly separated, before they merge into the
    shared carrier bar lower down.

    A plain "find the zero columns" approach is fragile: depending on the
    shot, one or two adjacent contacts can end up touching for their whole
    height (shadow, reflection, slight tilt), so a true zero gap may not
    exist between every pair. Instead, assume the `expected_n` contacts are
    roughly evenly spaced (true for a stamped strip) and, near each of the
    `expected_n - 1` expected boundaries, take the locally thinnest column
    as the split point -- that still works even where the gap never quite
    reaches zero.
    """
    x, y, w, h, _area = box
    band = mask[y:y + int(0.45 * h), x:x + w]
    column_sum = (band > 0).sum(axis=0).astype(float)
    smoothed = np.convolve(column_sum, np.ones(5) / 5, mode="same")

    window = max(5, w // (expected_n * 4))
    splits = []
    for k in range(1, expected_n):
        target = int(k * w / expected_n)
        lo, hi = max(0, target - window), min(w, target + window)
        splits.append(lo + int(np.argmin(smoothed[lo:hi])))

    edges = [0] + splits + [w]
    return [(x + a, y, b - a, h, (b - a) * h) for a, b in zip(edges, edges[1:])]


def find_objects(img, expected_n):
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    min_area = MIN_AREA_FRACTION * height * width
    max_area = 0.35 * height * width  # excludes glare/shadow blobs that reach the edge
    margin = 2

    raw_boxes = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        touches_border = x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin
        raw_boxes.append((x, y, w, h, area, touches_border))

    inside = [b for b in raw_boxes if not b[5]]
    pool = inside if inside else raw_boxes
    pool = [b[:5] for b in pool]
    pool.sort(key=lambda b: b[4], reverse=True)

    # A stray speck (watermark bleed, dust) can slip past min_area and make
    # `pool` look like it already has `expected_n` entries even though the
    # real objects are still fused into one blob -- so count only entries
    # at least plausibly close in size to "one slice of the whole thing",
    # not just entries that exist at all.
    big_area = pool[0][4] if pool else 0
    plausible = [b for b in pool if b[4] >= 0.4 * big_area / expected_n]

    if expected_n > 1 and len(plausible) < expected_n:
        # The biggest surviving blob is almost certainly the whole joined strip.
        boxes = split_gang(mask, pool[0], expected_n)
    else:
        boxes = pool[:expected_n]

    boxes.sort(key=lambda b: b[0])
    return boxes, mask


def crop_with_padding(img, box):
    x, y, w, h = box[:4]
    pad = int(round(PAD_FRACTION * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    return img[y0:y1, x0:x1]


def rotate(img, angle):
    h, w = img.shape[:2]
    background = int(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(background, background, background))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="write debug overlays to /tmp instead of assets/")
    args = parser.parse_args()

    preview_dir = Path("/tmp/crop_preview")
    if args.preview:
        preview_dir.mkdir(exist_ok=True)

    per_class_crop_count = {}

    for photo_name, (class_label, expected_n) in PHOTO_PLAN.items():
        photo_path = TARGET_OBJECTS_DIR / photo_name
        img = cv2.imread(str(photo_path))
        if img is None:
            print(f"  could not read {photo_path}")
            continue

        boxes, mask = find_objects(img, expected_n)
        print(f"{photo_name:<38} -> {class_label:<12} found {len(boxes)}/{expected_n} objects")

        if args.preview:
            overlay = img.copy()
            for (x, y, w, h, _area) in boxes:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 6)
            out = preview_dir / f"{Path(photo_name).stem}_overlay.png"
            cv2.imwrite(str(out), cv2.resize(overlay, None, fx=0.35, fy=0.35))
            cv2.imwrite(str(preview_dir / f"{Path(photo_name).stem}_mask.png"),
                       cv2.resize(mask, None, fx=0.35, fy=0.35))
            continue

        ref_dir = REFERENCES_DIR / class_label / "images"
        test_dir = TESTS_DIR / class_label / "images"
        ref_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for box in boxes:
            idx = per_class_crop_count.get(class_label, 0) + 1
            per_class_crop_count[class_label] = idx

            crop = crop_with_padding(img, box)
            ref_path = ref_dir / f"{class_label}_{idx}.png"
            cv2.imwrite(str(ref_path), crop)

            for angle in ROTATION_ANGLES:
                rotated = crop_with_padding(rotate(img, angle), box)
                test_path = test_dir / f"{class_label}_{idx}_rot{angle}.png"
                cv2.imwrite(str(test_path), rotated)

    if args.preview:
        print(f"\nOverlays written to {preview_dir} -- check them before running without --preview.")
    else:
        print("\nCrops written:")
        for label, n in sorted(per_class_crop_count.items()):
            print(f"  {label:<12} {n} reference image(s), {n * len(ROTATION_ANGLES)} synthetic test image(s)")


if __name__ == "__main__":
    main()
