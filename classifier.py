"""
Object classification from a backlight silhouette, plus the per-SKU
dimensional spec that lets a downstream measurement step check a real
measurement against tolerance.

Pipeline:

    image -> mask -> contour -> fingerprint -> shortlist -> matchShapes -> CLASS / UNKNOWN

Naming convention used everywhere in this file:
    functions defined here  = custom functions (our own code)
    cv2.something()         = real OpenCV API
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

import config


# ======================================================================
# Dataset loading
# ======================================================================

def load_dataset(root):
    """
    Find every sample image under `root` and use each class folder's name
    as the label.

    The dataset on disk looks like this:

        assets/references/
            98003P/images/98003P_1.png
            98661BBS-1/images/98661BBS-1_1.png
            98661BBS-1/images/98661BBS-1_2.png

    so the label of an image is the class folder name ("98003P"), and only
    the class folder's `images/` subfolder is scanned -- that keeps the
    class's `spec.json` and `drawing.png` (see load_class_assets) from
    accidentally being picked up as sample photos.

    Returns a list of (path, label), sorted so the order is deterministic.
    """
    root = Path(root)
    items = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = class_dir.name
        images_dir = class_dir / "images"
        if not images_dir.is_dir():
            continue
        for path in sorted(images_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS:
                items.append((path, label))
    return items


def load_class_assets(class_dir):
    """
    custom function. Read a class folder's spec.json and locate its
    drawing.png/jpg, if present -- see tools/build_specs.py for how those
    are generated from the manufacturer's inspection-report spreadsheets.

    Returns (spec, drawing_path); either is None when missing.
    """
    class_dir = Path(class_dir)

    spec = None
    spec_path = class_dir / "spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))

    drawing_path = None
    for candidate in class_dir.glob("drawing.*"):
        drawing_path = str(candidate)
        break

    return spec, drawing_path


# ======================================================================
# Step 1 - Backlight segmentation
# ======================================================================

def segment_backlight(img):
    """
    custom function.

    Turn a camera image into a binary mask:
        0   = background (bright, because of the backlight)
        255 = object     (dark)

    Uses cv2.cvtColor, cv2.threshold, cv2.getStructuringElement, cv2.morphologyEx.

    Returns (mask, otsu_threshold_value, gray).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu picks the threshold automatically from the image histogram.
    # THRESH_BINARY_INV because the object is the DARK side of the histogram.
    otsu_value, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    k = config.MORPH_KERNEL_SIZE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # OPEN  = erode then dilate -> removes small isolated white specks.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # CLOSE = dilate then erode -> fills very small gaps inside the object.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask, float(otsu_value), gray


# ======================================================================
# Step 2 - Find the main object
# ======================================================================

def largest_valid_contour(mask):
    """
    custom function.

    Pick the biggest sensible blob in the mask and return its outer contour.

    A contour is simply a list of boundary points [(x1,y1), (x2,y2), ...].

    Uses cv2.connectedComponentsWithStats (to throw away small components
    cheaply) and cv2.findContours.

    Returns (contour, object_mask) or (None, None) if nothing valid was found.
    `object_mask` contains ONLY the selected object, which makes hole counting
    easy in the next step.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    height, width = mask.shape[:2]
    margin = config.BORDER_MARGIN

    candidates = []
    for i in range(1, n_labels):          # label 0 is the background
        x, y, w, h, area = stats[i]

        if area < config.MIN_CONTOUR_AREA:
            continue                      # noise: paper texture, printed text
        if h == 0 or w == 0:
            continue
        aspect = w / h
        if not (config.MIN_BBOX_ASPECT < aspect < config.MAX_BBOX_ASPECT):
            continue                      # absurdly thin blob

        touches_border = (
            x <= margin or y <= margin
            or x + w >= width - margin or y + h >= height - margin
        )
        candidates.append((area, i, touches_border))

    if not candidates:
        return None, None

    # Prefer objects fully inside the image; a clipped object has a broken shape.
    inside = [c for c in candidates if not c[2]]
    pool = inside if inside else candidates
    _, best_label, _ = max(pool, key=lambda c: c[0])

    object_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    return contour, object_mask


def count_holes(object_mask, outer_area):
    """
    custom function.

    Count the significant holes inside the object.

    cv2.RETR_CCOMP gives a two-level hierarchy: outer boundaries and holes.
    A contour whose parent is not -1 is a hole. We ignore tiny holes, which
    come from threshold noise or from light reflecting inside metal parts.
    """
    contours, hierarchy = cv2.findContours(
        object_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return 0

    min_area = config.HOLE_MIN_AREA_RATIO * max(outer_area, 1.0)
    holes = 0
    for i, c in enumerate(contours):
        has_parent = hierarchy[0][i][3] != -1
        if has_parent and cv2.contourArea(c) >= min_area:
            holes += 1
    return holes


# ======================================================================
# Step 3 - Geometric fingerprint
# ======================================================================

def geometry_features(contour, object_mask):
    """
    custom function. "Fingerprint" is NOT an OpenCV API - it is our own
    feature vector that summarises the shape of the silhouette.

    Uses cv2.contourArea, cv2.arcLength, cv2.boundingRect, cv2.minAreaRect,
    cv2.convexHull, cv2.moments, cv2.HuMoments.

    Returns a plain dict.
    """
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    # Axis-aligned bounding box. Reported because it is easy to read, but it
    # is NOT rotation invariant, so it is not used for matching.
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0.0

    # Rotated bounding box -> rotation-invariant size and aspect.
    # This is what the matching actually uses, because the object can arrive
    # at any angle between 0 and 360 degrees.
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    long_side = float(max(rw, rh))
    short_side = float(min(rw, rh))
    rect_aspect = short_side / long_side if long_side > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    circularity = (4.0 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx, cy = float(x + w / 2), float(y + h / 2)

    # Hu moments: 7 global shape descriptors, roughly invariant to
    # translation, scale and rotation. The raw values span many orders of
    # magnitude, so we compress them with a signed log.
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    return {
        "area": area,
        "perimeter": perimeter,
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(aspect_ratio),
        "rect_long": long_side,
        "rect_short": short_side,
        "rect_aspect": float(rect_aspect),
        "solidity": float(solidity),
        "circularity": float(circularity),
        "hole_count": count_holes(object_mask, area),
        "centroid": [float(cx), float(cy)],
        "hu": [float(v) for v in hu_log],
    }


def describe_image(img):
    """
    custom function. Run the whole front half of the pipeline on one image:

        image -> mask -> contour -> fingerprint

    Returns a dict with everything the later stages and the plots need,
    or None if no object could be segmented.
    """
    mask, otsu_value, gray = segment_backlight(img)
    contour, object_mask = largest_valid_contour(mask)
    if contour is None:
        return None

    return {
        "gray": gray,
        "mask": mask,
        "object_mask": object_mask,
        "contour": contour,
        "otsu": otsu_value,
        "features": geometry_features(contour, object_mask),
    }


# ======================================================================
# Step 4 - Reference library
# ======================================================================

def build_reference_library(reference_items):
    """
    custom function.

    For every reference image: segment -> contour -> fingerprint, then group
    everything by SKU.

    Returns (library, contours, failures) where
        library[label]  = statistics + expected ranges for that SKU
        contours[label] = list of contours, kept for the detailed matching
        failures        = list of images where segmentation found no object
    """
    per_label_features = {}
    contours = {}
    failures = []

    for path, label in reference_items:
        img = cv2.imread(str(path))
        if img is None:
            failures.append((path, "could not read file"))
            continue

        described = describe_image(img)
        if described is None:
            failures.append((path, "no valid object found"))
            continue

        per_label_features.setdefault(label, []).append(described["features"])
        contours.setdefault(label, []).append(described["contour"])

    library = {}
    for label, feature_list in per_label_features.items():
        library[label] = _summarise_sku(label, feature_list)

    return library, contours, failures


def build_database(references_dir):
    """
    custom function. This is the one call that "digests" the whole known-
    object database: every class folder under `references_dir` contributes
    its sample images (-> fingerprint + contour, same as
    build_reference_library), its spec.json (-> tolerance table) and its
    drawing.png (-> kept as a path only, never used for matching).

        assets/references/<class>/images/*.png
        assets/references/<class>/spec.json
        assets/references/<class>/drawing.png

    Returns (library, contours, failures) exactly like build_reference_library,
    except library[label] additionally has "spec" and "drawing" keys (either
    can be None if that class has no spec.json / no drawing yet).
    """
    references_dir = Path(references_dir)
    reference_items = load_dataset(references_dir)
    library, contours, failures = build_reference_library(reference_items)

    # A class can have a spec.json (and possibly a drawing.png) without any
    # sample images yet -- keep it in the database so build_specs.py output
    # is visible even before photos exist, just with no contours to match.
    class_dirs = {p.name: p for p in references_dir.iterdir() if p.is_dir()}
    for label, class_dir in class_dirs.items():
        spec, drawing = load_class_assets(class_dir)
        if label not in library:
            if spec is None:
                continue
            library[label] = {"label": label, "n_references": 0}
            contours.setdefault(label, [])
        library[label]["spec"] = spec
        library[label]["drawing"] = drawing

    return library, contours, failures


def _summarise_sku(label, feature_list):
    """Mean / std / expected ranges for one SKU."""
    scalar_keys = [
        "area", "perimeter", "aspect_ratio", "rect_long", "rect_short",
        "rect_aspect", "solidity", "circularity", "hole_count",
    ]

    mean, std = {}, {}
    for key in scalar_keys:
        values = np.array([f[key] for f in feature_list], dtype=float)
        mean[key] = float(values.mean())
        std[key] = float(values.std())

    hu_matrix = np.array([f["hu"] for f in feature_list], dtype=float)
    mean["hu"] = [float(v) for v in hu_matrix.mean(axis=0)]
    std["hu"] = [float(v) for v in hu_matrix.std(axis=0)]

    areas = [f["area"] for f in feature_list]
    aspects = [f["rect_aspect"] for f in feature_list]
    holes = [f["hole_count"] for f in feature_list]

    return {
        "label": label,
        "n_references": len(feature_list),
        "feature_mean": mean,
        "feature_std": std,
        "expected_area_range": [float(min(areas)), float(max(areas))],
        "expected_aspect_range": [float(min(aspects)), float(max(aspects))],
        "expected_hole_count": int(round(float(np.median(holes)))),
    }


# ======================================================================
# Runtime stage 1 - Fingerprint shortlist
# ======================================================================

def fingerprint_distance(query, sku):
    """
    custom function. A small weighted distance between the query fingerprint
    and a SKU's average fingerprint. Lower = more similar.

    Each term is normalised so that the weights in config.py are comparable.
    """
    mean = sku["feature_mean"]

    def relative(key):
        ref = abs(mean[key])
        return abs(query[key] - mean[key]) / ref if ref > 1e-9 else 0.0

    d_area = relative("area")
    d_aspect = relative("rect_aspect")
    d_solidity = abs(query["solidity"] - mean["solidity"])
    d_circularity = abs(query["circularity"] - mean["circularity"])

    # Hu values are already log-compressed, so a plain mean difference works.
    hu_q = np.array(query["hu"], dtype=float)
    hu_r = np.array(mean["hu"], dtype=float)
    d_hu = float(np.mean(np.abs(hu_q - hu_r))) / 10.0

    return (
        config.W_AREA * d_area
        + config.W_ASPECT * d_aspect
        + config.W_SOLIDITY * d_solidity
        + config.W_CIRCULARITY * d_circularity
        + config.W_HU * d_hu
    )


def passes_hard_filters(query, sku):
    """
    custom function. Cheap reject test, run before the distance is computed.
    """
    lo, hi = sku["expected_area_range"]
    if not (lo * (1 - config.AREA_TOLERANCE) <= query["area"] <= hi * (1 + config.AREA_TOLERANCE)):
        return False

    lo, hi = sku["expected_aspect_range"]
    if not (lo * (1 - config.ASPECT_TOLERANCE) <= query["rect_aspect"] <= hi * (1 + config.ASPECT_TOLERANCE)):
        return False

    # Soft in practice: reflections on metal create extra "holes", so this
    # only rejects very large disagreements.
    if abs(query["hole_count"] - sku["expected_hole_count"]) > config.MAX_HOLE_DIFFERENCE:
        return False

    return True


def shortlist(query, library, top_k=None):
    """
    custom function. "Shortlist" is NOT an OpenCV API - it is our strategy to
    avoid running the expensive contour matching against every known SKU.

        many SKUs -> cheap feature comparison -> a few plausible candidates

    Returns a list of (label, distance), best first.
    """
    top_k = config.TOP_K if top_k is None else top_k

    passed, rejected = [], []
    for label, sku in library.items():
        if "feature_mean" not in sku:
            continue  # a spec-only class with no reference photos yet: unmatchable
        entry = (label, fingerprint_distance(query, sku))
        if passes_hard_filters(query, sku):
            passed.append(entry)
        else:
            rejected.append(entry)

    passed.sort(key=lambda item: item[1])
    rejected.sort(key=lambda item: item[1])

    # Always carry at least two candidates when the library has two or more
    # SKUs. The final decision compares the best score against the runner-up,
    # so a shortlist of length 1 would make that margin test meaningless and
    # every survivor would be accepted automatically.
    kept = passed
    minimum = min(2, len(library))
    if len(kept) < minimum:
        kept = kept + rejected[: minimum - len(kept)]

    return kept[:top_k]


# ======================================================================
# Runtime stage 2 - Fine shape matching
# ======================================================================

MATCH_METHODS = {
    "I1": cv2.CONTOURS_MATCH_I1,
    "I2": cv2.CONTOURS_MATCH_I2,
    "I3": cv2.CONTOURS_MATCH_I3,
}


def shape_distance(query_contour, reference_contours, method=None):
    """
    custom function. Compare the query contour against every stored contour
    of one SKU and keep the best (lowest) score.

    Uses cv2.matchShapes, which compares the Hu moments of the two contours.
    Lower score = more similar.
    """
    flag = MATCH_METHODS[method or config.MATCH_METHOD]
    scores = [cv2.matchShapes(query_contour, ref, flag, 0.0)
              for ref in reference_contours]
    return float(min(scores)) if scores else float("inf")


# ======================================================================
# Final decision - CLASS or UNKNOWN
# ======================================================================

def classify(features, contour, library, contours, thresholds, method=None):
    """
    custom function. Shortlist, then detailed matching, then accept or reject.

    We never force the best class: it is accepted only when the score is good
    AND clearly better than the runner-up.
    """
    candidates = shortlist(features, library)

    scored = []
    for label, fp_distance in candidates:
        scored.append({
            "label": label,
            "fingerprint_distance": fp_distance,
            "shape_score": shape_distance(contour, contours.get(label, []), method),
        })
    scored.sort(key=lambda c: c["shape_score"])

    best = scored[0]
    best_score = best["shape_score"]
    second_score = scored[1]["shape_score"] if len(scored) > 1 else float("inf")
    margin = second_score - best_score

    accepted = (
        best_score < thresholds["match_threshold"]
        and margin > thresholds["min_margin"]
    )

    return {
        "class": best["label"] if accepted else "UNKNOWN",
        "best_label": best["label"],
        "accepted": bool(accepted),
        "best_score": best_score,
        "second_score": second_score,
        "margin": margin,
        "shortlist": scored,
    }


def predict(img, library, contours, thresholds, method=None):
    """
    custom function. The end-to-end entry point for a brand-new photo:

        image -> classify() -> the matched class's spec.json

    so the caller gets back both the classification decision AND the
    tolerance table a measurement step needs to compare against, in one
    call. `library` must come from build_database() (a plain
    build_reference_library() library has no "spec" key to attach).

    Returns the same dict as classify(), plus:
        "spec"    the matched class's parsed dimensional spec, or None if
                  the class has none yet, segmentation failed, or the
                  result was UNKNOWN
        "drawing" path to that class's technical drawing, or None
    """
    described = describe_image(img)
    if described is None:
        return {
            "class": "UNKNOWN", "best_label": None, "accepted": False,
            "best_score": float("inf"), "second_score": float("inf"),
            "margin": 0.0, "shortlist": [], "spec": None, "drawing": None,
        }

    result = classify(described["features"], described["contour"],
                      library, contours, thresholds, method)

    entry = library.get(result["class"], {})
    result["spec"] = entry.get("spec")
    result["drawing"] = entry.get("drawing")
    return result


# ======================================================================
# Threshold calibration
# ======================================================================

def calibrate_thresholds(reference_items, library, contours, method=None):
    """
    custom function.

    Estimate MATCH_THRESHOLD and MIN_MARGIN from the reference data instead of
    inventing numbers, using leave-one-out:

        for each reference image:
            temporarily remove it from the library
            match it against everything that is left
            record the score of the CORRECT SKU  (genuine score)
            record the best score of a WRONG SKU (impostor score)

    A good threshold sits between the worst genuine score and the best
    impostor score. If those two distributions overlap, calibration is not
    possible and we keep the defaults from config.py.

    Returns (thresholds_dict, info_dict) - info is reported to the user.
    """
    genuine, impostor, margins = [], [], []

    for path, label in reference_items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        described = describe_image(img)
        if described is None:
            continue

        query_contour = described["contour"]

        # Leave this image out of its own SKU.
        held_out = {}
        for other_label, contour_list in contours.items():
            if other_label == label:
                remaining = _drop_one_matching(contour_list, query_contour)
                if remaining:
                    held_out[other_label] = remaining
            else:
                held_out[other_label] = contour_list

        if label not in held_out:
            continue   # this SKU had only one reference image

        scores = {lbl: shape_distance(query_contour, cs, method)
                  for lbl, cs in held_out.items()}

        genuine.append(scores[label])
        wrong = [s for lbl, s in scores.items() if lbl != label]
        if wrong:
            impostor.append(min(wrong))
            margins.append(min(wrong) - scores[label])

    info = {
        "calibrated": False,
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "worst_genuine_score": max(genuine) if genuine else None,
        "best_impostor_score": min(impostor) if impostor else None,
        "reason": "",
    }

    thresholds = {
        "match_threshold": config.MATCH_THRESHOLD,
        "min_margin": config.MIN_MARGIN,
    }

    if not genuine or not impostor:
        info["reason"] = "not enough reference images (need >=2 per SKU and >=2 SKUs)"
        return thresholds, info

    worst_genuine = max(genuine)
    best_impostor = min(impostor)

    if worst_genuine < best_impostor:
        # Clean separation: put the threshold halfway between the two groups.
        thresholds["match_threshold"] = float((worst_genuine + best_impostor) / 2.0)
        # Require half of the smallest margin we actually observed.
        thresholds["min_margin"] = float(max(0.0, min(margins) * 0.5))
        info["calibrated"] = True
        info["reason"] = "genuine and impostor scores are separated"
    else:
        info["reason"] = (
            "genuine and impostor scores overlap; keeping defaults from config.py"
        )

    return thresholds, info


# ======================================================================
# Self-check
# ======================================================================

def rotate_image(img, angle):
    """
    custom function. Rotate around the image centre, keeping the size.

    The new corners are filled with the median background grey instead of
    black, otherwise the fill would look like a huge dark object and the
    segmentation would latch onto it.

    Uses cv2.getRotationMatrix2D and cv2.warpAffine.
    """
    h, w = img.shape[:2]
    background = int(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(background, background, background),
    )


def rotation_self_check(reference_items, library, contours, thresholds, angles):
    """
    custom function.

    Rotate every reference image by a few odd angles and classify it again.
    The object is the same physical part, so the answer must not change.

    This matters because a dataset can contain only one photo per SKU (or
    duplicates of it). Rotating with interpolation creates genuinely new
    pixels, so the resulting scores are a REAL genuine-score distribution
    rather than a stream of perfect zeros.
    """
    total, correct, unknown = 0, 0, 0
    scores = []

    for path, label in reference_items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        for angle in angles:
            described = describe_image(rotate_image(img, angle))
            if described is None:
                total += 1
                continue
            result = classify(described["features"], described["contour"],
                              library, contours, thresholds)
            total += 1
            scores.append(result["best_score"])
            if result["class"] == label:
                correct += 1
            elif result["class"] == "UNKNOWN":
                unknown += 1

    return {
        "angles": list(angles),
        "tested": total,
        "correct": correct,
        "unknown": unknown,
        "wrong": total - correct - unknown,
        "accuracy": correct / total if total else 0.0,
        "worst_genuine_score": float(max(scores)) if scores else None,
        "median_genuine_score": float(np.median(scores)) if scores else None,
    }


def holdout_self_check(items, thresholds_from_config=False):
    """
    custom function.

    Remove one SKU from the library, then classify an image of that SKU.
    The correct answer is UNKNOWN, because the part is genuinely unknown.

    This is the only way to confirm the reject path works when every test
    image belongs to a SKU that is already in the library.
    """
    labels = sorted({label for _, label in items})
    if len(labels) < 3:
        return {"tested": 0, "unknown": 0, "leaked": [], "note": "needs >=3 SKUs"}

    tested, unknown, leaked = 0, 0, []
    for held in labels:
        reduced = [(p, l) for p, l in items if l != held]
        library, contours, _ = build_reference_library(reduced)
        if len(library) < 2:
            continue

        if thresholds_from_config:
            thresholds = {"match_threshold": config.MATCH_THRESHOLD,
                          "min_margin": config.MIN_MARGIN}
        else:
            thresholds, _ = calibrate_thresholds(reduced, library, contours)

        path = next(p for p, l in items if l == held)
        img = cv2.imread(str(path))
        if img is None:
            continue
        described = describe_image(img)
        if described is None:
            continue

        result = classify(described["features"], described["contour"],
                          library, contours, thresholds)
        tested += 1
        if result["class"] == "UNKNOWN":
            unknown += 1
        else:
            leaked.append({"held_out": held, "claimed": result["class"],
                           "best_score": result["best_score"]})

    return {"tested": tested, "unknown": unknown, "leaked": leaked, "note": ""}


def _drop_one_matching(contour_list, query_contour):
    """
    Remove the single stored contour that corresponds to `query_contour`.

    The reference image is read twice (once when building the library, once
    during calibration), so the two contour arrays are equal in value but not
    the same Python object. We drop the first one with identical shape and
    content.
    """
    remaining = []
    dropped = False
    for c in contour_list:
        same = (not dropped
                and c.shape == query_contour.shape
                and np.array_equal(c, query_contour))
        if same:
            dropped = True
            continue
        remaining.append(c)
    return remaining
