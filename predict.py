"""
Classify one new photo and print the matched class's dimensional spec --
the number a measurement step needs to compare a real measurement against.

    python predict.py path/to/photo.jpg

By default this builds the class database from assets/references/ and
calibrates thresholds from that same data (identical to what main.py does).
Run main.py at least once first if you want to sanity-check the database
before trusting single predictions.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

import config
import classifier as clf


def parse_args():
    parser = argparse.ArgumentParser(description="Classify one photo and print its spec.")
    parser.add_argument("image", help="photo of the part to identify")
    parser.add_argument("--reference-dir", default=str(config.REFERENCE_DIR),
                        help="class database folder (default: assets/references)")
    return parser.parse_args()


def main():
    args = parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"could not read {args.image}")

    reference_dir = Path(args.reference_dir)
    library, contours, failures = clf.build_database(reference_dir)
    for path, reason in failures:
        print(f"  skipped {path.name}: {reason}")
    if not any(contours.values()):
        sys.exit(f"no reference photos found under {reference_dir} -- "
                 "add images to assets/references/<class>/images/ first")

    reference_items = clf.load_dataset(reference_dir)
    thresholds, _info = clf.calibrate_thresholds(reference_items, library, contours)

    result = clf.predict(img, library, contours, thresholds)

    print(f"class      : {result['class']}")
    print(f"accepted   : {result['accepted']}")
    if result["shortlist"]:
        print(f"best score : {result['best_score']:.4f}  "
              f"(margin over runner-up: {result['margin']:.4f})")

    if result["spec"] is None:
        print("\nNo spec.json for this class yet (or the result was UNKNOWN) -- "
              "nothing to compare a measurement against.")
        return

    print("\nSpec (tools/build_specs.py output -- verify review_status before trusting it):")
    print(json.dumps(result["spec"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
