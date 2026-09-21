"""
All tunable settings for the object-classification module.

Everything you may want to change lives here, so you do not have to read
classifier.py to tune the pipeline.
"""

from pathlib import Path

# Folder that contains this file (object-classification/)
PROJECT_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# 1. Data paths
# ----------------------------------------------------------------------
# Each class lives in its own folder:
#   assets/references/<class>/images/*.png   sample photos (builds the database)
#   assets/references/<class>/spec.json      dimensional spec (see tools/build_specs.py)
#   assets/references/<class>/drawing.png    technical drawing, for humans only
#   assets/tests/<class>/images/*.png        photos to classify / evaluate against
REFERENCE_DIR = PROJECT_DIR / "assets" / "references"
TEST_DIR = PROJECT_DIR / "assets" / "tests"

# Where every result is written.
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Image file types we accept.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------------
# 2. Segmentation (Step 1)
# ----------------------------------------------------------------------
MORPH_KERNEL_SIZE = 3   # keep this small, a big kernel changes real geometry


# ----------------------------------------------------------------------
# 3. Main-object selection (Step 2)
# ----------------------------------------------------------------------
MIN_CONTOUR_AREA = 2000     # pixels; smaller blobs are noise (paper texture, text)
BORDER_MARGIN = 2           # a blob touching the image edge is probably clipped
MIN_BBOX_ASPECT = 0.02      # reject absurdly thin blobs
MAX_BBOX_ASPECT = 50.0

# A hole must be at least this fraction of the object area to be counted.
# Small holes are threshold noise or light reflections, not real geometry.
HOLE_MIN_AREA_RATIO = 0.005


# ----------------------------------------------------------------------
# 4. Shortlist (Runtime stage 1)
# ----------------------------------------------------------------------
TOP_K = 3   # how many candidate SKUs go to detailed matching

# Hard filters. A candidate SKU is dropped before the expensive matching
# when the query is clearly outside its expected range.
AREA_TOLERANCE = 0.35       # +/-35% around the SKU's expected area range
ASPECT_TOLERANCE = 0.35     # +/-35% around the SKU's expected aspect range
MAX_HOLE_DIFFERENCE = 3     # reject if hole count differs by more than this

# Weights of the normalised fingerprint distance.
# These are deliberately simple; tune them if a SKU is being shortlisted badly.
W_AREA = 1.0
W_ASPECT = 1.0
W_SOLIDITY = 0.5
W_CIRCULARITY = 0.5
W_HU = 0.5


# ----------------------------------------------------------------------
# 5. Fine shape matching (Runtime stage 2)
# ----------------------------------------------------------------------
# cv2.CONTOURS_MATCH_I1 / I2 / I3. main.py compares all three and prints
# which one separates the SKUs best; I1 is the default.
MATCH_METHOD = "I1"


# ----------------------------------------------------------------------
# 6. Final decision (accept or UNKNOWN)
# ----------------------------------------------------------------------
# Fallback values, used only when automatic calibration cannot run.
# When AUTO_CALIBRATE is True these are re-estimated from the reference
# data (leave-one-out) and the values actually used are printed and saved
# to outputs/summary.json.
MATCH_THRESHOLD = 0.25   # accept only if best score is below this
MIN_MARGIN = 0.05        # accept only if 2nd best is this much worse
AUTO_CALIBRATE = True


# ----------------------------------------------------------------------
# 7. Self-check
# ----------------------------------------------------------------------
# Two checks the normal evaluation cannot perform, especially on a small
# dataset. See README section 5 (outputs/summary.json).
#
#   rotation check : rotate reference images by odd angles and re-classify.
#                    Produces a REAL genuine-score distribution even when the
#                    dataset contains only one photo per SKU.
#   hold-out check : remove one SKU from the library and classify it. The
#                    answer must be UNKNOWN, which proves the reject path works.
RUN_SELF_CHECK = True
ROTATION_CHECK_ANGLES = (17, 37, 63, 128, 214)


# ----------------------------------------------------------------------
# 8. Visualisation
# ----------------------------------------------------------------------
MAX_VISUALIZATIONS = 12   # how many test images get a figure saved
