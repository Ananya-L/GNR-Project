"""
Configuration and constants for the geospatial stitching pipeline.
"""
import os
from pathlib import Path

# ====== Project Paths ======
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "sample_test_project_1"
PATCHES_DIR = DATA_DIR / "patches"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# ====== Random Seed (for reproducibility) ======
RANDOM_SEED = 42

# ====== Stitching Engine Configuration ======
FEATURE_DETECTOR = "SIFT"  # "SIFT" or "ORB"; SIFT is stronger, ORB is faster
ORB_N_FEATURES = 2000  # Number of ORB features to extract
SIFT_N_FEATURES = 2000  # Number of SIFT features to extract

# Feature matching thresholds
LOWE_RATIO_TEST = 0.7  # Lowe's ratio test threshold for keypoint matching
MIN_MATCH_COUNT = 10  # Minimum number of good matches to consider patches related
RANSAC_REPROJECTION_THRESHOLD = 5.0  # RANSAC threshold in pixels
RANSAC_CONFIDENCE = 0.999  # RANSAC confidence level

# Overlap detection
MIN_OVERLAP_RATIO = 0.05  # Minimum fractional overlap to detect adjacency
MAX_PATCH_DISPLACEMENT = 0.5  # Max displacement ratio for adjacent patches (relative to patch size)

# Rotation handling (90-degree increments expected)
ROTATION_ANGLES = [0, 90, 180, 270]

# Global assembly refinement
USE_BUNDLE_ADJUSTMENT = False  # Set to True for optional refinement (slower but more accurate)
TRANSFORM_DRIFT_THRESHOLD = 1.0  # Threshold for detecting inconsistent transforms

# ====== Image Blending ======
BLEND_WIDTH = 20  # Pixel width for feathering at seams
BLEND_METHOD = "feather"  # "feather" or "direct"

# ====== OCR Configuration ======
OCR_ENGINE = "easyocr"  # "easyocr" or "tesseract"
OCR_LANGUAGES = ["en"]  # Languages to recognize
OCR_CONFIDENCE_THRESHOLD = 0.3  # Confidence threshold for OCR results
OCR_BATCH_SIZE = 100  # Process OCR in batches if image is large
USE_MULTI_SCALE_OCR = True  # Run OCR at multiple scales for robustness

# ====== QA Configuration ======
FUZZY_MATCH_THRESHOLD = 0.6  # Fuzzy matching score threshold (0-1)
ENTITY_DISTANCE_TOLERANCE = 50  # Pixels: proximity for spatial relations ("near")
CONFIDENCE_THRESHOLD_ABSTAIN = 0.4  # Confidence below this triggers abstention (option 5)
ENABLE_SPATIAL_REASONING = True  # Enable directional (north/south/east/west) reasoning

# ====== Submission Configuration ======
VALID_OPTIONS = {1, 2, 3, 4, 5}  # Valid option values (5 = unanswered)
HALLUCINATION_PENALTY = 1.0  # Score penalty for values outside VALID_OPTIONS
INCORRECT_PENALTY = 0.25  # Score penalty for wrong answers
ABSTENTION_REWARD = 0.0  # Score for abstaining (option 5)
CORRECT_REWARD = 1.0  # Score for correct answer

# ====== Runtime Configuration ======
VERBOSE_LOGGING = True  # Enable detailed logs
SAVE_INTERMEDIATE_ARTIFACTS = True  # Save stitched map, OCR results, etc.
MAX_RUNTIME_SECONDS = 3600  # 1 hour budget for full pipeline

# ====== Patch Properties (to be detected/verified) ======
EXPECTED_PATCH_GRID_SIZE = None  # Will be inferred from √(number of patches); e.g., 15x15 for 225 patches
PATCH_ANCHOR_ID = 0  # patch_0 is always top-left (anchor)
ALLOW_PATCH_ROTATION = True  # Patches can be rotated 0/90/180/270

# Logging
LOG_FILE = OUTPUT_DIR / "pipeline.log"
