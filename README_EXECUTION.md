# GNR Geospatial Stitching & Analysis Project

## Overview

This project reconstructs large geospatial maps from shuffled, overlapping image patches and answers multiple-choice questions based on the reconstructed map.

**Pipeline stages:**
1. **Stitching**: Reconstruct the map from overlapping patches using feature matching (SIFT/ORB) and global placement
2. **OCR**: Extract textual information (landmark names, labels) from the stitched map
3. **QA**: Answer multiple-choice questions based on visual and textual analysis
4. **Submission**: Generate competition-valid predictions with strategic abstention

---

## Requirements

- **Python**: 3.9+
- **GPU** (recommended): NVIDIA GPU for faster stitching and OCR; CPU fallback supported
- **Memory**: 8GB RAM minimum for full pipeline
- **Runtime**: ~30-60 minutes for sample dataset on CPU; faster on GPU

---

## Installation

### Option 1: Using `requirements.txt` (pip)

```bash
pip install -r requirements.txt
```

### Option 2: Using `environment.yml` (conda)

```bash
conda env create -f environment.yml
conda activate gnr-stitching
```

### OCR Dependencies (Optional but Recommended)

For EasyOCR (default):
```bash
pip install easyocr
```

For Tesseract OCR (lighter setup):
```bash
# On Ubuntu/Debian
sudo apt-get install tesseract-ocr
pip install pytesseract

# On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

---

## Project Structure

```
GNR-Project/
├── src/
│   ├── __init__.py                 # Module init
│   ├── config.py                   # Configuration and constants
│   ├── models.py                   # Data models and interfaces
│   ├── io_utils.py                 # Input/output handling
│   ├── stitching.py                # Stitching engine (CV)
│   ├── qa.py                       # Question answering module
│   └── submission.py               # Submission generation
├── sample_test_project_1/
│   ├── test.csv                    # Test questions
│   ├── patches/                    # Image patches (patch_0.png, patch_1.png, ...)
│   └── sample_submission.csv       # Expected submission format
├── output/                         # Generated outputs (created automatically)
│   ├── submission.csv              # Final submission
│   ├── stitched_map.png            # Reconstructed map
│   ├── diagnostics.txt             # Stitching diagnostics
│   └── pipeline.log                # Detailed logs
├── script.py                       # Command-line entry point
├── notebook.ipynb                  # Jupyter notebook entry point
├── requirements.txt                # Python dependencies (pip)
├── environment.yml                 # Conda environment
└── README.md / README_EXECUTION.md # Documentation
```

---

## Running the Pipeline

### Command-Line Script

```bash
# With default paths (reads from sample_test_project_1/, writes to output/)
python script.py

# With custom paths
python script.py --input-dir /path/to/data --output-dir /path/to/output

# With custom seed for reproducibility
python script.py --seed 42
```

### Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```

The notebook provides the same pipeline in an interactive environment with intermediate visualization steps.

---

## Configuration

Edit `src/config.py` to customize behavior:

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `FEATURE_DETECTOR` | Feature extraction algorithm | `"SIFT"` |
| `LOWE_RATIO_TEST` | Feature match quality threshold | `0.7` |
| `MIN_MATCH_COUNT` | Minimum matches to link patches | `10` |
| `OCR_ENGINE` | Text extraction engine | `"easyocr"` |
| `OCR_CONFIDENCE_THRESHOLD` | OCR text confidence cutoff | `0.3` |
| `CONFIDENCE_THRESHOLD_ABSTAIN` | Abstain when below this confidence | `0.4` |
| `VERBOSE_LOGGING` | Detailed debug logs | `True` |

---

## Output Files

After running the pipeline:

- **submission.csv**: The final submission file in competition format
  - Columns: `id`, `question_num`, `option`
  - `option`: 1-4 (answered) or 5 (abstained/unanswered)
  
- **stitched_map.png**: Reconstructed full map image
  
- **diagnostics.txt**: Stitching diagnostics (coverage, placement confidence, etc.)
  
- **pipeline.log**: Detailed execution logs

---

## Offline Mode

This pipeline is designed to work **fully offline during inference**:

1. **All models pre-bundled**: EasyOCR/Tesseract models are downloaded once and cached locally
2. **No internet required**: Jupyter notebook execution doesn't make external API calls
3. **Reproducible**: Fixed random seed ensures deterministic output across runs

### First-time setup (may require internet):

```bash
# Download EasyOCR models (one-time, ~150MB)
python -c "import easyocr; reader = easyocr.Reader(['en']); print('Models cached locally')"
```

After this, all subsequent runs are fully offline.

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2'"
**Solution**: Install OpenCV
```bash
pip install opencv-python
```

### Issue: "EasyOCR model not found"
**Solution**: Download models on first run
```bash
python -c "import easyocr; reader = easyocr.Reader(['en'])"
```

### Issue: "Out of memory"
**Solution**: Reduce image resolution or batch size in `config.py`:
```python
OCR_BATCH_SIZE = 50  # Reduce from 100
```

### Issue: "Stitching produces disconnected components"
**Solution**: Adjust feature matching thresholds:
```python
LOWE_RATIO_TEST = 0.75  # More lenient
MIN_MATCH_COUNT = 8     # Lower threshold
```

---

## Strategy Notes

### Stitching Algorithm
- **Anchor**: `patch_0.png` is always at top-left (0, 0)
- **Matching**: SIFT (strongest) or ORB (faster) feature extraction
- **Alignment**: RANSAC-based homography estimation to filter outliers
- **Placement**: Breadth-first assembly using patch_0 as root
- **Blending**: Feathering at seams to reduce artifacts

### QA Strategy
- **OCR-first**: Extract text from map to identify landmarks
- **Fuzzy matching**: Flexible text matching for place names
- **Spatial reasoning**: Use bounding boxes to answer directional questions
- **Conservative**: Abstain (option 5) when confidence is low to maximize expected score

### Scoring Strategy
- Correct answer: +1
- Incorrect: -0.25
- Abstention (option 5): 0 (no penalty)
- Hallucinated (invalid value): -1
- **Expected value**: Abstain if confidence < 0.4 (threshold configurable)

---

## Performance & Optimization

| Component | Time (CPU) | Time (GPU) |
|-----------|-----------|-----------|
| Feature extraction | ~30s | ~5s |
| Patch matching | ~60s | ~10s |
| Global assembly | ~5s | ~5s |
| OCR | ~60s | ~15s |
| QA inference | ~5s | ~5s |
| **Total** | ~160s | ~40s |

To optimize:
1. Use GPU (set `cuda` availability in config)
2. Reduce `OCR_BATCH_SIZE` to save memory
3. Use ORB instead of SIFT (faster, slightly less robust)

---

## References

**Papers Used:**
- Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints" (SIFT)
- Rublee, E., et al. (2011). "ORB: An efficient alternative to SIFT or SURF"
- Brown, M., & Lowe, D. G. (2007). "Automatic Panoramic Image Stitching using Invariant Features"

---

## Author Notes

- No training data provided; this solution uses unsupervised feature matching
- Generalization tested on diverse geospatial imagery (maps with forests, water bodies, urban areas)
- Offline-first design ensures reproducibility and avoids API dependencies
- Strategic abstention reduces negative marking impact

---

## License & Submission

This project is developed for the GNR competition. 
- **Deadline**: 2 May 23:59 UTC
- **Submission Format**: Jupyter notebook + Python script + environment files
- **Runtime Budget**: < 1 hour on 48GB L40s GPU

---

For questions or issues, refer to logs in `output/pipeline.log` or adjust `config.py` parameters.
