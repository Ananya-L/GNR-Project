# -*- coding: utf-8 -*-
"""solution.ipynb

# Project 1: Geospatial Image Stitching & Analysis

**Pipeline:**
1. Load overlapping image patches from local disk
2. Reconstruct the map with global overlap matching and consistency checks
3. Answer MCQ questions using deterministic offline image/question heuristics
4. Write `submission.csv`

**Model policy:** internet is not available during final notebook execution. A small model would be acceptable only if fully downloaded beforehand and loaded locally, with no APIs or online inference. This notebook uses the preferred algorithmic route instead: no ML model is required.

---
## Cell 1 — Imports & Config
"""

import os
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path.cwd()
if not (PROJECT_DIR / 'patches').exists():
    for name in ('GNR_Project', 'GNR-Project'):
        candidate = PROJECT_DIR / name
        if (candidate / 'patches').exists():
            PROJECT_DIR = candidate
            break

os.chdir(PROJECT_DIR)

TEST_CSV_PATH = PROJECT_DIR / 'test.csv'
if not TEST_CSV_PATH.exists():
    TEST_CSV_PATH = PROJECT_DIR.parent / 'test.csv'
if not TEST_CSV_PATH.exists():
    raise FileNotFoundError(f'Could not find test.csv near {PROJECT_DIR}')

PATCH_DIR = 'patches'
STITCHED_PATH = 'stitched_map.png'
SUBMISSION_PATH = 'submission.csv'

print('Working directory:', os.getcwd())
print('Patch folder:', os.path.abspath(PATCH_DIR))
print('Test CSV:', TEST_CSV_PATH)

"""---
## Cell 2 — Download Model Weights
**Run ONCE with internet during environment setup. Skip this cell at inference.**
"""

# Offline model note:
# The notebook can use a local pre-downloaded VLM for map QA. Inference is kept
# offline with local_files_only=True and should run on GPU; CPU is intentionally
# avoided because it is too slow for this task.
print('Offline stitching + local GPU VLM QA solution.')

"""---
## Cell 3 — Load Patches
"""

def load_patches(folder):
    patches = {}
    for fname in os.listdir(folder):
        if fname.endswith('.png'):
            idx = int(fname.split('_')[1].split('.')[0])
            img = np.array(Image.open(os.path.join(folder, fname)).convert('RGB'))
            patches[idx] = img
    return patches

patches = load_patches(PATCH_DIR)
N       = len(patches)
GRID    = int(np.sqrt(N))
assert GRID * GRID == N, f'Non-square patch count: {N}'

PATCH_H, PATCH_W, _ = patches[0].shape
print(f'Total patches : {N}')
print(f'Grid          : {GRID} x {GRID}')
print(f'Patch size    : {PATCH_H} x {PATCH_W} px')
print('Final image size is computed after overlap inference.')

"""---
## Cell 4 — Rotations
"""

def get_rotations(img):
    """Return all four right-angle rotations of a patch."""
    return [np.rot90(img, k) for k in range(4)]

"""---
## Cell 5 — Edge Extraction
"""

def edges(img):
    """Extract the 4 edge pixel rows/columns of a patch."""
    return {
        'top'   : img[0,  :, :],
        'bottom': img[-1, :, :],
        'left'  : img[:,  0, :],
        'right' : img[:, -1, :]
    }

"""---
## Cell 6 — Edge Matching Score
"""

def edge_score(e1, e2):
    """Normalized mean absolute L1 difference."""
    return float(np.mean(np.abs(e1.astype(np.float32) - e2.astype(np.float32))) / 255.0)


def overlap_score(a, b, direction, overlap):
    """Normalized L1 score for overlapping patch regions.

    direction='right' means b is to the right of a.
    direction='bottom' means b is below a.
    """
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    if direction == 'right':
        return float(np.mean(np.abs(a[:, -overlap:, :] - b[:, :overlap, :])))
    if direction == 'bottom':
        return float(np.mean(np.abs(a[-overlap:, :, :] - b[:overlap, :, :])))
    raise ValueError(f'Unknown direction: {direction}')

"""---
## Cell 7 — Find Best Match (with rotation)
"""

def find_best_match(ref_variant_id, direction, graph):
    """Return the graph-consistent overlap match for an oriented patch variant."""
    match = graph.get(direction, {}).get(ref_variant_id)
    if match is None:
        return None
    return match

"""---
## Cell 8 — Stitch Grid
"""

def build_oriented_variants(patches):
    """Create one entry for every patch rotation."""
    variants = []
    lookup = {}
    for patch_id in sorted(patches):
        for rotation, img in enumerate(get_rotations(patches[patch_id])):
            variant_id = len(variants)
            lookup[(patch_id, rotation)] = variant_id
            variants.append({
                'patch_id': patch_id,
                'rotation': rotation,
                'image': img,
                'float': img.astype(np.float32) / 255.0,
            })
    return variants, lookup


def infer_overlap(patches, min_overlap=16, max_overlap=None, sample_count=12):
    """Infer the exact overlap width from deterministic patch samples."""
    if max_overlap is None:
        max_overlap = min(PATCH_H, PATCH_W) // 2

    patch_ids = sorted(patches)
    sample_ids = patch_ids[:min(sample_count, len(patch_ids))]
    float_patches = {idx: patches[idx].astype(np.float32) / 255.0 for idx in patch_ids}
    votes = []

    for a_id in sample_ids:
        a = float_patches[a_id]
        for direction in ('right', 'bottom'):
            best_score = float('inf')
            best_overlap = None
            for b_id in patch_ids:
                if b_id == a_id:
                    continue
                b = float_patches[b_id]
                for overlap in range(min_overlap, max_overlap + 1):
                    if direction == 'right':
                        score = float(np.mean(np.abs(a[:, -overlap:, :] - b[:, :overlap, :])))
                    else:
                        score = float(np.mean(np.abs(a[-overlap:, :, :] - b[:overlap, :, :])))
                    if score < best_score:
                        best_score = score
                        best_overlap = overlap
            if best_score <= 1e-7:
                votes.append(best_overlap)

    if not votes:
        raise RuntimeError('Could not infer a reliable exact overlap width.')

    values, counts = np.unique(votes, return_counts=True)
    return int(values[np.argmax(counts)])


def build_exact_overlap_graph(variants, overlap):
    """Build exact right/bottom overlap compatibility using all rotations."""
    right_graph = {}
    bottom_graph = {}

    left_map = {}
    top_map = {}
    for variant_id, variant in enumerate(variants):
        img = variant['float']
        left_key = img[:, :overlap, :].tobytes()
        top_key = img[:overlap, :, :].tobytes()
        left_map.setdefault(left_key, []).append(variant_id)
        top_map.setdefault(top_key, []).append(variant_id)

    for variant_id, variant in enumerate(variants):
        img = variant['float']
        patch_id = variant['patch_id']

        right_key = img[:, -overlap:, :].tobytes()
        right_matches = [
            candidate for candidate in left_map.get(right_key, [])
            if variants[candidate]['patch_id'] != patch_id
        ]
        if right_matches:
            right_graph[variant_id] = sorted(
                right_matches,
                key=lambda v: (variants[v]['rotation'], variants[v]['patch_id'], v)
            )

        bottom_key = img[-overlap:, :, :].tobytes()
        bottom_matches = [
            candidate for candidate in top_map.get(bottom_key, [])
            if variants[candidate]['patch_id'] != patch_id
        ]
        if bottom_matches:
            bottom_graph[variant_id] = sorted(
                bottom_matches,
                key=lambda v: (variants[v]['rotation'], variants[v]['patch_id'], v)
            )

    return {'right': right_graph, 'bottom': bottom_graph}


def exact_candidates(row, col, grid_ids, used_patch_ids, graph, variants):
    """Candidates that exactly satisfy already placed left/top neighbors."""
    candidates = None

    if col > 0:
        left_variant = grid_ids[row][col - 1]
        if left_variant is None:
            return []
        candidates = set(graph['right'].get(left_variant, []))

    if row > 0:
        top_variant = grid_ids[row - 1][col]
        if top_variant is None:
            return []
        below_top = set(graph['bottom'].get(top_variant, []))
        candidates = below_top if candidates is None else candidates & below_top

    if candidates is None:
        return []

    return sorted(
        [v for v in candidates if variants[v]['patch_id'] not in used_patch_ids],
        key=lambda v: (variants[v]['rotation'], variants[v]['patch_id'], v)
    )


def top_left_candidates(variants, graph):
    """Variants with no exact neighbor above or to the left."""
    right_in = {variant_id: 0 for variant_id in range(len(variants))}
    bottom_in = {variant_id: 0 for variant_id in range(len(variants))}
    for matches in graph['right'].values():
        for variant_id in matches:
            right_in[variant_id] += 1
    for matches in graph['bottom'].values():
        for variant_id in matches:
            bottom_in[variant_id] += 1

    starts = [
        variant_id for variant_id, variant in enumerate(variants)
        if right_in[variant_id] == 0 and bottom_in[variant_id] == 0
    ]
    return sorted(starts, key=lambda v: (variants[v]['rotation'], variants[v]['patch_id'], v))


def solve_exact_grid(variants, lookup, graph):
    """Backtracking constraint solve from a graph-derived top-left corner."""
    positions = [(r, c) for r in range(GRID) for c in range(GRID) if not (r == 0 and c == 0)]
    starts = top_left_candidates(variants, graph)
    if not starts:
        raise RuntimeError('Could not find a top-left candidate with no incoming top/left edges.')

    best_filled = 0
    for start_variant in starts:
        grid_ids = [[None] * GRID for _ in range(GRID)]
        grid_ids[0][0] = start_variant
        used_patch_ids = {variants[start_variant]['patch_id']}

        def search():
            nonlocal best_filled
            best_filled = max(best_filled, len(used_patch_ids))
            if len(used_patch_ids) == GRID * GRID:
                return True

            best_pos = None
            best_candidates = None
            for row, col in positions:
                if grid_ids[row][col] is not None:
                    continue
                if col > 0 and grid_ids[row][col - 1] is None:
                    continue
                if row > 0 and grid_ids[row - 1][col] is None:
                    continue

                candidates = exact_candidates(row, col, grid_ids, used_patch_ids, graph, variants)
                if best_candidates is None or len(candidates) < len(best_candidates):
                    best_pos = (row, col)
                    best_candidates = candidates
                    if len(candidates) <= 1:
                        break

            if best_pos is None or not best_candidates:
                return False

            row, col = best_pos
            for candidate in best_candidates:
                patch_id = variants[candidate]['patch_id']
                grid_ids[row][col] = candidate
                used_patch_ids.add(patch_id)
                if search():
                    return True
                used_patch_ids.remove(patch_id)
                grid_ids[row][col] = None

            return False

        if search():
            start = variants[start_variant]
            print(f"Top-left patch : {start['patch_id']} rotation={start['rotation']}")
            return grid_ids

    tried = [(variants[v]['patch_id'], variants[v]['rotation']) for v in starts]
    raise RuntimeError(f'Could not solve a full exact-overlap grid. Best filled {best_filled}/{GRID * GRID}. Tried starts: {tried}')


def stitch(patches):
    """Globally stitch overlapping patches using exact overlap constraints.

    Steps:
    1. Consider all patches and all four rotations.
    2. Infer the overlap width from exact L1 overlap matches.
    3. Build right/bottom compatibility graphs from exact overlapping strips.
    4. Start from a graph-derived top-left corner.
    5. Solve the full grid with deterministic constraint search.
    """
    variants, lookup = build_oriented_variants(patches)
    overlap = infer_overlap(patches)
    graph = build_exact_overlap_graph(variants, overlap)
    grid_ids = solve_exact_grid(variants, lookup, graph)

    used = [variants[grid_ids[r][c]]['patch_id'] for r in range(GRID) for c in range(GRID)]
    rotations = sorted({variants[grid_ids[r][c]]['rotation'] for r in range(GRID) for c in range(GRID)})
    assert len(used) == GRID * GRID, 'Grid is not fully filled.'
    assert len(set(used)) == GRID * GRID, 'A patch was reused.'

    print(f'Inferred overlap : {overlap} px')
    print(f'Stride           : {PATCH_W - overlap} px')
    print(f'Final image size : {PATCH_H + (GRID - 1) * (PATCH_H - overlap)} x {PATCH_W + (GRID - 1) * (PATCH_W - overlap)} px')
    print(f'Rotations used   : {rotations}')

    return [[variants[grid_ids[r][c]]['image'] for c in range(GRID)] for r in range(GRID)], overlap


grid, OVERLAP = stitch(patches)
print('Stitching complete.')

"""---
## Cell 9 — Merge & Save Stitched Image
"""

def merge(grid, overlap):
    stride_h = PATCH_H - overlap
    stride_w = PATCH_W - overlap
    out_h = PATCH_H + (GRID - 1) * stride_h
    out_w = PATCH_W + (GRID - 1) * stride_w

    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)

    for row in range(GRID):
        for col in range(GRID):
            y = row * stride_h
            x = col * stride_w
            img = grid[row][col].astype(np.float32)
            canvas[y:y + PATCH_H, x:x + PATCH_W] += img
            weight[y:y + PATCH_H, x:x + PATCH_W] += 1.0

    merged = canvas / np.maximum(weight, 1.0)
    return np.clip(merged, 0, 255).astype(np.uint8)

stitched = merge(grid, OVERLAP)
Image.fromarray(stitched).save(STITCHED_PATH)
print(f'Saved {STITCHED_PATH}  shape={stitched.shape}')

plt.figure(figsize=(10, 10))
plt.imshow(stitched)
plt.title('Reconstructed Map')
plt.axis('off')
plt.tight_layout()
plt.show()

"""---
## Cell 10 — Offline QA Helpers

"""

from qa_utils import answer_question, clear_qa_cache

clear_qa_cache()
print('Offline QA helper loaded. It will use the local Qwen2-VL model when needed.')

"""---
## Cell 12 — Generate Submission
"""

# test.csv columns  : id, question, option_1, option_2, option_3, option_4
# submission columns: id, question_num, option

df = pd.read_csv(TEST_CSV_PATH)
print(f'Loaded {len(df)} questions from test.csv')
print(df.head())

answers = []

for _, row in tqdm(df.iterrows(), total=len(df), desc='Answering'):
    q_id    = row['id']
    q       = row['question']
    options = [row['option_1'], row['option_2'], row['option_3'], row['option_4']]

    ans   = answer_question(q, options, stitched)
    label = options[ans - 1] if ans != 5 else 'SKIP'
    print(f'  [{q_id}] {q}')
    print(f'         => {ans} ({label})')

    answers.append({
        'id'          : q_id,
        'question_num': q_id,
        'option'      : ans
    })

sub = pd.DataFrame(answers)

# Safety check: only 1-5 are valid; anything else is penalised as hallucination
assert sub['option'].isin([1, 2, 3, 4, 5]).all(), 'Invalid option values detected!'

sub.to_csv(SUBMISSION_PATH, index=False)
print(f'\n{SUBMISSION_PATH} saved!')
print(sub)

def run_inference(test_dir):
    import os
    from pathlib import Path
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm

    from qa_utils import answer_question, clear_qa_cache

    PROJECT_DIR = Path(test_dir)

    TEST_CSV_PATH = PROJECT_DIR / 'test.csv'
    PATCH_DIR = PROJECT_DIR / 'patches'
    SUBMISSION_PATH = 'submission.csv'

    # ---- load patches ----
    patches = load_patches(PATCH_DIR)
    global GRID, PATCH_H, PATCH_W

    N = len(patches)
    GRID = int(np.sqrt(N))
    PATCH_H, PATCH_W, _ = patches[0].shape

    # ---- stitch ----
    grid, overlap = stitch(patches)
    stitched = merge(grid, overlap)

    # ---- QA ----
    clear_qa_cache()
    df = pd.read_csv(TEST_CSV_PATH)

    answers = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        options = [
            row['option_1'],
            row['option_2'],
            row['option_3'],
            row['option_4']
        ]

        ans = answer_question(row['question'], options, stitched)

        answers.append({
            "id": row["id"],
            "question_num": row["id"],
            "option": ans
        })

    pd.DataFrame(answers).to_csv(SUBMISSION_PATH, index=False)