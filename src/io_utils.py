"""
Input/output utilities for loading test data and writing submissions.
"""
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
from src.models import Question, SubmissionRow, PatchMetadata
from src import config


def load_test_csv(csv_path: Path = config.TEST_CSV) -> List[Question]:
    """
    Load test.csv and parse into Question objects.
    
    Expected columns: id, question, option_1, option_2, option_3, option_4
    """
    questions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = Question(
                question_id=row['id'].strip(),
                question_text=row['question'].strip(),
                options=[
                    row['option_1'].strip(),
                    row['option_2'].strip(),
                    row['option_3'].strip(),
                    row['option_4'].strip(),
                ]
            )
            questions.append(q)
    return questions


def load_patches(patches_dir: Path = config.PATCHES_DIR) -> Dict[int, PatchMetadata]:
    """
    Scan patches directory and create metadata for all patches.
    
    Returns dict mapping patch_id -> PatchMetadata.
    Patches are named patch_0.png, patch_1.png, ..., patch_N.png.
    """
    patches = {}
    patch_files = sorted(patches_dir.glob("patch_*.png"))
    
    for patch_file in patch_files:
        # Extract patch ID from filename
        stem = patch_file.stem  # "patch_123"
        patch_id = int(stem.split("_")[1])
        
        is_anchor = (patch_id == config.PATCH_ANCHOR_ID)
        
        metadata = PatchMetadata(
            patch_id=patch_id,
            image_path=patch_file,
            is_anchor=is_anchor
        )
        patches[patch_id] = metadata
    
    return patches


def save_stitched_image(image: np.ndarray, output_name: str = "stitched_map.png") -> Path:
    """Save stitched image to output directory."""
    output_path = config.OUTPUT_DIR / output_name
    cv2.imwrite(str(output_path), image)
    return output_path


def write_submission_csv(
    submission_rows: List[SubmissionRow],
    output_path: Path = config.OUTPUT_DIR / "submission.csv"
) -> Path:
    """
    Write submission rows to CSV in the required format.
    
    Columns: id, question_num, option
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'question_num', 'option'])
        writer.writeheader()
        for row in submission_rows:
            writer.writerow({
                'id': row.question_id,
                'question_num': row.question_num,
                'option': row.option
            })
    
    return output_path


def load_sample_submission(csv_path: Path = config.SAMPLE_SUBMISSION) -> List[SubmissionRow]:
    """Load the sample submission to verify format."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sr = SubmissionRow(
                question_id=row['id'].strip(),
                question_num=row['question_num'].strip(),
                option=int(row['option'].strip())
            )
            rows.append(sr)
    return rows


def validate_submission(rows: List[SubmissionRow], expected_count: int) -> Tuple[bool, List[str]]:
    """
    Validate submission format and content.
    
    Returns (is_valid, list_of_error_messages)
    """
    errors = []
    
    if len(rows) != expected_count:
        errors.append(f"Expected {expected_count} rows, got {len(rows)}")
    
    for i, row in enumerate(rows):
        if not row.is_valid():
            errors.append(f"Row {i}: invalid option {row.option} (must be 1-5)")
        if row.question_id != row.question_num:
            errors.append(f"Row {i}: question_id != question_num ({row.question_id} != {row.question_num})")
    
    return len(errors) == 0, errors


def save_diagnostic_report(diagnostics: Dict, output_path: Path = config.OUTPUT_DIR / "diagnostics.txt") -> Path:
    """Save diagnostic information to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for key, value in diagnostics.items():
            f.write(f"{key}: {value}\n")
    
    return output_path
