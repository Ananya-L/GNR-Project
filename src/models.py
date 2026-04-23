"""
Data models and interfaces for the geospatial stitching pipeline.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

# ====== Patch Model ======
@dataclass
class PatchMetadata:
    """Metadata for a single image patch."""
    patch_id: int
    image_path: Path
    image: Optional[np.ndarray] = None  # Raw image array (loaded on demand)
    rotation: int = 0  # Detected/corrected rotation (0, 90, 180, 270)
    is_anchor: bool = False  # True if this is patch_0 (top-left)
    
    def load_image(self) -> np.ndarray:
        """Load image from disk."""
        if self.image is None:
            import cv2
            self.image = cv2.imread(str(self.image_path))
            if self.image is None:
                raise ValueError(f"Failed to load image: {self.image_path}")
        return self.image
    
    def shape(self) -> Tuple[int, int, int]:
        """Get image shape (height, width, channels)."""
        img = self.load_image()
        return img.shape


# ====== Feature Matching Model ======
@dataclass
class FeatureMatch:
    """Represents a feature match between two patches."""
    patch_id_1: int
    patch_id_2: int
    good_matches: int  # Count of Lowe's ratio test matches
    inliers: int  # Count of RANSAC inliers
    homography: Optional[np.ndarray]  # 3x3 homography matrix
    reprojection_error: Optional[float]  # RANSAC reprojection error (pixels)
    quality_score: float  # Composite score (0-1) for match reliability
    rotation_offset: int = 0  # Detected rotation difference (0, 90, 180, 270)
    
    def is_valid(self, min_matches: int = 10, min_inliers: int = 8) -> bool:
        """Check if match meets quality thresholds."""
        return (
            self.good_matches >= min_matches and 
            self.inliers >= min_inliers and 
            self.homography is not None and
            self.quality_score > 0.5
        )


# ====== Patch Position Model ======
@dataclass
class PatchPosition:
    """Global position and transform for a placed patch."""
    patch_id: int
    global_x: int  # Top-left x in stitched canvas
    global_y: int  # Top-left y in stitched canvas
    rotation: int  # Final rotation applied (0, 90, 180, 270)
    transform: np.ndarray  # Affine or perspective transform
    confidence: float  # Placement confidence (0-1)
    edges_used: List[int] = field(default_factory=list)  # IDs of edges connecting this patch
    
    def bbox(self, patch_h: int, patch_w: int) -> Tuple[int, int, int, int]:
        """Get bounding box (x1, y1, x2, y2) for this patch in global coordinates."""
        return (self.global_x, self.global_y, self.global_x + patch_w, self.global_y + patch_h)


# ====== Stitching Result Model ======
@dataclass
class StitchingResult:
    """Result of the stitching process."""
    stitched_image: np.ndarray  # Final stitched map
    patch_positions: Dict[int, PatchPosition]  # Mapping from patch_id to placement
    overlap_graph: Dict[int, List[FeatureMatch]]  # Patch adjacency graph
    canvas_height: int
    canvas_width: int
    success: bool  # Whether stitching succeeded
    error_message: str = ""
    diagnostics: Dict = field(default_factory=dict)  # Metadata: coverage, seams, disconnected components, etc.
    
    def num_patches_placed(self) -> int:
        """Count how many patches were successfully placed."""
        return len(self.patch_positions)


# ====== OCR Result Model ======
@dataclass
class OCRResult:
    """Result of OCR on the stitched map."""
    text_regions: List[Dict]  # List of {text, bbox, confidence, language}
    extracted_text: str  # Concatenated text
    bounding_boxes: List[Tuple[int, int, int, int]]  # Text region bounding boxes
    confidence_scores: List[float]  # Confidence per region
    
    def get_high_confidence_text(self, threshold: float = 0.5) -> str:
        """Get only text regions above confidence threshold."""
        high_conf = [
            region['text'] for region in self.text_regions 
            if region.get('confidence', 0) >= threshold
        ]
        return ' '.join(high_conf)


# ====== Entity Model ======
@dataclass
class Entity:
    """A detected entity (location, landmark, etc.) from OCR or CV analysis."""
    name: str  # Extracted text or label
    bbox: Tuple[int, int, int, int]  # Bounding box (x1, y1, x2, y2) in stitched canvas
    confidence: float  # Detection confidence (0-1)
    source: str  # "ocr", "heuristic", or "cv_detector"
    centroid: Tuple[float, float] = field(default_factory=lambda: (0, 0))  # Center of bbox
    
    def __post_init__(self):
        """Auto-compute centroid."""
        x1, y1, x2, y2 = self.bbox
        self.centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def distance_to(self, other: "Entity") -> float:
        """Euclidean distance to another entity."""
        dx = self.centroid[0] - other.centroid[0]
        dy = self.centroid[1] - other.centroid[1]
        return (dx**2 + dy**2) ** 0.5


# ====== Question Model ======
@dataclass
class Question:
    """A multiple-choice question from the test set."""
    question_id: str  # e.g., "ques_1"
    question_text: str  # The question
    options: List[str]  # [option_1, option_2, option_3, option_4]
    predicted_option: Optional[int] = None  # 1, 2, 3, 4, or 5 (unanswered)
    confidence: float = 0.0  # Confidence in prediction (0-1)
    reasoning: str = ""  # Explanation for the choice


# ====== Answer Candidate Model ======
@dataclass
class AnswerCandidate:
    """Represents a candidate answer with evidence."""
    option_index: int  # 1, 2, 3, or 4
    option_text: str  # The answer text
    entity_matches: List[Entity]  # Matched entities supporting this option
    spatial_consistency: float  # How well entities align with spatial relations in question
    ocr_evidence: float  # Confidence from direct OCR text match
    composite_score: float  # Combined evidence score (0-1)
    reasoning: str = ""  # Human-readable explanation


# ====== Submission Row Model ======
@dataclass
class SubmissionRow:
    """A single row in the submission CSV."""
    question_id: str
    question_num: str  # Same as question_id in this format
    option: int  # 1, 2, 3, 4, or 5
    
    def is_valid(self) -> bool:
        """Check if row respects submission rules."""
        return self.option in {1, 2, 3, 4, 5} and self.question_id == self.question_num


# ====== Pipeline Run Summary ======
@dataclass
class PipelineSummary:
    """Summary statistics from a complete pipeline run."""
    total_patches: int
    patches_successfully_placed: int
    stitching_runtime_sec: float
    ocr_runtime_sec: float
    qa_runtime_sec: float
    total_runtime_sec: float
    questions_attempted: int  # Count where option != 5
    questions_abstained: int  # Count where option == 5
    answer_confidence_distribution: Dict[str, int] = field(default_factory=dict)  # Histogram of confidence levels
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
