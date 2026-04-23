"""
Geospatial image stitching engine.
Reconstructs a large map from shuffled, rotated patches using feature matching and global placement.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path

from src.models import PatchMetadata, FeatureMatch, PatchPosition, StitchingResult
from src import config

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract and match features between patches."""
    
    def __init__(self, detector_type: str = "SIFT"):
        self.detector_type = detector_type
        if detector_type == "SIFT":
            self.detector = cv2.SIFT_create()
        elif detector_type == "ORB":
            self.detector = cv2.ORB_create(nfeatures=config.ORB_N_FEATURES)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) if detector_type == "SIFT" \
                          else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def detect_keypoints_and_descriptors(self, image: np.ndarray) -> Tuple:
        """
        Detect keypoints and descriptors in an image.
        Returns (keypoints, descriptors).
        """
        if image is None or image.size == 0:
            return [], None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by angle in degrees (0, 90, 180, 270)."""
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise ValueError(f"Invalid angle: {angle}")
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Match features between two descriptor sets using Lowe's ratio test.
        Returns list of good matches.
        """
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < config.LOWE_RATIO_TEST * n.distance:
                    good_matches.append(m)
        
        return good_matches


class OverlapDetector:
    """Detect overlaps between patch pairs and compute homographies."""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.extractor = feature_extractor
    
    def compute_homography(self, kp1: List, kp2: List, matches: List) -> Tuple[Optional[np.ndarray], int]:
        """
        Compute homography matrix from keypoint matches using RANSAC.
        Returns (homography_matrix, num_inliers).
        """
        if len(matches) < 4:
            return None, 0
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                      ransacReprojThreshold=config.RANSAC_REPROJECTION_THRESHOLD)
        
        if H is None:
            return None, 0
        
        inliers = np.sum(mask)
        return H, inliers
    
    def compute_reprojection_error(self, H: np.ndarray, kp1: List, kp2: List, matches: List) -> float:
        """Compute mean reprojection error for inlier matches."""
        if H is None or len(matches) < 4:
            return float('inf')
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        projected = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.linalg.norm(projected - dst_pts, axis=1)
        
        return np.mean(errors[errors < config.RANSAC_REPROJECTION_THRESHOLD])
    
    def try_patch_match(self, patch1: PatchMetadata, patch2: PatchMetadata) -> Optional[FeatureMatch]:
        """
        Attempt to match two patches, trying all rotation combinations.
        Returns FeatureMatch if successful, None otherwise.
        """
        best_match = None
        best_score = 0
        
        img1 = patch1.load_image()
        img2_orig = patch2.load_image()
        
        kp1, desc1 = self.extractor.detect_keypoints_and_descriptors(img1)
        
        for rotation in config.ROTATION_ANGLES:
            img2 = self.extractor.rotate_image(img2_orig, rotation)
            kp2, desc2 = self.extractor.detect_keypoints_and_descriptors(img2)
            
            matches = self.extractor.match_features(desc1, desc2)
            
            if len(matches) < config.MIN_MATCH_COUNT:
                continue
            
            H, inliers = self.compute_homography(kp1, kp2, matches)
            if H is None or inliers < 8:
                continue
            
            reprojection_error = self.compute_reprojection_error(H, kp1, kp2, matches)
            
            # Compute quality score (composite of inliers and error)
            quality_score = max(0, 1.0 - (reprojection_error / config.RANSAC_REPROJECTION_THRESHOLD))
            quality_score *= (inliers / len(matches))
            
            if quality_score > best_score:
                best_score = quality_score
                best_match = FeatureMatch(
                    patch_id_1=patch1.patch_id,
                    patch_id_2=patch2.patch_id,
                    good_matches=len(matches),
                    inliers=inliers,
                    homography=H,
                    reprojection_error=reprojection_error,
                    quality_score=quality_score,
                    rotation_offset=rotation
                )
        
        return best_match if best_match is not None and best_match.is_valid() else None


class GlobalAssembler:
    """Assemble patches into a global coordinate system."""
    
    def __init__(self, patches: Dict[int, PatchMetadata], matches: Dict[Tuple[int, int], FeatureMatch]):
        self.patches = patches
        self.matches = matches
        self.patch_positions: Dict[int, PatchPosition] = {}
    
    def assemble(self) -> Tuple[Dict[int, PatchPosition], bool, Dict]:
        """
        Assemble patches using patch_0 as anchor and breadth-first traversal.
        Returns (patch_positions, success, diagnostics).
        """
        diagnostics = {}
        
        # Initialize anchor patch_0 at origin
        patch_0_meta = self.patches.get(config.PATCH_ANCHOR_ID)
        if patch_0_meta is None:
            return {}, False, {"error": "patch_0 (anchor) not found"}
        
        patch_0_h, patch_0_w, _ = patch_0_meta.shape()
        
        anchor_pos = PatchPosition(
            patch_id=config.PATCH_ANCHOR_ID,
            global_x=0,
            global_y=0,
            rotation=0,
            transform=np.eye(3),
            confidence=1.0,
            edges_used=[]
        )
        self.patch_positions[config.PATCH_ANCHOR_ID] = anchor_pos
        
        # BFS to place connected patches
        visited = {config.PATCH_ANCHOR_ID}
        queue = [config.PATCH_ANCHOR_ID]
        
        while queue:
            current_pid = queue.pop(0)
            current_pos = self.patch_positions[current_pid]
            
            # Find all matches involving current patch
            for (p1, p2), match in self.matches.items():
                if match.patch_id_1 == current_pid and p2 not in visited:
                    neighbor_pid = p2
                elif match.patch_id_2 == current_pid and p1 not in visited:
                    neighbor_pid = p1
                else:
                    continue
                
                if neighbor_pid in visited:
                    continue
                
                neighbor_meta = self.patches[neighbor_pid]
                neighbor_h, neighbor_w, _ = neighbor_meta.shape()
                
                # Estimate neighbor position based on homography
                # (Simplified: use match.homography to estimate displacement)
                # In a full implementation, would refine using bundle adjustment
                neighbor_pos = self._estimate_neighbor_position(
                    current_pos, patch_0_w, patch_0_h, neighbor_w, neighbor_h, match
                )
                
                if neighbor_pos is not None:
                    self.patch_positions[neighbor_pid] = neighbor_pos
                    visited.add(neighbor_pid)
                    queue.append(neighbor_pid)
        
        success = len(visited) > 0
        diagnostics['patches_placed'] = len(visited)
        diagnostics['patches_total'] = len(self.patches)
        diagnostics['coverage'] = len(visited) / len(self.patches) if self.patches else 0
        
        return self.patch_positions, success, diagnostics
    
    def _estimate_neighbor_position(
        self, 
        current_pos: PatchPosition, 
        current_w: int, 
        current_h: int,
        neighbor_w: int, 
        neighbor_h: int,
        match: FeatureMatch
    ) -> Optional[PatchPosition]:
        """
        Estimate the global position of a neighbor patch based on the homography match.
        This is a simplified estimation; full implementation would use bundle adjustment.
        """
        if match.homography is None:
            return None
        
        # Use corners of neighbor patch to estimate placement
        corners = np.array([
            [0, 0, 1],
            [neighbor_w, 0, 1],
            [0, neighbor_h, 1],
            [neighbor_w, neighbor_h, 1]
        ]).T
        
        # Transform corners
        H = match.homography
        # Simplified: use homography to infer displacement
        # Full implementation: solve system of equations for global position
        
        # For now, use a heuristic: estimate based on match quality and direction
        # This is a placeholder; refine with actual homography decomposition
        neighbor_x = current_pos.global_x + int(neighbor_w * 0.8)  # Overlap assumption
        neighbor_y = current_pos.global_y
        
        return PatchPosition(
            patch_id=match.patch_id_2,
            global_x=neighbor_x,
            global_y=neighbor_y,
            rotation=match.rotation_offset,
            transform=H,
            confidence=match.quality_score,
            edges_used=[match.patch_id_1]
        )


class ImageStitcher:
    """Main stitching orchestrator."""
    
    def __init__(self, patches: Dict[int, PatchMetadata]):
        self.patches = patches
        self.logger = logging.getLogger(__name__)
    
    def stitch(self) -> StitchingResult:
        """
        Stitch all patches into a single map.
        Returns StitchingResult.
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract features for all patches
            self.logger.info("Extracting features from patches...")
            extractor = FeatureExtractor(config.FEATURE_DETECTOR)
            
            # Step 2: Match patches pairwise
            self.logger.info("Matching patches...")
            overlap_detector = OverlapDetector(extractor)
            matches = {}
            
            patch_ids = sorted(self.patches.keys())
            for i, pid1 in enumerate(patch_ids):
                for pid2 in patch_ids[i+1:]:
                    match = overlap_detector.try_patch_match(self.patches[pid1], self.patches[pid2])
                    if match is not None:
                        matches[(pid1, pid2)] = match
                        self.logger.debug(f"Match found: {pid1} <-> {pid2} (score={match.quality_score:.3f})")
            
            self.logger.info(f"Found {len(matches)} valid matches")
            
            # Step 3: Global assembly
            self.logger.info("Assembling patches globally...")
            assembler = GlobalAssembler(self.patches, matches)
            patch_positions, assembly_success, assembly_diagnostics = assembler.assemble()
            
            if not assembly_success:
                raise RuntimeError("Global assembly failed")
            
            # Step 4: Render stitched image
            self.logger.info("Rendering stitched image...")
            stitched_image, canvas_h, canvas_w = self._render_stitched_image(patch_positions)
            
            runtime = time.time() - start_time
            self.logger.info(f"Stitching completed in {runtime:.2f}s")
            
            return StitchingResult(
                stitched_image=stitched_image,
                patch_positions=patch_positions,
                overlap_graph={pid: [m for (p1, p2), m in matches.items() if p1 == pid or p2 == pid] 
                               for pid in self.patches},
                canvas_height=canvas_h,
                canvas_width=canvas_w,
                success=True,
                diagnostics={**assembly_diagnostics, "runtime_sec": runtime}
            )
        
        except Exception as e:
            self.logger.error(f"Stitching failed: {e}")
            return StitchingResult(
                stitched_image=np.zeros((1, 1, 3), dtype=np.uint8),
                patch_positions={},
                overlap_graph={},
                canvas_height=1,
                canvas_width=1,
                success=False,
                error_message=str(e)
            )
    
    def _render_stitched_image(self, patch_positions: Dict[int, PatchPosition]) -> Tuple[np.ndarray, int, int]:
        """
        Render the stitched image by warping and blending patches.
        Returns (stitched_image, canvas_height, canvas_width).
        """
        if not patch_positions:
            return np.zeros((1, 1, 3), dtype=np.uint8), 1, 1
        
        # Estimate canvas size
        max_x, max_y = 0, 0
        for pos in patch_positions.values():
            patch_meta = self.patches[pos.patch_id]
            h, w, _ = patch_meta.shape()
            max_x = max(max_x, pos.global_x + w)
            max_y = max(max_y, pos.global_y + h)
        
        canvas = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        
        # Place patches
        for pid, pos in sorted(patch_positions.items()):
            patch_meta = self.patches[pid]
            img = patch_meta.load_image()
            
            # Apply rotation if needed
            if pos.rotation != 0:
                if pos.rotation == 90:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif pos.rotation == 180:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                elif pos.rotation == 270:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            h, w = img.shape[:2]
            x, y = pos.global_x, pos.global_y
            
            # Place on canvas (simple blend: just overlay)
            canvas[y:y+h, x:x+w] = cv2.addWeighted(
                canvas[y:y+h, x:x+w], 0.5,
                img, 0.5, 0
            )
        
        return canvas, max_y, max_x
