"""
MCQ question answering module using visual analysis and OCR.
"""
import logging
from typing import List, Dict, Optional
import numpy as np
import cv2
from pathlib import Path

from src.models import Question, AnswerCandidate, Entity, OCRResult
from src import config

logger = logging.getLogger(__name__)


class OCREngine:
    """Extract text from stitched map using OCR."""
    
    def __init__(self, engine_type: str = "easyocr"):
        self.engine_type = engine_type
        self.reader = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize OCR engine."""
        try:
            if self.engine_type == "easyocr":
                import easyocr
                self.reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=True)
            elif self.engine_type == "tesseract":
                import pytesseract
                self.tesseract = pytesseract
            else:
                raise ValueError(f"Unknown OCR engine: {self.engine_type}")
            logger.info(f"OCR engine ({self.engine_type}) initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OCR engine: {e}. Will proceed without OCR.")
            self.reader = None
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract text from image using OCR.
        Returns OCRResult with text regions, bounding boxes, and confidence scores.
        """
        if self.reader is None:
            logger.warning("OCR engine not available; returning empty result")
            return OCRResult(
                text_regions=[],
                extracted_text="",
                bounding_boxes=[],
                confidence_scores=[]
            )
        
        try:
            if self.engine_type == "easyocr":
                return self._extract_easyocr(image)
            elif self.engine_type == "tesseract":
                return self._extract_tesseract(image)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return OCRResult(
                text_regions=[],
                extracted_text="",
                bounding_boxes=[],
                confidence_scores=[]
            )
    
    def _extract_easyocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using EasyOCR."""
        results = self.reader.readtext(image)
        
        text_regions = []
        bounding_boxes = []
        confidence_scores = []
        all_text = []
        
        for (bbox, text, confidence) in results:
            if confidence < config.OCR_CONFIDENCE_THRESHOLD:
                continue
            
            bbox_array = np.array(bbox, dtype=np.int32)
            x_coords = bbox_array[:, 0]
            y_coords = bbox_array[:, 1]
            x1, y1 = x_coords.min(), y_coords.min()
            x2, y2 = x_coords.max(), y_coords.max()
            
            text_regions.append({
                'text': text,
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence
            })
            bounding_boxes.append((x1, y1, x2, y2))
            confidence_scores.append(confidence)
            all_text.append(text)
        
        return OCRResult(
            text_regions=text_regions,
            extracted_text=' '.join(all_text),
            bounding_boxes=bounding_boxes,
            confidence_scores=confidence_scores
        )
    
    def _extract_tesseract(self, image: np.ndarray) -> OCRResult:
        """Extract text using Tesseract OCR."""
        try:
            data = self.tesseract.image_to_data(image, output_type=self.tesseract.Output.DICT)
            
            text_regions = []
            bounding_boxes = []
            confidence_scores = []
            all_text = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if not text:
                    continue
                
                confidence = int(data['conf'][i]) / 100.0
                if confidence < config.OCR_CONFIDENCE_THRESHOLD:
                    continue
                
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = (x, y, x + w, y + h)
                
                text_regions.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': confidence
                })
                bounding_boxes.append(bbox)
                confidence_scores.append(confidence)
                all_text.append(text)
            
            return OCRResult(
                text_regions=text_regions,
                extracted_text=' '.join(all_text),
                bounding_boxes=bounding_boxes,
                confidence_scores=confidence_scores
            )
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return OCRResult(
                text_regions=[],
                extracted_text="",
                bounding_boxes=[],
                confidence_scores=[]
            )


class EntityMatcher:
    """Match entities from OCR to answer options."""
    
    @staticmethod
    def fuzzy_match(text1: str, text2: str, threshold: float = 0.6) -> float:
        """
        Fuzzy match two text strings.
        Returns similarity score (0-1).
        """
        try:
            from rapidfuzz import fuzz
            return fuzz.token_set_ratio(text1.lower(), text2.lower()) / 100.0
        except ImportError:
            # Fallback to simple string matching
            return float(text1.lower() == text2.lower())
    
    @staticmethod
    def extract_entities_from_ocr(ocr_result: OCRResult) -> List[Entity]:
        """Convert OCR results to Entity objects."""
        entities = []
        for region in ocr_result.text_regions:
            entity = Entity(
                name=region['text'],
                bbox=region['bbox'],
                confidence=region['confidence'],
                source='ocr'
            )
            entities.append(entity)
        return entities
    
    @staticmethod
    def find_matching_entities(option_text: str, entities: List[Entity], threshold: float = 0.6) -> List[Entity]:
        """
        Find entities that match an option text.
        Returns list of matching entities sorted by score.
        """
        matches = []
        for entity in entities:
            score = EntityMatcher.fuzzy_match(option_text, entity.name, threshold)
            if score >= threshold:
                matches.append((entity, score))
        
        return [e for e, _ in sorted(matches, key=lambda x: x[1], reverse=True)]


class QuestionAnalyzer:
    """Analyze questions and generate answers."""
    
    def __init__(self, ocr_engine: OCREngine):
        self.ocr = ocr_engine
        self.entity_matcher = EntityMatcher()
    
    def analyze_question(self, question: Question, stitched_image: np.ndarray, entities: List[Entity]) -> int:
        """
        Analyze a single question and predict the answer (1-5).
        Returns predicted option (1, 2, 3, 4, or 5 for unanswered).
        """
        logger.debug(f"Analyzing question {question.question_id}: {question.question_text}")
        
        # Score each option
        option_scores = {}
        for option_idx, option_text in enumerate(question.options, start=1):
            score = self._score_option(option_text, question.question_text, entities)
            option_scores[option_idx] = score
            logger.debug(f"  Option {option_idx}: {option_text} -> score {score:.3f}")
        
        # Select best option or abstain
        best_option = max(option_scores, key=option_scores.get)
        best_score = option_scores[best_option]
        
        if best_score < config.CONFIDENCE_THRESHOLD_ABSTAIN:
            logger.debug(f"  Decision: ABSTAIN (best score {best_score:.3f} < threshold)")
            return 5
        
        logger.debug(f"  Decision: Option {best_option} (score {best_score:.3f})")
        return best_option
    
    def _score_option(self, option_text: str, question_text: str, entities: List[Entity]) -> float:
        """
        Score how likely an option is the answer.
        Considers OCR matches, keyword presence, spatial relations.
        """
        # Find matching entities
        matches = self.entity_matcher.find_matching_entities(option_text, entities)
        
        if not matches:
            # No direct entity match; check if option text appears in question
            if option_text.lower() in question_text.lower():
                return 0.3  # Low confidence
            return 0.0
        
        # Score based on match confidence
        match_score = sum(e.confidence for e in matches) / len(matches)
        
        # Check for spatial relations in question
        spatial_boost = 0.0
        spatial_keywords = ['north', 'south', 'east', 'west', 'near', 'between', 'visible', 'shown']
        if any(kw in question_text.lower() for kw in spatial_keywords):
            spatial_boost = 0.1
        
        return min(1.0, match_score + spatial_boost)


# Stub for integration with main pipeline
def analyze_and_answer_questions(
    questions: List[Question],
    stitched_image: np.ndarray
) -> List[Question]:
    """
    Analyze stitched map and answer all questions.
    Returns questions with predicted_option and confidence filled.
    """
    logger.info("Initializing QA engine...")
    
    ocr_engine = OCREngine(config.OCR_ENGINE)
    ocr_result = ocr_engine.extract_text(stitched_image)
    entities = EntityMatcher.extract_entities_from_ocr(ocr_result)
    
    logger.info(f"Extracted {len(entities)} entities from OCR")
    
    analyzer = QuestionAnalyzer(ocr_engine)
    
    for question in questions:
        predicted_option = analyzer.analyze_question(question, stitched_image, entities)
        question.predicted_option = predicted_option
        question.confidence = 0.5 if predicted_option != 5 else 0.0
    
    return questions
