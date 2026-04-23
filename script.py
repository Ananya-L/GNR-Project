"""
Main entry point script for running the geospatial stitching pipeline.
Run as: python script.py --input-dir <path> --output-dir <path>
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.io_utils import load_test_csv, load_patches, write_submission_csv, save_stitched_image
from src.stitching import ImageStitcher
from src.qa import analyze_and_answer_questions
from src.submission import generate_submission


def setup_logging(log_file: Optional[Path] = None):
    """Configure logging."""
    log_level = logging.DEBUG if config.VERBOSE_LOGGING else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )


def run_pipeline(input_dir: Path, output_dir: Path) -> bool:
    """
    Run the complete stitching and QA pipeline.
    Returns True if successful.
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("GNR Geospatial Stitching Pipeline Starting")
    logger.info("=" * 60)
    
    try:
        # Load test questions and patches
        logger.info("Loading test data...")
        questions = load_test_csv(input_dir / "test.csv")
        patches = load_patches(input_dir / "patches")
        
        logger.info(f"Loaded {len(questions)} questions and {len(patches)} patches")
        
        # Stitch patches
        logger.info("Running stitching engine...")
        stitcher = ImageStitcher(patches)
        stitch_result = stitcher.stitch()
        
        if not stitch_result.success:
            logger.error(f"Stitching failed: {stitch_result.error_message}")
            return False
        
        logger.info(f"Stitching successful: {stitch_result.num_patches_placed()}/{len(patches)} patches placed")
        
        # Save stitched image
        stitch_image_path = save_stitched_image(stitch_result.stitched_image, "stitched_map.png")
        logger.info(f"Stitched map saved to {stitch_image_path}")
        
        # Answer questions
        logger.info("Analyzing questions and generating answers...")
        answered_questions = analyze_and_answer_questions(questions, stitch_result.stitched_image)
        
        abstained_count = sum(1 for q in answered_questions if q.predicted_option == 5)
        logger.info(f"Questions answered: {len(answered_questions) - abstained_count}/{len(answered_questions)}, abstained: {abstained_count}")
        
        # Generate submission
        logger.info("Generating submission...")
        submission_rows = generate_submission(answered_questions)
        
        # Write submission
        submission_path = write_submission_csv(submission_rows, output_dir / "submission.csv")
        logger.info(f"Submission written to {submission_path}")
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
        return True
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return False


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="GNR Geospatial Stitching Pipeline")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=config.DATA_DIR,
        help="Path to test data directory (containing test.csv and patches/)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR,
        help="Path to output directory for submission and artifacts"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.output_dir / "pipeline.log")
    
    # Set random seed
    import numpy as np
    np.random.seed(args.seed)
    
    # Run pipeline
    success = run_pipeline(args.input_dir, args.output_dir)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
