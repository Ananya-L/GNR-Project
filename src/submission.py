"""
Submission generation and validation module.
"""
from typing import List
from src.models import Question, SubmissionRow
from src import config


def generate_submission(questions: List[Question]) -> List[SubmissionRow]:
    """
    Convert answered questions into submission rows.
    """
    submission_rows = []
    
    for question in questions:
        option = question.predicted_option if question.predicted_option is not None else 5
        
        # Validate option
        if option not in config.VALID_OPTIONS:
            option = 5  # Default to abstention if invalid
        
        row = SubmissionRow(
            question_id=question.question_id,
            question_num=question.question_id,  # Same as question_id in this format
            option=option
        )
        
        submission_rows.append(row)
    
    return submission_rows
