#!/usr/bin/env python3
"""
Simple Question Matcher
Ultra-simple version - just change the file paths and run!
"""

from question_answer_matcher import QuestionAnswerMatcher
from simple_document_parser import DOCUMENT_PATH
# ============================================================================
# CHANGE THESE PATHS TO YOUR FILES
# ============================================================================
QUESTIONS_FILE = "pdf_esg_social_ethical_questionnaire_extracted_questions.json"  # <-- YOUR QUESTIONS JSON
QA_PAIRS_FILE = "pdf-qa-generator/output/qa_pairs.json"  # <-- YOUR Q&A PAIRS JSON
TOP_ANSWERS = 3  # <-- NUMBER OF TOP ANSWERS TO FIND
# ============================================================================

def main():
    """Match questions to answers using the specified files"""
    
    matcher = QuestionAnswerMatcher()
    
    print(f"📝 Questions from: {QUESTIONS_FILE}")
    print(f"💡 Answers from: {QA_PAIRS_FILE}")
    print(f"🎯 Finding top {TOP_ANSWERS} matches per question")
    print()
    
    # Run the matching
    matcher.match_questions_to_answers(QUESTIONS_FILE, QA_PAIRS_FILE, TOP_ANSWERS)

if __name__ == "__main__":
    main()
