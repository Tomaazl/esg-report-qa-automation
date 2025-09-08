#!/usr/bin/env python3
"""
Document Question Parser using Docling
Parses questions from PDF, DOCX, PPTX, XLSX files and formats them like test_questions.json
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    print("Warning: Docling not available. Please install with: pip install docling")
    DOCLING_AVAILABLE = False

@dataclass
class ExtractedQuestion:
    """Data class for extracted questions"""
    id: int
    question: str
    source_file: str = ""
    confidence: float = 1.0

class DocumentQuestionParser:
    """Parser for extracting questions from various document formats"""
    
    def __init__(self):
        """Initialize the parser with Docling converter"""
        if DOCLING_AVAILABLE:
            self.doc_converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.PPTX,
                    InputFormat.XLSX
                ]
            )
        else:
            self.doc_converter = None
        
        # Question patterns - various ways questions might appear
        self.question_patterns = [
            # Direct questions ending with ?
            r'([A-Z][^?]*\?)',
            # Questions starting with question words
            r'((?:What|How|When|Where|Why|Who|Which|Does|Do|Did|Can|Could|Will|Would|Should|Is|Are|Was|Were)[^?]*\?)',
            # Numbered questions
            r'(\d+[\.\)]\s*[^?]*\?)',
            # Questions in quotes
            r'"([^"]*\?)"',
            # Questions after colons
            r':\s*([^?]*\?)',
        ]
    
    def extract_questions_from_text(self, text: str, source_file: str = "") -> List[ExtractedQuestion]:
        """Extract questions from plain text using regex patterns"""
        questions = []
        question_id = 1
        
        # Clean the text
        text = self._clean_text(text)
        
        for pattern in self.question_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                cleaned_question = self._clean_question(match)
                if self._is_valid_question(cleaned_question):
                    # Check for duplicates
                    if not any(q.question.lower() == cleaned_question.lower() for q in questions):
                        questions.append(ExtractedQuestion(
                            id=question_id,
                            question=cleaned_question,
                            source_file=source_file
                        ))
                        question_id += 1
        
        return questions
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better question extraction"""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        # Remove common document artifacts
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove section headers that might get attached to questions
        text = re.sub(r'[A-Z][^:]*:\s*\d+\.', '', text)
        return text.strip()
    
    def _clean_question(self, question: str) -> str:
        """Clean and format individual questions"""
        # Remove leading numbers, bullets, section headers, etc.
        question = re.sub(r'^[\d\.\)\-\*\•\s]+', '', question)
        question = re.sub(r'^[A-Z][^:]*:\s*[\-\•]?\s*', '', question)
        question = re.sub(r'^[A-Z][^:]*:\s*\d+\.\s*', '', question)
        # Remove extra whitespace
        question = re.sub(r'\s+', ' ', question).strip()
        # Ensure it ends with a question mark
        if not question.endswith('?'):
            question += '?'
        # Capitalize first letter
        if question:
            question = question[0].upper() + question[1:]
        return question
    
    def _is_valid_question(self, question: str) -> bool:
        """Validate if the extracted text is a proper question"""
        # Minimum length check
        if len(question) < 10:
            return False
        
        # Must end with question mark
        if not question.endswith('?'):
            return False
        
        # Should contain at least one question word or be structured as a question
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 
                         'does', 'do', 'did', 'can', 'could', 'will', 'would', 
                         'should', 'is', 'are', 'was', 'were']
        
        question_lower = question.lower()
        has_question_word = any(word in question_lower for word in question_words)
        
        # Also accept questions that don't start with question words but are structured properly
        # (like "Your company employs children?")
        if not has_question_word and len(question.split()) < 5:
            return False
        
        return True
    
    def parse_document(self, file_path: str) -> List[ExtractedQuestion]:
        """Parse a single document and extract questions"""
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is not available. Please install with: pip install docling")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Processing document: {file_path.name}")
        
        try:
            # Convert document using Docling
            conv_result = self.doc_converter.convert(str(file_path))
            
            # Export to markdown for better text structure
            doc_text = conv_result.document.export_to_markdown()
            
            # Extract questions from the text
            questions = self.extract_questions_from_text(doc_text, str(file_path))
            
            print(f"Found {len(questions)} questions in {file_path.name}")
            return questions
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            return []
    
    def parse_documents(self, file_paths: List[str]) -> List[ExtractedQuestion]:
        """Parse multiple documents and extract all questions"""
        all_questions = []
        current_id = 1
        
        for file_path in file_paths:
            questions = self.parse_document(file_path)
            # Renumber questions to be sequential
            for question in questions:
                question.id = current_id
                current_id += 1
            all_questions.extend(questions)
        
        return all_questions
    
    def save_to_json(self, questions: List[ExtractedQuestion], output_file: str = "extracted_questions.json"):
        """Save questions to JSON file in the same format as test_questions.json"""
        # Convert to the same format as test_questions.json
        formatted_questions = {
            "test_questions": [
                {
                    "id": q.id,
                    "question": q.question
                }
                for q in questions
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_questions, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(questions)} questions to {output_file}")
    
    def save_detailed_json(self, questions: List[ExtractedQuestion], output_file: str = "detailed_questions.json"):
        """Save questions with additional metadata"""
        detailed_questions = {
            "extracted_questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "source_file": q.source_file,
                    "confidence": q.confidence
                }
                for q in questions
            ],
            "metadata": {
                "total_questions": len(questions),
                "extraction_method": "docling + regex patterns"
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_questions, f, indent=2, ensure_ascii=False)
        
        print(f"Saved detailed information for {len(questions)} questions to {output_file}")

def main():
    """Main function to demonstrate usage"""
    parser = DocumentQuestionParser()
    
    # Example usage - you would replace these with your actual document paths
    sample_documents = [
        # "path/to/your/document.pdf",
        # "path/to/your/document.docx",
        # "path/to/your/document.pptx",
        # "path/to/your/document.xlsx"
    ]
    
    # For demonstration, let's also show how to extract from a text sample
    sample_text = """
    ESG Questionnaire Sample:
    
    1. What are your company's climate targets?
    2. How does your organization measure carbon emissions?
    3. Does your company have a diversity policy in place?
    4. What percentage of your board consists of women?
    5. How do you ensure supply chain sustainability?
    
    Additional questions:
    - Are all employees trained on anti-corruption policies?
    - What initiatives does your company have for renewable energy?
    """
    
    print("Extracting questions from sample text:")
    questions = parser.extract_questions_from_text(sample_text, "sample_text")
    
    for q in questions:
        print(f"{q.id}: {q.question}")
    
    # Save to JSON files
    if questions:
        parser.save_to_json(questions, "sample_extracted_questions.json")
        parser.save_detailed_json(questions, "sample_detailed_questions.json")
    
    # If you have actual documents, uncomment and modify this:
    # if sample_documents:
    #     all_questions = parser.parse_documents(sample_documents)
    #     parser.save_to_json(all_questions, "extracted_questions.json")
    #     parser.save_detailed_json(all_questions, "detailed_questions.json")

if __name__ == "__main__":
    main()
