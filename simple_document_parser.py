#!/usr/bin/env python3
"""
Simple Document Parser
A general-purpose parser that can handle Excel, PDF, PowerPoint, Word, and other Docling-supported formats.
Extracts questions and saves them to JSON format similar to excel_extracted_questions.json.

Usage:
    python simple_document_parser.py
    # Then modify the DOCUMENT_PATH variable in the script to point to your file
"""

import json
import os
from pathlib import Path
from document_question_parser import DocumentQuestionParser

# =============================================================================
# CONFIGURATION - MODIFY THIS SECTION
# =============================================================================

# Put the path to your document here
DOCUMENT_PATH = "pdf_esg_social_ethical_questionnaire.pdf"  # Change this to your document path

# Optional: Choose output format (both formats will be generated)
GENERATE_SIMPLE_FORMAT = True    # Generates format like excel_extracted_questions.json
GENERATE_DETAILED_FORMAT = True  # Generates format like excel_detailed_questions.json

# =============================================================================

class SimpleDocumentParser:
    """Simple wrapper around DocumentQuestionParser for easy document processing"""
    
    def __init__(self):
        """Initialize the parser"""
        self.parser = DocumentQuestionParser()
    
    def parse_document_to_json(self, document_path: str):
        """
        Parse a document and save to JSON files with the same base name
        
        Args:
            document_path (str): Path to the document to parse
        """
        document_path = Path(document_path)
        
        # Check if file exists
        if not document_path.exists():
            print(f"❌ Error: File not found: {document_path}")
            return False
        
        print(f"📄 Processing document: {document_path.name}")
        print(f"📁 Full path: {document_path.absolute()}")
        
        # Get supported formats
        supported_extensions = {'.pdf', '.docx', '.pptx', '.xlsx', '.doc', '.ppt', '.xls'}
        file_extension = document_path.suffix.lower()
        
        if file_extension not in supported_extensions:
            print(f"⚠️  Warning: File extension '{file_extension}' may not be supported.")
            print(f"✅ Supported formats: {', '.join(supported_extensions)}")
        
        try:
            # Extract questions using the document parser
            questions = self.parser.parse_document(str(document_path))
            
            if not questions:
                print("⚠️  No questions found in the document.")
                return False
            
            print(f"✅ Successfully extracted {len(questions)} questions")
            
            # Generate output filenames based on input filename
            base_name = document_path.stem
            
            # Generate simple format (like excel_extracted_questions.json)
            if GENERATE_SIMPLE_FORMAT:
                simple_output = f"{base_name}_extracted_questions.json"
                self._save_simple_format(questions, simple_output)
            
            # Generate detailed format (like excel_detailed_questions.json)
            if GENERATE_DETAILED_FORMAT:
                detailed_output = f"{base_name}_detailed_questions.json"
                self._save_detailed_format(questions, detailed_output, str(document_path))
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing document: {str(e)}")
            return False
    
    def _save_simple_format(self, questions, output_file):
        """Save questions in simple format (like excel_extracted_questions.json)"""
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
        
        print(f"💾 Saved simple format: {output_file}")
    
    def _save_detailed_format(self, questions, output_file, source_file):
        """Save questions in detailed format (like excel_detailed_questions.json)"""
        detailed_questions = {
            "extracted_questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "source_file": os.path.basename(source_file),
                    "confidence": q.confidence
                }
                for q in questions
            ],
            "metadata": {
                "total_questions": len(questions),
                "extraction_method": "docling + regex patterns",
                "source_file": os.path.basename(source_file)
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_questions, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved detailed format: {output_file}")
    
    def batch_process(self, document_paths):
        """Process multiple documents at once"""
        print(f"🔄 Batch processing {len(document_paths)} documents...")
        
        results = []
        for doc_path in document_paths:
            print(f"\n{'='*60}")
            success = self.parse_document_to_json(doc_path)
            results.append((doc_path, success))
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        
        successful = sum(1 for _, success in results if success)
        total = len(results)
        
        for doc_path, success in results:
            status = "✅ Success" if success else "❌ Failed"
            print(f"{status}: {doc_path}")
        
        print(f"\n📈 Results: {successful}/{total} documents processed successfully")


def main():
    """Main function - modify DOCUMENT_PATH at the top of the file"""
    parser = SimpleDocumentParser()
    
    print("🚀 Simple Document Parser")
    print("=" * 50)
    
    # Check if the document path is set
    if DOCUMENT_PATH == "esg_social_ethical_questionnaire.xlsx":
        print("⚠️  Please modify the DOCUMENT_PATH variable at the top of this script")
        print("   to point to your document file.")
        print(f"   Current path: {DOCUMENT_PATH}")
        
        # Check if the default file exists
        if not Path(DOCUMENT_PATH).exists():
            print(f"❌ Default file not found: {DOCUMENT_PATH}")
            print("\n📝 To use this script:")
            print("1. Open this file in a text editor")
            print("2. Change the DOCUMENT_PATH variable to your file path")
            print("3. Run the script again")
            return
    
    # Process the document
    success = parser.parse_document_to_json(DOCUMENT_PATH)
    
    if success:
        print(f"\n🎉 Document processing completed successfully!")
        print(f"📂 Check the current directory for the generated JSON files.")
    else:
        print(f"\n❌ Document processing failed.")


def example_batch_processing():
    """Example of how to process multiple documents at once"""
    parser = SimpleDocumentParser()
    
    # List of documents to process
    documents = [
        "document1.pdf",
        "document2.xlsx", 
        "document3.pptx",
        "document4.docx"
    ]
    
    # Filter to only existing files
    existing_documents = [doc for doc in documents if Path(doc).exists()]
    
    if existing_documents:
        parser.batch_process(existing_documents)
    else:
        print("No documents found for batch processing")


if __name__ == "__main__":
    # Run the main function
    main()
    
    # Uncomment the line below if you want to try batch processing instead
    # example_batch_processing()
