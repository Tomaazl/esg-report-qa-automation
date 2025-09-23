#!/usr/bin/env python3
"""
Simple Document Parser with Azure OpenAI
A general-purpose parser that can handle Excel, PDF, PowerPoint, Word, and other Docling-supported formats.
Uses Azure OpenAI to intelligently extract ESG-related questions from documents.

Usage:
    python simple_document_parser.py
    # Then modify the DOCUMENT_PATH variable in the script to point to your file
    # Make sure to set up your Azure OpenAI credentials in pdf-qa-generator/.env
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Azure OpenAI imports
from openai import AzureOpenAI
from dotenv import load_dotenv

# Document processing imports
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    print("Warning: Docling not available. Please install with: pip install docling")
    DOCLING_AVAILABLE = False

# =============================================================================
# CONFIGURATION - MODIFY THIS SECTION
# =============================================================================

# Put the path to your document here
DOCUMENT_PATH = "pdf_esg_social_ethical_questionnaire.pdf"  # Change this to your document path

# Optional: Choose output format (both formats will be generated)
GENERATE_SIMPLE_FORMAT = True    # Generates format like excel_extracted_questions.json
GENERATE_DETAILED_FORMAT = True  # Generates format like excel_detailed_questions.json

# =============================================================================

# Load environment variables from pdf-qa-generator/.env
load_dotenv(os.path.join(os.path.dirname(__file__), 'pdf-qa-generator', '.env'))

@dataclass
class ExtractedQuestion:
    """Data class for extracted questions"""
    id: int
    question: str
    source_file: str = ""
    confidence: float = 1.0
    category: str = ""

class ESGQuestion(BaseModel):
    """Pydantic model for ESG questions extracted by Azure OpenAI"""
    question: str = Field(description="The ESG-related question extracted from the document")
    category: str = Field(description="ESG category: Environmental, Social, or Governance")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0", default=1.0)

class ESGQuestionList(BaseModel):
    """List of ESG questions"""
    questions: List[ESGQuestion] = Field(description="List of ESG-related questions found in the document")

class SimpleDocumentParser:
    """Document parser using Azure OpenAI to extract ESG-related questions"""
    
    def __init__(self):
        """Initialize the parser with Azure OpenAI client and document converter"""
        # Initialize Azure OpenAI client using the same pattern as openai_qa_gen.py
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        
        if not all([self.api_key, self.endpoint, self.api_version]):
            raise ValueError(
                "Missing Azure OpenAI configuration. Please ensure the following environment variables are set in pdf-qa-generator/.env:\n"
                "- AZURE_OPENAI_API_KEY\n"
                "- AZURE_OPENAI_ENDPOINT\n"
                "- AZURE_OPENAI_API_VERSION"
            )
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version
        )
        
        # Initialize document converter if available
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
    
    def extract_text_from_document(self, document_path: Path) -> str:
        """Extract text from document using Docling"""
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is not available. Please install with: pip install docling")
        
        try:
            # Convert document using Docling
            conv_result = self.doc_converter.convert(str(document_path))
            # Export to markdown for better text structure
            doc_text = conv_result.document.export_to_markdown()
            return doc_text
        except Exception as e:
            print(f"❌ Error extracting text from {document_path.name}: {str(e)}")
            return ""

    def call_openai_structured(self, prompt: str, deployment: str, system_prompt: str = None, temperature: float = 0.3) -> List[ESGQuestion]:
        """Call Azure OpenAI with structured output - following openai_qa_gen.py pattern"""
        if system_prompt is None:
            system_prompt = """You are an ESG (Environmental, Social, Governance) expert. Your task is to identify and extract ESG-related questions from the provided document text.

Focus on questions that relate to:
- Environmental: Climate change, carbon emissions, renewable energy, waste management, water usage, biodiversity, environmental policies
- Social: Employee welfare, diversity & inclusion, human rights, labor practices, community impact, health & safety, supply chain ethics
- Governance: Board composition, executive compensation, business ethics, anti-corruption, transparency, risk management, regulatory compliance

Extract only clear, well-formed questions that end with a question mark. Categorize each question as Environmental, Social, or Governance. Provide a confidence score based on how clearly ESG-related the question is."""

        try:
            response = self.client.beta.chat.completions.parse(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=deployment,
                response_format=ESGQuestionList,
                temperature=temperature,
                max_tokens=16384
            )
            
            if response.choices[0].message.parsed:
                return response.choices[0].message.parsed.questions
            else:
                print("⚠️  No structured response received from Azure OpenAI")
                return []
                
        except Exception as e:
            print(f"❌ Error calling Azure OpenAI: {str(e)}")
            return []

    def extract_esg_questions_with_ai(self, text: str, deployment: str = "gpt-4") -> List[ESGQuestion]:
        """Use Azure OpenAI to extract ESG-related questions from text"""
        user_prompt = f"""Please analyze the following document text and extract all ESG-related questions. For each question, provide:
1. The exact question text
2. ESG category (Environmental, Social, or Governance)
3. Confidence score (0.0-1.0)

Document text:
{text}

Please extract only legitimate questions that are clearly related to ESG topics."""

        return self.call_openai_structured(user_prompt, deployment)

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
            # Extract text from document
            print("🔄 Extracting text from document...")
            document_text = self.extract_text_from_document(document_path)
            
            if not document_text.strip():
                print("⚠️  No text content found in the document.")
                return False
            
            print(f"✅ Extracted {len(document_text)} characters of text")
            
            # Use Azure OpenAI to extract ESG questions
            print("🤖 Using Azure OpenAI to identify ESG questions...")
            
            # Use the known deployment name
            deployment_name = "esg-qa"
            print(f"   Using deployment: {deployment_name}")
            esg_questions = self.extract_esg_questions_with_ai(document_text, deployment_name)
            
            if not esg_questions:
                print("⚠️  No ESG-related questions found in the document.")
                return False
            
            print(f"✅ Successfully extracted {len(esg_questions)} ESG questions")
            
            # Convert to ExtractedQuestion format
            questions = []
            for i, esg_q in enumerate(esg_questions, 1):
                questions.append(ExtractedQuestion(
                    id=i,
                    question=esg_q.question,
                    source_file=str(document_path),
                    confidence=esg_q.confidence,
                    category=esg_q.category
                ))
            
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
                    "confidence": q.confidence,
                    "esg_category": q.category
                }
                for q in questions
            ],
            "metadata": {
                "total_questions": len(questions),
                "extraction_method": "docling + azure openai",
                "source_file": os.path.basename(source_file),
                "esg_categories": list(set(q.category for q in questions if q.category))
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
    print("🚀 Simple Document Parser with Azure OpenAI")
    print("=" * 60)
    print("🤖 Uses AI to intelligently extract ESG-related questions")
    print("=" * 60)
    
    try:
        parser = SimpleDocumentParser()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n📝 Setup Instructions:")
        print("1. Make sure you have a '.env' file in the 'pdf-qa-generator' directory")
        print("2. Ensure your Azure OpenAI credentials are set:")
        print("   AZURE_OPENAI_API_KEY=your_api_key_here")
        print("   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        print("   AZURE_OPENAI_API_VERSION=2024-02-15-preview")
        print("3. The script will automatically try common deployment names")
        print("4. Run the script again")
        return
    
    # Check if the document path is set to default
    if DOCUMENT_PATH == "pdf_esg_social_ethical_questionnaire.pdf":
        print("ℹ️  Using default document path. You can modify DOCUMENT_PATH at the top of this script.")
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
        print(f"✨ ESG questions were intelligently extracted using Azure OpenAI")
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
