#!/usr/bin/env python3
"""
Complete Workflow Example
Shows how to use all the document processing tools together:
1. Parse documents to extract questions
2. Match questions to answers from Q&A pairs
3. Generate reports

This is a complete end-to-end example.
"""

import os
from pathlib import Path
from simple_document_parser import SimpleDocumentParser
from question_answer_matcher import QuestionAnswerMatcher

def workflow_example():
    """Complete workflow example"""
    
    print("🚀 COMPLETE DOCUMENT PROCESSING WORKFLOW")
    print("=" * 60)
    
    # Step 1: Document parsing
    print("\n📋 STEP 1: DOCUMENT PARSING")
    print("-" * 30)
    
    parser = SimpleDocumentParser()
    
    # Example documents to process
    documents_to_process = [
        "esg_social_ethical_questionnaire.xlsx",
        "pdf_esg_social_ethical_questionnaire.pdf",
        "questionnaire.docx"
    ]
    
    # Filter to existing documents
    existing_docs = [doc for doc in documents_to_process if Path(doc).exists()]
    
    if existing_docs:
        print(f"Found {len(existing_docs)} documents to process:")
        for doc in existing_docs:
            print(f"  📄 {doc}")
        
        # Process documents
        for doc in existing_docs:
            print(f"\n🔄 Processing: {doc}")
            success = parser.parse_document_to_json(doc)
            if success:
                print(f"✅ Successfully processed: {doc}")
            else:
                print(f"❌ Failed to process: {doc}")
    else:
        print("⚠️  No documents found to process")
    
    # Step 2: Question-Answer matching
    print("\n\n🎯 STEP 2: QUESTION-ANSWER MATCHING")
    print("-" * 40)
    
    matcher = QuestionAnswerMatcher()
    
    # Look for generated question files
    question_files = list(Path(".").glob("*_extracted_questions.json"))
    qa_pairs_file = "pdf-qa-generator/output/qa_pairs.json"
    
    if question_files and Path(qa_pairs_file).exists():
        print(f"Found {len(question_files)} question files:")
        for qf in question_files:
            print(f"  📝 {qf}")
        
        print(f"Q&A pairs file: {qa_pairs_file}")
        
        # Process each question file
        for question_file in question_files:
            print(f"\n🔄 Matching questions from: {question_file.name}")
            matcher.match_questions_to_answers(str(question_file), qa_pairs_file, 3)
    else:
        print("⚠️  No question files or Q&A pairs found for matching")
        if not Path(qa_pairs_file).exists():
            print(f"   Missing: {qa_pairs_file}")
        if not question_files:
            print("   No *_extracted_questions.json files found")
    
    # Step 3: Summary
    print("\n\n📊 WORKFLOW SUMMARY")
    print("-" * 20)
    
    # Count generated files
    json_files = list(Path(".").glob("*.json"))
    excel_files = list(Path(".").glob("*.xlsx"))
    question_files = [f for f in json_files if "_extracted_questions.json" in f.name]
    detailed_files = [f for f in json_files if "_detailed_questions.json" in f.name]
    matched_json_files = [f for f in json_files if "_matched_answers.json" in f.name]
    matched_excel_files = [f for f in excel_files if "_matched_answers.xlsx" in f.name]
    
    print(f"📄 Generated question files: {len(question_files)}")
    print(f"📋 Generated detailed files: {len(detailed_files)}")
    print(f"🎯 Generated matched answer JSON files: {len(matched_json_files)}")
    print(f"📊 Generated matched answer Excel files: {len(matched_excel_files)}")
    
    if matched_json_files or matched_excel_files:
        print(f"\n✅ Workflow completed successfully!")
        print(f"🎉 Check these files for your results:")
        for mf in matched_json_files:
            print(f"   📁 {mf}")
        for mf in matched_excel_files:
            print(f"   📊 {mf}")
    else:
        print(f"\n⚠️  Workflow partially completed - no matched answers generated")

def quick_single_document_example():
    """Quick example for processing a single document"""
    
    print("\n" + "=" * 60)
    print("🚀 QUICK SINGLE DOCUMENT EXAMPLE")
    print("=" * 60)
    
    # Change this to your document
    document_path = "esg_social_ethical_questionnaire.xlsx"
    qa_pairs_path = "pdf-qa-generator/output/qa_pairs.json"
    
    if not Path(document_path).exists():
        print(f"❌ Document not found: {document_path}")
        return
    
    if not Path(qa_pairs_path).exists():
        print(f"❌ Q&A pairs not found: {qa_pairs_path}")
        return
    
    # Step 1: Parse document
    print(f"📄 Parsing document: {document_path}")
    parser = SimpleDocumentParser()
    success = parser.parse_document_to_json(document_path)
    
    if not success:
        print("❌ Failed to parse document")
        return
    
    # Step 2: Match questions to answers
    base_name = Path(document_path).stem
    questions_file = f"{base_name}_extracted_questions.json"
    
    if Path(questions_file).exists():
        print(f"🎯 Matching questions to answers...")
        matcher = QuestionAnswerMatcher()
        matcher.match_questions_to_answers(questions_file, qa_pairs_path, 3)
        
        # Check for results
        results_json_file = f"{base_name}_matched_answers.json"
        results_excel_file = f"{base_name}_matched_answers.xlsx"
        
        if Path(results_json_file).exists():
            print(f"✅ Complete! JSON results in: {results_json_file}")
            if Path(results_excel_file).exists():
                print(f"✅ Complete! Excel results in: {results_excel_file}")
        else:
            print("⚠️  No results file generated")
    else:
        print(f"❌ Questions file not found: {questions_file}")

def show_usage_guide():
    """Show usage guide for all tools"""
    
    print("\n" + "=" * 60)
    print("📖 USAGE GUIDE")
    print("=" * 60)
    
    guide = """
🔧 TOOL OVERVIEW:

1. simple_document_parser.py
   - Parses documents (PDF, Excel, Word, PowerPoint)
   - Extracts questions using Docling + regex
   - Outputs: [filename]_extracted_questions.json

2. question_answer_matcher.py  
   - Takes extracted questions
   - Matches them to Q&A pairs using TF-IDF similarity
   - Outputs: [filename]_matched_answers.json

3. Simple versions:
   - simple_question_matcher.py (just change paths and run)
   - parse_any_document.py (just change path and run)

📝 QUICK START:

Option 1 - Use simple versions:
   1. Edit parse_any_document.py - set your document path
   2. Run: python parse_any_document.py
   3. Edit simple_question_matcher.py - set the generated JSON path
   4. Run: python simple_question_matcher.py

Option 2 - Use main scripts:
   1. Edit DOCUMENT_PATH in simple_document_parser.py
   2. Run: python simple_document_parser.py
   3. Edit paths in question_answer_matcher.py
   4. Run: python question_answer_matcher.py

Option 3 - Full workflow:
   Run: python complete_workflow_example.py

📁 OUTPUT FILES:
   - [name]_extracted_questions.json (simple format)
   - [name]_detailed_questions.json (with metadata)
   - [name]_matched_answers.json (questions + top 3 answers)
   - [name]_matched_answers.xlsx (Excel format with multiple sheets)

📊 EXCEL OUTPUT FEATURES:
   - Summary sheet: Best matches overview
   - All_Matches sheet: Detailed view of all matches
   - Metadata sheet: Processing information
   - Auto-adjusted column widths for readability
    """
    
    print(guide)

def main():
    """Run the complete workflow example"""
    
    # Show usage guide first
    show_usage_guide()
    
    # Run quick single document example
    quick_single_document_example()
    
    # Run full workflow
    workflow_example()

if __name__ == "__main__":
    main()
