#!/usr/bin/env python3
"""
Streamlit App Demo Script
Demonstrates the features of the Q&A Document Processor web app
"""

import os
from pathlib import Path

def show_demo_instructions():
    """Show demo instructions for the Streamlit app"""
    
    print("🎬 ESG Q&A Document Processor - Demo Guide")
    print("=" * 60)
    
    print("\n🚀 GETTING STARTED:")
    print("1. Run: python run_streamlit_app.py")
    print("2. Your browser will open automatically")
    print("3. If not, go to: http://localhost:8501")
    
    print("\n📤 DEMO WORKFLOW:")
    print("Step 1: Check Knowledge Base")
    print("  - Sidebar shows Q&A pairs file status")
    print("  - Default: pdf-qa-generator/output/qa_pairs.json")
    print("  - Should show ✅ if file exists")
    
    print("\nStep 2: Upload Document")
    print("  - Click 'Browse files' or drag & drop")
    print("  - Try these sample files:")
    
    # Check for available demo files
    demo_files = [
        "esg_social_ethical_questionnaire.xlsx",
        "pdf_esg_social_ethical_questionnaire.pdf",
        "Sustainability_Highlights_2024.pdf"
    ]
    
    available_files = [f for f in demo_files if Path(f).exists()]
    
    if available_files:
        print("  📁 Available demo files:")
        for f in available_files:
            print(f"    - {f}")
    else:
        print("  ⚠️  No demo files found in current directory")
        print("  - Upload your own PDF, Excel, Word, or PowerPoint file")
    
    print("\nStep 3: Process Document")
    print("  - Click '🚀 Process Document' button")
    print("  - Watch progress bars for:")
    print("    • Question extraction")
    print("    • Answer matching")
    
    print("\nStep 4: View Results")
    print("  - Summary table shows best matches")
    print("  - Check 'Show Detailed Matches' for all results")
    print("  - View statistics: total questions, match rate")
    
    print("\nStep 5: Download Results")
    print("  - 📄 Download JSON: Complete results with metadata")
    print("  - 📊 Download Excel: Multi-sheet workbook")
    
    print("\n🎛️ CONFIGURATION OPTIONS:")
    print("- Adjust 'Number of top matches' (1-10)")
    print("- Change Q&A pairs file path if needed")
    print("- Results update automatically")
    
    print("\n📊 WHAT TO EXPECT:")
    print("- Questions extracted from your document")
    print("- Each question matched to similar Q&A pairs")
    print("- Similarity scores (0.0 - 1.0, higher is better)")
    print("- Interactive tables for easy exploration")
    
    print("\n🔍 EXAMPLE RESULTS:")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ Question: 'Does your company have anti-corruption...'   │")
    print("│ Best Match: 'What initiatives does Prysmian have...'    │")
    print("│ Score: 0.418                                            │")
    print("│ Answer: 'Prysmian embraces circular economy...'         │")
    print("└─────────────────────────────────────────────────────────┘")
    
    print("\n💡 TIPS:")
    print("- Larger documents take longer to process")
    print("- Higher similarity scores indicate better matches")
    print("- Use Excel export for detailed analysis")
    print("- Check different top-k values for more/fewer matches")
    
    print("\n🛑 TO STOP THE APP:")
    print("- Close browser tab")
    print("- Press Ctrl+C in terminal")
    
    print("\n🎉 Ready to try the demo? Run the app and start exploring!")

def check_demo_environment():
    """Check if demo environment is ready"""
    
    print("\n🔍 ENVIRONMENT CHECK:")
    print("-" * 30)
    
    # Check required files
    required_files = [
        "streamlit_qa_app.py",
        "simple_document_parser.py", 
        "question_answer_matcher.py",
        "document_question_parser.py"
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    # Check Q&A pairs
    qa_file = "pdf-qa-generator/output/qa_pairs.json"
    if Path(qa_file).exists():
        print(f"✅ {qa_file}")
    else:
        print(f"❌ {qa_file}")
        missing_files.append(qa_file)
    
    # Check demo documents
    demo_docs = [
        "esg_social_ethical_questionnaire.xlsx",
        "pdf_esg_social_ethical_questionnaire.pdf"
    ]
    
    available_docs = []
    for doc in demo_docs:
        if Path(doc).exists():
            print(f"📄 {doc} (demo file)")
            available_docs.append(doc)
    
    # Summary
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} required files")
        print("Some features may not work properly")
    else:
        print(f"\n✅ All required files present!")
    
    if available_docs:
        print(f"📁 {len(available_docs)} demo documents available")
    else:
        print("📁 No demo documents - you can upload your own files")
    
    return len(missing_files) == 0

def main():
    """Main demo function"""
    
    print("🎬 Streamlit Q&A App Demo")
    
    # Check environment
    env_ok = check_demo_environment()
    
    # Show instructions
    show_demo_instructions()
    
    if not env_ok:
        print("\n⚠️  Some files are missing. The app may not work correctly.")
        print("Please ensure all required files are present.")
    
    print("\n" + "=" * 60)
    print("🚀 TO START THE DEMO:")
    print("   python run_streamlit_app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
