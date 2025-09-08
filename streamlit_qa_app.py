#!/usr/bin/env python3
"""
Streamlit Question-Answer Matching App
A web interface for uploading documents, extracting questions, and matching them to answers.
"""

import streamlit as st
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from io import BytesIO
import zipfile

# Import our custom modules
from simple_document_parser import SimpleDocumentParser
from question_answer_matcher import QuestionAnswerMatcher

# Configure page
st.set_page_config(
    page_title="ESG Q&A Document Processor",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables"""
    if 'processed_questions' not in st.session_state:
        st.session_state.processed_questions = None
    if 'matched_results' not in st.session_state:
        st.session_state.matched_results = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def process_document(file_path, file_name):
    """Process document to extract questions"""
    try:
        with st.spinner(f"Processing {file_name}..."):
            parser = SimpleDocumentParser()
            
            # Create a progress bar
            progress_bar = st.progress(0)
            st.write("🔄 Extracting questions from document...")
            progress_bar.progress(30)
            
            # Parse document
            success = parser.parse_document_to_json(file_path)
            progress_bar.progress(70)
            
            if success:
                # Load the generated questions
                base_name = Path(file_path).stem
                questions_file = f"{base_name}_extracted_questions.json"
                
                if os.path.exists(questions_file):
                    with open(questions_file, 'r', encoding='utf-8') as f:
                        questions_data = json.load(f)
                    
                    progress_bar.progress(100)
                    st.success(f"✅ Successfully extracted questions from {file_name}")
                    
                    # Clean up temp files
                    try:
                        os.unlink(file_path)
                        os.unlink(questions_file)
                        detailed_file = f"{base_name}_detailed_questions.json"
                        if os.path.exists(detailed_file):
                            os.unlink(detailed_file)
                    except:
                        pass
                    
                    return questions_data
                else:
                    st.error("❌ No questions file generated")
                    return None
            else:
                progress_bar.progress(100)
                st.error("❌ Failed to process document")
                return None
                
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

def match_questions_to_answers(questions_data, qa_pairs_file, top_k=3):
    """Match extracted questions to Q&A pairs"""
    try:
        with st.spinner("Matching questions to answers..."):
            # Create temporary files for processing
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_questions:
                json.dump(questions_data, tmp_questions, indent=2, ensure_ascii=False)
                questions_temp_path = tmp_questions.name
            
            # Create progress bar
            progress_bar = st.progress(0)
            st.write("🎯 Finding best matching answers...")
            progress_bar.progress(20)
            
            # Initialize matcher
            matcher = QuestionAnswerMatcher()
            progress_bar.progress(40)
            
            # Load data
            questions = matcher.load_extracted_questions(questions_temp_path)
            qa_items = matcher.load_qa_pairs(qa_pairs_file)
            progress_bar.progress(60)
            
            if not questions:
                st.error("❌ No questions to process")
                return None
            
            if not qa_items:
                st.error("❌ No Q&A pairs available for matching")
                return None
            
            # Build TF-IDF encoder and match
            qa_questions = [item.question for item in qa_items]
            token_to_idf, doc_vectors = matcher.build_encoder(qa_questions)
            progress_bar.progress(80)
            
            # Process each question
            all_matches = []
            for question in questions:
                matches = matcher.find_best_answers(
                    question.question, 
                    token_to_idf, 
                    doc_vectors, 
                    qa_items, 
                    top_k
                )
                
                question_result = {
                    "original_question": {
                        "id": question.id,
                        "question": question.question,
                        "source_file": question.source_file
                    },
                    "matched_answers": matches
                }
                all_matches.append(question_result)
            
            progress_bar.progress(100)
            
            # Clean up temp file
            try:
                os.unlink(questions_temp_path)
            except:
                pass
            
            st.success(f"✅ Successfully matched {len(questions)} questions to answers")
            return all_matches
            
    except Exception as e:
        st.error(f"Error matching questions: {str(e)}")
        return None

def create_results_dataframe(matched_results):
    """Create pandas DataFrame from matched results for display"""
    data = []
    detailed_data = []
    
    for result in matched_results:
        original_q = result["original_question"]
        matches = result["matched_answers"]
        
        # Summary data (one row per question with best match)
        if matches:
            top_match = matches[0]
            data.append({
                "Question ID": original_q["id"],
                "Original Question": original_q["question"][:100] + ("..." if len(original_q["question"]) > 100 else ""),
                "Best Match Score": f"{top_match['similarity_score']:.3f}",
                "Best Match Question": top_match["matched_question"][:80] + ("..." if len(top_match["matched_question"]) > 80 else ""),
                "Best Match Answer": top_match["answer"][:150] + ("..." if len(top_match["answer"]) > 150 else ""),
                "Total Matches": len(matches)
            })
        else:
            data.append({
                "Question ID": original_q["id"],
                "Original Question": original_q["question"][:100] + ("..." if len(original_q["question"]) > 100 else ""),
                "Best Match Score": "0.000",
                "Best Match Question": "No matches found",
                "Best Match Answer": "No matches found",
                "Total Matches": 0
            })
        
        # Detailed data (one row per match)
        for i, match in enumerate(matches, 1):
            detailed_data.append({
                "Question ID": original_q["id"],
                "Original Question": original_q["question"],
                "Match Rank": i,
                "Similarity Score": f"{match['similarity_score']:.3f}",
                "Matched Question": match["matched_question"],
                "Answer": match["answer"]
            })
    
    summary_df = pd.DataFrame(data)
    detailed_df = pd.DataFrame(detailed_data)
    
    return summary_df, detailed_df

def download_results(matched_results, file_format="json"):
    """Create downloadable results in specified format"""
    if file_format == "json":
        output_data = {
            "question_answer_matches": matched_results,
            "metadata": {
                "total_questions": len(matched_results),
                "matching_method": "TF-IDF cosine similarity",
                "generated_by": "Streamlit Q&A App"
            }
        }
        return json.dumps(output_data, indent=2, ensure_ascii=False).encode('utf-8')
    
    elif file_format == "excel":
        summary_df, detailed_df = create_results_dataframe(matched_results)
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            detailed_df.to_excel(writer, sheet_name='All_Matches', index=False)
            
            # Metadata sheet
            metadata_df = pd.DataFrame([
                ["Total Questions", len(matched_results)],
                ["Matching Method", "TF-IDF cosine similarity"],
                ["Generated By", "Streamlit Q&A App"]
            ], columns=["Metric", "Value"])
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        output.seek(0)
        return output.getvalue()

def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Header
    st.title("📋 ESG Question-Answer Document Processor")
    st.markdown("Upload documents to extract questions and match them with relevant answers from your knowledge base.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Q&A pairs file selection
        st.subheader("📚 Knowledge Base")
        qa_pairs_file = st.text_input(
            "Q&A Pairs File Path", 
            value="pdf-qa-generator/output/qa_pairs.json",
            help="Path to your Q&A pairs JSON file"
        )
        
        # Check if Q&A pairs file exists
        if os.path.exists(qa_pairs_file):
            st.success("✅ Q&A pairs file found")
            try:
                with open(qa_pairs_file, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                qa_pairs = qa_data.get("qa_pairs", qa_data)
                if isinstance(qa_pairs, list):
                    qa_count = len(qa_pairs)
                elif isinstance(qa_pairs, dict):
                    qa_count = len(qa_pairs)
                else:
                    qa_count = 0
                st.info(f"📊 {qa_count} Q&A pairs available")
            except Exception as e:
                st.error(f"❌ Error reading Q&A file: {str(e)}")
        else:
            st.error("❌ Q&A pairs file not found")
        
        # Matching settings
        st.subheader("🎯 Matching Settings")
        top_k = st.slider("Number of top matches per question", 1, 10, 3)
        
        # Supported formats info
        st.subheader("📄 Supported Formats")
        st.markdown("""
        - **PDF** (.pdf)
        - **Excel** (.xlsx, .xls)
        - **Word** (.docx, .doc)
        - **PowerPoint** (.pptx, .ppt)
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=['pdf', 'xlsx', 'xls', 'docx', 'doc', 'pptx', 'ppt'],
            help="Upload a document to extract questions from"
        )
        
        if uploaded_file is not None:
            st.success(f"📁 File uploaded: {uploaded_file.name}")
            
            # File details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size:,} bytes",
                "File type": uploaded_file.type
            }
            st.json(file_details)
            
            # Process button
            if st.button("🚀 Process Document", type="primary"):
                if not os.path.exists(qa_pairs_file):
                    st.error("❌ Please ensure Q&A pairs file exists before processing")
                else:
                    # Save uploaded file temporarily
                    temp_file_path = save_uploaded_file(uploaded_file)
                    
                    if temp_file_path:
                        # Process document
                        questions_data = process_document(temp_file_path, uploaded_file.name)
                        
                        if questions_data:
                            st.session_state.processed_questions = questions_data
                            st.session_state.uploaded_file_name = uploaded_file.name
                            
                            # Match questions to answers
                            matched_results = match_questions_to_answers(questions_data, qa_pairs_file, top_k)
                            
                            if matched_results:
                                st.session_state.matched_results = matched_results
                                st.rerun()
    
    with col2:
        st.header("📊 Results")
        
        if st.session_state.matched_results:
            st.success(f"✅ Processed: {st.session_state.uploaded_file_name}")
            
            # Create results DataFrames
            summary_df, detailed_df = create_results_dataframe(st.session_state.matched_results)
            
            # Display summary
            st.subheader("📋 Summary - Best Matches")
            st.dataframe(
                summary_df,
                use_container_width=True,
                height=400
            )
            
            # Statistics
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                st.metric("Total Questions", len(st.session_state.matched_results))
            
            with col_stats2:
                questions_with_matches = sum(1 for r in st.session_state.matched_results if r["matched_answers"])
                st.metric("Questions with Matches", questions_with_matches)
            
            with col_stats3:
                if len(st.session_state.matched_results) > 0:
                    match_rate = (questions_with_matches / len(st.session_state.matched_results)) * 100
                    st.metric("Match Rate", f"{match_rate:.1f}%")
            
            # Detailed view toggle
            if st.checkbox("📝 Show Detailed Matches"):
                st.subheader("🔍 All Matches Detail")
                st.dataframe(
                    detailed_df,
                    use_container_width=True,
                    height=600
                )
            
            # Download options
            st.subheader("💾 Download Results")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # JSON download
                json_data = download_results(st.session_state.matched_results, "json")
                st.download_button(
                    label="📄 Download JSON",
                    data=json_data,
                    file_name=f"{Path(st.session_state.uploaded_file_name).stem}_matched_answers.json",
                    mime="application/json"
                )
            
            with col_dl2:
                # Excel download
                try:
                    excel_data = download_results(st.session_state.matched_results, "excel")
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name=f"{Path(st.session_state.uploaded_file_name).stem}_matched_answers.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Excel download not available: {str(e)}")
        
        else:
            st.info("👆 Upload and process a document to see results here")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. 📤 Upload your document (PDF, Excel, Word, PowerPoint)
    2. 🔍 Questions are automatically extracted using AI
    3. 🎯 Questions are matched to your Q&A knowledge base using TF-IDF similarity
    4. 📊 View results in an interactive table
    5. 💾 Download results as JSON or Excel
    """)

if __name__ == "__main__":
    main()
