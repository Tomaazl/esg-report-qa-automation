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
    if 'selected_qa_indices' not in st.session_state:
        st.session_state.selected_qa_indices = None  # Will be set to all indices by default
    if 'qa_pairs_data' not in st.session_state:
        st.session_state.qa_pairs_data = None
    if 'qa_pairs_file_path' not in st.session_state:
        st.session_state.qa_pairs_file_path = None

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

def match_questions_to_answers(questions_data, qa_pairs_file, top_k=3, selected_indices=None):
    """Match extracted questions to Q&A pairs
    
    Args:
        questions_data: The extracted questions data
        qa_pairs_file: Path to the Q&A pairs file
        top_k: Number of top matches to return
        selected_indices: List of indices of selected Q&A pairs to use (None = use all)
    """
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
            
            # Filter Q&A items based on selected indices
            if selected_indices is not None and len(selected_indices) > 0:
                qa_items = [qa_items[i] for i in selected_indices if i < len(qa_items)]
                st.info(f"Using {len(qa_items)} selected Q&A pairs for matching")
            
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

def load_qa_pairs_for_selection(qa_pairs_file):
    """Load Q&A pairs for the selection interface"""
    try:
        with open(qa_pairs_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        qa_pairs = qa_data.get("qa_pairs", qa_data)
        if isinstance(qa_pairs, list):
            return qa_pairs
        return []
    except Exception as e:
        st.error(f"Error loading Q&A pairs: {str(e)}")
        return []

def render_qa_selection_page():
    """Render the Q&A pair selection page"""
    st.header("📚 Knowledge Base Selection")
    st.markdown("Select which question-answer pairs from the knowledge base should be used for matching.")
    
    # Q&A pairs file selection
    col1, col2 = st.columns([3, 1])
    with col1:
        qa_pairs_file = st.text_input(
            "Q&A Pairs File Path", 
            value=st.session_state.qa_pairs_file_path or "pdf-qa-generator/output/qa_pairs.json",
            help="Path to your Q&A pairs JSON file"
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        load_button = st.button("Load Q&A Pairs", type="primary")
    
    # Load Q&A pairs when button is clicked or path changes
    if load_button or (qa_pairs_file != st.session_state.qa_pairs_file_path):
        if os.path.exists(qa_pairs_file):
            qa_pairs = load_qa_pairs_for_selection(qa_pairs_file)
            if qa_pairs:
                st.session_state.qa_pairs_data = qa_pairs
                st.session_state.qa_pairs_file_path = qa_pairs_file
                # Initialize all as selected by default
                if st.session_state.selected_qa_indices is None:
                    st.session_state.selected_qa_indices = list(range(len(qa_pairs)))
                st.success(f"✅ Loaded {len(qa_pairs)} Q&A pairs")
            else:
                st.error("❌ No Q&A pairs found in the file")
        else:
            st.error("❌ File not found")
    
    # Display Q&A pairs for selection
    if st.session_state.qa_pairs_data:
        st.markdown("---")
        
        # Selection controls
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        with col1:
            if st.button("Select All"):
                st.session_state.selected_qa_indices = list(range(len(st.session_state.qa_pairs_data)))
                st.rerun()
        with col2:
            if st.button("Deselect All"):
                st.session_state.selected_qa_indices = []
                st.rerun()
        with col3:
            if st.button("Invert Selection"):
                current = set(st.session_state.selected_qa_indices or [])
                all_indices = set(range(len(st.session_state.qa_pairs_data)))
                st.session_state.selected_qa_indices = list(all_indices - current)
                st.rerun()
        with col4:
            selected_count = len(st.session_state.selected_qa_indices or [])
            total_count = len(st.session_state.qa_pairs_data)
            st.metric("Selected Q&A Pairs", f"{selected_count} / {total_count}")
        
        st.markdown("---")
        
        # Search/filter functionality
        search_term = st.text_input("🔍 Search Q&A pairs", placeholder="Enter keywords to filter questions or answers...")
        
        # Display Q&A pairs in a table with checkboxes
        st.subheader("Q&A Pairs")
        
        # Create a container for the scrollable area
        with st.container():
            # Create columns for the header
            header_cols = st.columns([0.5, 3, 4, 1])
            with header_cols[0]:
                st.markdown("**Select**")
            with header_cols[1]:
                st.markdown("**Question**")
            with header_cols[2]:
                st.markdown("**Answer**")
            with header_cols[3]:
                st.markdown("**Index**")
            
            st.markdown("---")
            
            # Filter Q&A pairs based on search term
            filtered_indices = []
            for i, qa_pair in enumerate(st.session_state.qa_pairs_data):
                if search_term:
                    search_lower = search_term.lower()
                    if (search_lower not in qa_pair.get('question', '').lower() and 
                        search_lower not in qa_pair.get('answer', '').lower()):
                        continue
                filtered_indices.append(i)
            
            # Display filtered Q&A pairs
            if filtered_indices:
                # Use a form to batch checkbox updates
                with st.form("qa_selection_form"):
                    new_selected = []
                    
                    for idx in filtered_indices:
                        qa_pair = st.session_state.qa_pairs_data[idx]
                        cols = st.columns([0.5, 3, 4, 1])
                        
                        with cols[0]:
                            is_selected = st.checkbox(
                                "",
                                value=idx in (st.session_state.selected_qa_indices or []),
                                key=f"qa_select_{idx}"
                            )
                            if is_selected:
                                new_selected.append(idx)
                        
                        with cols[1]:
                            # Display truncated question
                            question = qa_pair.get('question', '')
                            display_question = question[:150] + "..." if len(question) > 150 else question
                            st.markdown(f"<small>{display_question}</small>", unsafe_allow_html=True)
                        
                        with cols[2]:
                            # Display truncated answer
                            answer = qa_pair.get('answer', '')
                            display_answer = answer[:200] + "..." if len(answer) > 200 else answer
                            st.markdown(f"<small>{display_answer}</small>", unsafe_allow_html=True)
                        
                        with cols[3]:
                            st.markdown(f"<small>{idx}</small>", unsafe_allow_html=True)
                        
                        st.markdown("---")
                    
                    # Submit button
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.form_submit_button("Update Selection", type="primary"):
                            # Update selected indices
                            # Keep non-filtered selections and add new filtered selections
                            non_filtered = [i for i in (st.session_state.selected_qa_indices or []) 
                                           if i not in filtered_indices]
                            st.session_state.selected_qa_indices = non_filtered + new_selected
                            st.success(f"✅ Selection updated: {len(st.session_state.selected_qa_indices)} Q&A pairs selected")
            else:
                st.info("No Q&A pairs match your search criteria.")
        
        # Display selected Q&A pairs summary
        if st.session_state.selected_qa_indices:
            with st.expander(f"View Selected Q&A Pairs ({len(st.session_state.selected_qa_indices)})" ):
                for idx in st.session_state.selected_qa_indices[:10]:  # Show first 10
                    if idx < len(st.session_state.qa_pairs_data):
                        qa_pair = st.session_state.qa_pairs_data[idx]
                        st.markdown(f"**Q{idx + 1}:** {qa_pair.get('question', '')[:100]}...")
                if len(st.session_state.selected_qa_indices) > 10:
                    st.markdown(f"*... and {len(st.session_state.selected_qa_indices) - 10} more*")
    else:
        st.info("👆 Please load a Q&A pairs file to begin selection")

def render_qa_matching_page():
    """Render the main Q&A matching page"""
    # Header
    st.header("📋 Question-Answer Matching")
    st.markdown("Upload documents to extract questions and match them with selected answers from your knowledge base.")
    
    # Check if Q&A pairs are loaded and selected
    if st.session_state.qa_pairs_file_path and st.session_state.selected_qa_indices:
        qa_info = st.info(f"Using {len(st.session_state.selected_qa_indices)} selected Q&A pairs from {st.session_state.qa_pairs_file_path}")
    elif st.session_state.qa_pairs_file_path:
        st.warning("⚠️ No Q&A pairs selected. Please go to the Knowledge Base tab to select Q&A pairs.")
    else:
        st.warning("⚠️ No knowledge base loaded. Please go to the Knowledge Base tab to load and select Q&A pairs.")
    
    # Main content area (rest of the original main function content)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Document")
        
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
                if not st.session_state.qa_pairs_file_path:
                    st.error("❌ Please load a knowledge base first in the Knowledge Base tab")
                elif not st.session_state.selected_qa_indices:
                    st.error("❌ Please select at least one Q&A pair in the Knowledge Base tab")
                else:
                    # Save uploaded file temporarily
                    temp_file_path = save_uploaded_file(uploaded_file)
                    
                    if temp_file_path:
                        # Process document
                        questions_data = process_document(temp_file_path, uploaded_file.name)
                        
                        if questions_data:
                            st.session_state.processed_questions = questions_data
                            st.session_state.uploaded_file_name = uploaded_file.name
                            
                            # Match questions to answers using selected Q&A pairs
                            matched_results = match_questions_to_answers(
                                questions_data, 
                                st.session_state.qa_pairs_file_path, 
                                top_k=3,
                                selected_indices=st.session_state.selected_qa_indices
                            )
                            
                            if matched_results:
                                st.session_state.matched_results = matched_results
                                st.rerun()
    
    with col2:
        st.subheader("📊 Results")
        
        if st.session_state.matched_results:
            st.success(f"✅ Processed: {st.session_state.uploaded_file_name}")
            
            # Create results DataFrames
            summary_df, detailed_df = create_results_dataframe(st.session_state.matched_results)
            
            # Display summary
            st.markdown("### 📋 Summary - Best Matches")
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
                st.markdown("### 🔍 All Matches Detail")
                st.dataframe(
                    detailed_df,
                    use_container_width=True,
                    height=600
                )
            
            # Download options
            st.markdown("### 💾 Download Results")
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

def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Header
    st.title("📋 ESG Question-Answer Document Processor")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📚 Knowledge Base", "📋 Q&A Matching"])
    
    with tab1:
        render_qa_selection_page()
    
    with tab2:
        render_qa_matching_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. 📚 Go to the Knowledge Base tab to load and select Q&A pairs
    2. 📤 Upload your document (PDF, Excel, Word, PowerPoint) in the Q&A Matching tab
    3. 🔍 Questions are automatically extracted using AI
    4. 🎯 Questions are matched to your selected Q&A pairs using TF-IDF similarity
    5. 📊 View results in interactive tables
    6. 💾 Download results as JSON or Excel
    
    **Features:**
    - ✅ Select specific Q&A pairs from your knowledge base
    - 🔍 Search and filter Q&A pairs
    - 📋 View summary and detailed matching results
    - 📊 Export results in multiple formats
    """)

if __name__ == "__main__":
    main() 
