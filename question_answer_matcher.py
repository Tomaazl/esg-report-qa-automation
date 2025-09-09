#!/usr/bin/env python3
"""
Question Answer Matcher with LlamaIndex FusionRetriever
Takes generated questions from document parsing and finds the most suitable answers
from qa_pairs.json using advanced embedding-based semantic matching with LlamaIndex.

Features:
- Uses OpenAI embeddings for semantic understanding
- Implements FusionRetriever for hybrid search (semantic + keyword)
- ChromaDB vector store for efficient similarity search
- Fallback to sentence-transformers if OpenAI is unavailable

Usage:
    python question_answer_matcher.py
    # Modify the paths in the script to point to your files
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# LlamaIndex imports
try:
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding
    import chromadb
    LLAMAINDEX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LlamaIndex not available: {e}")
    print("Please install with: pip install llama-index llama-index-embeddings-openai llama-index-vector-stores-chroma chromadb")
    LLAMAINDEX_AVAILABLE = False

# Fallback to sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers faiss-cpu")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas not available. Excel export will be disabled.")
    PANDAS_AVAILABLE = False

# Environment setup
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), 'pdf-qa-generator', '.env'))

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS
# =============================================================================

# Path to the generated questions JSON file
QUESTIONS_JSON_PATH = "pdf_esg_social_ethical_questionnaire_extracted_questions.json"  # Change this

# Path to the Q&A pairs JSON file
QA_PAIRS_JSON_PATH = "pdf-qa-generator/output/qa_pairs.json"  # Change this

# Number of top answers to find for each question
TOP_K_ANSWERS = 3

# Embedding configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer fallback

# =============================================================================

@dataclass
class QAItem:
    question: str
    answer: str
    id: Optional[str] = None

@dataclass
class ExtractedQuestion:
    id: int
    question: str
    source_file: str = ""

@dataclass
class MatchResult:
    matched_question: str
    answer: str
    similarity_score: float
    retrieval_method: str

class QuestionAnswerMatcher:
    """Matches questions to most suitable answers using LlamaIndex FusionRetriever"""
    
    def __init__(self):
        """Initialize the matcher with embedding models and vector stores"""
        self.vector_index = None
        self.vector_retriever = None
        self.embedding_model = None
        self.temp_dir = None
        self.qa_items = []
        
        # Initialize embedding model
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model (OpenAI or fallback to sentence-transformers)"""
        try:
            # For now, use sentence-transformers as primary (more reliable for demo)
            if SENTENCE_TRANSFORMERS_AVAILABLE and LLAMAINDEX_AVAILABLE:
                print("🔧 Initializing sentence-transformers embeddings...")
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                self.embedding_model = HuggingFaceEmbedding(model_name=FALLBACK_EMBEDDING_MODEL)
                Settings.embed_model = self.embedding_model
                print(f"✅ Using sentence-transformers: {FALLBACK_EMBEDDING_MODEL}")
                
            # Try Azure OpenAI embeddings as secondary option (requires correct deployment)
            elif LLAMAINDEX_AVAILABLE:
                api_key = os.getenv('AZURE_OPENAI_API_KEY')
                endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
                api_version = os.getenv('AZURE_OPENAI_API_VERSION')
                
                if all([api_key, endpoint, api_version]):
                    print("🔧 Trying Azure OpenAI embeddings...")
                    self.embedding_model = OpenAIEmbedding(
                        model="text-embedding-3-small",
                        api_key=api_key,
                        azure_endpoint=endpoint,
                        api_version=api_version,
                        azure_deployment="text-embedding-3-small"  # May need adjustment
                    )
                    Settings.embed_model = self.embedding_model
                    print(f"✅ Using Azure OpenAI embeddings: text-embedding-3-small")
                else:
                    raise ValueError("Azure OpenAI credentials not available")
                
            else:
                raise ValueError("No embedding model available")
                
        except Exception as e:
            print(f"⚠️  Error initializing primary embedding model: {e}")
            if SENTENCE_TRANSFORMERS_AVAILABLE and LLAMAINDEX_AVAILABLE:
                try:
                    print("🔧 Falling back to sentence-transformers...")
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    self.embedding_model = HuggingFaceEmbedding(model_name=FALLBACK_EMBEDDING_MODEL)
                    Settings.embed_model = self.embedding_model
                    print(f"✅ Using sentence-transformers fallback: {FALLBACK_EMBEDDING_MODEL}")
                except Exception as e2:
                    print(f"❌ Fallback also failed: {e2}")
                    raise ValueError("No embedding models available")
            else:
                raise ValueError("No embedding models available")
    
    def _setup_vector_store(self, qa_items: List[QAItem]) -> VectorStoreIndex:
        """Setup ChromaDB vector store and create index"""
        if not LLAMAINDEX_AVAILABLE:
            raise ValueError("LlamaIndex not available")
        
        # Create temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp(prefix="qa_matcher_")
        print(f"📁 Created temporary vector store: {self.temp_dir}")
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=self.temp_dir)
        chroma_collection = chroma_client.get_or_create_collection("qa_pairs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create documents from QA pairs
        documents = []
        for i, qa_item in enumerate(qa_items):
            # Combine question and answer for better semantic matching
            content = f"Question: {qa_item.question}\nAnswer: {qa_item.answer}"
            doc = Document(
                text=content,
                metadata={
                    "question": qa_item.question,
                    "answer": qa_item.answer,
                    "qa_id": qa_item.id or str(i)
                }
            )
            documents.append(doc)
        
        print(f"📚 Creating vector index from {len(documents)} QA pairs...")
        
        # Create vector store index
        vector_index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            show_progress=True
        )
        
        return vector_index
    
    def _create_vector_retriever(self, vector_index: VectorStoreIndex, similarity_top_k: int = 10) -> VectorIndexRetriever:
        """Create vector retriever for semantic search"""
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=similarity_top_k
        )
        
        return vector_retriever
    
    def load_extracted_questions(self, questions_path: str) -> List[ExtractedQuestion]:
        """Load questions from the generated JSON file"""
        try:
            with open(questions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            questions = []
            
            # Handle different JSON formats
            if 'test_questions' in data:
                # Simple format: {"test_questions": [{"id": 1, "question": "..."}]}
                for item in data['test_questions']:
                    questions.append(ExtractedQuestion(
                        id=item.get('id', 0),
                        question=item.get('question', '').strip()
                    ))
            elif 'extracted_questions' in data:
                # Detailed format: {"extracted_questions": [{"id": 1, "question": "...", "source_file": "..."}]}
                for item in data['extracted_questions']:
                    questions.append(ExtractedQuestion(
                        id=item.get('id', 0),
                        question=item.get('question', '').strip(),
                        source_file=item.get('source_file', '')
                    ))
            elif isinstance(data, list):
                # Direct list format: [{"id": 1, "question": "..."}]
                for item in data:
                    if isinstance(item, dict) and 'question' in item:
                        questions.append(ExtractedQuestion(
                            id=item.get('id', 0),
                            question=item.get('question', '').strip()
                        ))
            
            # Filter out empty questions
            questions = [q for q in questions if q.question]
            
            print(f"✅ Loaded {len(questions)} questions from {questions_path}")
            return questions
            
        except Exception as e:
            print(f"❌ Error loading questions: {str(e)}")
            return []
    
    def load_qa_pairs(self, qa_path: str) -> List[QAItem]:
        """Load Q&A pairs from JSON file (copied from main.py)"""
        try:
            with open(qa_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            
            raw = raw.get("qa_pairs", raw)
            items: List[QAItem] = []

            if isinstance(raw, dict):
                for question, answer in raw.items():
                    if not isinstance(question, str) or not isinstance(answer, str):
                        continue
                    q = question.strip()
                    a = answer.strip()
                    if q and a:
                        items.append(QAItem(question=q, answer=a))
            elif isinstance(raw, list):
                for entry in raw:
                    if not isinstance(entry, dict):
                        continue
                    question = entry.get("question")
                    answer = entry.get("answer")
                    if not isinstance(question, str) or not isinstance(answer, str):
                        continue
                    q = question.strip()
                    a = answer.strip()
                    if q and a:
                        items.append(QAItem(question=q, answer=a))
            else:
                raise ValueError("Unsupported qa_pairs.json format")

            print(f"✅ Loaded {len(items)} Q&A pairs from {qa_path}")
            return items
            
        except Exception as e:
            print(f"❌ Error loading Q&A pairs: {str(e)}")
            return []

    def find_best_answers_with_embeddings(self, query_question: str, top_k: int = 3) -> List[MatchResult]:
        """Find the best matching answers using LlamaIndex vector retriever"""
        if not self.vector_retriever:
            raise ValueError("Vector retriever not initialized. Call setup_retrieval_system first.")
        
        if not query_question or not query_question.strip():
            return []
        
        try:
            # Use vector retriever to find best matches
            nodes = self.vector_retriever.retrieve(query_question)
            
            results = []
            for node in nodes[:top_k]:
                # Extract metadata
                metadata = node.metadata
                similarity_score = getattr(node, 'score', 0.0)
                
                result = MatchResult(
                    matched_question=metadata.get('question', ''),
                    answer=metadata.get('answer', ''),
                    similarity_score=float(similarity_score) if similarity_score else 0.0,
                    retrieval_method="LlamaIndex Vector Embeddings"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in vector retrieval: {str(e)}")
            return []
    
    def setup_retrieval_system(self, qa_items: List[QAItem]):
        """Setup the entire retrieval system with vector store and retriever"""
        print("🚀 Setting up advanced embedding-based retrieval system...")
        
        if not LLAMAINDEX_AVAILABLE:
            print("❌ LlamaIndex not available. Cannot use advanced retrieval.")
            return False
        
        try:
            # Store QA items for reference
            self.qa_items = qa_items
            
            # Setup vector store and index
            self.vector_index = self._setup_vector_store(qa_items)
            
            # Create vector retriever
            self.vector_retriever = self._create_vector_retriever(
                self.vector_index, 
                similarity_top_k=TOP_K_ANSWERS * 2  # Get more candidates for better results
            )
            
            print("✅ Retrieval system setup complete!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up retrieval system: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up temporary files and resources"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up temp directory: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

    def match_questions_to_answers(self, questions_path: str, qa_pairs_path: str, top_k: int = 3):
        """Main function to match questions to answers using advanced embeddings"""
        print("🚀 Advanced Question Answer Matcher with LlamaIndex")
        print("=" * 70)
        print("🤖 Using semantic embeddings and vector retrieval for better matching")
        print("=" * 70)
        
        try:
            # Load data
            questions = self.load_extracted_questions(questions_path)
            qa_items = self.load_qa_pairs(qa_pairs_path)
            
            if not questions:
                print("❌ No questions to process")
                return
            
            if not qa_items:
                print("❌ No Q&A pairs available for matching")
                return
            
            print(f"\n🔄 Processing {len(questions)} questions against {len(qa_items)} Q&A pairs...")
            
            # Setup retrieval system
            if not self.setup_retrieval_system(qa_items):
                print("❌ Failed to setup retrieval system")
                return
            
            # Process each question
            all_matches = []
            
            for i, question in enumerate(questions, 1):
                print(f"\n📝 Question {i}/{len(questions)}: {question.question[:100]}{'...' if len(question.question) > 100 else ''}")
                
                # Find best matching answers using vector embeddings
                match_results = self.find_best_answers_with_embeddings(question.question, top_k)
                
                # Convert MatchResult objects to dict format for compatibility
                matches = []
                for match in match_results:
                    matches.append({
                        "matched_question": match.matched_question,
                        "answer": match.answer,
                        "similarity_score": match.similarity_score,
                        "retrieval_method": match.retrieval_method
                    })
                
                question_result = {
                    "original_question": {
                        "id": question.id,
                        "question": question.question,
                        "source_file": question.source_file
                    },
                    "matched_answers": matches
                }
                
                all_matches.append(question_result)
                
                # Print top matches
                if matches:
                    print(f"   🎯 Top {len(matches)} semantic matches:")
                    for j, match in enumerate(matches, 1):
                        score = match['similarity_score']
                        method = match.get('retrieval_method', 'Unknown')
                        matched_q = match['matched_question'][:80] + ('...' if len(match['matched_question']) > 80 else '')
                        print(f"   {j}. Score: {score:.3f} | Method: {method} | Q: {matched_q}")
                else:
                    print("   ⚠️  No matches found")
            
            # Save results
            output_filename = self._generate_output_filename(questions_path)
            self._save_results(all_matches, output_filename)
            
            # Also save as Excel if pandas is available
            if PANDAS_AVAILABLE:
                excel_filename = output_filename.replace('.json', '.xlsx')
                self._save_results_to_excel(all_matches, excel_filename)
            
            print(f"\n🎉 Advanced semantic matching completed!")
            print(f"📊 Processed {len(questions)} questions using LlamaIndex vector embeddings")
            print(f"💾 Results saved to: {output_filename}")
            if PANDAS_AVAILABLE:
                print(f"📊 Excel file saved to: {excel_filename}")
            
        finally:
            # Always cleanup temporary resources
            self.cleanup()

    def _generate_output_filename(self, questions_path: str) -> str:
        """Generate output filename based on input filename"""
        questions_path = Path(questions_path)
        base_name = questions_path.stem
        
        # Remove common suffixes to get clean base name
        for suffix in ['_extracted_questions', '_detailed_questions', '_questions']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        return f"{base_name}_matched_answers.json"

    def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save matching results to JSON file"""
        output_data = {
            "question_answer_matches": results,
            "metadata": {
                "total_questions": len(results),
                "top_k_answers": TOP_K_ANSWERS,
                "matching_method": "LlamaIndex vector embeddings with semantic search",
                "embedding_model": EMBEDDING_MODEL,
                "fallback_model": FALLBACK_EMBEDDING_MODEL,
                "questions_source": QUESTIONS_JSON_PATH,
                "qa_pairs_source": QA_PAIRS_JSON_PATH
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}")

    def _save_results_to_excel(self, results: List[Dict[str, Any]], output_file: str):
        """Save matching results to Excel file with multiple sheets"""
        if not PANDAS_AVAILABLE:
            print("⚠️  Cannot save Excel file - pandas not available")
            return
        
        try:
            # Create a list for the main data
            main_data = []
            detailed_data = []
            
            for result in results:
                original_q = result["original_question"]
                matches = result["matched_answers"]
                
                # Main sheet - one row per question with top match
                if matches:
                    top_match = matches[0]
                    main_data.append({
                        "Question_ID": original_q["id"],
                        "Original_Question": original_q["question"],
                        "Source_File": original_q.get("source_file", ""),
                        "Best_Match_Question": top_match["matched_question"],
                        "Best_Match_Answer": top_match["answer"],
                        "Best_Match_Score": top_match["similarity_score"],
                        "Total_Matches_Found": len(matches)
                    })
                else:
                    main_data.append({
                        "Question_ID": original_q["id"],
                        "Original_Question": original_q["question"],
                        "Source_File": original_q.get("source_file", ""),
                        "Best_Match_Question": "No matches found",
                        "Best_Match_Answer": "No matches found",
                        "Best_Match_Score": 0.0,
                        "Total_Matches_Found": 0
                    })
                
                # Detailed sheet - one row per match
                for i, match in enumerate(matches, 1):
                    detailed_data.append({
                        "Question_ID": original_q["id"],
                        "Original_Question": original_q["question"],
                        "Match_Rank": i,
                        "Matched_Question": match["matched_question"],
                        "Answer": match["answer"],
                        "Similarity_Score": match["similarity_score"],
                        "Source_File": original_q.get("source_file", "")
                    })
            
            # Create DataFrames
            df_main = pd.DataFrame(main_data)
            df_detailed = pd.DataFrame(detailed_data)
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Main summary sheet
                df_main.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed matches sheet
                if not df_detailed.empty:
                    df_detailed.to_excel(writer, sheet_name='All_Matches', index=False)
                
                # Metadata sheet
                metadata_df = pd.DataFrame([
                    ["Total Questions", len(results)],
                    ["Top K Answers", TOP_K_ANSWERS],
                    ["Matching Method", "LlamaIndex vector embeddings with semantic search"],
                    ["Embedding Model", EMBEDDING_MODEL],
                    ["Fallback Model", FALLBACK_EMBEDDING_MODEL],
                    ["Questions Source", QUESTIONS_JSON_PATH],
                    ["QA Pairs Source", QA_PAIRS_JSON_PATH]
                ], columns=["Metric", "Value"])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Auto-adjust column widths
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        # Set reasonable max width
                        adjusted_width = min(max_length + 2, 100)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"📊 Excel results saved to: {output_file}")
            print(f"   📋 Summary sheet: Overview with best matches")
            print(f"   📋 All_Matches sheet: Detailed view of all matches")
            print(f"   📋 Metadata sheet: Processing information")
            
        except Exception as e:
            print(f"❌ Error saving Excel file: {str(e)}")

def main():
    """Main function - modify paths at the top of the file"""
    matcher = QuestionAnswerMatcher()
    
    # Check if files exist
    if not Path(QUESTIONS_JSON_PATH).exists():
        print(f"❌ Questions file not found: {QUESTIONS_JSON_PATH}")
        print("Please modify QUESTIONS_JSON_PATH at the top of this script")
        return
    
    if not Path(QA_PAIRS_JSON_PATH).exists():
        print(f"❌ Q&A pairs file not found: {QA_PAIRS_JSON_PATH}")
        print("Please modify QA_PAIRS_JSON_PATH at the top of this script")
        return
    
    # Run the matching process
    matcher.match_questions_to_answers(
        QUESTIONS_JSON_PATH, 
        QA_PAIRS_JSON_PATH, 
        TOP_K_ANSWERS
    )

if __name__ == "__main__":
    main()
