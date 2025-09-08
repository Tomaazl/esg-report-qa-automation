#!/usr/bin/env python3
"""
Question Answer Matcher
Takes generated questions from document parsing and finds the most suitable answers
from qa_pairs.json using TF-IDF similarity matching.

Usage:
    python question_answer_matcher.py
    # Modify the paths in the script to point to your files
"""

import json
import math
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable
from dataclasses import dataclass

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas not available. Excel export will be disabled.")
    PANDAS_AVAILABLE = False

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS
# =============================================================================

# Path to the generated questions JSON file
QUESTIONS_JSON_PATH = "pdf_esg_social_ethical_questionnaire_extracted_questions.json"  # Change this

# Path to the Q&A pairs JSON file
QA_PAIRS_JSON_PATH = "pdf-qa-generator/output/qa_pairs.json"  # Change this

# Number of top answers to find for each question
TOP_K_ANSWERS = 3

# =============================================================================

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)

@dataclass
class QAItem:
    question: str
    answer: str

@dataclass
class ExtractedQuestion:
    id: int
    question: str
    source_file: str = ""

class QuestionAnswerMatcher:
    """Matches questions to most suitable answers using TF-IDF similarity"""
    
    def __init__(self):
        """Initialize the matcher"""
        pass
    
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

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text (copied from main.py)"""
        text = text.lower()
        tokens = TOKEN_PATTERN.findall(text)
        return tokens

    def generate_ngrams(self, tokens: List[str], n: int) -> Iterable[str]:
        """Generate n-grams from tokens (copied from main.py)"""
        if n <= 1:
            for tok in tokens:
                yield tok
            return
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i:i + n])

    def make_features(self, text: str) -> List[str]:
        """Extract features (unigrams + bigrams) from text (copied from main.py)"""
        tokens = self.tokenize(text)
        return list(self.generate_ngrams(tokens, 1)) + list(self.generate_ngrams(tokens, 2))

    def build_encoder(self, questions: List[str]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """Build TF-IDF encoder (copied from main.py)"""
        # Compute document frequencies
        doc_count = len(questions)
        token_to_document_frequency: Dict[str, int] = {}
        for q in questions:
            seen = set(self.make_features(q))
            for tok in seen:
                token_to_document_frequency[tok] = token_to_document_frequency.get(tok, 0) + 1

        # IDF with smoothing
        token_to_idf: Dict[str, float] = {}
        for tok, df in token_to_document_frequency.items():
            idf = math.log((doc_count + 1) / (df + 1)) + 1.0
            token_to_idf[tok] = idf

        # Build TF-IDF vectors for documents
        doc_vectors: List[Dict[str, float]] = []
        for q in questions:
            features = self.make_features(q)
            if not features:
                doc_vectors.append({})
                continue
            term_counts: Dict[str, int] = {}
            for t in features:
                term_counts[t] = term_counts.get(t, 0) + 1
            max_tf = max(term_counts.values())
            vec: Dict[str, float] = {}
            for t, tf in term_counts.items():
                idf = token_to_idf.get(t, 0.0)
                vec[t] = (tf / max_tf) * idf
            doc_vectors.append(vec)

        return token_to_idf, doc_vectors

    def vectorize_query(self, query_text: str, token_to_idf: Dict[str, float]) -> Dict[str, float]:
        """Convert query text to TF-IDF vector (copied from main.py)"""
        features = self.make_features(query_text)
        if not features:
            return {}
        term_counts: Dict[str, int] = {}
        for t in features:
            term_counts[t] = term_counts.get(t, 0) + 1
        max_tf = max(term_counts.values())
        vec: Dict[str, float] = {}
        for t, tf in term_counts.items():
            idf = token_to_idf.get(t)
            if idf is None:
                continue
            vec[t] = (tf / max_tf) * idf
        return vec

    def cosine_similarity_sparse(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """Calculate cosine similarity between sparse vectors (copied from main.py)"""
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        dot = 0.0
        for k, va in a.items():
            vb = b.get(k)
            if vb is not None:
                dot += va * vb
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def find_best_answers(self, query_question: str, token_to_idf: Dict[str, float], 
                         doc_vectors: List[Dict[str, float]], qa_items: List[QAItem], 
                         top_k: int = 3) -> List[Dict[str, Any]]:
        """Find the best matching answers for a question (adapted from main.py)"""
        if not query_question or not query_question.strip():
            return []

        q_vec = self.vectorize_query(query_question, token_to_idf)
        if not q_vec:
            return []

        scores: List[Tuple[int, float]] = []
        for idx, d_vec in enumerate(doc_vectors):
            s = self.cosine_similarity_sparse(q_vec, d_vec)
            scores.append((idx, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        num_candidates = min(top_k, len(qa_items))
        results: List[Dict[str, Any]] = []
        for idx, score in scores[:num_candidates]:
            results.append({
                "matched_question": qa_items[idx].question,
                "answer": qa_items[idx].answer,
                "similarity_score": float(round(score, 6)),
            })
        return results

    def match_questions_to_answers(self, questions_path: str, qa_pairs_path: str, top_k: int = 3):
        """Main function to match questions to answers"""
        print("🚀 Question Answer Matcher")
        print("=" * 60)
        
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
        
        # Build TF-IDF encoder from Q&A questions
        qa_questions = [item.question for item in qa_items]
        token_to_idf, doc_vectors = self.build_encoder(qa_questions)
        
        # Process each question
        all_matches = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Question {i}/{len(questions)}: {question.question[:100]}{'...' if len(question.question) > 100 else ''}")
            
            # Find best matching answers
            matches = self.find_best_answers(
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
            
            # Print top matches
            if matches:
                print(f"   🎯 Top {len(matches)} matches:")
                for j, match in enumerate(matches, 1):
                    score = match['similarity_score']
                    matched_q = match['matched_question'][:80] + ('...' if len(match['matched_question']) > 80 else '')
                    print(f"   {j}. Score: {score:.3f} | Q: {matched_q}")
            else:
                print("   ⚠️  No matches found")
        
        # Save results
        output_filename = self._generate_output_filename(questions_path)
        self._save_results(all_matches, output_filename)
        
        # Also save as Excel if pandas is available
        if PANDAS_AVAILABLE:
            excel_filename = output_filename.replace('.json', '.xlsx')
            self._save_results_to_excel(all_matches, excel_filename)
        
        print(f"\n🎉 Processing completed!")
        print(f"📊 Processed {len(questions)} questions")
        print(f"💾 Results saved to: {output_filename}")
        if PANDAS_AVAILABLE:
            print(f"📊 Excel file saved to: {excel_filename}")

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
                "matching_method": "TF-IDF cosine similarity",
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
                    ["Matching Method", "TF-IDF cosine similarity"],
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
