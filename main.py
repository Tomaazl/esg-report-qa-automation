#!/usr/bin/env python3

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Iterable


TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class QAItem:
	question: str
	answer: str


def load_q_a(knowledge_base_path: str) -> List[QAItem]:
	with open(knowledge_base_path, "r", encoding="utf-8") as f:
		raw = json.load(f)
	raw = raw.get("qa_pairs", raw)
	items: List[QAItem] = []

	if isinstance(raw, dict):
		for question, answer in raw.items():
			if not isinstance(question, str):
				continue
			if not isinstance(answer, str):
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
		raise ValueError("Unsupported qa_pairs.json format. Use either {question: answer} or a list of {question, answer} objects.")

	return items


def tokenize(text: str) -> List[str]:
	text = text.lower()
	tokens = TOKEN_PATTERN.findall(text)
	return tokens


def generate_ngrams(tokens: List[str], n: int) -> Iterable[str]:
	if n <= 1:
		for tok in tokens:
			yield tok
		return
	for i in range(len(tokens) - n + 1):
		yield " ".join(tokens[i:i + n])


def make_features(text: str) -> List[str]:
	# Unigrams + bigrams
	tokens = tokenize(text)
	return list(generate_ngrams(tokens, 1)) + list(generate_ngrams(tokens, 2))


def build_encoder(questions: List[str]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
	# Compute document frequencies
	doc_count = len(questions)
	token_to_document_frequency: Dict[str, int] = {}
	for q in questions:
		seen = set(make_features(q))
		for tok in seen:
			token_to_document_frequency[tok] = token_to_document_frequency.get(tok, 0) + 1

	# IDF with smoothing
	token_to_idf: Dict[str, float] = {}
	for tok, df in token_to_document_frequency.items():
		# idf = log((N + 1) / (df + 1)) + 1 to keep > 0
		idf = math.log((doc_count + 1) / (df + 1)) + 1.0
		token_to_idf[tok] = idf

	# Build TF-IDF vectors for documents
	doc_vectors: List[Dict[str, float]] = []
	for q in questions:
		features = make_features(q)
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
			# Normalize TF by max TF to reduce bias for longer texts
			vec[t] = (tf / max_tf) * idf
		doc_vectors.append(vec)

	return token_to_idf, doc_vectors


def vectorize_query(query_text: str, token_to_idf: Dict[str, float]) -> Dict[str, float]:
	features = make_features(query_text)
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


def cosine_similarity_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
	if not a or not b:
		return 0.0
	# Compute dot product over intersection of keys
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


def search_candidates(
	query_text: str,
	token_to_idf: Dict[str, float],
	doc_vectors: List[Dict[str, float]],
	items: List[QAItem],
	top_k: int = 5,
) -> List[Dict[str, Any]]:
	if not query_text or not query_text.strip():
		return []

	q_vec = vectorize_query(query_text, token_to_idf)
	if not q_vec:
		return []

	scores: List[Tuple[int, float]] = []
	for idx, d_vec in enumerate(doc_vectors):
		s = cosine_similarity_sparse(q_vec, d_vec)
		scores.append((idx, s))

	scores.sort(key=lambda x: x[1], reverse=True)
	num_candidates = min(top_k, len(items))
	results: List[Dict[str, Any]] = []
	for idx, score in scores[:num_candidates]:
		results.append(
			{
				"question": items[idx].question,
				"answer": items[idx].answer,
				"score": float(round(score, 6)),
			}
		)
	return results


def print_results(results: List[Dict[str, Any]], json_output: bool) -> None:
	if json_output:
		print(json.dumps(results, ensure_ascii=False, indent=2))
		return

	if not results:
		print("No candidates found.")
		return

	for i, r in enumerate(results, start=1):
		print(f"{i}. score={r['score']}")
		print(f"   Q: {r['question']}")
		print(f"   A: {r['answer']}\n")


def parse_args(argv: List[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Embed KB questions with pure-Python TF-IDF and retrieve best answer candidates for input queries.",
	)
	parser.add_argument(
		"-k",
		"--kb",
		type=str,
		default="pdf-qa-generator/output/qa_pairs.json",
		help="Path to knowledge base JSON (list[{question, answer}] or {question: answer}).",
	)
	parser.add_argument(
		"-q",
		"--query",
		type=str,
		default=None,
		help="Query string. If omitted, runs in interactive mode.",
	)
	parser.add_argument(
		"-t",
		"--topk",
		type=int,
		default=5,
		help="Number of top candidates to return.",
	)
	parser.add_argument(
		"-j",
		"--json-output",
		action="store_true",
		help="Output results as JSON.",
	)
	return parser.parse_args(argv)


def main(argv: List[str]) -> int:
	args = parse_args(argv)

	try:
		items = load_q_a(args.kb)
	except FileNotFoundError:
		print(f"Knowledge base not found at: {args.kb}", file=sys.stderr)
		return 2
	except Exception as exc:
		print(f"Failed to load knowledge base: {exc}", file=sys.stderr)
		return 2

	if not items:
		print("Knowledge base is empty or invalid.", file=sys.stderr)
		return 2

	questions = [it.question for it in items]
	token_to_idf, doc_vectors = build_encoder(questions)

	if args.query:
		results = search_candidates(args.query, token_to_idf, doc_vectors, items, top_k=args.topk)
		print_results(results, args.json_output)
		return 0

	print("Interactive mode. Type your question (Ctrl-D to exit).\n")
	try:
		while True:
			try:
				query_text = input("? ").strip()
			except EOFError:
				print()
				break
			if not query_text:
				continue
			results = search_candidates(query_text, token_to_idf, doc_vectors, items, top_k=args.topk)
			print_results(results, args.json_output)
	except KeyboardInterrupt:
		print()

	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))