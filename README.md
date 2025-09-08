## ESG Answer Ranker

Suggest standardized ESG answer candidates for questionnaire questions with simple scoring.

### Install

No external dependencies. Requires Python 3.9+.

### Usage

```python
from esg_answer_ranker import ESGAnswerRanker

questions = [
    "Do you track Scope 1 and Scope 2 emissions?",
    "What are your targets for net zero?",
]

ranker = ESGAnswerRanker()
ranked = ranker.rank_answer_candidates(questions, top_k=3)

for q, candidates in zip(questions, ranked):
    print(q)
    for text, score in candidates:
        print(" -", score, text)
```

Or run the example:

```bash
python example_esg_ranking.py
```

### API

- `ESGAnswerRanker.rank_answer_candidates_for_question(question: str, top_k: int = 3) -> List[Tuple[str, float]]`
- `ESGAnswerRanker.rank_answer_candidates(questions: Sequence[str], top_k: int = 3) -> List[List[Tuple[str, float]]]`

Returns tuples of `(answer_text, score)` sorted by descending score.

### Notes

- The model uses keyword overlap plus surface text similarity (difflib). It is simple by design and dependency-free. Extend `DEFAULT_ANSWER_BANK` in `esg_answer_ranker.py` to add domain-specific answers/keywords.

## QA Retriever (Pure-Python TF-IDF)

A small Python utility that embeds knowledge base questions from `q_a.json` using a lightweight, pure-Python TF-IDF implementation (unigrams + bigrams) and retrieves the most similar answers for new questions.

### Install

No external dependencies required. Requires Python 3.8+.

### Prepare knowledge base

The knowledge base is a JSON file that can be either of the following shapes:

- List of objects:

```json
[
  {"question": "How do I reset my password?", "answer": "Use the Forgot password link."}
]
```

- Object mapping questions to answers:

```json
{
  "How do I reset my password?": "Use the Forgot password link."
}
```

A sample file is provided at `q_a.json`.

### Usage

- One-off query:

```bash
python main.py --kb q_a.json --query "How can I change my password?" --topk 3
```

- JSON output:

```bash
python main.py -k q_a.json -q "international shipping" -t 5 -j
```

- Interactive mode (no `--query`):

```bash
python main.py -k q_a.json -t 5
```

### Notes

- Uses TF-IDF with unigrams and bigrams, cosine similarity.
- Scores are in [0, 1].