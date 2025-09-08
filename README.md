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