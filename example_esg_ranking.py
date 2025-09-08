from esg_answer_ranker import ESGAnswerRanker


def main() -> None:
    questions = [
        "Do you track Scope 1 and Scope 2 emissions?",
        "What are your targets for net zero and when will you achieve them?",
        "How do you ensure anti-corruption compliance across suppliers?",
        "Describe your data privacy and security practices.",
        "What steps are you taking to increase diversity and pay equity?",
    ]

    ranker = ESGAnswerRanker()
    results = ranker.rank_answer_candidates(questions, top_k=3)

    for i, (question, suggestions) in enumerate(zip(questions, results), start=1):
        print(f"Q{i}. {question}")
        for j, (answer, score) in enumerate(suggestions, start=1):
            print(f"  {j}) score={score:.3f}  candidate=\"{answer}\"")
        print()


if __name__ == "__main__":
    main()

