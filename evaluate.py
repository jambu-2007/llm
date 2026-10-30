"""Evaluate both supported embedding models on the sample test cases."""

import json
from pathlib import Path

import pandas as pd

from embedder import MODEL_NAMES, embed
from similarity import cosine_similarity, rank

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"


def main() -> None:
    """Run evaluations and write detailed and summary CSV files."""
    with (DATA_DIR / "sentences.json").open(encoding="utf-8") as file:
        sentences = json.load(file)
    with (DATA_DIR / "test_cases.json").open(encoding="utf-8") as file:
        test_cases = json.load(file)

    sentence_by_id = {item["id"]: item for item in sentences}
    sentence_texts = [item["sentence"] for item in sentences]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_details: list[dict[str, object]] = []
    comparison: list[dict[str, object]] = []

    for model_name in MODEL_NAMES:
        print(f"\nEvaluating {model_name} (CPU)...")
        corpus_vectors = embed(sentence_texts, model_name)
        query_texts = [item["query"] for item in test_cases["queries"]]
        query_vectors = embed(query_texts, model_name)

        related_scores: list[float] = []
        for query_case, query_vector in zip(test_cases["queries"], query_vectors):
            for relevant_id in query_case["relevant_ids"]:
                related_scores.append(
                    cosine_similarity(
                        query_vector,
                        corpus_vectors[
                            next(
                                index
                                for index, item in enumerate(sentences)
                                if item["id"] == relevant_id
                            )
                        ],
                    )
                )

        unrelated_scores = [
            cosine_similarity(
                corpus_vectors[
                    next(
                        index
                        for index, item in enumerate(sentences)
                        if item["id"] == pair["sentence_id_a"]
                    )
                ],
                corpus_vectors[
                    next(
                        index
                        for index, item in enumerate(sentences)
                        if item["id"] == pair["sentence_id_b"]
                    )
                ],
            )
            for pair in test_cases["unrelated_pairs"]
        ]

        top1_hits = 0
        top3_hits = 0
        for query_case, query_vector in zip(test_cases["queries"], query_vectors):
            indices, scores = rank(query_vector, corpus_vectors)
            top_ids = [sentences[index]["id"] for index in indices[:3]]
            relevant_ids = query_case["relevant_ids"]
            hit_top1 = bool(top_ids and top_ids[0] in relevant_ids)
            hit_top3 = any(sentence_id in relevant_ids for sentence_id in top_ids)
            top1_hits += int(hit_top1)
            top3_hits += int(hit_top3)

            detail = {
                "model": model_name,
                "query_id": query_case["query_id"],
                "query": query_case["query"],
                "relevant_ids": ";".join(relevant_ids),
                "top_1_id": top_ids[0] if top_ids else "",
                "top_1_score": scores[0] if scores else 0.0,
                "top_3_ids": ";".join(top_ids),
                "top_1_correct": hit_top1,
                "top_3_correct": hit_top3,
            }
            all_details.append(detail)

            if not hit_top3:
                nearest = (
                    sentence_by_id[top_ids[0]]["sentence"] if top_ids else "No result"
                )
                print(
                    f"Failure {query_case['query_id']}: no expected ID "
                    f"{relevant_ids} appeared in the Top 3. "
                    f"{query_case['explanation']} Nearest result: {nearest}"
                )

        query_count = len(test_cases["queries"])
        summary = {
            "model": model_name,
            "average_related_cosine": sum(related_scores) / len(related_scores),
            "average_unrelated_cosine": sum(unrelated_scores) / len(unrelated_scores),
            "top_1_accuracy": top1_hits / query_count,
            "top_3_accuracy": top3_hits / query_count,
        }
        comparison.append(summary)
        print(
            f"Related average: {summary['average_related_cosine']:.4f}; "
            f"unrelated average: {summary['average_unrelated_cosine']:.4f}; "
            f"Top-1: {summary['top_1_accuracy']:.1%}; "
            f"Top-3: {summary['top_3_accuracy']:.1%}"
        )

    pd.DataFrame(all_details).to_csv(
        RESULTS_DIR / "evaluation_details.csv", index=False
    )
    pd.DataFrame(comparison).to_csv(
        RESULTS_DIR / "model_comparison.csv", index=False
    )
    print(f"\nSaved evaluation CSV files in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
