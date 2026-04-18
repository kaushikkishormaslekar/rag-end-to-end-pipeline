from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_rag import AdvancedRAGEngine


def load_golden_set(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Golden test set must be a JSON array.")
    return data


def _keyword_hit_count(text: str, expected_keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for kw in expected_keywords if kw.lower() in lowered)


def evaluate_retrieval(
    engine: AdvancedRAGEngine,
    golden_cases: list[dict[str, Any]],
    k: int = 5,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []

    for case in golden_cases:
        query = str(case["query"])
        expected_chunk_ids = set(case.get("expected_chunk_ids", []))
        expected_keywords = case.get("expected_keywords", [])

        retrieved = engine.retrieve(query, top_k=k)
        results = retrieved.get("results", [])

        hit_positions: list[int] = []
        keyword_hits = 0

        for rank, item in enumerate(results, start=1):
            chunk_text = str(item.get("text", ""))
            chunk_id = str(item.get("chunk_id", ""))

            if expected_chunk_ids and chunk_id in expected_chunk_ids:
                hit_positions.append(rank)
            if expected_keywords:
                keyword_hits += int(_keyword_hit_count(chunk_text, expected_keywords) > 0)

        recall_at_k = 0.0
        mrr = 0.0

        if expected_chunk_ids:
            recall_at_k = 1.0 if hit_positions else 0.0
            mrr = 1.0 / min(hit_positions) if hit_positions else 0.0
        elif expected_keywords:
            recall_at_k = 1.0 if keyword_hits > 0 else 0.0
            mrr = 1.0 / 1.0 if keyword_hits > 0 else 0.0

        rows.append(
            {
                "case_id": case.get("id"),
                "query": query,
                "recall_at_k": recall_at_k,
                "mrr": mrr,
                "retrieval_confidence": retrieved.get("confidence", 0.0),
                "retrieved_count": len(results),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "mean_recall_at_k": float(df["recall_at_k"].mean()) if not df.empty else 0.0,
        "mean_mrr": float(df["mrr"].mean()) if not df.empty else 0.0,
        "mean_retrieval_confidence": float(df["retrieval_confidence"].mean()) if not df.empty else 0.0,
    }
    return df, summary


def evaluate_answers(
    engine: AdvancedRAGEngine,
    golden_cases: list[dict[str, Any]],
    llm_callable,
    k: int = 5,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []

    for case in golden_cases:
        query = str(case["query"])
        expected_keywords = case.get("expected_keywords", [])

        response = engine.answer(query=query, llm_callable=llm_callable, top_k=k)
        answer = str(response.get("answer", ""))
        abstained = bool(response.get("abstained", False))
        citations = response.get("citations", [])

        keyword_support = _keyword_hit_count(answer, expected_keywords) if expected_keywords else 0
        precision_proxy = 0.0
        if expected_keywords:
            precision_proxy = keyword_support / max(len(expected_keywords), 1)

        rows.append(
            {
                "case_id": case.get("id"),
                "query": query,
                "abstained": abstained,
                "confidence": response.get("confidence", 0.0),
                "citation_count": len(citations),
                "faithfulness_proxy": float(precision_proxy),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "abstain_rate": float(df["abstained"].mean()) if not df.empty else 0.0,
        "mean_confidence": float(df["confidence"].mean()) if not df.empty else 0.0,
        "mean_faithfulness_proxy": float(df["faithfulness_proxy"].mean()) if not df.empty else 0.0,
        "mean_citation_count": float(df["citation_count"].mean()) if not df.empty else 0.0,
    }
    return df, summary


def build_dashboard(
    retrieval_df: pd.DataFrame,
    answer_df: pd.DataFrame,
    output_dir: str | Path = "reports",
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}

    if not retrieval_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        retrieval_df[["recall_at_k", "mrr"]].mean().plot(kind="bar", ax=ax, color=["#2E86AB", "#F18F01"])
        ax.set_title("Retrieval Metrics")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        fig.tight_layout()
        path = output / "retrieval_metrics.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files["retrieval_chart"] = str(path)

    if not answer_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        cols = ["mean_confidence", "mean_faithfulness_proxy", "mean_citation_count"]
        values = [
            float(answer_df["confidence"].mean()),
            float(answer_df["faithfulness_proxy"].mean()),
            float(answer_df["citation_count"].mean()),
        ]
        scaled_citation = min(values[2] / max(answer_df["citation_count"].max(), 1), 1.0)
        y_vals = [values[0], values[1], scaled_citation]
        ax.bar(cols, y_vals, color=["#5C4B8A", "#2A9D8F", "#E76F51"])
        ax.set_ylim(0, 1)
        ax.set_title("Answer Quality Dashboard (normalized)")
        fig.tight_layout()
        path = output / "answer_metrics.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files["answer_chart"] = str(path)

    retrieval_csv = output / "retrieval_eval.csv"
    answer_csv = output / "answer_eval.csv"
    retrieval_df.to_csv(retrieval_csv, index=False)
    answer_df.to_csv(answer_csv, index=False)
    files["retrieval_csv"] = str(retrieval_csv)
    files["answer_csv"] = str(answer_csv)

    return files
