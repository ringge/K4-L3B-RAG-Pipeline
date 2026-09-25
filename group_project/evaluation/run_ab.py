"""Reproduce the paired dense vs. dense+BM25/RRF evaluation.

Run from the repository root with ``.venv/bin/python -m group_project.evaluation.run_ab``.
The retrieval stage uses a temporary copy of Chroma so it does not modify the
tracked database. Generation and judging use the configured OpenAI-compatible
endpoint and fail explicitly if that endpoint is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src import task4_chunking_indexing as indexing
from src import task6_lexical_search as lexical
from src.task7_reranking import rerank_rrf
from src.task10_generation import (
    SYSTEM_PROMPT,
    TOP_P,
    TEMPERATURE,
    call_llm,
    format_context,
    reorder_for_llm,
)


ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "group_project/evaluation/golden_dataset.json"
OUTPUT = ROOT / "group_project/evaluation/ab_results.json"
TOP_K = 5
THRESHOLD = 0.3  # Recorded for comparison; fallback is disabled in both arms.
RRF_K = 60


def ngrams(text: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    return set(zip(*(tokens[i:] for i in range(size))))


def context_metrics(gold: str, results: list[dict]) -> tuple[float, float]:
    expected = ngrams(gold)
    if not expected:
        raise ValueError("Expected context has fewer than five words")
    found = [ngrams(result["content"]) for result in results]
    recall = len(expected.intersection(set().union(*found))) / len(expected)
    precision = sum(len(expected.intersection(chunk)) >= 3 for chunk in found) / TOP_K
    return recall, precision


def dense_results(collection, vector: list[float]) -> list[dict]:
    response = collection.query(
        query_embeddings=[vector],
        n_results=TOP_K * 2,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for item_id, content, metadata, distance in zip(
        response["ids"][0], response["documents"][0],
        response["metadatas"][0], response["distances"][0],
    ):
        results.append({
            "id": item_id,
            "content": content,
            "metadata": metadata,
            "score": max(0.0, 1.0 - float(distance)),
            "retrieval_method": "dense",
        })
    return results


def corpus_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "data/standardized").rglob("*.md")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save(data: dict) -> None:
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def retrieve_all(data: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="rag-ab-chroma-") as tmp:
        shutil.copytree(ROOT / "chroma_db", Path(tmp) / "chroma_db")
        indexing.CHROMA_DIR = Path(tmp) / "chroma_db"
        collection = indexing.get_collection()
        all_items = collection.get(include=["documents", "metadatas"])
        lexical.CORPUS[:] = [
            {"id": item_id, "content": content, "metadata": metadata}
            for item_id, content, metadata in zip(
                all_items["ids"], all_items["documents"], all_items["metadatas"]
            )
        ]
        data["indexed_chunks"] = len(lexical.CORPUS)
        for case in data["cases"]:
            start = time.perf_counter()
            vector = indexing.embed_texts([case["question"]])[0]
            dense = dense_results(collection, vector)
            dense_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            sparse = lexical.lexical_search(case["question"], top_k=TOP_K * 2)
            hybrid = rerank_rrf([dense, sparse], top_k=TOP_K, k=RRF_K)
            hybrid_extra_ms = (time.perf_counter() - start) * 1000
            for name, selected, latency in [
                ("A", dense[:TOP_K], dense_ms),
                ("B", hybrid, dense_ms + hybrid_extra_ms),
            ]:
                recall, precision = context_metrics(case["expected_context"], selected)
                case[name] = {
                    "retrieval_ms": round(latency, 2),
                    "context_recall": round(recall, 4),
                    "context_precision": round(precision, 4),
                    "ids": [item["id"] for item in selected],
                    "scores": [round(item["score"], 5) for item in selected],
                }
            print(case["id"], "A", case["A"]["context_recall"],
                  case["A"]["context_precision"], "B",
                  case["B"]["context_recall"], case["B"]["context_precision"], flush=True)
        data["retrieval_complete"] = True
        save(data)


def judge(client: OpenAI, question: str, answer: str, context: str) -> dict:
    rubric = (
        "Evaluate one RAG response. Return only a JSON object with numeric keys "
        "faithfulness and answer_relevance, each from 0 to 1, and short string "
        "keys faithfulness_reason and relevance_reason. Faithfulness: proportion "
        "of factual claims in ANSWER supported by CONTEXT; unsupported or "
        "contradicted claims score 0 for those claims. A refusal with available "
        "answer evidence scores 0. Answer relevance: how completely ANSWER "
        "addresses QUESTION, including requested entities and quantities. "
        "Ignore citation formatting for relevance. Apply the same rubric to all cases."
    )
    response = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"],
        temperature=0,
        messages=[
            {"role": "system", "content": rubric},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"},
        ],
    )
    raw = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Evaluator did not return JSON: {raw[:200]}")
    result = json.loads(match.group())
    for key in ("faithfulness", "answer_relevance"):
        value = result[key]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {key}: {value}")
    return result


def complete(data: dict) -> None:
    if not data.get("retrieval_complete"):
        raise ValueError("Run retrieval stage first")
    load_dotenv(ROOT / ".env")
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=45,
    )
    # Rehydrate the exact indexed passages by ID, without touching tracked Chroma.
    with tempfile.TemporaryDirectory(prefix="rag-ab-chroma-") as tmp:
        shutil.copytree(ROOT / "chroma_db", Path(tmp) / "chroma_db")
        indexing.CHROMA_DIR = Path(tmp) / "chroma_db"
        collection = indexing.get_collection()
        for case in data["cases"]:
            for name in ("A", "B"):
                arm = case[name]
                if "faithfulness" in arm:
                    continue
                records = collection.get(ids=arm["ids"], include=["documents", "metadatas"])
                by_id = dict(zip(records["ids"], zip(records["documents"], records["metadatas"])))
                chunks = [
                    {"id": item_id, "content": by_id[item_id][0],
                     "metadata": by_id[item_id][1]}
                    for item_id in arm["ids"]
                ]
                context = format_context(reorder_for_llm(chunks))
                message = f"Context:\n{context}\n\nQuestion: {case['question']}"
                start = time.perf_counter()
                answer = call_llm(SYSTEM_PROMPT, message)
                arm["generation_ms"] = round((time.perf_counter() - start) * 1000, 2)
                arm["answer"] = answer
                result = judge(client, case["question"], answer, context)
                arm.update(result)
                save(data)
                print(case["id"], name, "faithfulness", result["faithfulness"],
                      "relevance", result["answer_relevance"], flush=True)
    data["generation_complete"] = True
    save(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["retrieval", "complete"], required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if args.stage == "retrieval":
        gold = json.loads(GOLD.read_text(encoding="utf-8"))
        for case in gold:
            source = ROOT / case["source"]
            if case["expected_context"] not in source.read_text(encoding="utf-8"):
                raise ValueError(f"Gold context is absent from corpus: {case['id']}")
        data = {
            "date": time.strftime("%Y-%m-%d", time.gmtime()),
            "corpus_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "corpus_sha256": corpus_hash(),
            "dataset_sha256": hashlib.sha256(GOLD.read_bytes()).hexdigest(),
            "embedding_provider": os.environ.get("EMBEDDING_PROVIDER", "sentence_transformers"),
            "embedding_model": indexing.JINA_EMBEDDING_MODEL if os.environ.get("EMBEDDING_PROVIDER") == "jina" else os.environ.get("EMBEDDING_MODEL", indexing.EMBEDDING_MODEL),
            "generator_model": os.environ.get("OPENAI_MODEL"),
            "evaluator_model": os.environ.get("OPENAI_MODEL"),
            "top_k": TOP_K,
            "candidate_k": TOP_K * 2,
            "rrf_k": RRF_K,
            "threshold": THRESHOLD,
            "fallback_enabled": False,
            "generator_temperature": TEMPERATURE,
            "generator_top_p": TOP_P,
            "cases": gold,
        }
        retrieve_all(data)
    else:
        complete(json.loads(OUTPUT.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
