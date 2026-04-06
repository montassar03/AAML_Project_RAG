#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment script 1:
- vary chunk size: 32, 128, 256
- keep top_k fixed at 5

This script is intentionally kept close to the testing notebook logic:
- same retrieval idea
- same prompt style
- same generation function
- same evaluation metrics

Outputs are saved under:
    /home/a/arfaoui/rag_project/Output/chunk_<size>_topk_<k>/
"""

import os
import json
import time
import string
import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Paths and fixed experiment settings
# ============================================================

DATA_DIR = Path("/home/a/arfaoui/rag_project/data")
OUTPUT_DIR = Path("/home/a/arfaoui/rag_project/Output")

DATASET_PATH = DATA_DIR / "hotpotqa_sample_500.json"

CHUNK_SIZES = [32, 128, 256]
FIXED_TOP_K = 5

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_NAME = "meta-llama/Llama-3.2-3B-Instruct"


# ============================================================
# Utility functions
# ============================================================

def ensure_dir(path: Path):
    """Create directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    """Save one JSON object."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, records):
    """Save one record per line in JSONL format."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_csv(path: Path, records):
    """
    Save records as CSV.

    Lists are flattened to readable strings so the file is easy to inspect
    later in pandas or Excel.
    """
    flat_records = []

    for record in records:
        row = record.copy()

        if "retrieved_titles" in row and isinstance(row["retrieved_titles"], list):
            row["retrieved_titles"] = " | ".join(row["retrieved_titles"])

        flat_records.append(row)

    df = pd.DataFrame(flat_records)
    df.to_csv(path, index=False)


# ============================================================
# Metric functions
# ============================================================

def normalize_text(text):
    """
    Normalize text for fair comparison.

    - lowercase
    - remove punctuation
    - strip whitespace
    """

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()

    return text


def compute_em(prediction, correct_answer):
    """Exact match after normalization."""
    return int(normalize_text(prediction) == normalize_text(correct_answer))


def compute_f1(prediction, correct_answer):
    """
    Token-level F1 score between normalized prediction and gold answer.
    """
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(correct_answer).split()

    # Handle empty cases safely
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common_tokens = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


# ============================================================
# Retrieval metrics
# ============================================================

def compute_recall_at_k_answer(retrieved_chunks, correct_answer):
    """
    Simple answer-based Recall@k.

    Returns 1 if the normalized answer string appears in at least one
    retrieved chunk, else 0.
    """
    gold_norm = normalize_text(correct_answer)

    for chunk in retrieved_chunks:
        chunk_text_norm = normalize_text(chunk["text"])
        if gold_norm in chunk_text_norm:
            return 1

    return 0


# Improved Recall@k based on HotpotQA supporting facts (titles)
# -----------------------------------------------------------
# Unlike the simple answer-based Recall@k (which checks if the answer string
# appears in retrieved chunks), this metric evaluates whether the retriever
# successfully retrieved the correct supporting documents.
#
# In HotpotQA, each question is associated with one or more supporting titles
# (documents) required to answer the question. We consider retrieval successful
# if at least one of the retrieved chunks comes from one of these gold titles.
#
# This provides a more meaningful measure of retrieval quality, especially for
# multi-hop questions, where correct reasoning depends on retrieving the right
# sources rather than just matching the answer string.
def compute_recall_at_k_supporting_titles(retrieved_chunks, supporting_facts):
    """
    Supporting-title Recall@k.

    Returns 1 if at least one retrieved chunk title matches a gold
    supporting title, else 0.
    """
    gold_titles = set(supporting_facts["title"])

    for chunk in retrieved_chunks:
        if chunk.get("title") in gold_titles:
            return 1

    return 0


# ============================================================
# Retrieval and generation functions
# ============================================================

def retrieve_top_k(question, index, metadata, embed_model, k=5):
    """
    Given a question, retrieve top-k most similar chunks.

    Returns:
    - retrieved_chunks: list of metadata entries
    - scores: similarity scores
    """
    # BGE models expect "query: " prefix for queries
    query_text = "query: " + question

    # Encode query
    query_embedding = embed_model.encode([query_text])

    # Convert to float32 for FAISS
    query_embedding = np.array(query_embedding).astype("float32")

    # Search in FAISS index
    distances, indices = index.search(query_embedding, k)

    # Retrieve corresponding chunks
    retrieved_chunks = [metadata[idx] for idx in indices[0]]
    scores = distances[0]

    return retrieved_chunks, scores


def build_retrieved_context(retrieved_chunks):
    """
    Concatenate retrieved chunks into one context string for the prompt.
    """
    parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        parts.append(f"[Chunk {i}] Title: {chunk['title']}\n{chunk['text']}")

    return "\n\n".join(parts)


def build_prompt(question, context):
    """
    Fixed prompt from the notebook.

    We keep this prompt unchanged because it gave the best results
    during notebook testing.
    """
    prompt = f"""Context:
{context}

Question: {question}

Short answer (just the key phrase, no full sentences):"""
    return prompt


def generate_answer(prompt, tokenizer, model, max_new_tokens=32):
    """
    Generate a short answer from the LLM.

    Decode only the newly generated continuation, not the entire prompt.
    Keep only the first line.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode only the newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Keep only the first line
    answer = answer.split("\n")[0].strip()

    return answer


# ============================================================
# Loading resources
# ============================================================

def load_chunk_resources(chunk_size):
    """
    Load the FAISS index and chunk metadata for a given chunk size.
    """
    index_path = DATA_DIR / f"faiss_index_{chunk_size}.index"
    metadata_path = DATA_DIR / f"chunks_metadata_{chunk_size}.json"

    index = faiss.read_index(str(index_path))
    metadata = load_json(metadata_path)

    return index, metadata


def load_dataset(max_examples=None):
    """
    Load the saved HotpotQA sample.

    If max_examples is provided, only use the first N examples.
    """
    data = load_json(DATASET_PATH)

    if max_examples is not None:
        data = data[:max_examples]

    return data


# ============================================================
# Single example and experiment loop
# ============================================================

def run_single_example(example, chunk_size, top_k, index, metadata, embed_model, llm_tokenizer, llm_model):
    """
    Run the full RAG pipeline for one example.

    Returns a dictionary with prediction, metrics, settings, and titles.
    """
    question = example["question"]
    correct_answer = example["answer"]

    # Total pipeline latency for this example
    start_time = time.time()

    # Retrieval
    retrieved_chunks, scores = retrieve_top_k(
        question=question,
        index=index,
        metadata=metadata,
        embed_model=embed_model,
        k=top_k
    )

    # Build context and prompt
    context = build_retrieved_context(retrieved_chunks)
    prompt = build_prompt(question, context)

    # Generation
    prediction = generate_answer(
        prompt=prompt,
        tokenizer=llm_tokenizer,
        model=llm_model
    )

    latency = time.time() - start_time

    # Metrics
    em = compute_em(prediction, correct_answer)
    f1 = compute_f1(prediction, correct_answer)
    recall_answer = compute_recall_at_k_answer(retrieved_chunks, correct_answer)
    recall_support_titles = compute_recall_at_k_supporting_titles(
        retrieved_chunks,
        example["supporting_facts"]
    )

    # Keep retrieved titles for later inspection
    retrieved_titles = [chunk["title"] for chunk in retrieved_chunks]

    return {
        "question": question,
        "correct_answer": correct_answer,
        "prediction": prediction,
        "EM": em,
        "F1": f1,
        "Recall@k_answer": recall_answer,
        "Recall@k_support_titles": recall_support_titles,
        "latency": latency,
        "chunk_size": chunk_size,
        "top_k": top_k,
        "retrieved_titles": retrieved_titles
    }


def summarize_results(results):
    """
    Compute aggregate metrics for one experiment run.
    """
    em_values = [r["EM"] for r in results]
    f1_values = [r["F1"] for r in results]
    recall_answer_values = [r["Recall@k_answer"] for r in results]
    recall_support_values = [r["Recall@k_support_titles"] for r in results]
    latencies = [r["latency"] for r in results]

    summary = {
        "num_examples": len(results),
        "avg_EM": float(np.mean(em_values)) if results else 0.0,
        "avg_F1": float(np.mean(f1_values)) if results else 0.0,
        "avg_Recall@k_answer": float(np.mean(recall_answer_values)) if results else 0.0,
        "avg_Recall@k_support_titles": float(np.mean(recall_support_values)) if results else 0.0,
        "latency_p50": float(np.percentile(latencies, 50)) if results else 0.0,
        "latency_p95": float(np.percentile(latencies, 95)) if results else 0.0,
    }

    return summary


def run_one_config(chunk_size, top_k, dataset, embed_model, llm_tokenizer, llm_model):
    """
    Run one full experiment configuration and save outputs.
    """
    print("=" * 80)
    print(f"Running configuration: chunk_size={chunk_size}, top_k={top_k}")

    # Load the correct resources for this chunk size
    index, metadata = load_chunk_resources(chunk_size)

    # Output folder for this configuration
    run_dir = OUTPUT_DIR / f"chunk_{chunk_size}_topk_{top_k}"
    ensure_dir(run_dir)

    # Save config for reproducibility
    config = {
        "chunk_size": chunk_size,
        "top_k": top_k,
        "dataset_path": str(DATASET_PATH),
        "num_examples": len(dataset),
        "embed_model_name": EMBED_MODEL_NAME,
        "llm_name": LLM_NAME,
    }
    save_json(run_dir / "config.json", config)

    # Run full loop
    results = []

    for i, example in enumerate(dataset, start=1):
        print(f"[{i}/{len(dataset)}] question: {example['question'][:80]}")

        result = run_single_example(
            example=example,
            chunk_size=chunk_size,
            top_k=top_k,
            index=index,
            metadata=metadata,
            embed_model=embed_model,
            llm_tokenizer=llm_tokenizer,
            llm_model=llm_model
        )

        results.append(result)

    # Save detailed results
    save_jsonl(run_dir / "predictions.jsonl", results)
    save_csv(run_dir / "predictions.csv", results)

    # Save summary
    summary = summarize_results(results)
    summary["chunk_size"] = chunk_size
    summary["top_k"] = top_k

    save_json(run_dir / "summary.json", summary)

    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return results, summary


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RAG experiment: chunk size study with fixed top_k=5")
    parser.add_argument("--max-examples", type=int, default=None, help="Use only the first N examples")
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_dir(OUTPUT_DIR)

    print("Loading dataset...")
    dataset = load_dataset(max_examples=args.max_examples)
    print(f"Loaded {len(dataset)} examples")

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"Embedding model loaded: {EMBED_MODEL_NAME}")

    print("Loading LLM tokenizer and model...")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    print(f"LLM loaded: {LLM_NAME}")
    print(f"Model device: {llm_model.device}")

    # Script 1: vary chunk size, keep top_k fixed at 5
    all_summaries = []

    for chunk_size in CHUNK_SIZES:
        _, summary = run_one_config(
            chunk_size=chunk_size,
            top_k=FIXED_TOP_K,
            dataset=dataset,
            embed_model=embed_model,
            llm_tokenizer=llm_tokenizer,
            llm_model=llm_model
        )
        all_summaries.append(summary)

    # Save one global summary file for the whole script
    save_json(OUTPUT_DIR / "summary_script1_chunksize_study.json", all_summaries)
    pd.DataFrame(all_summaries).to_csv(
        OUTPUT_DIR / "summary_script1_chunksize_study.csv",
        index=False
    )

    print("\nAll configurations completed.")


if __name__ == "__main__":
    main()
