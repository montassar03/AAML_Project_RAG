# RAG Evaluation on HotpotQA

## Overview

This project implements and evaluates a **Retrieval-Augmented Generation (RAG)** pipeline on the HotpotQA dataset.
The main goal is to study how different retrieval configurations (chunk size and top-k) affect system behavior.

The repository is organized to clearly separate:

* data preparation
* indexing
* generation & evaluation
* experiment outputs
* analysis

---

## Project Structure

```text
rag_project/
│
├── 500_Sample/
├── data/
├── logs/
├── notebooks/
├── Output_1B/
├── scripts/
└── README.md
```

---

## Folder Details

### 1. `500_Sample/`

Contains the **pilot experiment (500 samples)**:

* used for debugging and fast experimentation
* includes outputs generated on the small subset
* helps validate the pipeline before scaling to full dataset

---

### 2. `data/`

This folder contains all **intermediate and processed data** used by the pipeline.

Typical contents:

* **Dataset**


* **Corpus**

  * processed context passages extracted from HotpotQA

* **Chunks**

  * `hotpotqa_chunks_<size>.json`
  * token-based chunks for each chunk size

* **Embeddings**

  * `embeddings_<size>.npy`
  * vector representations of chunks

* **FAISS Indexes**

  * `faiss_index_<size>.index`
  * used for similarity search

* **Metadata**

  * `chunks_metadata_<size>.json`
  * mapping between chunks and original documents

-> In summary:
`data/` contains the **output of every preprocessing step** (corpus → chunks → embeddings → index)

---

### 3. `logs/`

Contains **SLURM job logs**:

* `.log` files for each experiment run
* used for:

  * debugging errors
  * monitoring runtime
  * checking GPU usage and failures

---

### 4. `notebooks/`

Contains all **development and analysis notebooks**:

```text
00_environment_check.ipynb
01_Data_Load_Preparation.ipynb
02_build_corpus.ipynb
03_Chunking.ipynb
04_embeddings.ipynb
05_Build_Index.ipynb
06_generation_evaluation.ipynb
07_All_experiment_summaries.ipynb
08_BERTScore.ipynb
```

#### Notebook roles

* **00** – environment setup and dependency checks
* **01** – load and inspect dataset
* **02** – build corpus from raw context
* **03** – chunking strategy implementation
* **04** – generate embeddings
* **05** – build FAISS index
* **06** – full RAG pipeline (retrieval + generation + evaluation)
* **07** – aggregate experiment results
* **08** – compute BERTScore

---

### Important note

Inside `notebooks/`:

* `.ipynb` → used for development and testing
* `.py` → production scripts for SLURM

Specifically:

```text
06_full_generation_evaluation.py
06_full_generation_evaluation_TopK_variation.py
```

👉 These `.py` files are **converted versions of Notebook 06**,
because SLURM requires execution as a Python script.

---

### 5. `Output_1B/`

Contains **final experiment outputs using LLaMA 3.2 1B model**.

Structure:

```text
Output_1B/
  chunk_<size>_topk_<k>/
    predictions.jsonl
    summary.json
```

#### Files:

* `predictions.jsonl`

  * per-question results:

    * question
    * prediction
    * ground truth
    * metrics

* `summary.json`

  * aggregated metrics:

    * EM
    * F1
    * Recall@k
    * latency

Additionally:

* aggregated CSV files (experiment summaries)
* BERTScore results

---

### 6. `scripts/`

Contains **SLURM job scripts** used to run experiments on GPU.

* defines:

  * job configuration
  * GPU allocation
  * runtime limits
* executes `.py` experiment scripts from `notebooks/`

---

## Execution Flow

The pipeline is executed in the following order:

1. Prepare data (`01–02`)
2. Chunk corpus (`03`)
3. Generate embeddings (`04`)
4. Build FAISS index (`05`)
5. Run RAG pipeline (`06`)
6. Aggregate results (`07`)
7. Compute BERTScore (`08`)

---

## Notes

* Development is done in notebooks for flexibility
* Final experiments are executed via `.py` scripts on SLURM
* Intermediate outputs are stored to avoid recomputation
* The project is modular and reproducible

---

## Author

Montassar Arfaoui.
