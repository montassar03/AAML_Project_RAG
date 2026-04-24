# RAG Evaluation on HotpotQA

## Overview

This project implements and evaluates a **Retrieval-Augmented Generation (RAG)** pipeline on the HotpotQA dataset.
The main objective is to study how different retrieval configurations, especially **chunk size** and **top-k**, influence answer generation and evaluation.

The project was developed step by step in notebooks and later adapted into `.py` scripts for execution with **SLURM** on GPU.

---

## Important Note

The original project also contains a **`data/` folder**, but it is **not included here** because it is too large to upload.

That folder originally stores all intermediate and generated artifacts of the pipeline, including:

* prepared dataset files
* built corpus files
* chunked documents
* embeddings
* FAISS indexes
* chunk metadata
* outputs generated between preprocessing steps

Because this folder is not included, the **paths inside the scripts and notebooks must be adapted** before running the project in a new environment.

If you reuse this project, please first check and update all path definitions in the scripts and notebooks to match your local setup.

---
## Environment & Dependencies

The project was developed and executed in the following environment:

- Python: 3.10.20  
- PyTorch: 2.11.0  
- Transformers: 4.44.2  
- Datasets: 2.20.0  
- Sentence-Transformers: 3.0.1  
- FAISS: 1.8.0  
- NumPy: 1.26.4  
- Pandas: 2.2.2  

### Hardware

Experiments were run on LMU CIP servers:
- CPU: Intel Core i9-9900 (8 cores / 16 threads)
- RAM: 64 GB  
- GPU: AMD Radeon RX 5500M (via SLURM cluster)

Note: GPU resources are accessed through SLURM jobs. 

### Execution Notes

- Preprocessing steps (corpus construction, chunking, embeddings, and FAISS index building) are executed on CPU.
- Generation and large-scale experiments are executed via SLURM jobs.
- All configurations are run sequentially to manage memory constraints.

## Project Structure

```text id="j47cbf"
rag_project/
│
├── 500_Sample/
├── logs/
├── notebooks/
├── Output_1B/
├── scripts/
└── README.md
```

---

## Folder Details

### `500_Sample/`

Contains the **pilot experiment based on 500 questions**.

This part was used for:

* fast testing
* debugging
* first controlled experiments before scaling to the full validation set

It also contains outputs generated on the small sample.

---

### `logs/`

Contains **SLURM log files** for experiment runs.

These logs are useful for:

* monitoring running jobs
* debugging errors
* checking runtime behavior
* identifying GPU memory issues

---

### `notebooks/`

Contains all development and analysis notebooks.

Main notebooks:

```text id="ldm8jd"
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

#### What they do

* **00** – environment setup and dependency checks
* **01** – dataset loading and preparation
* **02** – corpus construction
* **03** – token-based chunking
* **04** – embedding generation
* **05** – FAISS index construction
* **06** – full retrieval, generation, and evaluation pipeline
* **07** – aggregation of experiment summaries
* **08** – BERTScore computation

#### `.ipynb` and `.py`

The notebooks were used during development and testing.

Some `.py` files are also stored in this folder, for example:

```text id="mgthzc"
06_full_generation_evaluation.py
06_full_generation_evaluation_TopK_variation.py
```

These files are **converted versions of Notebook 06**, created because **SLURM expects Python scripts (`.py`) rather than notebooks (`.ipynb`)**.

---

### `Output_1B/`

Contains the experiment outputs generated with the **LLaMA 3.2 1B model**.

Typical structure:

```text id="rffdx0"
Output_1B/
  chunk_<size>_topk_<k>/
    predictions.jsonl
    summary.json
```

#### Files

* **`predictions.jsonl`**

  * stores per-question results, such as:

    * question
    * prediction
    * ground truth
    * evaluation metrics

* **`summary.json`**

  * stores aggregated metrics for one configuration

This folder may also contain:

* aggregated CSV files
* BERTScore summary files

---

### `scripts/`

Contains the **SLURM job scripts** used to launch experiments on GPU.

These scripts define:

* job name
* partition
* runtime
* logging
* execution of the corresponding `.py` experiment file

---

## Original Pipeline

The project was built in the following order:

1. environment setup
2. dataset loading
3. corpus building
4. chunking
5. embedding generation
6. FAISS indexing
7. full generation + evaluation
8. experiment aggregation
9. BERTScore computation

---

## Path Configuration

Since the original `data/` folder is not uploaded, you will likely need to update path variables in the notebooks and scripts.

Typical examples include paths such as:

```python id="xsme6g"
DATA_DIR = Path("/home/a/arfaoui/rag_project/data")
OUTPUT_DIR = Path("/home/a/arfaoui/rag_project/Output_1B")
```

If these paths do not match your environment, modify them before running the project.

It is recommended to first review:

* dataset paths
* embedding paths
* FAISS index paths
* output paths
* SLURM script paths

This helps avoid file-not-found and path-related errors.

---

## Notes

* The project was developed interactively in notebooks and later adapted to scripts for batch execution.
* Intermediate data products are expected to exist even though they are not included here.
* The repository structure reflects the full workflow of the RAG evaluation pipeline.

---

## Author

Montassar Arfaoui.