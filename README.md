# COURT

**COURT** means **Categorization Of Unstructured Records and Texts**.

This repository contains the implementation for a Natural Language Processing project focused on classifying Brazilian legal documents from the VICTOR/STF dataset. The target task is a Kaggle-style multiclass classification problem with five document categories:

| Label | Class |
| --- | --- |
| 0 | Acordao |
| 1 | Agravo de Recurso Extraordinario (ARE) |
| 2 | Despacho |
| 3 | Recurso Extraordinario (RE) |
| 4 | Sentenca |

The competition metric is F1-score, with special attention to macro F1 because the dataset is imbalanced.

## What Is Included

The project is organized so the notebook remains explanatory and the reusable logic lives in Python modules:

```text
template-implementacao/
├── main.ipynb
├── PLANO_EXECUCAO.md
├── requirements.txt
├── requirements-advanced.txt
├── requirements-nvidia.txt
├── data/.gitkeep
├── models/.gitkeep
├── outputs/.gitkeep
├── scripts/
│   ├── analise_exploratoria.py
│   ├── analise_resultados.py
│   ├── experimentos.py
│   └── preprocessamento.py
└── tests/
    └── test_experimentos.py
```

The repository intentionally does **not** include `train.csv`, `test.csv`, `sample_submission.csv`, generated model files, notebook checkpoints, or generated submissions. Place the Kaggle CSV files in `template-implementacao/data/` or in the project root before running the notebook.

## Pipeline

The notebook runs the full pipeline without manual flags:

1. Load and validate `train.csv`, `test.csv`, and `sample_submission.csv`.
2. Split valid labels from `Category = -1` rows.
3. Analyze class distribution, text length, and top terms.
4. Clean OCR-heavy legal text into `Body_clean`.
5. Extract legal and basic NLP features.
6. Train classical TF-IDF baselines and stronger sparse models.
7. Audit possible noisy labels using out-of-fold predictions.
8. Pseudo-label high-confidence unlabeled samples.
9. Train the final classical model with the augmented training set.
10. Build chunked SentenceTransformer embeddings for long documents.
11. Fine-tune a Transformer classifier, using `cuda`, `mps`, or `cpu` automatically.
12. Generate Kaggle submission CSVs in `template-implementacao/outputs/`.

## Device Handling

The project detects the available accelerator automatically:

- NVIDIA GPU: uses `cuda`, mixed precision when supported, and larger batches.
- Apple Silicon: uses `mps`.
- No accelerator: uses `cpu` with smaller batches.

The heavy stages are still part of the pipeline on every machine. On CPU they can take a long time, but the code path remains executable.

## Setup

From the repository root:

```bash
cd template-implementacao
python3 -m venv ../.venv
../.venv/bin/pip install -r requirements.txt
../.venv/bin/pip install -r requirements-advanced.txt
```

For NVIDIA/Linux experiments with 4-bit quantization:

```bash
../.venv/bin/pip install -r requirements-nvidia.txt
```

Then open `template-implementacao/main.ipynb` and run all cells.

## Validation

The current implementation includes lightweight unit tests for the newest pipeline pieces:

```bash
.venv/bin/python -m unittest template-implementacao/tests/test_experimentos.py
```

These tests cover:

- Hugging Face `Trainer` compatibility across `tokenizer` and `processing_class` APIs.
- Pseudo-label selection by confidence.
- Conservative label-audit corrections.
- Chunking behavior for long legal texts.

## Notes On Sensitive Data

The `.gitignore` blocks competition CSV files, generated outputs, saved models, virtual environments, caches, and notebook checkpoints. Keep raw data local or on Kaggle; only implementation files should be committed.
