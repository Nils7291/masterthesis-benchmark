# Masterthesis Benchmark: Encoder vs. LLM for ERP Contract Classification

This repository contains the reproducible evaluation pipeline for the empirical
chapter of the Master's thesis "Encoder-Based Classification versus Proprietary
Large Language Models for Automated ERP Contract Review" (HWR Berlin, BIPM).

The benchmark compares two systems on the same set of 64 manually labelled
contract sections against a 76-entry requirement catalogue:

1. An embedding-based zero-shot classifier built on
   `paraphrase-multilingual-MiniLM-L12-v2` with mean pooling, computing cosine
   similarity against catalogue reference texts (Notebook 01).
2. A proprietary large language model accessed through the OpenAI API,
   prompted in a force-choice formulation with structured outputs (Notebook 02).

Both systems are evaluated with identical inputs, identical metrics, and
identical statistical machinery (bootstrap confidence intervals with B=1000,
stability runs with N=5).

## Repository structure

```
masterthesis-benchmark/
├── data/
│   ├── sections_labeled_manually.xlsx     64 sections with gold catalog_id
│   ├── catalogue_clean_with_aspects.xlsx  76 catalogue requirements
│   └── README.md                          data dictionary
├── notebooks/
│   ├── 01_encoder_benchmark.ipynb         encoder pipeline + metrics
│   └── 02_llm_benchmark.ipynb             OpenAI pipeline + metrics
├── src/
│   ├── __init__.py
│   └── metrics.py                         bootstrap CIs, stability score
├── results/
│   ├── encoder/                           predictions, metrics, plots
│   └── llm/                               predictions, metrics, plots
├── .env.example                           template for OPENAI_API_KEY
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Tested on Python 3.11. Uses `uv` as the package manager, consistent with
the original prototype repository.

If `uv` is not installed yet, see https://github.com/astral-sh/uv.

```bash
uv venv .venv --python 3.11

# Activate it
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\Activate.ps1         # Windows

uv pip install -r requirements.txt
```

## Running the benchmark

Open the notebooks in VS Code (Jupyter extension) and run top to bottom.
Each notebook writes predictions, metrics, and plots into `results/encoder/`
or `results/llm/` respectively.

Both notebooks are deterministic where the underlying system allows. The
encoder is configured with `model.eval()` and fixed seeds for `torch` and
`numpy`. The LLM is called with `temperature=0` and a fixed `seed` parameter;
documented residual non-determinism on the API side is empirically measured
through the N=5 stability protocol.

## Methodological notes

Statistical inference uses bootstrap confidence intervals (B=1000) on the
full 64-section set rather than cross-validation. No training occurs in
either system, so the standard rationale for CV does not apply.

The labelled set covers 62 of the 76 catalogue classes. The remaining 14
classes have no test instances and therefore cannot contribute to per-class
metrics. This is a property of the available labelled data, not of the
evaluation design, and is reported transparently in Chapter 6.2 of the
thesis.

## Author

Nils Taglieber, HWR Berlin, Business Intelligence and Process Management.
