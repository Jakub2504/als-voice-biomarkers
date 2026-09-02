# Acoustic Voice Biomarkers for ALS

**An end-to-end system that analyses voice recordings to support the monitoring of ALS** (amyotrophic lateral sclerosis): acoustic biomarkers → ML classification → SHAP explainability → LLM-drafted clinical reports → REST API and web portal.

`Python` · `scikit-learn` · `XGBoost / LightGBM / CatBoost` · `Optuna` · `SHAP` · `Llama-3.3-70B (Groq)` · `FastAPI` · `MLflow`

> A tool to **support clinical interpretation**, not to diagnose. Every prediction needs review by a qualified health professional.

**What this project demonstrates:** taking a clinical research problem end to end — feature engineering, model selection and validation, explainability, LLM report generation, and the API and portal that make it usable.

Bachelor's thesis (TFG), graded 10/10 · **Jakub Wysocki** · BSc in Computer Engineering, Universitat de Lleida (Igualada campus) · Supervisor: **Alberto Tena**

---

## What it does

**Clinical research support** — Builds a reproducible dataset from the acoustic features of five sustained vowels, integrates the clinical labels, and applies semi-supervised relabeling (S4VM) as an exploratory analysis.

**ML classification** — Six models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM) with stratified 5-fold cross-validation and Optuna hyperparameter optimization.

**Explainability** — SHAP values at the level of the individual biomarker, so each prediction can be traced back to the acoustic features that drove it.

**LLM report generation** — Llama-3.3-70B (via Groq) converts the classification results and SHAP-derived information into a structured clinical report, constrained to the supplied outputs and always flagged for professional review.

**API & software engineering** — A FastAPI REST API and a web portal expose subjects, results, SHAP importance and generated reports end to end.

## Architecture

```text
Voice recordings
       ↓
Acoustic feature extraction
       ↓
Data preprocessing & label integration
       ↓
ML models + cross-validation
       ↓
SHAP explainability
       ↓
Structured model outputs
       ↓
Llama-3.3-70B report generation
       ↓
FastAPI REST API
       ↓
Web portal
```

## Main result

For discriminating **bulbar vs non-bulbar** involvement, the best model (Random Forest with S4VM relabeling) reaches an **AUC-ROC of 0.910 ± 0.066**. SHAP points to the vowel /e/ and the time-frequency descriptors as the most informative features.

The S4VM figure should be read as an experimental upper bound on this corpus rather than clinical performance: the relabeling and the evaluation share the same data. What it supports is a falsifiable hypothesis, not a validated result.

## Limitations

- **63 subjects.** Enough for a proof of concept and for generating hypotheses, not for claiming generalisation.
- **No external validation set.** All results come from cross-validation on a single corpus.
- **The S4VM finding is exploratory.** It needs neurologist-verified labels and an independent test set before it means anything clinically.
- **The report module was evaluated qualitatively**, on the three report types rather than with a full quantitative rubric.
- **This is a support tool, not a diagnostic system.** Every output requires review by a qualified professional.

## Structure

```
src/
  preprocessing.py       Load and merge the acoustic features (5 vowels)
  label_loader.py        Clinical label integration and S4VM relabeling
  api.py                 REST API (FastAPI): data, results, SHAP and LLM reports
  frontend/index.html    Web portal
notebooks/
  data_preparation.ipynb   Dataset construction
  eda.ipynb                Exploratory data analysis
  label_integration.ipynb  Label reconciliation
  ml_experiments.ipynb     ML experiments, Optuna and SHAP
  llm_reports.ipynb        Report generation and evaluation
requirements.txt
.env.example
```

## What I learned

- Model performance and clinical usefulness are different problems.
- Explainability matters when model outputs need to be inspected by domain experts.
- LLM-generated reports are only useful when grounded in structured model outputs.
- Turning research code into an API and usable interface introduces a different set of engineering constraints from experimentation alone.

## Data

The clinical corpus (voice recordings from patients at **Hospital Universitari de Bellvitge**; 63 subjects: 45 with ALS and 18 controls) is **not included in this repository**, for data-protection reasons (GDPR).

The pipeline expects a local `data/` folder with:
- `data/raw/caracteristicas_vocal_{1..5}.xls` — features exported from MATLAB.
- `data/processed/dataset_final.csv` — the merged, labeled dataset (produced by preprocessing).

## Getting started

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Environment variables
cp .env.example .env               # then fill in GROQ_API_KEY

# 3. Run the API + portal
uvicorn src.api:app --reload
# Portal:  http://localhost:8000
# Swagger: http://localhost:8000/docs
```

## Context

The thesis also includes **ELA-Monitor**, an Android app for at-home follow-up (voice, motor and touch input), built as a separate component and not included in this repository.
