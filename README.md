# Acoustic Voice Biomarkers for ALS

Bachelor's thesis (TFG) — **Jakub Wysocki**
BSc in Computer Engineering (GEI-IGD) · Universitat de Lleida — Igualada campus
Supervisor: **Alberto Tena**

A support system for monitoring **amyotrophic lateral sclerosis (ALS)** through voice analysis. It runs end to end: it extracts acoustic biomarkers, classifies bulbar involvement with machine learning, explains the model's decisions, and generates clinical reports automatically with a large language model — all served through a REST API and a web portal.

> A tool to **support clinical interpretation**, not to diagnose. Every prediction needs review by a qualified health professional.

## What it does

A reproducible pipeline in six stages:

1. **Preprocessing** of the acoustic features from five sustained vowels, merged into a single dataset.
2. **Label integration** (clinical labels) and semi-supervised relabeling (S4VM).
3. **Classification** with six models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM), stratified 5-fold cross-validation and Optuna hyperparameter optimization.
4. **Explainability** with SHAP, down to the individual biomarker.
5. **Clinical reports** written in natural language by an LLM (Llama-3.3-70B via Groq).
6. **REST API** (FastAPI) and a **web portal** to explore subjects, results, SHAP importance and reports.

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

## Main result

For discriminating **bulbar vs non-bulbar** involvement, the best model (Random Forest with S4VM relabeling) reaches an **AUC-ROC of 0.910 ± 0.066**. SHAP points to the vowel /e/ and the time-frequency descriptors as the most informative features.

## Context

The thesis also includes **ELA-Monitor**, an Android app for at-home follow-up (voice, motor and touch input), built as a separate component and not included in this repository.
