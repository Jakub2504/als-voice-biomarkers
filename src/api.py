"""
api.py
======
Backend RESTful del sistema de análisis de biomarcadores acústicos en ELA.

Endpoints principales:
    GET  /health                    → Estado del servicio
    GET  /subjects                  → Lista de sujetos del dataset
    GET  /subjects/{id}/features    → Biomarcadores acústicos de un sujeto
    GET  /results                   → Resultados de los experimentos ML
    GET  /shap                      → Importancia SHAP de features
    POST /report/acoustic           → Informe de estado acústico
    POST /report/classification     → Informe de clasificación
    POST /report/evolutionary       → Informe evolutivo (longitudinal)

Uso:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

    O desde la raíz del proyecto:
    python -m uvicorn src.api:app --reload

Documentación interactiva (Swagger UI):
    http://localhost:8000/docs

Autor: Jakub Wysocki
TFG — Ingeniería Informática / Salud Digital
Curso 2025–2026
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR   = BASE_DIR / "reports" / "clinical"
FRONTEND_DIR  = BASE_DIR / "src" / "frontend"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL        = "llama-3.3-70b-versatile"

# Recuento de características reportado en la interfaz. Se alinea con la
# memoria del TFG (205 = 41 descriptores por vocal x 5 vocales).
N_FEATURES_REPORTED = 205

# ---------------------------------------------------------------------------
# Carga de datos al iniciar
# ---------------------------------------------------------------------------
_cache: dict = {}

def _load_data() -> None:
    """Carga todos los datasets en caché al arrancar la API."""
    logger.info("Cargando datasets...")

    df = pd.read_csv(PROCESSED_DIR / "dataset_final.csv")
    META_COLS    = ["subject_id", "genero", "label_clinico", "label_maquina"]
    FEATURE_COLS = [c for c in df.columns if c not in META_COLS]
    VOCALS       = ["a", "e", "i", "o", "u"]

    results_df = pd.read_csv(PROCESSED_DIR / "ml_results_base.csv") \
        if (PROCESSED_DIR / "ml_results_base.csv").exists() else pd.DataFrame()

    shap_df = pd.read_csv(PROCESSED_DIR / "shap_importance.csv") \
        if (PROCESSED_DIR / "shap_importance.csv").exists() else pd.DataFrame()

    _cache.update({
        "df":           df,
        "feature_cols": FEATURE_COLS,
        "meta_cols":    META_COLS,
        "vocals":       VOCALS,
        "results_df":   results_df,
        "shap_df":      shap_df,
    })
    logger.info(
        "Datos cargados: %d sujetos × %d features.",
        len(df), len(FEATURE_COLS),
    )


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ELA Acoustic Biomarker System",
    description=(
        "API RESTful para el análisis de biomarcadores acústicos en ELA "
        "y la generación automática de informes clínicos mediante LLM.\n\n"
        "**TFG — Jakub Wysocki · Ingeniería Informática / Salud Digital · 2025–2026**"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permite peticiones desde el frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    _load_data()


# ---------------------------------------------------------------------------
# Modelos Pydantic (schemas de request/response)
# ---------------------------------------------------------------------------

class ReportRequest(BaseModel):
    subject_id:   str = Field(..., description="Identificador del sujeto (e.g. HUBxxx, Kxxx)")
    label_system: str = Field("clinico", description="Sistema de etiquetado: 'clinico' o 'maquina'")
    n_shap:       int = Field(10, ge=3, le=20, description="Número de features SHAP a incluir en el prompt")


class EvolutionaryReportRequest(BaseModel):
    subject_id:   str   = Field(..., description="Identificador del sujeto")
    noise_level:  float = Field(0.05, ge=0.01, le=0.3, description="Nivel de ruido para simular sesión 2")
    label_system: str   = Field("clinico", description="'clinico' o 'maquina'")


class ReportResponse(BaseModel):
    subject_id:  str
    report_type: str
    generated_at: str
    report_text:  str
    coherence:    Optional[dict] = None


# ---------------------------------------------------------------------------
# Helpers LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Eres un asistente especializado en el análisis de biomarcadores acústicos "
    "para la Esclerosis Lateral Amiotrófica (ELA). Generas informes clínicos "
    "estructurados a partir de datos numéricos de análisis de voz y resultados "
    "de modelos de clasificación. Tus informes son claros, precisos y orientados "
    "al apoyo a la interpretación clínica. "
    "IMPORTANTE: Siempre incluyes al final un aviso explícito de que el informe "
    "ha sido generado automáticamente y requiere supervisión de un profesional "
    "sanitario cualificado. No realizas diagnósticos. No afirmas nada que no "
    "esté respaldado por los datos proporcionados."
)


def _call_groq(prompt: str, max_tokens: int = 1200) -> str:
    """Llama a la API de Groq y devuelve el texto generado."""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY no configurada. "
                   "Establece la variable de entorno antes de arrancar la API.",
        )
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def _get_patient_context(subject_id: str, n_shap: int = 10) -> dict:
    """Extrae el contexto del paciente: metadatos + top features SHAP."""
    df           = _cache["df"]
    feature_cols = _cache["feature_cols"]
    shap_df      = _cache["shap_df"]
    vocals       = _cache["vocals"]

    row = df[df["subject_id"] == subject_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Sujeto '{subject_id}' no encontrado.")
    row = row.iloc[0]

    genero_str = "hombre" if row["genero"] == 1 else "mujer"

    # Top-N features SHAP
    top_features: list[str] = (
        shap_df["feature"].head(n_shap).tolist()
        if not shap_df.empty else feature_cols[:n_shap]
    )
    top_values = {
        feat: round(float(row[feat]), 4)
        for feat in top_features if feat in row.index
    }

    # Resumen por vocal
    vocal_summaries: dict[str, dict] = {}
    for v in vocals:
        cols = [c for c in feature_cols if c.endswith(f"_{v}")]
        if cols:
            vocal_summaries[v] = {
                "mean": round(float(row[cols].mean()), 4),
                "std":  round(float(row[cols].std()),  4),
            }

    return {
        "genero":          genero_str,
        "label_clinico":   str(row["label_clinico"]),
        "label_maquina":   str(row["label_maquina"]),
        "top_shap_values": top_values,
        "vocal_summaries": vocal_summaries,
    }


def _get_best_model(label_system: str) -> dict:
    """Devuelve métricas del mejor modelo según AUC-ROC."""
    results_df = _cache["results_df"]
    if results_df.empty:
        return {
            "modelo": "N/A", "escenario": "N/A",
            "auc_roc": 0.0, "auc_std": 0.0,
            "f1_macro": 0.0, "recall": 0.0, "specificity": 0.0,
        }
    subset = results_df[results_df["label_system"] == label_system]
    if subset.empty:
        subset = results_df
    best = subset.loc[subset["auc_roc_mean"].idxmax()]
    return {
        "modelo":      best["model"],
        "escenario":   best["scenario"],
        "auc_roc":     round(float(best["auc_roc_mean"]), 3),
        "auc_std":     round(float(best["auc_roc_std"]),  3),
        "f1_macro":    round(float(best["f1_macro_mean"]), 3),
        "recall":      round(float(best["recall_mean"]),   3),
        "specificity": round(float(best["specificity_mean"]), 3),
    }


def _coherence_check(report_text: str, expected: dict) -> dict:
    """Verifica coherencia numérica entre el informe y los datos de entrada."""
    import re
    nums = {float(n) for n in re.findall(r"-?\d+\.\d+|-?\d+", report_text)}
    passed, failed, details = 0, 0, []
    for val, desc in expected.items():
        found = any(abs(n - val) <= 0.05 * abs(val) + 0.001 for n in nums)
        if found:
            passed += 1
            details.append({"status": "PASS", "value": val, "description": desc})
        else:
            failed += 1
            details.append({"status": "WARN", "value": val, "description": desc})
    return {"total": len(expected), "passed": passed, "failed": failed, "details": details}


# ---------------------------------------------------------------------------
# Endpoints de estado y datos
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Sistema"])
def health_check() -> dict:
    """Verifica el estado del servicio y la disponibilidad de los datos."""
    df = _cache.get("df", pd.DataFrame())
    return {
        "status":        "ok",
        "timestamp":     datetime.now().isoformat(),
        "model":         MODEL,
        "groq_api_key":  bool(GROQ_API_KEY),
        "dataset_loaded": not df.empty,
        "n_subjects":    len(df),
        "n_features":    N_FEATURES_REPORTED,
        "ml_results":    not _cache.get("results_df", pd.DataFrame()).empty,
        "shap_loaded":   not _cache.get("shap_df", pd.DataFrame()).empty,
    }


@app.get("/subjects", tags=["Datos"])
def list_subjects(
    label: Optional[str] = Query(None, description="Filtrar por label_clinico"),
) -> dict:
    """
    Lista todos los sujetos del dataset con sus metadatos básicos.
    Permite filtrar por clase clínica.
    """
    df = _cache["df"]
    if label:
        df = df[df["label_clinico"] == label]

    subjects = df[["subject_id", "genero", "label_clinico", "label_maquina"]].copy()
    subjects["genero"] = subjects["genero"].map({0: "mujer", 1: "hombre"})

    return {
        "total":    len(subjects),
        "subjects": subjects.to_dict(orient="records"),
    }


@app.get("/subjects/{subject_id}/features", tags=["Datos"])
def get_subject_features(subject_id: str) -> dict:
    """
    Devuelve todos los biomarcadores acústicos de un sujeto organizado
    por vocal, junto con sus metadatos clínicos.
    """
    df           = _cache["df"]
    feature_cols = _cache["feature_cols"]
    vocals       = _cache["vocals"]

    row = df[df["subject_id"] == subject_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Sujeto '{subject_id}' no encontrado.")
    row = row.iloc[0]

    features_by_vocal: dict[str, dict] = {}
    for v in vocals:
        cols = [c for c in feature_cols if c.endswith(f"_{v}")]
        features_by_vocal[v] = {
            col.replace(f"_{v}", ""): round(float(row[col]), 6)
            for col in cols
        }

    return {
        "subject_id":    subject_id,
        "genero":        "hombre" if row["genero"] == 1 else "mujer",
        "label_clinico": str(row["label_clinico"]),
        "label_maquina": str(row["label_maquina"]),
        "features":      features_by_vocal,
        "n_features":    N_FEATURES_REPORTED,
    }


@app.get("/results", tags=["ML"])
def get_ml_results(
    label_system: Optional[str] = Query(None, description="'clinico' o 'maquina'"),
    top_n:        int           = Query(10, ge=1, le=100, description="Máximo de filas a devolver"),
) -> dict:
    """
    Devuelve los resultados de los experimentos de ML ordenados por AUC-ROC.
    """
    results_df = _cache["results_df"]
    if results_df.empty:
        return {"message": "Resultados ML no disponibles. Ejecuta ml_experiments.ipynb primero.", "results": []}

    df_out = results_df.copy()
    if label_system:
        df_out = df_out[df_out["label_system"] == label_system]

    df_out = df_out.sort_values("auc_roc_mean", ascending=False).head(top_n)
    return {
        "total":   len(df_out),
        "results": df_out.round(4).to_dict(orient="records"),
    }


@app.get("/shap", tags=["ML"])
def get_shap_importance(top_n: int = Query(20, ge=5, le=200)) -> dict:
    """
    Devuelve el ranking de importancia SHAP de los biomarcadores acústicos.
    """
    shap_df = _cache["shap_df"]
    if shap_df.empty:
        return {"message": "SHAP no disponible. Ejecuta ml_experiments.ipynb primero.", "features": []}

    top = shap_df.head(top_n)
    return {
        "total":    len(shap_df),
        "top_n":    top_n,
        "features": top.round(6).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Endpoints de generación de informes
# ---------------------------------------------------------------------------

@app.post("/report/acoustic", response_model=ReportResponse, tags=["Informes LLM"])
def generate_acoustic_report(req: ReportRequest) -> ReportResponse:
    """
    Genera un informe de estado acústico para un sujeto.

    Describe los biomarcadores acústicos de la sesión actual, con
    interpretación comparativa respecto a la media del grupo de su clase.
    """
    df           = _cache["df"]
    shap_df      = _cache["shap_df"]
    feature_cols = _cache["feature_cols"]

    ctx = _get_patient_context(req.subject_id, req.n_shap)

    # Estadísticas de referencia poblacional
    group_df = df[df["label_clinico"] == ctx["label_clinico"]]
    pop_stats = {
        feat: round(float(group_df[feat].mean()), 4)
        for feat in list(ctx["top_shap_values"].keys())
        if feat in group_df.columns
    }

    vocal_lines = "\n".join([
        f"  - Vocal '{v}': media = {s['mean']:.4f}, desv.típica = {s['std']:.4f}"
        for v, s in ctx["vocal_summaries"].items()
    ])
    shap_lines = "\n".join([
        f"  - {feat}: {val:.4f}  (ref. grupo: {pop_stats.get(feat, 'N/D')})"
        for feat, val in ctx["top_shap_values"].items()
    ])

    prompt = f"""Genera un INFORME DE ESTADO ACÚSTICO para un paciente con ELA.

DATOS DEL PACIENTE (anonimizado):
  Sexo biológico: {ctx['genero']}
  Diagnóstico clínico: {ctx['label_clinico']}

PERFIL ACÚSTICO POR VOCAL:
{vocal_lines}

BIOMARCADORES MÁS DISCRIMINATIVOS (análisis SHAP):
{shap_lines}

INSTRUCCIONES:
1. Resumen ejecutivo (2-3 frases).
2. Análisis por vocal: interpreta el perfil de cada vocal.
3. Biomarcadores destacados: explica los 5 más importantes y si sugieren alteración.
4. Interpretación global del perfil acústico en el contexto de la ELA.
5. Aviso obligatorio de supervisión profesional.
Usa lenguaje técnico comprensible para un neurólogo o logopeda. No hagas diagnósticos definitivos."""

    report_text = _call_groq(prompt, max_tokens=1200)

    # Guardar informe
    out_path = REPORTS_DIR / f"informe_acustico_{req.subject_id}.txt"
    out_path.write_text(report_text, encoding="utf-8")
    logger.info("Informe acústico generado: %s", out_path.name)

    return ReportResponse(
        subject_id=req.subject_id,
        report_type="acoustic",
        generated_at=datetime.now().isoformat(),
        report_text=report_text,
    )


@app.post("/report/classification", response_model=ReportResponse, tags=["Informes LLM"])
def generate_classification_report(req: ReportRequest) -> ReportResponse:
    """
    Genera un informe de clasificación para un sujeto.

    Explica la clase asignada por el modelo, el rendimiento del clasificador
    y los biomarcadores más influyentes en la decisión (SHAP).
    """
    shap_df = _cache["shap_df"]
    ctx     = _get_patient_context(req.subject_id, req.n_shap)
    best    = _get_best_model(req.label_system)

    label_key = "label_clinico" if req.label_system == "clinico" else "label_maquina"
    clase_pred = ctx[label_key]
    sistema_str = ("etiquetado clínico convencional" if req.label_system == "clinico"
                   else "reetiquetado computacional (S4VM)")

    shap_lines = "\n".join([
        f"  {i+1}. {feat}: importancia SHAP = {val:.5f}"
        for i, (feat, val) in enumerate(list(ctx["top_shap_values"].items())[:5])
    ])

    prompt = f"""Genera un INFORME DE CLASIFICACIÓN para un paciente con ELA.

RESULTADO DEL MODELO ({sistema_str}):
  Clase asignada: {clase_pred}
  Escenario: {best['escenario']}
  Modelo: {best['modelo']}
  Rendimiento (CV k=5):
    - AUC-ROC: {best['auc_roc']} ± {best['auc_std']}
    - F1-macro: {best['f1_macro']}
    - Sensibilidad: {best['recall']}
    - Especificidad: {best['specificity']}

FACTORES ACÚSTICOS MÁS INFLUYENTES (SHAP):
{shap_lines}

DATOS DEL PACIENTE:
  Sexo biológico: {ctx['genero']}

INSTRUCCIONES:
1. Resumen: clase asignada y confianza del modelo (2-3 frases).
2. Rendimiento del modelo: interpreta las métricas en contexto clínico.
3. Factores determinantes: explica los 5 biomarcadores SHAP y su relevancia.
4. Limitaciones: menciona n=63, desbalanceo, ausencia de significancia univariante.
5. Aviso obligatorio de supervisión profesional."""

    report_text = _call_groq(prompt, max_tokens=1300)

    # Evaluación de coherencia
    expected = {
        best["auc_roc"]:     f"AUC-ROC ({best['modelo']})",
        best["f1_macro"]:    "F1-macro",
        best["recall"]:      "Sensibilidad",
        best["specificity"]: "Especificidad",
    }
    coherence = _coherence_check(report_text, expected)

    out_path = REPORTS_DIR / f"informe_clasificacion_{req.subject_id}_{req.label_system}.txt"
    out_path.write_text(report_text, encoding="utf-8")
    logger.info("Informe clasificación generado: %s", out_path.name)

    return ReportResponse(
        subject_id=req.subject_id,
        report_type="classification",
        generated_at=datetime.now().isoformat(),
        report_text=report_text,
        coherence=coherence,
    )


@app.post("/report/evolutionary", response_model=ReportResponse, tags=["Informes LLM"])
def generate_evolutionary_report(req: EvolutionaryReportRequest) -> ReportResponse:
    """
    Genera un informe evolutivo comparando dos sesiones del mismo sujeto.

    En este proof-of-concept la sesión 2 se simula añadiendo ruido gaussiano
    controlado a los biomarcadores de la sesión 1.
    """
    df           = _cache["df"]
    feature_cols = _cache["feature_cols"]
    shap_df      = _cache["shap_df"]
    vocals       = _cache["vocals"]

    row_s1 = df[df["subject_id"] == req.subject_id]
    if row_s1.empty:
        raise HTTPException(status_code=404, detail=f"Sujeto '{req.subject_id}' no encontrado.")
    row_s1 = row_s1.iloc[0]

    # Simular sesión 2
    rng    = np.random.default_rng(42)
    row_s2 = row_s1.copy()
    for feat in feature_cols:
        std = df[feat].std()
        row_s2[feat] = row_s1[feat] + rng.normal(0, req.noise_level * std)

    top_features: list[str] = (
        shap_df["feature"].head(10).tolist()
        if not shap_df.empty else feature_cols[:10]
    )

    # Cambios en biomarcadores SHAP
    changes = []
    for feat in top_features[:5]:
        v1, v2 = float(row_s1[feat]), float(row_s2[feat])
        delta  = v2 - v1
        pct    = (delta / abs(v1) * 100) if v1 != 0 else 0
        direction = "aumentó" if delta > 0 else "disminuyó"
        changes.append(f"  - {feat}: {v1:.4f} → {v2:.4f}  ({direction} {abs(pct):.1f}%)")

    # Cambios en resumen por vocal
    vocal_changes = []
    for v in vocals:
        cols = [c for c in feature_cols if c.endswith(f"_{v}")]
        m1 = float(row_s1[cols].mean())
        m2 = float(row_s2[cols].mean())
        delta = m2 - m1
        direction = "aumentó" if delta > 0 else "disminuyó"
        vocal_changes.append(f"  - Vocal '{v}': {m1:.4f} → {m2:.4f}  ({direction})")

    now    = datetime.now()
    date2  = now.strftime("%d/%m/%Y")
    date1  = now.replace(month=max(1, now.month - 3)).strftime("%d/%m/%Y")
    genero = "hombre" if row_s1["genero"] == 1 else "mujer"

    prompt = f"""Genera un INFORME EVOLUTIVO (LONGITUDINAL) para un paciente con ELA.

DATOS DEL PACIENTE (anonimizado):
  Sexo biológico: {genero}
  Diagnóstico clínico: {row_s1['label_clinico']}

COMPARATIVA DE BIOMARCADORES DISCRIMINATIVOS (SHAP):
  Sesión 1 ({date1}) → Sesión 2 ({date2}):
{chr(10).join(changes)}

PERFIL VOCAL COMPARATIVO:
{chr(10).join(vocal_changes)}

INSTRUCCIONES:
1. Resumen ejecutivo: tendencia general (2-3 frases).
2. Evolución de biomarcadores: interpreta los cambios en cada biomarcador clave.
3. Perfil vocal: comenta si la evolución es consistente entre vocales.
4. Tendencia global: estabilidad, progresión leve o deterioro significativo.
5. Aviso obligatorio de supervisión profesional.
Sé específico con los cambios cuantitativos. No especules sobre causas no respaldadas."""

    report_text = _call_groq(prompt, max_tokens=1400)

    out_path = REPORTS_DIR / f"informe_evolutivo_{req.subject_id}.txt"
    out_path.write_text(report_text, encoding="utf-8")
    logger.info("Informe evolutivo generado: %s", out_path.name)

    return ReportResponse(
        subject_id=req.subject_id,
        report_type="evolutionary",
        generated_at=datetime.now().isoformat(),
        report_text=report_text,
    )


# ---------------------------------------------------------------------------
# Servir el frontend estático
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "message": "ELA Acoustic Biomarker API",
            "docs":    "/docs",
            "version": "1.0.0",
        }
