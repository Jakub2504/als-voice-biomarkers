"""
label_loader.py
===============
Módulo de carga y procesamiento de la base de datos clínica para el pipeline
de análisis de biomarcadores acústicos en ELA.

Estructura del Excel clínico (clinical_data.xlsx):
    - Hoja 'Pacientes' : 45 sujetos ELA válidos para el estudio.
    - Hoja 'Controles' : 18 sujetos sanos (C001...C018 en Excel → Kxxx...Kxxx en MATLAB).

El Excel ya contiene exactamente los sujetos correctos para el estudio.
No se aplica ningún filtrado ni modificación sobre sus datos.

Flujo de IDs:
    HUBxxx...HUBxxx  →  pacientes ELA — mismo ID en Excel y en MATLAB
    C001...C018      →  controles en Excel clínico
    Kxxx...Kxxx      →  mismos controles en dataset MATLAB / acústico

Reconciliación con el dataset acústico:
    El CSV acústico (dataset_raw_fusionado.csv) contiene 68 filas porque MATLAB
    procesó todas las carpetas de audio disponibles (50 HUB + 18 K).
    Al aplicar el label_map sobre el CSV, las filas HUB sin entrada en el Excel
    quedan con NaN y se descartan mediante dropna en label_integration.ipynb.
    El Excel no se modifica en ningún momento.

Autor: Jakub Wysocki
TFG — Ingeniería Informática / Salud Digital
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SHEET_PACIENTES   = "Pacientes"   # nombre exacto de la hoja en el Excel
SHEET_CONTROLES   = "Controles"   # nombre exacto de la hoja en el Excel

COL_ID            = "ID paciente"
COL_LABEL_MAQUINA = "Afectación bulbar con S4VM"

SI_VALUES = {"sí", "si", "yes", "s", "1"}
NO_VALUES = {"no", "n", "0"}

LABEL_ELA_BULBAR    = "ELA_bulbar"
LABEL_ELA_NO_BULBAR = "ELA_no_bulbar"
LABEL_CONTROL       = "Control"

PREFIX_CONTROL_XLSX = "C"
PREFIX_CONTROL_MAT  = "K"

# ---------------------------------------------------------------------------
# SUBJECT_ORDER
# ---------------------------------------------------------------------------
# Orden alfanumérico en el que MATLAB procesó las carpetas de audio.
# 68 entradas: 50 HUB + 18 K. Corresponde fila a fila con dataset_raw_fusionado.csv.
SUBJECT_ORDER: list[str] = (
    [f"HUB{i:03d}" for i in range(1, 51)] +
    [f"K{i:03d}"   for i in range(1, 19)]
)


# ---------------------------------------------------------------------------
# 1. Carga de la hoja 'Pacientes'
# ---------------------------------------------------------------------------

def load_pacientes_sheet(filepath: str | Path) -> pd.DataFrame:
    """
    Carga la hoja 'Pacientes' del Excel clínico.

    El Excel ya contiene exactamente los 45 pacientes ELA del estudio.
    No se aplica ningún filtrado.

    Gestiona la duplicidad de 'Afectación bulbar' (aparece dos veces):
        - Primera aparición  → campo clínico general (sin uso en el pipeline)
        - Segunda aparición  → etiqueta diagnóstica → '_label_clinico_raw'

    Parameters
    ----------
    filepath : str | Path
        Ruta a clinical_data.xlsx.

    Returns
    -------
    pd.DataFrame
        45 filas con columnas de ID y etiquetas de afectación bulbar.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si faltan columnas obligatorias o no se encuentra la etiqueta clínica.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Excel clínico no encontrado: {filepath}\n"
            f"Ruta esperada: C:\\TFG\\data\\clinical_data.xlsx"
        )

    df = pd.read_excel(filepath, sheet_name=SHEET_PACIENTES, dtype=str)
    logger.info("Hoja '%s': %d filas × %d columnas.", SHEET_PACIENTES, *df.shape)

    required = [COL_ID, COL_LABEL_MAQUINA]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas no encontradas en hoja '{SHEET_PACIENTES}': {missing}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Pandas renombra automáticamente la 2ª 'Afectación bulbar' → 'Afectación bulbar.1'
    if "Afectación bulbar.1" in df.columns:
        df = df.rename(columns={"Afectación bulbar.1": "_label_clinico_raw"})
    elif "Afectación bulbar" in df.columns:
        df = df.rename(columns={"Afectación bulbar": "_label_clinico_raw"})
    else:
        raise ValueError(
            "No se encontró la columna de etiqueta clínica 'Afectación bulbar'.\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    logger.info("Pacientes cargados: %d.", len(df))
    return df


# ---------------------------------------------------------------------------
# 2. Carga de la hoja 'Controles'
# ---------------------------------------------------------------------------

def load_controles_sheet(filepath: str | Path) -> pd.DataFrame:
    """
    Carga la hoja 'Controles' del Excel clínico.

    Normaliza los IDs de C001...C018 a Kxxx...Kxxx para que coincidan
    con los identificadores del dataset MATLAB.

    Parameters
    ----------
    filepath : str | Path
        Ruta a clinical_data.xlsx.

    Returns
    -------
    pd.DataFrame
        18 filas con columna 'subject_id_matlab' (Kxxx...Kxxx).

    Raises
    ------
    ValueError
        Si no se encuentra columna de ID en la hoja.
    """
    df = pd.read_excel(filepath, sheet_name=SHEET_CONTROLES, dtype=str)
    logger.info("Hoja '%s': %d filas × %d columnas.", SHEET_CONTROLES, *df.shape)

    # Localizar columna de ID (nombre puede variar ligeramente)
    id_col = next(
        (c for c in (COL_ID, "ID", "ID control", "id", "Id", "subject") if c in df.columns),
        None,
    )
    if id_col is None:
        raise ValueError(
            f"Columna de ID no encontrada en hoja '{SHEET_CONTROLES}'.\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    if id_col != COL_ID:
        df = df.rename(columns={id_col: COL_ID})

    df["subject_id_matlab"] = df[COL_ID].apply(_c_to_k_id)
    logger.info(
        "Controles cargados: %d. Muestra ID: %s → %s",
        len(df),
        df[COL_ID].head(2).tolist(),
        df["subject_id_matlab"].head(2).tolist(),
    )
    return df


def _c_to_k_id(raw_id: str) -> str:
    """Convierte C001 → Kxxx. IDs HUB no se modifican."""
    raw_id = str(raw_id).strip()
    if raw_id.upper().startswith(PREFIX_CONTROL_XLSX):
        digits = re.sub(r"[^0-9]", "", raw_id)
        return f"{PREFIX_CONTROL_MAT}{digits}"
    return raw_id


# ---------------------------------------------------------------------------
# 3. Codificación de etiquetas
# ---------------------------------------------------------------------------

def _encode_label(raw_value: str, is_control: bool) -> str:
    """Convierte sí/no en etiqueta canónica del proyecto."""
    if is_control:
        return LABEL_CONTROL
    if pd.isna(raw_value):
        raise ValueError("Valor nulo inesperado en etiqueta de paciente ELA.")
    normalized = str(raw_value).strip().lower()
    if normalized in SI_VALUES:
        return LABEL_ELA_BULBAR
    if normalized in NO_VALUES:
        return LABEL_ELA_NO_BULBAR
    raise ValueError(
        f"Valor no reconocido: '{raw_value}'. Se esperaba 'sí' o 'no'."
    )


def encode_labels_pacientes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera 'label_clinico' y 'label_maquina' para los pacientes ELA.

    Parameters
    ----------
    df : pd.DataFrame
        Hoja 'Pacientes' con columnas '_label_clinico_raw' y COL_LABEL_MAQUINA.

    Returns
    -------
    pd.DataFrame
        DataFrame original con las dos columnas de etiqueta añadidas.
    """
    df = df.copy()
    errors: list[str] = []
    labels_c, labels_m = [], []

    for idx, row in df.iterrows():
        sid = row.get(COL_ID, f"fila_{idx}")
        try:
            lc = _encode_label(row.get("_label_clinico_raw", np.nan), False)
            lm = _encode_label(row.get(COL_LABEL_MAQUINA,    np.nan), False)
        except ValueError as e:
            errors.append(f"  {sid}: {e}")
            lc = lm = np.nan
        labels_c.append(lc)
        labels_m.append(lm)

    df["label_clinico"] = labels_c
    df["label_maquina"]  = labels_m

    if errors:
        logger.error("%d errores en codificación:\n%s", len(errors), "\n".join(errors))
    else:
        logger.info("Etiquetas de pacientes codificadas sin errores.")

    logger.info("label_clinico:\n%s", df["label_clinico"].value_counts().to_string())
    logger.info("label_maquina:\n%s",  df["label_maquina"].value_counts().to_string())
    return df


def encode_labels_controles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna 'Control' como etiqueta a todos los sujetos sanos (ambos sistemas).

    Parameters
    ----------
    df : pd.DataFrame
        Hoja 'Controles'.

    Returns
    -------
    pd.DataFrame
        DataFrame con 'label_clinico' = 'label_maquina' = 'Control'.
    """
    df = df.copy()
    df["label_clinico"] = LABEL_CONTROL
    df["label_maquina"]  = LABEL_CONTROL
    logger.info("Controles etiquetados: %d × 'Control'.", len(df))
    return df


# ---------------------------------------------------------------------------
# 4. Construcción del label_map
# ---------------------------------------------------------------------------

def build_label_map(
    df_pac:  pd.DataFrame,
    df_ctrl: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """
    Construye el label_map para preprocessing.assign_labels().

    Cubre los sujetos presentes en el Excel clínico: 45 ELA + 18 controles.
    Las filas del CSV acústico sin entrada en este mapa quedarán con NaN
    al aplicar assign_labels() y se descartarán con dropna en el notebook.

    Parameters
    ----------
    df_pac : pd.DataFrame
        Hoja 'Pacientes' con etiquetas codificadas.
    df_ctrl : pd.DataFrame
        Hoja 'Controles' con etiquetas codificadas y 'subject_id_matlab'.

    Returns
    -------
    dict[str, dict[str, str]]
        {'HUBxxx': {'label_clinico': ..., 'label_maquina': ...}, ...}
        63 entradas totales.
    """
    label_map: dict[str, dict[str, str]] = {}

    for _, row in df_pac.iterrows():
        mid = str(row[COL_ID]).strip()
        label_map[mid] = {
            "label_clinico": row["label_clinico"],
            "label_maquina":  row["label_maquina"],
        }

    for _, row in df_ctrl.iterrows():
        mid = str(row["subject_id_matlab"]).strip()
        label_map[mid] = {
            "label_clinico": row["label_clinico"],
            "label_maquina":  row["label_maquina"],
        }

    logger.info("label_map: %d entradas (45 ELA + 18 controles).", len(label_map))
    return label_map


# ---------------------------------------------------------------------------
# 5. Pipeline principal
# ---------------------------------------------------------------------------

def run_clinical_pipeline(
    clinical_excel_path: str | Path,
) -> dict[str, dict[str, str]]:
    """
    Carga el Excel clínico y construye el label_map.

    Pasos:
        1. Cargar hoja 'Pacientes' (45 ELA, sin modificaciones).
        2. Cargar hoja 'Controles' (18 sanos, sin modificaciones).
        3. Codificar etiquetas sí/no → canónico.
        4. Construir label_map con los 63 sujetos del Excel.

    Parameters
    ----------
    clinical_excel_path : str | Path
        Ruta a clinical_data.xlsx.

    Returns
    -------
    dict[str, dict[str, str]]
        label_map con 63 entradas listo para preprocessing.assign_labels().

    Example
    -------
    >>> from src.label_loader import run_clinical_pipeline
    >>> label_map = run_clinical_pipeline(r'C:/TFG/data/clinical_data.xlsx')
    >>> len(label_map)   # 63
    """
    logger.info("=" * 55)
    logger.info("PIPELINE CLÍNICO — inicio")
    logger.info("=" * 55)

    logger.info("[1/3] Hoja 'Pacientes'...")
    df_pac = encode_labels_pacientes(load_pacientes_sheet(clinical_excel_path))

    logger.info("[2/3] Hoja 'Controles'...")
    df_ctrl = encode_labels_controles(load_controles_sheet(clinical_excel_path))

    logger.info("[3/3] Construyendo label_map...")
    label_map = build_label_map(df_pac, df_ctrl)

    logger.info("PIPELINE CLÍNICO — completado (%d entradas)", len(label_map))
    logger.info("=" * 55)
    return label_map


# ---------------------------------------------------------------------------
# Punto de entrada directo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    CLINICAL_PATH = Path(r"C:\TFG\data\clinical_data.xlsx")
    label_map = run_clinical_pipeline(CLINICAL_PATH)
    print(f"\nlabel_map: {len(label_map)} entradas")
    for k, v in list(label_map.items())[:5]:
        print(f"  {k}: {v}")
