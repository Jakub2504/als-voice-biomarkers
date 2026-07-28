"""
preprocessing.py
================
Módulo de preprocesamiento para el pipeline de análisis de biomarcadores
acústicos en ELA (Esclerosis Lateral Amiotrófica).

Pipeline:
    1. Carga de los 5 archivos Excel (vocal a/e/i/o/u).
    2. Unificación y validación de la columna Genero.
    3. Fusión horizontal de las 5 vocales (205 → 200 features + Genero + ID).
    4. Adición de columnas de etiqueta placeholder (label_clinico, label_maquina).
    5. Validación de integridad y guardado del dataset fusionado.

Nota: La asignación de etiquetas clínicas se realiza en label_loader.py.
    La reconciliación 68→63 filas se hace mediante dropna en label_integration.ipynb.

Dependencias:
    pandas >= 1.5, numpy >= 1.23, xlrd >= 2.0 (para .xls)

Autor: Jakub Wysocki
TFG — Ingeniería Informática / Salud Digital
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración del logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes del proyecto
# ---------------------------------------------------------------------------
# Archivos Excel generados por MATLAB (formato .xls legacy)
VOCAL_FILES: dict[str, str] = {
    "a": "caracteristicas_vocal_1.xls",
    "e": "caracteristicas_vocal_2.xls",
    "i": "caracteristicas_vocal_3.xls",
    "o": "caracteristicas_vocal_4.xls",
    "u": "caracteristicas_vocal_5.xls",
}

# Nombre de la columna de género en los archivos MATLAB originales
GENDER_COL_MATLAB: str = "Genero"

# Nombre canónico de género en el dataset final (0 = mujer, 1 = hombre)
GENDER_COL_FINAL: str = "genero"

# Sujetos totales procesados por MATLAB (50 ELA + 18 controles = 68)
# Nota: incluye los 5 ELA excluidos, que se filtran en label_integration.ipynb
EXPECTED_ROWS_RAW: int = 68

# Sujetos válidos tras filtrado (45 ELA + 18 controles = 63)
EXPECTED_SUBJECTS_FINAL: int = 63

# Features por vocal (sin Genero): 40 biomarcadores acústicos
EXPECTED_FEATURES_PER_VOCAL: int = 40

# Etiquetas canónicas
LABEL_ENCODING: dict[str, int] = {
    "ELA_bulbar":    0,
    "ELA_no_bulbar": 1,
    "Control":       2,
}


# ---------------------------------------------------------------------------
# 1. Carga de datos brutos
# ---------------------------------------------------------------------------

def load_raw_excel_files(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Carga los 5 archivos .xls de características acústicas (una por vocal).

    Los archivos son exportaciones de MATLAB en formato Excel 97-2003 (.xls).
    Requiere la librería xlrd instalada en el entorno (pip install xlrd).

    Parameters
    ----------
    raw_dir : str | Path
        Ruta al directorio que contiene los archivos .xls brutos.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario {'a': df_a, 'e': df_e, ...} con los DataFrames cargados.

    Raises
    ------
    FileNotFoundError
        Si alguno de los archivos esperados no existe en raw_dir.
    ImportError
        Si la librería xlrd no está instalada.
    """
    raw_dir = Path(raw_dir)
    dataframes: dict[str, pd.DataFrame] = {}

    for vocal, filename in VOCAL_FILES.items():
        filepath = raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {filepath}\n"
                f"Verifica que 'data/raw/' contiene '{filename}'."
            )
        try:
            df = pd.read_excel(filepath, engine="xlrd")
        except ImportError:
            raise ImportError(
                "La librería 'xlrd' es necesaria para leer archivos .xls.\n"
                "Instálala con: pip install xlrd"
            )

        logger.info(
            "Vocal '%s' cargada: %d sujetos × %d columnas.",
            vocal, df.shape[0], df.shape[1],
        )
        dataframes[vocal] = df

    return dataframes


# ---------------------------------------------------------------------------
# 2. Validación y unificación de la columna Genero
# ---------------------------------------------------------------------------

def validate_and_unify_gender(
    dataframes: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """
    Verifica que la columna Genero es idéntica en las 5 vocales y extrae
    una única Serie canónica.

    La columna Genero en MATLAB representa el sexo biológico del sujeto
    (0 = mujer, 1 = hombre) y es constante entre vocales del mismo sujeto.
    Las inconsistencias se registran como warning con los índices afectados.

    Parameters
    ----------
    dataframes : dict[str, pd.DataFrame]
        DataFrames cargados por load_raw_excel_files.

    Returns
    -------
    dataframes_sin_genero : dict[str, pd.DataFrame]
        DataFrames sin la columna Genero.
    genero_serie : pd.Series
        Serie única con la columna Genero unificada (nombre: 'genero').

    Raises
    ------
    KeyError
        Si algún DataFrame no contiene la columna 'Genero'.
    ValueError
        Si los valores de Genero contienen entradas fuera de {0, 1}.
    """
    gender_columns: dict[str, pd.Series] = {}

    for vocal, df in dataframes.items():
        if GENDER_COL_MATLAB not in df.columns:
            raise KeyError(
                f"Columna '{GENDER_COL_MATLAB}' no encontrada en vocal '{vocal}'. "
                f"Columnas disponibles: {list(df.columns)}"
            )
        gender_columns[vocal] = df[GENDER_COL_MATLAB].copy()

    # Verificar que todos los valores son {0, 1}
    all_values = pd.concat(gender_columns.values())
    invalid_values = set(all_values.unique()) - {0, 1}
    if invalid_values:
        raise ValueError(
            f"Valores inesperados en columna Genero: {invalid_values}. "
            f"Solo se aceptan 0 (mujer) y 1 (hombre)."
        )

    # Verificar consistencia entre vocales (vocal 'a' como referencia)
    ref_series = gender_columns["a"]
    for vocal, series in gender_columns.items():
        if vocal == "a":
            continue
        mismatches = ref_series.index[ref_series != series].tolist()
        if mismatches:
            logger.warning(
                "Inconsistencia en Genero entre vocal 'a' y vocal '%s' "
                "en índices: %s. Se usa vocal 'a' como referencia.",
                vocal, mismatches,
            )

    logger.info(
        "Columna Genero unificada en una única columna 'genero'."
    )

    dataframes_sin_genero = {
        vocal: df.drop(columns=[GENDER_COL_MATLAB])
        for vocal, df in dataframes.items()
    }
    genero_serie = ref_series.rename(GENDER_COL_FINAL)
    return dataframes_sin_genero, genero_serie


# ---------------------------------------------------------------------------
# 3. Fusión de las 5 vocales
# ---------------------------------------------------------------------------

def merge_vocal_features(
    dataframes_sin_genero: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Añade el sufijo de vocal a cada columna y fusiona horizontalmente los
    5 DataFrames.

    Ejemplo: 'f_Cres1' en vocal 'a' → 'f_Cres1_a'

    Parameters
    ----------
    dataframes_sin_genero : dict[str, pd.DataFrame]
        DataFrames ya sin la columna Genero.

    Returns
    -------
    pd.DataFrame
        DataFrame fusionado con n_sujetos × 200 columnas de features.
    """
    dfs_con_sufijo: list[pd.DataFrame] = []

    for vocal, df in dataframes_sin_genero.items():
        df_sufijo = df.add_suffix(f"_{vocal}")
        logger.info(
            "Vocal '%s': %d features renombradas con sufijo '_%s'.",
            vocal, df_sufijo.shape[1], vocal,
        )
        dfs_con_sufijo.append(df_sufijo)

    df_merged = pd.concat(dfs_con_sufijo, axis=1)
    logger.info(
        "Fusión completada: %d sujetos × %d features de voz.",
        df_merged.shape[0], df_merged.shape[1],
    )
    return df_merged


# ---------------------------------------------------------------------------
# 4. Construcción del dataset fusionado (sin etiquetas aún)
# ---------------------------------------------------------------------------

def build_raw_dataset(
    df_features: pd.DataFrame,
    genero_serie: pd.Series,
) -> pd.DataFrame:
    """
    Construye el dataset fusionado añadiendo columnas de metadatos placeholder.

    Columnas generadas:
        - 'subject_id'    : identificador provisional S001...S068
                            (se reemplazará por HUB/K en label_integration)
        - 'genero'        : sexo biológico (0=mujer, 1=hombre)
        - 'label_clinico' : etiqueta diagnóstica (placeholder = NaN)
        - 'label_maquina' : reetiquetado computacional (placeholder = NaN)

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame fusionado de features acústicas (200 columnas).
    genero_serie : pd.Series
        Serie unificada de género.

    Returns
    -------
    pd.DataFrame
        Dataset fusionado: 4 columnas metadatos + 200 features.
    """
    n = len(df_features)

    subject_ids  = pd.Series([f"S{i+1:03d}" for i in range(n)], name="subject_id")
    label_clinico = pd.Series([np.nan] * n, dtype="object", name="label_clinico")
    label_maquina = pd.Series([np.nan] * n, dtype="object", name="label_maquina")

    df_final = pd.concat(
        [
            subject_ids,
            genero_serie.reset_index(drop=True),
            label_clinico,
            label_maquina,
            df_features.reset_index(drop=True),
        ],
        axis=1,
    )

    logger.info(
        "Dataset fusionado: %d filas × %d columnas "
        "(4 metadatos + %d features).",
        df_final.shape[0], df_final.shape[1], df_features.shape[1],
    )
    return df_final


# ---------------------------------------------------------------------------
# 5. Asignación de etiquetas (llamada desde label_loader)
# ---------------------------------------------------------------------------

def assign_labels(
    df: pd.DataFrame,
    label_map: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """
    Asigna las etiquetas clínica y de reetiquetado a cada sujeto.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con columna 'subject_id' y columnas 'label_clinico',
        'label_maquina' = NaN.
    label_map : dict[str, dict[str, str]]
        Estructura:
            {
                'HUBxxx': {'label_clinico': 'ELA_bulbar',
                           'label_maquina': 'ELA_bulbar'},
                'Kxxx':   {'label_clinico': 'Control',
                           'label_maquina': 'Control'},
                ...
            }
        Valores aceptados: 'ELA_bulbar', 'ELA_no_bulbar', 'Control'.

    Returns
    -------
    pd.DataFrame
        Dataset con etiquetas asignadas.
    """
    valid_labels = set(LABEL_ENCODING.keys())
    df = df.copy()
    assigned = 0

    for subject_id, labels in label_map.items():
        mask = df["subject_id"] == subject_id
        if not mask.any():
            logger.warning(
                "subject_id '%s' del label_map no encontrado en el dataset.",
                subject_id,
            )
            continue

        for col in ("label_clinico", "label_maquina"):
            val = labels.get(col)
            if val not in valid_labels:
                raise ValueError(
                    f"Etiqueta inválida '{val}' para {col} en sujeto "
                    f"{subject_id}. Valores aceptados: {valid_labels}"
                )
            df.loc[mask, col] = val

        assigned += 1

    missing = df["label_clinico"].isna().sum()
    logger.info(
        "Etiquetas asignadas: %d sujetos. Sin etiqueta: %d sujetos.",
        assigned, missing,
    )
    if missing > 0:
        missing_ids = df.loc[df["label_clinico"].isna(), "subject_id"].tolist()
        logger.warning(
            "Sujetos sin etiqueta (excluidos o no mapeados): %s", missing_ids,
        )
    return df


# ---------------------------------------------------------------------------
# 6. Validación de integridad
# ---------------------------------------------------------------------------

def validate_dataset(
    df: pd.DataFrame,
    expected_rows: int = EXPECTED_SUBJECTS_FINAL,
) -> dict[str, object]:
    """
    Ejecuta checks de integridad sobre el dataset y devuelve un informe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a validar.
    expected_rows : int
        Número de filas esperado (63 tras filtrar excluidos, 68 antes).

    Returns
    -------
    dict
        Informe con claves: n_subjects, n_features, null_total,
        null_by_column, duplicates, gender_anomalies, zero_variance_cols.
    """
    meta_cols    = ("subject_id", "genero", "label_clinico", "label_maquina")
    feature_cols = [c for c in df.columns if c not in meta_cols]
    numeric_df   = df[feature_cols].select_dtypes(include="number")

    report: dict[str, object] = {}

    # Número de filas
    report["n_subjects"] = len(df)
    if len(df) != expected_rows:
        logger.warning(
            "Número de filas: %d (esperado: %d).", len(df), expected_rows,
        )
    else:
        logger.info("Número de filas correcto: %d.", len(df))

    report["n_features"] = len(feature_cols)
    logger.info("Features acústicas: %d.", len(feature_cols))

    # Nulos en features
    null_total  = numeric_df.isnull().sum().sum()
    null_by_col = numeric_df.isnull().sum()
    null_by_col = null_by_col[null_by_col > 0]
    report["null_total"]      = int(null_total)
    report["null_by_column"]  = null_by_col.to_dict()
    if null_total == 0:
        logger.info("Sin valores nulos en features.")
    else:
        logger.warning(
            "%d valores nulos en features. Columnas: %s",
            null_total, null_by_col.index.tolist(),
        )

    # Duplicados
    n_dup = df.duplicated().sum()
    report["duplicates"] = int(n_dup)
    if n_dup == 0:
        logger.info("Sin filas duplicadas.")
    else:
        logger.warning("%d filas duplicadas.", n_dup)

    # Anomalías de género
    gender_anomalies = df[~df["genero"].isin([0, 1])]["subject_id"].tolist()
    report["gender_anomalies"] = gender_anomalies
    if not gender_anomalies:
        logger.info("Columna 'genero' sin anomalías.")
    else:
        logger.warning("Sujetos con género inesperado: %s", gender_anomalies)

    # Features de varianza cero
    zero_var = numeric_df.columns[numeric_df.var() == 0].tolist()
    report["zero_variance_cols"] = zero_var
    if not zero_var:
        logger.info("Sin features de varianza cero.")
    else:
        logger.warning(
            "%d features constantes (varianza cero): %s",
            len(zero_var), zero_var,
        )

    logger.info("Validación completada.")
    return report


# ---------------------------------------------------------------------------
# 7. Guardado
# ---------------------------------------------------------------------------

def save_dataset(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "dataset_final.csv",
) -> Path:
    """
    Guarda el dataset en formato CSV en el directorio indicado.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a guardar.
    output_dir : str | Path
        Directorio de salida (data/processed/).
    filename : str
        Nombre del archivo CSV de salida.

    Returns
    -------
    Path
        Ruta completa al archivo guardado.
    """
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    df.to_csv(output_path, index=False)
    logger.info(
        "Dataset guardado: %s  (%d filas × %d columnas)",
        output_path, df.shape[0], df.shape[1],
    )
    return output_path


# ---------------------------------------------------------------------------
# 8. Pipeline principal
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    raw_dir: str | Path,
    processed_dir: str | Path,
    output_filename: str = "dataset_raw_fusionado.csv",
) -> pd.DataFrame:
    """
    Ejecuta el pipeline de preprocesamiento completo de forma reproducible.

    Genera el dataset acústico fusionado (68 filas × 204 columnas) con
    columnas de etiqueta = NaN. La asignación de etiquetas y el filtrado
    de los 5 sujetos excluidos se realiza en label_integration.ipynb.

    Parameters
    ----------
    raw_dir : str | Path
        Directorio con los archivos .xls brutos.
    processed_dir : str | Path
        Directorio de salida para el dataset procesado.
    output_filename : str
        Nombre del CSV de salida.

    Returns
    -------
    pd.DataFrame
        Dataset fusionado (68 filas, sin etiquetas).

    Example
    -------
    >>> from src.preprocessing import run_preprocessing_pipeline
    >>> df = run_preprocessing_pipeline(
    ...     raw_dir='data/raw',
    ...     processed_dir='data/processed',
    ... )
    >>> print(df.shape)
    (68, 204)   # 4 metadatos + 200 features
    """
    logger.info("=" * 60)
    logger.info("INICIO DEL PIPELINE DE PREPROCESAMIENTO")
    logger.info("=" * 60)

    logger.info("[1/5] Cargando archivos .xls...")
    dataframes = load_raw_excel_files(raw_dir)

    logger.info("[2/5] Validando y unificando columna Genero...")
    dataframes_sin_genero, genero_serie = validate_and_unify_gender(dataframes)

    logger.info("[3/5] Fusionando las 5 vocales...")
    df_features = merge_vocal_features(dataframes_sin_genero)

    logger.info("[4/5] Construyendo dataset fusionado...")
    df_raw = build_raw_dataset(df_features, genero_serie)

    logger.info("[5/5] Validando integridad (pre-filtrado, 68 filas)...")
    validate_dataset(df_raw, expected_rows=EXPECTED_ROWS_RAW)

    save_dataset(df_raw, processed_dir, output_filename)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETADO — siguiente paso: label_integration.ipynb")
    logger.info("=" * 60)

    return df_raw


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR      = Path(__file__).resolve().parents[1]
    RAW_DIR       = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

    df = run_preprocessing_pipeline(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
    )

    print(f"\nShape     : {df.shape}")
    print(f"Columnas  : {list(df.columns[:6])} ... [{df.shape[1]} total]")
    print(f"Nulos     : {df.isnull().sum().sum()}")
    print(f"Género    : {df['genero'].value_counts().to_dict()}")
