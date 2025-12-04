"""
Utilities for cargar y diagnosticar el inventario de activos abiertos.

El módulo expone funciones reutilizables para:
* Limpieza y enriquecimiento del CSV
* Cálculo de métricas para los objetivos específicos OE1-OE3
* Ejecución de un modelo ML sencillo (clustering por temática)

# NOTA IMPORTANTE:
Si falla al ejecutar por gpu revisar:
- Versión cuda compatible con la grafica
- Versión BitsAndBytes compatible con la grafica
- Versión torch torchvision torchaudio compatibles con la grafica

Esto puede variar dependiendo la gráfica y sus drivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ctransformers import AutoModelForCausalLM as CAutoModel
import string 
import numpy as np
import pandas as pd
import re
import requests
import unicodedata
import time
import pickle
import os
from huggingface_hub import login
import gc
import torch
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from context import GOV_ACRONYMS, GUIA_CALIDAD_TEXT


def ensure_hf_login() -> None:
    """Intenta login no interactivo usando variable de entorno HF_TOKEN o HUGGINGFACEHUB_API_TOKEN."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        # Sin token no intentamos login para evitar prompt interactivo en contenedores.
        return
    try:
        login(token=token, add_to_git_credential=False)
    except Exception as exc:  # pragma: no cover - solo loguea
        print(f"Aviso: no se pudo iniciar sesion en HuggingFace: {exc}")


ensure_hf_login()

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover - se maneja en runtime
    KMeans = None
    TfidfVectorizer = None

# Guardará el dataframe localmente para evitar recargas innecesarias.
CACHE_DIR = Path("dataframe")
CACHE_FILE = CACHE_DIR / "inventory_cache.pkl"
# guardará el modelo localmente para evitar recargas innecesarias.
CACHE_DIR_MODEL = "./model"
# url de la API del inventario
URL = "https://www.datos.gov.co/resource/uzcf-b9dh.json"

METADATA_FIELDS = [
    "name",
    "Descripción",
    "Dueño",
    "Categoría",
    "Etiqueta",
    "Información de Datos: Frecuencia de Actualización",
    "Información de Datos: Cobertura Geográfica",
    "Información de Datos: Idioma",
    "Información de la Entidad: Nombre de la Entidad",
    "Common Core: Contact Email",
    "email",
    "Common Core: License",
    "Common Core: Public Access Level",
]

DATE_COLUMNS = {
    "Fecha de creación (UTC)",
    "Fecha de última actualización de metadatos (UTC)",
    "Fecha de última actualización de datos (UTC)",
    "Common Core: Issued",
    "Common Core: Last Update",
}

# Lista breve de stopwords en español para la búsqueda aproximada y clustering
STOP_WORDS_ES = [
    "de",
    "la",
    "el",
    "en",
    "y",
    "a",
    "los",
    "del",
    "por",
    "para",
    "con",
    "un",
    "una",
    "las",
    "se",
    "que",
    "al",
    "su",
    "sobre",
    "lo",
    "como",
    "datos",
    "informacion",
    "información",
    "datos obligatorios",
    "obligatorios",
    "activo",
    "activos",
    "div",
    "span",
    "style",
    "font",
    "http",
    "https",
    "www",
]

# Normaliza nombres de columnas provenientes de CSV (esp) y API (en) al formato usado en la app
COLUMN_ALIASES = {
    "uid": "UID",
    "name": "name",
    "title": "name",
    "titulo": "name",
    "description": "Descripción",
    "descripcion": "Descripción",
    "owner": "Dueño",
    "dueno": "Dueño",
    "category": "Categoría",
    "categoria": "Categoría",
    "tags": "Etiqueta",
    "etiqueta": "Etiqueta",
    "audience": "Público",
    "publico": "Público",
    "commoncore_publicaccesslevel": "Common Core: Public Access Level",
    "common_core_public_access_level": "Common Core: Public Access Level",
    "informacindelaentidad_nombredelaentidad": "Información de la Entidad: Nombre de la Entidad",
    "informacion_de_la_entidad_nombre_de_la_entidad": "Información de la Entidad: Nombre de la Entidad",
    "informacindelaentidad_sector": "Información de la Entidad: Sector",
    "informacion_de_la_entidad_sector": "Información de la Entidad: Sector",
    "commoncore_theme": "Common Core: Theme",
    "common_core_theme": "Common Core: Theme",
    "visits": "Vistas",
    "visitas": "Vistas",
    "downloads": "Descargas",
    "descargas": "Descargas",
    "last_data_updated_date": "Fecha de última actualización de datos (UTC)",
    "fecha_de_ultima_actualizacion_de_datos_utc": "Fecha de última actualización de datos (UTC)",
    "last_metadata_updated_date": "Fecha de última actualización de metadatos (UTC)",
    "fecha_de_ultima_actualizacion_de_metadatos_utc": "Fecha de última actualización de metadatos (UTC)",
    "creation_date": "Fecha de creación (UTC)",
    "fecha_de_creacion_utc": "Fecha de creación (UTC)",
    "commoncore_issued": "Common Core: Issued",
    "common_core_issued": "Common Core: Issued",
    "commoncore_lastupdate": "Common Core: Last Update",
    "common_core_last_update": "Common Core: Last Update",
    "informacindedatos_frecuenciadeactualizacin": "Información de Datos: Frecuencia de Actualización",
    "informacion_de_datos_frecuencia_de_actualizacion": "Información de Datos: Frecuencia de Actualización",
    "informacindedatos_coberturageogrfica": "Información de Datos: Cobertura Geográfica",
    "informacindedatos_idioma": "Información de Datos: Idioma",
    "informacion_de_datos_cobertura_geografica": "Información de Datos: Cobertura Geográfica",
    "informacion_de_datos_idioma": "Información de Datos: Idioma",
    "commoncore_contactemail": "Common Core: Contact Email",
    "common_core_contact_email": "Common Core: Contact Email",
    "contact_email": "email", 
    "email": "Contact email",
    "commoncore_license": "Common Core: License",
    "common_core_license": "Common Core: License",
}

def load_api(path: str, batch_size: int = 5000, offset: int = 0) -> pd.DataFrame:
    """ carga el inventario desde la API en modo batch 
    """
    all_records = []
    session = requests.Session()
    
    try:
        while True:
            params = {"$limit": batch_size, "$offset": offset}
            response = session.get(path, params=params, timeout=100)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                break
            
            all_records.extend(data)
            offset += batch_size

    finally:
        session.close()

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    cleanup_map = {
        "": np.nan,
        " ": np.nan,
        "NA": np.nan,
        "N/A": np.nan,
        "-": np.nan,
        "null": np.nan,
        "None": np.nan
    }
    df = df.replace(cleanup_map)
    df = df.infer_objects(copy=False)
    
    print(f"Columnas cargadas: {list(df.columns)}")
    return df


def load_inventory(path: str = URL, batch_size: int = 5000, offset: int = 0, force_update: bool = False) -> pd.DataFrame:
    """
    Carga el inventario. 
    Verifica si existe la carpeta 'dataframe', si no, la crea.
    Busca el archivo .pkl.
    Si el archivo existe y tiene menos de 12 horas, lo carga.
    Si es viejo (>12h) o no existe, descarga de la API y lo guarda.
    """
    
    # Asegurar que el directorio de caché exista
    if not CACHE_DIR.exists():
        print(f"Creando directorio de caché: {CACHE_DIR}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Definir expiración 12 horas en segundos
    CACHE_DURATION_SECONDS = 12 * 3600
    
    # Verificar estado del archivo en disco
    file_exists = CACHE_FILE.exists()
    is_fresh = False
    
    if file_exists:
        last_modified = CACHE_FILE.stat().st_mtime
        time_since_update = time.time() - last_modified
        is_fresh = time_since_update < CACHE_DURATION_SECONDS
        
        if is_fresh:
            print(f"Caché válida (actualizada hace {time_since_update/3600:.1f} horas).")
        else:
            print(f"Caché expirada (hace {time_since_update/3600:.1f} horas). Se actualizará.")

    # Lógica de carga
    if file_exists and is_fresh and not force_update:
        print(f"Cargando inventario desde: {CACHE_FILE} ...")
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error leyendo el archivo local ({e}), intentando descargar...")

    # Descarga desde API (si no hay caché, está vieja o force_update=True)
    print("Iniciando descarga de datos actualizados desde la API...")
    try:
        df = load_api(path, batch_size=batch_size, offset=offset)
        df = enrich_inventory(df)
        
        # Guardar en disco
        print(f"Guardando datos actualizados en {CACHE_FILE} ...")
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(df, f)
            
        return df
        
    except Exception as e:
        # Si falla la API pero tenemos un archivo viejo, usémoslo como emergencia
        if file_exists:
            print(f"Error conectando a API ({e}). Usando caché antigua por seguridad.")
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        raise e

def enrich_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza campos clave y genera columnas derivadas."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = _apply_aliases(df)

    if "type" in df.columns:
        df["derived_view"] = df["type"].apply(lambda x: str(x).lower() not in ["dataset", "table"])
    elif "Tipo" in df.columns:
        df["derived_view"] = df["Tipo"].apply(lambda x: str(x).lower() not in ["dataset", "table"])
    else:
        df["derived_view"] = False

    if "domain" not in df.columns and "Dominio" not in df.columns:
        df["domain"] = "www.datos.gov.co"

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
                format="mixed",
                utc=True,  
            )

    audience_col = "Público" if "Público" in df.columns else "audience"
    df["is_public"] = (
        df[audience_col].astype(str).str.lower().eq("public") if audience_col in df.columns else False
    )
    if "Common Core: Public Access Level" in df.columns:
        df["public_access_level"] = (
            df["Common Core: Public Access Level"].astype(str).str.lower().fillna("desconocido")
        )
    else:
        df["public_access_level"] = "desconocido"

    if "type" in df.columns:
        type_series = df["type"] = df["type"].astype(str).str.lower()
    elif "Tipo" in df.columns:
        type_series = df["Tipo"].astype(str).str.lower()
    else:
        type_series = pd.Series("", index=df.index)
    
    is_federated = type_series.str.contains("federated_href", na=False)
    # si es federado comprueba si es publico
    strict_check = (df["is_public"] & df["public_access_level"].str.contains("public")|~df["is_public"] & ~df["public_access_level"].str.contains("public"))

    # para no penalizar los archivos no federados
    df["coherence_flag"] = (~is_federated) | strict_check

    entidad_series = df.get("Información de la Entidad: Nombre de la Entidad")
    owner_series = df.get("Dueño")
    if entidad_series is None:
        entidad_series = pd.Series("Entidad sin registro", index=df.index)
    df["entidad"] = entidad_series.fillna(owner_series).fillna("Entidad sin registro")

    sector_series = df.get("Información de la Entidad: Sector")
    df["sector"] = sector_series.fillna("Sector sin registro") if sector_series is not None else "Sector sin registro"

    theme_series = df.get("Common Core: Theme")
    category_series = df.get("Categoría")
    tag_series = df.get("Etiqueta")
    theme_group = theme_series if theme_series is not None else pd.Series(index=df.index, dtype=object)
    if category_series is not None:
        theme_group = theme_group.fillna(category_series)
    if tag_series is not None:
        theme_group = theme_group.fillna(tag_series)
    df["theme_group"] = theme_group.fillna("Tema sin registro")

    df["metadata_completeness"] = _compute_row_completeness(df)
    df["metadata_segment"] = pd.cut(
        df["metadata_completeness"],
        bins=[-0.01, 0.5, 0.75, 0.9, 1.0],
        labels=["Crítico (<50%)", "Bajo (50%-75%)", "Medio (75%-90%)", "Óptimo (>90%)"],
    )

    df["views"] = pd.to_numeric(df.get("Vistas"), errors="coerce")
    df["downloads"] = pd.to_numeric(df.get("Descargas"), errors="coerce")

    date_candidates = [
        col
        for col in [
            "Fecha de última actualización de datos (UTC)",
            "Fecha de última actualización de metadatos (UTC)",
            "Common Core: Last Update",
        ]
        if col in df.columns
    ]

    if date_candidates:
        df["effective_updated_date"] = df[date_candidates].max(axis=1)
        if "Fecha de creación (UTC)" in df.columns:
            df["effective_updated_date"] = df["effective_updated_date"].fillna(
                df["Fecha de creación (UTC)"]
            )
    else:
        df["effective_updated_date"] = pd.Timestamp.utcnow()

    ref_date = df["effective_updated_date"].max() or pd.Timestamp.utcnow()
    delta = ref_date - df["effective_updated_date"]
    df["days_since_update"] = delta.dt.days.fillna(9999)
    df["freshness_bucket"] = pd.cut(
        df["days_since_update"],
        bins=[-1, 30, 90, 180, 365, 10_000],
        labels=[
            "≤30 días",
            "31-90 días",
            "91-180 días",
            "181-365 días",
            ">365 días",
        ],
    )

    if "Información de Datos: Frecuencia de Actualización" in df.columns:
        df["update_frequency_norm"] = (
            df["Información de Datos: Frecuencia de Actualización"]
            .astype(str)
            .str.lower()
            .str.strip()
            .fillna("sin registro")
        )
    else:
        df["update_frequency_norm"] = "sin registro"

    return df


def _normalize_label(label: str) -> str:
    cleaned = unicodedata.normalize("NFKD", str(label))
    cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", cleaned.lower())
    return cleaned.strip("_")


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    normalized_cols = {_normalize_label(col): col for col in df.columns}
    rename_map = {}
    for alias_key, target in COLUMN_ALIASES.items():
        if target in df.columns:
            continue
        source_col = normalized_cols.get(alias_key)
        if source_col:
            rename_map[source_col] = target

    if rename_map:
        df = df.rename(columns=rename_map)

    if "name" not in df.columns:
        fallback_col = normalized_cols.get("titulo") or normalized_cols.get("title")
        if fallback_col and fallback_col in df.columns:
            df["name"] = df[fallback_col]

    return df


def _compute_row_completeness(df: pd.DataFrame) -> pd.Series:
    available_cols = [c for c in METADATA_FIELDS if c in df.columns]
    if not available_cols:
        return pd.Series(0.0, index=df.index)

    mask = (
        df[available_cols]
        .fillna("")
        .apply(lambda s: s.astype(str).str.strip().ne(""), axis=0)
    )
    return mask.mean(axis=1).round(3)


def compute_summary(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "total_assets": int(len(df)),
        "public_assets": int(df["is_public"].sum()),
        "coherence_ratio": float(df["coherence_flag"].mean()),
        "avg_metadata_completeness": float(df["metadata_completeness"].mean()),
        "recent_updates": int(df["days_since_update"].le(90).sum()),
        "median_views": float(df["views"].median(skipna=True) or 0),
    }


def completeness_by_entity(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.groupby("entidad")
        .agg(
            assets=("UID", "count"),
            avg_completeness=("metadata_completeness", "mean"),
            median_views=("views", "median"),
        )
        .sort_values("avg_completeness", ascending=False)
        .reset_index()
    )
    return pivot


def frequency_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["update_frequency_norm", "freshness_bucket"], observed=False
        )  # observed=False mantiene comportamiento actual y silencia FutureWarning
        .size()
        .reset_index(name="assets")
    )


def theme_coverage(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["theme_group", "sector"])
        .size()
        .reset_index(name="assets")
        .sort_values("assets", ascending=False)
    )


def metadata_gaps(df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    columns = [
        "UID",
        "name",
        "entidad",
        "sector",
        "metadata_completeness",
        "Información de Datos: Frecuencia de Actualización",
        "Common Core: Contact Email",
        "email",
        "Common Core: License",
        "Common Core: Public Access Level"
    ]
    present_cols = [c for c in columns if c in df.columns]
    return (
        df.nsmallest(limit, "metadata_completeness")[present_cols]
        .reset_index(drop=True)
    )


@dataclass
class ClusterSummary:
    labels: Optional[pd.Series]
    keywords: List[str]
    counts: Optional[pd.DataFrame]
    available: bool
    message: str


def run_basic_clustering(df: pd.DataFrame, n_clusters: int = 6) -> ClusterSummary:
    """Agrupa activos por similitud textual para apoyar OE1/OE3."""
    if TfidfVectorizer is None or KMeans is None:
        return ClusterSummary(
            labels=None,
            keywords=[],
            counts=None,
            available=False,
            message="Scikit-learn no está instalado. Instala los requisitos para habilitar el clustering.",
        )

    text = (
        df["name"].fillna("")
        + " "
        + df["Descripción"].fillna("")
        + " "
        + df["Etiqueta"].fillna("")
    ).str.lower()
    text = text.str.replace(r"<[^>]+>", " ", regex=True)
    text = text.str.replace(r"&nbsp;", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True)
    valid_idx = text.str.strip().ne("")
    corpus = text[valid_idx]
    if corpus.empty:
        return ClusterSummary(
            labels=None,
            keywords=[],
            counts=None,
            available=False,
            message="No hay texto disponible para agrupar.",
        )

    clusters = min(n_clusters, corpus.nunique(), 10)
    vectorizer = TfidfVectorizer(
        max_features=1500, ngram_range=(1, 2), stop_words=STOP_WORDS_ES
    )
    matrix = vectorizer.fit_transform(corpus)
    km = KMeans(n_clusters=clusters, random_state=42, n_init="auto").fit(matrix)
    labels = pd.Series(km.labels_, index=corpus.index, name="cluster_id")
    df.loc[labels.index, "cluster_id"] = labels

    keywords = _top_keywords_per_cluster(km, vectorizer, stop_words=STOP_WORDS_ES)
    counts = (
        df.groupby("cluster_id")
        .agg(
            assets=("UID", "count"),
            avg_completeness=("metadata_completeness", "mean"),
            median_views=("views", "median"),
        )
        .reset_index()
        .sort_values("assets", ascending=False)
    )

    return ClusterSummary(
        labels=labels,
        keywords=keywords,
        counts=counts,
        available=True,
        message="Clustering completado con éxito.",
    )


def _top_keywords_per_cluster(
    model: KMeans,
    vectorizer: TfidfVectorizer,
    top_n: int = 5,
    stop_words: Optional[List[str]] = None,
) -> List[str]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords = []
    for centroid in model.cluster_centers_:
        idx = np.argsort(centroid)[::-1]
        cleaned = []
        for i in idx:
            token = feature_names[i]
            if stop_words and token in stop_words:
                continue
            if token.isdigit():
                continue
            cleaned.append(token)
            if len(cleaned) >= top_n:
                break
        keywords.append(", ".join(cleaned))
    return keywords


def build_diagnostic_markdown(
    summary: Dict[str, float],
    theme_df: pd.DataFrame,
    freq_df: pd.DataFrame,
) -> str:
    top_theme = (
        theme_df.groupby("theme_group")["assets"].sum().idxmax()
        if not theme_df.empty
        else "Sin tema predominante"
    )
    freq_rank = (
        freq_df.groupby("update_frequency_norm")["assets"]
        .sum()
        .sort_values(ascending=False)
    )
    top_freq = freq_rank.index[0] if not freq_rank.empty else "sin registro"

    return f"""
### Diagnóstico Ejecutivo
- Inventario total: **{summary['total_assets']:,}** activos ({summary['public_assets']:,} públicos).
- Coherencia publicación vs. acceso: **{summary['coherence_ratio']:.1%}** de los activos mantiene consistencia.
- Completitud promedio de metadatos: **{summary['avg_metadata_completeness']:.1%}**, con {summary['recent_updates']:,} activos actualizados en ≤90 días.
- Tema con mayor cobertura: **{top_theme}**. Frecuencia declarada más común: **{top_freq}**.
- Mediana de vistas por activo: **{summary['median_views']:.0f}**.

Estas señales permiten priorizar acciones de mejora continua y orientar aperturas futuras.
""".strip()


def agent_flow(summary: Dict[str, float]) -> List[Dict[str, str]]:
    """Describe un flujo básico de agente que consume las métricas calculadas."""
    return [
        {
            "title": "OE1 · Diagnóstico",
            "status": "Completado",
            "detail": f"{summary['coherence_ratio']:.1%} coherencia y {summary['total_assets']:,} activos analizados.",
        },
        {
            "title": "OE2 · Métricas",
            "status": "En ejecución",
            "detail": "Completitud promedio y frecuencia de actualización listadas para monitoreo.",
        },
        {
            "title": "OE3 · Informe",
            "status": "Disponible",
            "detail": "Panel interactivo y markdown de diagnóstico generados.",
        },
    ]


def _build_text_corpus(df: pd.DataFrame) -> pd.Series:
    """Concatena y limpia campos relevantes para búsqueda aproximada."""
    combined = (
        df["name"].fillna("")
        + " "
        + df["Descripción"].fillna("")
        + " "
        + df["Etiqueta"].fillna("")
        + " "
        + df["Categoría"].fillna("")
        + " "
        + df["theme_group"].fillna("")
    ).str.lower()
    # Limpia etiquetas HTML y entidades básicas
    combined = combined.str.replace(r"<[^>]+>", " ", regex=True)
    combined = combined.str.replace(r"&nbsp;", " ", regex=True)
    combined = combined.str.replace(r"\s+", " ", regex=True)
    return combined


def _filter_search_scope(
    df: pd.DataFrame, allowed_types: Optional[Sequence[str]] = ("dataset",)
) -> pd.DataFrame:
    """
    Restringe el universo de búsqueda a datasets públicos con UID válido.

    - type/Tipo == "dataset"
    - Excluye UIDs con formato "uid:<n>" o vacíos
    - publication_storage/Público marcado como público
    """
    filtered = df.copy()

    type_column = "type" if "type" in filtered.columns else "Tipo" if "Tipo" in filtered.columns else None
    if allowed_types is not None and type_column:
        allowed = {t.lower() for t in allowed_types}
        filtered = filtered[filtered[type_column].fillna("").str.lower().isin(allowed)]

    if "UID" in filtered.columns:
        uid_series = filtered["UID"].astype(str)
        valid_uid = uid_series.str.strip().ne("") & ~uid_series.str.match(
            r"uid:\d+$", case=False, na=False
        )
        filtered = filtered[valid_uid]

    if "publication_storage" in filtered.columns:
        storage = filtered["publication_storage"].fillna("").str.lower()
        filtered = filtered[storage.isin({"publico", "public"})]
    elif "Público" in filtered.columns:
        filtered = filtered[filtered["Público"].fillna("").str.lower().eq("public")]

    return filtered


def get_dataset_by_uid(uid: str, df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el dataset que coincide con el UID, aplicando el filtro de alcance."""
    uid = (uid or "").strip().lower()
    if not uid:
        return pd.DataFrame()

    scoped = _filter_search_scope(df, allowed_types=None)
    if "UID" not in scoped.columns or scoped.empty:
        return pd.DataFrame()

    matches = scoped[scoped["UID"].fillna("").str.lower() == uid]
    return matches


def get_public_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el universo filtrado para reportería masiva (UID válido y público)."""
    return _filter_search_scope(df, allowed_types=None)


def search_inventory(query: str, df: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    """Búsqueda aproximada usando TF-IDF; fallback a filtro por palabras clave."""
    query = (query or "").strip()
    if not query:
        return pd.DataFrame()

    df = _filter_search_scope(df)
    if df.empty:
        return pd.DataFrame()

    uid_pattern = r'^[a-z0-9]{4}-[a-z0-9]{4}$'

    columns = [
        "UID",
        "name",
        "entidad",
        "sector",
        "theme_group",
        "metadata_completeness",
        "days_since_update",
        "coherence_flag",
        "update_frequency_norm",
        "Vistas",
        "Descargas",
        "similarity",
        "Descripción",       
        "url",               
        "Tipo",              
        "type",              
        "Dominio",           
        "domain",            
        "Licencia",          
        "Common Core: License", 
        "Common Core: Public Access Level",
        "derived_view",      
        "Categoría",         
        "row_count",         
        "api_endpoint",      
        "Público",           
        "Common Core: Public Access Level",
        "Common Core: Contact Email",
        "email",
        "Información de Datos: Cobertura Geográfica",
        "Etiqueta",
        "Información de Datos: Frecuencia de Actualización",
        "Información de Datos: Idioma",
        "Información de la Entidad: Nombre de la Entidad",
    ]

    if re.match(uid_pattern, query.lower()):
        exact_match = get_dataset_by_uid(query, df)
        if not exact_match.empty:
            results = exact_match.copy()
            results["similarity"] = 1.0

            present_cols = [c for c in columns if c in results.columns]
            return results.loc[:, present_cols].reset_index(drop=True)

    corpus = _build_text_corpus(df)
    scores = pd.Series(0.0, index=df.index)
    tfidf_max = 0.0
    if TfidfVectorizer is not None:
        try:
            vectorizer = TfidfVectorizer(
                max_features=2000, ngram_range=(1, 2), stop_words=STOP_WORDS_ES
            )
            matrix = vectorizer.fit_transform(corpus)
            query_vec = vectorizer.transform([query.lower()])
            tfidf_array = (matrix @ query_vec.T).toarray().ravel()
            scores = pd.Series(tfidf_array, index=df.index)
            tfidf_max = float(scores.max()) if len(scores) else 0.0
        except Exception:
            # fallback silencioso cuando TF-IDF falla
            scores = pd.Series(0.0, index=df.index)
            tfidf_max = 0.0

    keywords = [w for w in re.split(r"\W+", query.lower()) if len(w) > 2]

    # Fallback adicional cuando TF-IDF no aporta similitud
    if tfidf_max == 0:
        keyword_scores = corpus.apply(
            lambda text: sum(kw in text for kw in keywords)
        ).astype(float)
        scores = pd.Series(keyword_scores, index=df.index)
        # si aún así todo es cero, usa coincidencia de substring simple
        if scores.max() == 0 and keywords:
            scores = corpus.str.contains("|".join(map(re.escape, keywords)), case=False).astype(
                float
            )
        # si seguimos sin señal, intenta coincidencia directa con el query completo
        if scores.max() == 0:
            scores = corpus.str.contains(re.escape(query.lower()), case=False).astype(float)
            # última defensa: asigna pequeña señal a todos para devolver top_k
            if scores.max() == 0:
                scores = pd.Series(1.0, index=corpus.index)

    results = df.copy()
    results["similarity"] = scores
    results["metadata_completeness"] = results["metadata_completeness"].fillna(0.0)
    results["days_since_update"] = results["days_since_update"].fillna(9999)

    present_cols = [c for c in columns if c in results.columns]

    return (
        results.sort_values("similarity", ascending=False)
        .head(top_k)
        .loc[:, present_cols]
        .reset_index(drop=True)
    )


def build_search_report(
    query: str, results: pd.DataFrame, threshold: float = 0.05, row_index: int = 0
) -> str:
    """Genera un breve reporte sobre el match seleccionado."""
    if results.empty:
        return f"No se encontraron activos relacionados con la consulta: **{query}**."

    idx = row_index if 0 <= row_index < len(results) else 0
    top = results.iloc[idx]
    confidence = top.get("similarity", 0)
    match_note = (
        "Hallazgo sólido" if confidence >= threshold else "Hallazgo aproximado"
    )

    vistas = pd.to_numeric(top.get("Vistas"), errors="coerce")
    descargas = pd.to_numeric(top.get("Descargas"), errors="coerce")
    vistas_int = int(vistas) if pd.notna(vistas) else 0
    descargas_int = int(descargas) if pd.notna(descargas) else 0

    return f"""
### Recomendación · {match_note}
- Consulta: "**{query}**"
- Dataset sugerido: **{top['name']}** ({top['entidad']})
- Similitud estimada: **{confidence:.0%}** · Completitud: **{top['metadata_completeness']:.0%}**
- Sector/tema: **{top['sector']}** · Eje temático: **{top['theme_group']}**
- Actualizado hace **{int(top['days_since_update'])} días** (frecuencia declarada: {top['update_frequency_norm']})
- Consumo: **{vistas_int:,} vistas** · **{descargas_int:,} descargas**

Selecciona otra fila en la tabla para actualizar el reporte; se listan hasta {len(results)} coincidencias ordenadas por similitud.
""".strip()


# CLASE ORBIT

class OrbiAssistant:
    def __init__(self, model_id="NicolasRodriguez/manaba_gemma_2_2b"):
        # carga el modelo gemma
        print("\nINICIANDO MANABA...")

        # Cargar Base de Conocimiento RAG
        print("Indexando Guía de Calidad...")
        self.guide_chunks = [c.strip() for c in GUIA_CALIDAD_TEXT.split('\n\n') if c.strip()]
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.guide_embeddings = self.embedder.encode(self.guide_chunks, convert_to_tensor=True)

        # Cargar Inventario de Datos
        print("Cargando Inventario de Datos...")
        self.inventory_df = load_inventory()

        # Cargar Modelo Gemma
        print(f"Descargando y cargando modelo LLM ({model_id})...")
        try:
            # GPU con BitsAndBytes 
            print("Intentando carga con GPU (4-bit)...")
            # compresion de 4 bits para GPUs con poca memoria
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=CACHE_DIR_MODEL)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                cache_dir=CACHE_DIR_MODEL
            )
            print("Modelo cargado exitosamente en GPU.")
            
        except Exception as e:
                print(f"Advertencia: Falló la carga en GPU. Causa: {e}")
                print("Cambiando a modo CPU...")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=CACHE_DIR_MODEL)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        cache_dir=CACHE_DIR_MODEL
                    )
                    print(">>> Modelo cargado exitosamente en modo CPU.")
                except Exception as e2:
                    print(f"Error crítico: No se pudo cargar el modelo ni en GPU ni en CPU.")
                    print(f"Detalle del error: {e2}")
                    raise e2

    def _retrieve_guide_info(self, query):
        """Busca información en la guía normativa."""
        if not self.guide_chunks: return None
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.guide_embeddings)[0]
        top_results = torch.topk(scores, k=3)

        context = []
        for score, idx in zip(top_results.values, top_results.indices):
            if score > 0.25:
                context.append(self.guide_chunks[idx])
        return "\n".join(context) if context else None

    def _expand_acronyms(self, text):
        """Mejora la consulta expandiendo siglas conocidas."""
        words = text.lower().split()
        expanded_query = []
        found_meaning = None
        
        for w in words:
            # Limpiar puntuación
            clean_w = w.strip(string.punctuation)
            if clean_w in GOV_ACRONYMS:
                meaning = GOV_ACRONYMS[clean_w]
                expanded_query.append(f"{w} ({meaning})")
                found_meaning = meaning
            else:
                expanded_query.append(w)
        
        return " ".join(expanded_query), found_meaning

    def _generate_response(self, system_instruction, user_query, context_data="", history=[]):
        """
        Genera respuesta con Gemma usando un prompt estructurado para evitar alucinaciones.
        """
        history_str = ""
        if history:
            recent_history = history[-6:]
            for msg in recent_history:
                role = "model" if msg['role'] == "orbit" else "user"
                # Limpiamos el contenido histórico para no pasar contexto basura
                content = msg['content'].replace("\n", " ")[:200] 
                history_str += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"

        # Prompt
        prompt = f"""<start_of_turn>user
Eres Manaba, un oso de anteojos experto en Datos Abiertos de Colombia.
Recuerda ya saludaste al usuario.
Responde al usuario con un sonido de oso (*Grrr*) 
Tu misión es ayudar a encontrar datasets o explicar normas de calidad.
NO inventes significados de palabras. Si te dan contexto de datos, ÚSALO.

CONTEXTO DE DATOS ENCONTRADOS:
{context_data if context_data else "No se encontraron datasets específicos en esta búsqueda."}

INSTRUCCIÓN:
{system_instruction}

PREGUNTA DEL USUARIO:
{user_query}

Responde brevemente como un oso amable.<end_of_turn>
<start_of_turn>model
"""     
        device = self.model.device 
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=4096, temperature=0.2, repetition_penalty=1.1)
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("model\n")[-1].strip()

    def chat(self, message, history):
        """Función principal."""
        gc.collect()
        torch.cuda.empty_cache()

        msg_lower = message.lower()
        msg_clean = msg_lower.translate(str.maketrans('', '', string.punctuation))
        
        # SALUDO INICIAL
        greetings = ["hola", "buenos dias", "buenas", "hi", "hello", "holis"]
        is_greeting = msg_clean in greetings or (len(msg_clean.split()) <= 2 and "hola" in msg_clean)

        if is_greeting:
            return self._generate_response(
                "pregúntale qué dato público de Colombia busca hoy.", 
                message, 
                "",
                history
            )

        # NORMATIVA DE CALIDAD
        quality_keywords = ["error", "err", "calidad", "criterio", "norma", "guía", "sello", "iso", "interoperabilidad"]
        if any(k in msg_lower for k in quality_keywords):
            context = self._retrieve_guide_info(message)
            if context:
                return self._generate_response(
                    "Responde la duda técnica usando estrictamente el contexto de la guía proporcionado.", 
                    message, 
                    context, 
                    history
                )
        
        # Expansión de siglas
        search_query, acronym_meaning = self._expand_acronyms(message)
        
        # Buscar en el inventario con la query expandida
        # Usamos la query la expandida para asegurar match en TF-IDF
        results = search_inventory(search_query, self.inventory_df, top_k=10)

        if not results.empty:
            # Construir contexto enriquecido
            data_context = ""
            for i, row in results.iterrows():
                # Añadimos métricas clave para que el modelo pueda opinar sobre la calidad
                data_context += f"- Dataset: {row['name']}\n  Entidad: {row.get('entidad', 'N/A')}\n  Actualizado hace: {int(row.get('days_since_update', 999))} días\n  UID: {row.get('UID', 'N/A')}\n"
            
            instruction = "El usuario busca datos. Dile que has encontrado estos recursos en el bosque de datos. Menciona el nombre de los primeros 3 datasets con su explicación, su uid y su entidad. Si alguno está muy desactualizado (más de 365 días), adviértelo amablemente."
            
            # Si detectamos una sigla, reforzamos el contexto
            if acronym_meaning:
                instruction += f" Nota: El usuario buscó '{message}', que entiendo como '{acronym_meaning}'."

            return self._generate_response(instruction, message, data_context, history)

        # No encontró datos
        else:
            # Si detectamos una sigla pero no hubo datos
            if acronym_meaning:
                fallback_msg = f"Grrr... busqué datos sobre **{acronym_meaning}** ({message}) pero no encontré nada específico en el inventario cargado. ¿Quizás intentar con la entidad (ej: Alcaldía de...)? 🐻"
                return fallback_msg
            
            # Respuesta genérica controlada para evitar alucinaciones
            return self._generate_response(
                "No encontraste datasets exactos. Pide al usuario que sea más específico con la entidad o el tema. NO intentes adivinar qué significa la palabra si no la sabes.",
                message,
                "",
                history
            )
