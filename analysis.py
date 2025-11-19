"""
Utilities for cargar y diagnosticar el inventario de activos abiertos.

El módulo expone funciones reutilizables para:
* Limpieza y enriquecimiento del CSV
* Cálculo de métricas para los objetivos específicos OE1-OE3
* Ejecución de un modelo ML sencillo (clustering por temática)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover - se maneja en runtime
    KMeans = None
    TfidfVectorizer = None


DATA_PATH = (
    Path(__file__).resolve().parent / "datasets" / "Asset_Inventory_Public_20251119.csv"
)

METADATA_FIELDS = [
    "Titulo",
    "Descripción",
    "Dueño",
    "Categoría",
    "Etiqueta",
    "Información de Datos: Frecuencia de Actualización",
    "Información de Datos: Cobertura Geográfica",
    "Información de Datos: Idioma",
    "Información de la Entidad: Nombre de la Entidad",
    "Common Core: Contact Email",
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


def load_inventory(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga el inventario y aplica enriquecimiento."""
    df = pd.read_csv(
        path,
        na_values=["", " ", "NA", "N/A", "-", "null", "None"],
        keep_default_na=True,
    )
    df = enrich_inventory(df)
    return df


def enrich_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza campos clave y genera columnas derivadas."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["is_public"] = df["Público"].str.lower().eq("public")
    df["public_access_level"] = (
        df["Common Core: Public Access Level"].str.lower().fillna("desconocido")
    )
    df["coherence_flag"] = (
        df["is_public"] & df["public_access_level"].str.contains("public")
    ) | (~df["is_public"] & ~df["public_access_level"].str.contains("public"))

    df["entidad"] = (
        df["Información de la Entidad: Nombre de la Entidad"]
        .fillna(df["Dueño"])
        .fillna("Entidad sin registro")
    )
    df["sector"] = df["Información de la Entidad: Sector"].fillna("Sector sin registro")
    df["theme_group"] = (
        df["Common Core: Theme"]
        .fillna(df["Categoría"])
        .fillna(df["Etiqueta"])
        .fillna("Tema sin registro")
    )

    df["metadata_completeness"] = _compute_row_completeness(df)
    df["metadata_segment"] = pd.cut(
        df["metadata_completeness"],
        bins=[-0.01, 0.5, 0.75, 0.9, 1.0],
        labels=["Crítico (<50%)", "Bajo (50%-75%)", "Medio (75%-90%)", "Óptimo (>90%)"],
    )

    df["views"] = pd.to_numeric(df.get("Vistas"), errors="coerce")
    df["downloads"] = pd.to_numeric(df.get("Descargas"), errors="coerce")

    ref_date = (
        df["Fecha de última actualización de datos (UTC)"].max()
        or pd.Timestamp.utcnow()
    )
    delta = ref_date - df["Fecha de última actualización de datos (UTC)"]
    df["days_since_update"] = delta.dt.days
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

    df["update_frequency_norm"] = (
        df["Información de Datos: Frecuencia de Actualización"]
        .str.lower()
        .str.strip()
        .fillna("sin registro")
    )

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
        df.groupby(["update_frequency_norm", "freshness_bucket"])
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
        "Titulo",
        "entidad",
        "sector",
        "metadata_completeness",
        "Información de Datos: Frecuencia de Actualización",
        "Common Core: Contact Email",
        "Common Core: License",
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
        df["Titulo"].fillna("")
        + " "
        + df["Descripción"].fillna("")
        + " "
        + df["Etiqueta"].fillna("")
    )
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
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    km = KMeans(n_clusters=clusters, random_state=42, n_init="auto")
    labels = pd.Series(km.labels_, index=corpus.index, name="cluster_id")
    df.loc[labels.index, "cluster_id"] = labels

    keywords = _top_keywords_per_cluster(km, vectorizer)
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
    model: KMeans, vectorizer: TfidfVectorizer, top_n: int = 5
) -> List[str]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords = []
    for centroid in model.cluster_centers_:
        idx = np.argsort(centroid)[::-1][:top_n]
        keywords.append(", ".join(feature_names[idx]))
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

