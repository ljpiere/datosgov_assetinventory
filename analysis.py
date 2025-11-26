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
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import re

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


def load_inventory(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga el inventario y aplica enriquecimiento."""
    df = pd.read_csv(
        path,
        na_values=["", " ", "NA", "N/A", "-", "null", "None"],
        keep_default_na=True,
        low_memory=False,  # evita DtypeWarning por columnas mixtas
    )
    df = enrich_inventory(df)
    return df


def enrich_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza campos clave y genera columnas derivadas."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in DATE_COLUMNS:
        if col in df.columns:
            # Se usa format="mixed" para silenciar advertencias al inferir fechas heterogéneas
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
                format="mixed",
                utc=True,  # evita FutureWarning por zonas horarias mezcladas
            )

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
        df["Titulo"].fillna("")
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


def search_inventory(query: str, df: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    """Búsqueda aproximada usando TF-IDF; fallback a filtro por palabras clave."""
    query = (query or "").strip()
    if not query:
        return pd.DataFrame()

    df = _filter_search_scope(df)
    if df.empty:
        return pd.DataFrame()

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

    columns = [
        "UID",
        "Titulo",
        "entidad",
        "sector",
        "theme_group",
        "metadata_completeness",
        "days_since_update",
        "update_frequency_norm",
        "Vistas",
        "Descargas",
        "similarity",
    ]
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
- Dataset sugerido: **{top['Titulo']}** ({top['entidad']})
- Similitud estimada: **{confidence:.0%}** · Completitud: **{top['metadata_completeness']:.0%}**
- Sector/tema: **{top['sector']}** · Eje temático: **{top['theme_group']}**
- Actualizado hace **{int(top['days_since_update'])} días** (frecuencia declarada: {top['update_frequency_norm']})
- Consumo: **{vistas_int:,} vistas** · **{descargas_int:,} descargas**

Selecciona otra fila en la tabla para actualizar el reporte; se listan hasta {len(results)} coincidencias ordenadas por similitud.
""".strip()
