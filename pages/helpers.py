from __future__ import annotations

import plotly.express as px
from dash import dash_table

from analysis import (
    build_diagnostic_markdown,
    completeness_by_entity,
    frequency_distribution,
    metadata_gaps,
    run_basic_clustering,
    theme_coverage,
)
from components.ui import metric_card


def build_metric_cards(summary: dict) -> list:
    return [
        metric_card("Activos analizados", f"{summary['total_assets']:,}", "Total en inventario"),
        metric_card(
            "Completitud promedio",
            f"{summary['avg_metadata_completeness']:.1%}",
            "Campos críticos diligenciados",
        ),
        metric_card(
            "Coherencia publicación",
            f"{summary['coherence_ratio']:.1%}",
            "Público vs acceso declarado",
        ),
        metric_card(
            "Activos recientes",
            f"{summary['recent_updates']:,}",
            "Actualizados en ≤90 días",
        ),
    ]


def build_figures(df):
    entity_stats = completeness_by_entity(df)
    freq_stats = frequency_distribution(df)
    theme_stats = theme_coverage(df)

    hist_fig = px.histogram(
        df,
        x="metadata_completeness",
        nbins=20,
        color="metadata_segment",
        opacity=0.8,
        title="Distribución de completitud de metadatos",
    )
    hist_fig.update_layout(bargap=0.05, xaxis_tickformat=".0%")

    entity_fig = px.bar(
        entity_stats.head(15),
        x="avg_completeness",
        y="entidad",
        orientation="h",
        color="assets",
        title="Entidades con mayor completitud",
        labels={"avg_completeness": "Completitud promedio", "entidad": "Entidad"},
    )
    entity_fig.update_layout(xaxis_tickformat=".0%")

    freshness_fig = px.bar(
        freq_stats,
        x="update_frequency_norm",
        y="assets",
        color="freshness_bucket",
        title="Frecuencia declarada vs. frescura observada",
        labels={"update_frequency_norm": "Frecuencia declarada", "assets": "Activos"},
    )

    theme_fig = px.treemap(
        theme_stats,
        path=["sector", "theme_group"],
        values="assets",
        title="Cobertura temática por sector",
    )

    scatter_fig = px.scatter(
        df.sample(min(5000, len(df)), random_state=42),
        x="days_since_update",
        y="views",
        color="metadata_segment",
        hover_data=["Titulo", "entidad", "metadata_completeness"],
        title="Relación entre frescura y consumo",
        labels={
            "days_since_update": "Días desde la última actualización de datos",
            "views": "Vistas",
        },
    )

    return {
        "hist": hist_fig,
        "entity": entity_fig,
        "freshness": freshness_fig,
        "theme": theme_fig,
        "scatter": scatter_fig,
    }


def build_gap_table(df):
    gaps = metadata_gaps(df)
    return dash_table.DataTable(
        data=gaps.to_dict("records"),
        columns=[{"name": col, "id": col} for col in gaps.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
    )


def build_cluster_outputs(df):
    cluster_summary = run_basic_clustering(df)
    cluster_table = None
    cluster_keywords = "Instala scikit-learn para habilitar el agrupamiento."
    cluster_message = cluster_summary.message
    if cluster_summary.available and cluster_summary.counts is not None:
        cluster_table = dash_table.DataTable(
            data=cluster_summary.counts.to_dict("records"),
            columns=[
                {"name": "Cluster", "id": "cluster_id"},
                {"name": "Activos", "id": "assets"},
                {"name": "Completitud Promedio", "id": "avg_completeness", "type": "numeric"},
                {"name": "Mediana de Vistas", "id": "median_views", "type": "numeric"},
            ],
            style_cell={"textAlign": "center"},
            style_data_conditional=[
                {"if": {"column_id": "avg_completeness"}, "backgroundColor": "#e0f7fa"}
            ],
        )
        cluster_keywords = "\n".join(
            f"- Cluster {idx + 1}: {keywords}"
            for idx, keywords in enumerate(cluster_summary.keywords)
        )
    return cluster_summary, cluster_table, cluster_keywords, cluster_message


def build_diagnostic(summary, df):
    theme_stats = theme_coverage(df)
    freq_stats = frequency_distribution(df)
    return build_diagnostic_markdown(summary, theme_stats, freq_stats)
