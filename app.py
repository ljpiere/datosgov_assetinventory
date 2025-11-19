from __future__ import annotations

import dash
from dash import Dash, dash_table, dcc, html
import plotly.express as px

from analysis import (
    agent_flow,
    build_diagnostic_markdown,
    completeness_by_entity,
    frequency_distribution,
    load_inventory,
    metadata_gaps,
    run_basic_clustering,
    theme_coverage,
    compute_summary,
)


STYLES = {
    "container": {"padding": "2rem", "fontFamily": "Inter, Arial, sans-serif"},
    "metric_grid": {
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
        "gap": "1rem",
        "marginBottom": "1rem",
    },
    "metric_card": {
        "border": "1px solid #e0e0e0",
        "borderRadius": "10px",
        "padding": "1rem",
        "backgroundColor": "#f8f9fb",
    },
    "graph_grid": {
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
        "gap": "1.5rem",
    },
    "agent_grid": {
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
        "gap": "1rem",
        "marginBottom": "1.5rem",
    },
}


def metric_card(title: str, value: str, detail: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div(value, className="metric-value"),
            html.Small(detail, className="metric-detail"),
        ],
        style=STYLES["metric_card"],
    )


def agent_step_card(step: dict) -> html.Div:
    return html.Div(
        [
            html.Div(step["title"], className="agent-title"),
            html.Div(step["status"], className="agent-status"),
            html.Small(step["detail"], className="agent-detail"),
        ],
        style={
            "border": "1px solid #dfe3eb",
            "borderRadius": "8px",
            "padding": "0.75rem",
            "backgroundColor": "#ffffff",
        },
    )


def build_figures():
    df = load_inventory()
    summary = compute_summary(df)
    entity_stats = completeness_by_entity(df)
    freq_stats = frequency_distribution(df)
    theme_stats = theme_coverage(df)
    gaps = metadata_gaps(df)
    cluster_summary = run_basic_clustering(df)
    diagnostic_md = build_diagnostic_markdown(summary, theme_stats, freq_stats)
    agent_steps = agent_flow(summary)

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

    cluster_table = None
    cluster_keywords = "Instala scikit-learn para habilitar el agrupamiento."
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
                {
                    "if": {"column_id": "avg_completeness"},
                    "backgroundColor": "#e0f7fa",
                }
            ],
        )
        cluster_keywords = "\n".join(
            f"- Cluster {idx + 1}: {keywords}"
            for idx, keywords in enumerate(cluster_summary.keywords)
        )

    gap_table = dash_table.DataTable(
        data=gaps.to_dict("records"),
        columns=[{"name": col, "id": col} for col in gaps.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
    )

    cards = [
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

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H1("Diagnóstico inteligente de activos de datos abiertos"),
            html.P(
                "Flujo básico de agente para apoyar la planificación sectorial con métricas de coherencia, "
                "completitud y cobertura.",
                className="subtitle",
            ),
            html.Div(cards, style=STYLES["metric_grid"]),
            dcc.Markdown(
                diagnostic_md,
                style={
                    "border": "1px solid #dfe3eb",
                    "borderRadius": "10px",
                    "padding": "1rem",
                    "backgroundColor": "#ffffff",
                },
            ),
            html.H2("Flujo de agente"),
            html.Div([agent_step_card(step) for step in agent_steps], style=STYLES["agent_grid"]),
            html.H2("Métricas claves (OE1 y OE2)"),
            html.Div(
                [
                    dcc.Graph(figure=hist_fig),
                    dcc.Graph(figure=entity_fig),
                    dcc.Graph(figure=freshness_fig),
                    dcc.Graph(figure=theme_fig),
                    dcc.Graph(figure=scatter_fig),
                ],
                style=STYLES["graph_grid"],
            ),
            html.H2("Activos con brechas de metadatos"),
            gap_table,
            html.H2("Modelo ML básico"),
            dcc.Markdown(cluster_summary.message),
            dcc.Markdown(cluster_keywords),
            cluster_table if cluster_table else html.Div(
                "Aún no hay resultados de clustering para mostrar.", className="placeholder"
            ),
        ],
        style=STYLES["container"],
    )

    return app


if __name__ == "__main__":
    app = build_figures()
    app.run_server(debug=True, host="0.0.0.0", port=8050)
