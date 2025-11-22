from __future__ import annotations

from dash import dcc, html

from components.ui import STYLES, navbar
from pages.helpers import build_cluster_outputs


def layout(df):
    cluster_summary, cluster_table, cluster_keywords, cluster_message = build_cluster_outputs(df)
    cluster_available = cluster_summary.available
    return html.Div(
        [
            navbar(active_path="/ml"),
            html.H1("Modelo ML básico"),
            dcc.Markdown(cluster_message),
            dcc.Markdown(cluster_keywords),
            cluster_table
            if cluster_table is not None
            else (
                html.Div(
                    "Aún no hay resultados de clustering para mostrar.", className="placeholder"
                )
                if not cluster_available
                else html.Div()
            ),
        ],
        style=STYLES["container"],
    )
