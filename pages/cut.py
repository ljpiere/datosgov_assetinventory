from __future__ import annotations

from dash import dcc, html

from components.ui import STYLES, navbar


def layout(df):
    return html.Div(
        [
            navbar(active_path="/cut"),
            html.H1("Corte de métricas"),
            html.P(
                "Genera un corte masivo en CSV con las métricas de la Guía 2025, metadatos clave, "
                "fecha de corte y mes/año de generación para todos los UID públicos.",
                className="subtitle",
            ),
            html.Div(
                [
                    html.Button(
                        "Generar corte CSV",
                        id="cut-generate-button",
                        n_clicks=0,
                        style={"height": "48px", "padding": "0 1.5rem"},
                    ),
                ],
                style={"marginBottom": "1rem"},
            ),
            html.Div(
                id="cut-status",
                style=STYLES["search_card"],
                children=html.Div(
                    "Pulsa \"Generar corte CSV\" para descargar el archivo con todos los datasets.",
                ),
            ),
            dcc.Download(id="cut-download"),
        ],
        style=STYLES["container"],
    )
