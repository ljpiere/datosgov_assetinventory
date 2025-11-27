from __future__ import annotations

from dash import dcc, html

from components.ui import STYLES, navbar


def _placeholder():
    return html.Div(
        [
            html.H4("Genera un reporte por UID"),
            html.P(
                "Ingresa el UID exacto del dataset. Si se encuentra, verás un resumen visual "
                "alineado a la Guía de Calidad e Interoperabilidad 2025 con metadatos y métricas clave.",
                className="subtitle",
            ),
        ],
        style={"padding": "1rem 0"},
    )


def layout(df):
    return html.Div(
        [
            navbar(active_path="/report"),
            html.H1("Reporte por UID"),
            html.P(
                "Consulta rápida de un dataset específico y exporta el reporte en PDF.",
                className="subtitle",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("UID del dataset"),
                            dcc.Input(
                                id="report-uid-input",
                                placeholder="Ej: xkc6-akp5",
                                type="text",
                                style={"width": "100%"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Buscar UID",
                                id="report-search-button",
                                n_clicks=0,
                                style={"width": "100%"},
                            ),
                            html.Button(
                                "Exportar PDF",
                                id="report-download-button",
                                n_clicks=0,
                                style={"width": "100%", "marginTop": "0.5rem"},
                                disabled=True,
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column"},
                    ),
                ],
                style=STYLES["search_box"],
            ),
            html.Div(id="report-content", children=_placeholder(), style=STYLES["search_card"]),
            dcc.Store(id="report-record-store"),
            dcc.Download(id="report-download"),
        ],
        style=STYLES["container"],
    )