from __future__ import annotations

from dash import html

from analysis import compute_summary
from components.ui import STYLES, navbar
from pages.helpers import build_metric_cards


def layout(df):
    summary = compute_summary(df)
    cards = build_metric_cards(summary)
    highlights = [
        "Búsqueda en lenguaje natural para hallar datasets y obtener un mini-reporte.",
        "Métricas rápidas de completitud, coherencia y frescura de metadatos.",
        "Visualizaciones listas para priorizar aperturas y mejoras.",
    ]
    return html.Div(
        [
            navbar(active_path="/"),
            html.Div(
                [
                    html.H1("Bienvenido al inventario inteligente de datos abiertos"),
                    html.P(
                        "Explora, busca y diagnostica los activos de datos para orientar decisiones "
                        "sectoriales. Usa el panel para navegar y el agente de búsqueda para llegar rápido "
                        "al dataset que necesitas.",
                        className="subtitle",
                    ),
                    html.Ul([html.Li(item) for item in highlights]),
                ],
                style={"marginBottom": "1.5rem"},
            ),
            html.H2("Estado rápido del inventario"),
            html.Div(cards, style=STYLES["metric_grid"]),
            html.Div(
                [
                    html.H3("¿Qué puedes hacer?"),
                    html.Ul(
                        [
                            html.Li("Ir a Métricas para ver visualizaciones y flujo de agente."),
                            html.Li("Probar el Agente de búsqueda con frases libres."),
                            html.Li("Revisar Brechas de metadatos o el Modelo ML básico."),
                        ]
                    ),
                ],
                style={
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "10px",
                    "padding": "1rem",
                    "backgroundColor": "#ffffff",
                },
            ),
        ],
        style=STYLES["container"],
    )
