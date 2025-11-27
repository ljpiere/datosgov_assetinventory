from __future__ import annotations

from dash import dcc, html

from analysis import compute_summary
from components.ui import STYLES, build_search_section, build_search_table, navbar
from pages.helpers import build_metric_cards


def layout(df):
    summary = compute_summary(df)
    cards = build_metric_cards(summary)
    search_table = build_search_table()
    return html.Div(
        [
            navbar(active_path="/search"),
            html.H1("Agente de búsqueda y reportería"),
            html.P(
                "Describe lo que necesitas y obtén datasets recomendados con un mini-reporte automático.",
                className="subtitle",
            ),
            build_search_section(search_table),
            html.Div(
                [
                    html.Button(
                        "Descargar Informe ASPA (Word)",
                        id="btn-download-report",
                        style={"display": "none"}
                    ),
                    dcc.Download(id="download-component")
                ],
                style={"marginTop": "20px", "textAlign": "center", "marginBottom": "30px"}
            ),
            html.H3("Estado del inventario"),
            html.Div(cards, style=STYLES["metric_grid"]),
            dcc.Store(id="search-results-store"),
        ],
        style=STYLES["container"],
    )
