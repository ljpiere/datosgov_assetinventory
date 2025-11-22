from __future__ import annotations

from dash import html

from components.ui import STYLES, navbar
from pages.helpers import build_gap_table


def layout(df):
    gap_table = build_gap_table(df)
    return html.Div(
        [
            navbar(active_path="/gaps"),
            html.H1("Activos con brechas de metadatos"),
            html.P(
                "Prioriza la mejora de metadatos en los activos con menor completitud.",
                className="subtitle",
            ),
            gap_table,
        ],
        style=STYLES["container"],
    )
