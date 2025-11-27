from __future__ import annotations

from dash import Dash, dcc, html

from analysis import load_inventory
from callbacks import (
    register_page_routing,
    register_report_callbacks,
    register_search_callbacks,
    register_cut_callbacks,
)

def create_app() -> Dash:
    """Inicializa la app Dash, layout y callbacks."""
    df = load_inventory()
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            html.Div(id="page-container"),
        ]
    )
    register_page_routing(app, df)
    register_search_callbacks(app, df)
    register_report_callbacks(app, df)
    register_cut_callbacks(app, df)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=8050)
