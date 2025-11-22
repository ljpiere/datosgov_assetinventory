from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, callback_context

from analysis import build_search_report, search_inventory
from pages import (
    gaps_layout,
    metrics_layout,
    ml_layout,
    search_layout,
    welcome_layout,
)


def _format_table_rows(results: pd.DataFrame) -> pd.DataFrame:
    table_rows = results.copy()
    table_rows["metadata_completeness"] = (
        pd.to_numeric(table_rows["metadata_completeness"], errors="coerce")
        .fillna(0)
        .mul(100)
        .round(1)
    )
    table_rows["similarity"] = (
        pd.to_numeric(table_rows["similarity"], errors="coerce").fillna(0).round(3)
    )
    table_rows["days_since_update"] = (
        pd.to_numeric(table_rows["days_since_update"], errors="coerce")
        .fillna(9999)
        .astype(int)
    )
    return table_rows


def register_search_callbacks(app, df):
    @app.callback(
        [
            Output("search-table", "data"),
            Output("search-results-store", "data"),
            Output("search-table", "selected_rows"),
            Output("search-report", "children"),
        ],
        [
            Input("search-button", "n_clicks"),
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
            Input("search-table", "active_cell"),
=======
            Input("search-table", "selected_rows"),
>>>>>>> theirs
=======
            Input("search-table", "selected_rows"),
>>>>>>> theirs
=======
            Input("search-table", "selected_rows"),
>>>>>>> theirs
        ],
        [
            State("search-query", "value"),
            State("search-results-store", "data"),
            State("search-table", "selected_rows"),
        ],
    )
    def _run_search(n_clicks: int, selected_rows, query: str, stored_results, download_clicks):
        triggered = {t["prop_id"] for t in callback_context.triggered}

        if not n_clicks and not triggered:
            return [], None, [], "Describe un dataset para ver recomendaciones."

        # Botón de búsqueda presionado: recalcula resultados
        if "search-button.n_clicks" in triggered:
            try:
                results = search_inventory(query, df, top_k=8)
            except Exception as exc:  # pragma: no cover - defensa ante errores inesperados
                return [], None, [], f"Error al ejecutar la búsqueda: {exc}"

            if results.empty:
                return [], None, [], f"No se encontraron activos relacionados con \"{query}\"."

            display_rows = _format_table_rows(results).to_dict("records")
            selected = [0]
            report = build_search_report(query, results, row_index=0)
            return display_rows, results.to_dict("records"), selected, report

        # Cambio de selección en la tabla o descarga
        if stored_results:
            results = pd.DataFrame(stored_results)
            display_rows = _format_table_rows(results).to_dict("records")
            selected_idx = selected_rows[0] if selected_rows else 0
            report = build_search_report(query or "", results, row_index=selected_idx)
            # Exportar CSV si se presionó el botón
            if "download-search-btn.n_clicks" in triggered:
                rows_to_export = results.iloc[selected_rows] if selected_rows else results
                return (
                    display_rows,
                    stored_results,
                    selected_rows or [],
                    report,
                    send_data_frame(rows_to_export.to_csv, "busqueda_datasets.csv", index=False),
                )
            return display_rows, stored_results, selected_rows or [], report, None

        return [], None, [], "Describe un dataset para ver recomendaciones.", None

    return app


def register_page_routing(app, df):
    @app.callback(
        Output("page-container", "children"),
        Input("url", "pathname"),
    )
    def _render_page(pathname: str):
        if pathname in ("/", "/welcome", None):
            return welcome_layout(df)
        if pathname.startswith("/search"):
            return search_layout(df)
        if pathname.startswith("/metrics"):
            return metrics_layout(df)
        if pathname.startswith("/gaps"):
            return gaps_layout(df)
        if pathname.startswith("/ml"):
            return ml_layout(df)
        return welcome_layout(df)

    return app
