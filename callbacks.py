from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, callback_context, dcc

from analysis import build_search_report, search_inventory
from reporting import create_aspa_report # <--- NUEVO IMPORT
from pages import (
    gaps_layout,
    metrics_layout,
    ml_layout,
    search_layout,
    welcome_layout,
)


def _format_table_rows(results: pd.DataFrame) -> pd.DataFrame:
    table_rows = results.copy()
    # Aseguramos que similarity exista aunque venga vacía
    if "similarity" not in table_rows.columns:
        table_rows["similarity"] = 0.0
        
    table_rows["metadata_completeness"] = (
        pd.to_numeric(table_rows.get("metadata_completeness", 0), errors="coerce")
        .fillna(0)
        .mul(100)
        .round(1)
    )
    table_rows["similarity"] = (
        pd.to_numeric(table_rows["similarity"], errors="coerce").fillna(0).round(3)
    )
    table_rows["days_since_update"] = (
        pd.to_numeric(table_rows.get("days_since_update", 9999), errors="coerce")
        .fillna(9999)
        .astype(int)
    )
    return table_rows


def register_search_callbacks(app, df):
    
    # Callback de Búsqueda y Selección (Maneja la tabla y el reporte en texto)
    @app.callback(
        [
            Output("search-table", "data"),
            Output("search-results-store", "data"),
            Output("search-table", "selected_rows"),
            Output("search-report", "children"),
            Output("btn-download-report", "style"), # Controlamos visibilidad del botón
        ],
        [
            Input("search-button", "n_clicks"),
            Input("search-table", "selected_rows"),
        ],
        [
            State("search-query", "value"),
            State("search-results-store", "data"),
        ],
    )
    def _run_search(n_clicks: int, selected_rows, query: str, stored_results):
        triggered = {t["prop_id"] for t in callback_context.triggered}
        
        # Estilo base del botón (oculto)
        btn_style_hidden = {"display": "none"}
        btn_style_visible = {
            "marginTop": "10px", "padding": "10px 20px", 
            "backgroundColor": "#28a745", "color": "white", 
            "border": "none", "borderRadius": "5px", 
            "cursor": "pointer", "fontWeight": "bold", 
            "display": "inline-block"
        }

        # Caso inicial
        if not n_clicks and not triggered:
            return [], None, [], "Describe un dataset para ver recomendaciones.", btn_style_hidden

        # Caso: Botón Buscar presionado
        if "search-button.n_clicks" in triggered:
            try:
                results = search_inventory(query, df, top_k=8)
            except Exception as exc:
                return [], None, [], f"Error al ejecutar la búsqueda: {exc}", btn_style_hidden

            if results.empty:
                return [], None, [], f"No se encontraron activos relacionados con \"{query}\".", btn_style_hidden

            display_rows = _format_table_rows(results).to_dict("records")
            selected = [0] # Seleccionar el primero por defecto
            report = build_search_report(query, results, row_index=0)
            return display_rows, results.to_dict("records"), selected, report, btn_style_visible

        # Caso: Fila seleccionada en la tabla
        if stored_results:
            try:
                results = pd.DataFrame(stored_results)
                display_rows = _format_table_rows(results).to_dict("records")
                selected_rows = selected_rows or []
                
                if not selected_rows:
                    return display_rows, stored_results, [], "Selecciona un dataset.", btn_style_hidden
                
                selected_idx = selected_rows[0]
                if selected_idx >= len(results):
                    selected_idx = 0

                report = build_search_report(query or "", results, row_index=selected_idx)
                return display_rows, stored_results, selected_rows, report, btn_style_visible
            except Exception as exc:
                return [], stored_results, selected_rows or [], f"Error en selección: {exc}", btn_style_hidden

        return [], None, [], "Describe un dataset para ver recomendaciones.", btn_style_hidden

    # Nuevo Callback para Generar el Word
    @app.callback(
        Output("download-component", "data"),
        Input("btn-download-report", "n_clicks"),
        State("search-table", "selected_rows"),
        State("search-results-store", "data"),
        prevent_initial_call=True,
    )
    def _download_aspa_report(n_clicks, selected_rows, stored_data):
        if not n_clicks or not stored_data or not selected_rows:
            return None
        
        try:
            # Obtener datos de la fila seleccionada
            idx = selected_rows[0]
            dataset = stored_data[idx]
            
            # Nombre de la entidad (usamos el campo 'entidad' si existe, o 'owner')
            entity_name = dataset.get("entidad", dataset.get("owner", "Entidad Desconocida"))
            
            # Generar el Blob del Word
            docx_buffer = create_aspa_report(dataset, entity_name)
            
            # Nombre del archivo
            filename = f"Informe_ASPA_2025_{dataset.get('uid', 'report')}.docx"
            
            return dcc.send_bytes(docx_buffer.getvalue(), filename)
            
        except Exception as e:
            print(f"Error generando reporte: {e}")
            return None

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