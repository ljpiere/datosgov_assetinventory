from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, callback_context, dcc, html, no_update
from dash import dash_table

from analysis import (
    build_search_report,
    get_dataset_by_uid,
    get_public_scope,
    search_inventory,
)
from components.ui import STYLES
from reporting import (
    build_agent_analysis,
    build_dataset_metrics,
    build_metadata_pairs,
    build_pdf_document,
    build_quality_summary,
    build_cut_csv,
    create_aspa_report
)
from pages import (
    gaps_layout,
    metrics_layout,
    ml_layout,
    report_layout,
    search_layout,
    cut_layout,
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
            Output("btn-download-report", "style"),    
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
        
        btn_style_hidden = {"display": "none"}
        btn_style_visible = {
            "marginTop": "10px", "padding": "10px 20px", 
            "backgroundColor": "#a1a1a1", "color": "white", 
            "border": "none", "borderRadius": "5px", 
            "cursor": "pointer", "fontWeight": "bold", 
            "display": "inline-block"
        }

        if not n_clicks and not triggered:
            return [], None, [], "Describe un dataset para ver recomendaciones.", btn_style_hidden

        if "search-button.n_clicks" in triggered:
            try:
                results = search_inventory(query, df, top_k=8)
            except Exception as exc:
                return [], None, [], f"Error al ejecutar la búsqueda: {exc}", btn_style_hidden

            if results.empty:
                return [], None, [], f"No se encontraron activos relacionados con \"{query}\".", btn_style_hidden

            display_rows = _format_table_rows(results).to_dict("records")
            selected = [0] # Seleccionamos el primer resultado por defecto
            report = build_search_report(query, results, row_index=0)
            
            # hacemos visible el botón
            return display_rows, results.to_dict("records"), selected, report, btn_style_visible

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

        # Fallback final
        return [], None, [], "Describe un dataset para ver recomendaciones.", btn_style_hidden

    # Generar y Descargar Reporte en Word
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
            
            # Determinar nombre de la entidad
            entity_name = dataset.get("entidad", dataset.get("owner", "Entidad Desconocida"))
            
            # Generar el Blob del Word (buffer en memoria)
            docx_buffer = create_aspa_report(dataset, entity_name)
            
            # Definir nombre del archivo
            uid_str = str(dataset.get('uid', 'report'))
            filename = f"Informe_ASPA_2025_{uid_str}.docx"
            
            # Enviar al componente dcc.Download
            return dcc.send_bytes(docx_buffer.getvalue(), filename)
            
        except Exception as e:
            # Aquí podrías agregar un log o print para depuración
            print(f"Error generando reporte: {e}")
            return None

    return app


def _normalize_record(row: pd.Series) -> dict:
    record = {}
    for key, value in row.items():
        if pd.isna(value):
            record[key] = None
        elif isinstance(value, pd.Timestamp):
            record[key] = value.isoformat()
        else:
            record[key] = value
    return record


def _report_placeholder():
    return html.Div(
        [
            html.H4("Esperando un UID..."),
            html.P(
                "Ingresa el UID exacto del dataset y pulsa \"Buscar UID\" para generar el reporte.",
                className="subtitle",
            ),
        ]
    )


def _build_report_view(record: dict):
    quality = build_quality_summary(record)
    metadata_pairs = build_metadata_pairs(record)
    metrics = build_dataset_metrics(record)
    agent = build_agent_analysis(record)

    header = html.Div(
        [
            html.H3(record.get("name", "Sin título")),
            html.Div(
                [
                    html.Span(f"UID: {record.get('UID', 'N/D')}"),
                    html.Span(f"Entidad: {record.get('entidad', 'Sin registro')}"),
                    html.Span(f"Sector: {record.get('sector', 'Sin registro')}"),
                ],
                style={
                    "display": "flex",
                    "gap": "0.5rem",
                    "flexWrap": "wrap",
                    "fontSize": "0.9rem",
                    "color": "#374151",
                },
            ),
            html.P(record.get("Descripción") or "Sin descripción disponible."),
            html.Small(
                "Resumen alineado a la Guía de Calidad e Interoperabilidad 2025.",
                style={"color": "#4b5563"},
            ),
        ]
    )

    quality_cards = html.Div(
        [
            html.Div(
                [
                    html.Div(item["label"], className="metric-title"),
                    html.Div(item["value"], className="metric-value"),
                    html.Small(item["detail"], className="metric-detail"),
                ],
                style=STYLES["metric_card"],
            )
            for item in quality
        ],
        style=STYLES["metric_grid"],
    )

    agent_summary = html.Div(
        [
            html.Div(
                [
                    html.Div("Estado del dataset", className="metric-title"),
                    html.Div(agent["status"], className="metric-value"),
                    html.Small("Diagnostico automatico del agente."),
                ],
                style=STYLES["metric_card"],
            ),
            html.Div(
                [
                    html.Div("Resumen rapido", className="metric-title"),
                    html.Div(
                        ", ".join(f"{k}: {v}" for k, v in agent["summary"].items()),
                        style={"color": "#374151", "fontSize": "0.9rem"},
                    ),
                    html.Small("Valores clave usados para el diagnostico."),
                ],
                style=STYLES["metric_card"],
            ),
        ],
        style=STYLES["metric_grid"],
    )

    warning_list = html.Ul([html.Li(w) for w in agent["warnings"]])
    action_list = html.Ul([html.Li(a) for a in agent["actions"]])
    agent_panel = html.Div(
        [
            html.H4("Acciones recomendadas (Agente)"),
            html.Div(
                [
                    html.Div([html.Strong("Alertas:"), warning_list]),
                    html.Div([html.Strong("Proximos pasos:"), action_list]),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
                    "gap": "1rem",
                },
            ),
        ],
        style={"marginBottom": "1rem"},
    )

    metadata_table = dash_table.DataTable(
        data=metadata_pairs,
        columns=[{"name": "Campo", "id": "Campo"}, {"name": "Valor", "id": "Valor"}],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "0.5rem"},
    )

    metrics_table = dash_table.DataTable(
        data=metrics,
        columns=[
            {"name": "Métrica", "id": "Métrica"},
            {"name": "Categoría", "id": "Categoría"},
            {"name": "Puntaje", "id": "Puntaje"},
            {"name": "Definición", "id": "Definición"},
        ],
        page_size=8,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "0.4rem"},
    )

    return html.Div(
        [
            header,
            html.H4("Métricas clave"),
            quality_cards,
            agent_summary,
            agent_panel,
            html.H4("Métricas de la Guía 2025"),
            metrics_table,
            html.H4("Metadatos disponibles"),
            metadata_table,
        ]
    )


def register_report_callbacks(app, df):
    @app.callback(
        [
            Output("report-content", "children"),
            Output("report-record-store", "data"),
            Output("report-download-button", "disabled"),
        ],
        Input("report-search-button", "n_clicks"),
        State("report-uid-input", "value"),
    )
    def _render_report(n_clicks: int, uid: str):
        if not n_clicks:
            return _report_placeholder(), None, True

        matches = get_dataset_by_uid(uid, df)
        if matches.empty:
            msg = html.Div(
                [
                    html.H4("UID no encontrado"),
                    html.P(
                        f"No se encontró un dataset público con UID \"{uid}\". "
                        "Verifica el identificador y vuelve a intentarlo."
                    ),
                ]
            )
            return msg, None, True

        record = _normalize_record(matches.iloc[0])
        view = _build_report_view(record)
        return view, {"record": record}, False

    @app.callback(
        Output("report-download", "data"),
        Input("report-download-button", "n_clicks"),
        State("report-record-store", "data"),
        prevent_initial_call=True,
    )
    def _download_report(n_clicks: int, stored):
        if not n_clicks:
            return no_update
        if not stored or "record" not in stored:
            return no_update

        record = stored["record"]
        quality = build_quality_summary(record)
        metadata_pairs = build_metadata_pairs(record)
        metric_scores = build_dataset_metrics(record)
        try:
            pdf_bytes = build_pdf_document(record, quality, metadata_pairs, metric_scores)
            filename = f"reporte_{record.get('UID', 'dataset')}.pdf"
            return dcc.send_bytes(lambda buf: buf.write(pdf_bytes), filename=filename)
        except Exception as exc:  # pragma: no cover - manejo defensivo
            error_msg = f"Error al generar PDF: {exc}"
            return dcc.send_string(error_msg, filename="reporte_error.txt")

    return app


def register_cut_callbacks(app, df):
    @app.callback(
        [Output("cut-download", "data"), Output("cut-status", "children")],
        Input("cut-generate-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def _generate_cut(n_clicks: int):
        if not n_clicks:
            return no_update, no_update

        scoped = get_public_scope(df)
        if scoped.empty:
            return (
                no_update,
                html.Div("No hay datasets públicos disponibles para generar el corte."),
            )

        csv_text = build_cut_csv(scoped)
        filename = f"corte_metricas_{pd.Timestamp.utcnow():%Y%m%d}.csv"
        download = dcc.send_string(csv_text, filename=filename)
        status = html.Div("Corte generado. La descarga debería iniciarse automáticamente.")
        return download, status

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
        if pathname.startswith("/report"):
            return report_layout(df)
        if pathname.startswith("/cut"):
            return cut_layout(df)
        if pathname.startswith("/metrics"):
            return metrics_layout(df)
        if pathname.startswith("/gaps"):
            return gaps_layout(df)
        if pathname.startswith("/ml"):
            return ml_layout(df)
        return welcome_layout(df)

    return app