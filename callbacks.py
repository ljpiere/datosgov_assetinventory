from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, callback_context, dcc, html, no_update
from dash import dash_table
from datetime import datetime

from analysis import (
    build_search_report,
    get_dataset_by_uid,
    get_public_scope,
    search_inventory,
)
from reporting import (
    build_agent_analysis,
    build_dataset_metrics,
    build_metadata_pairs,
    build_pdf_document,
    build_quality_summary,
    build_cut_csv,
    create_aspa_report
)
from components.ui import STYLES
from pages import (
    gaps_layout,
    metrics_layout,
    ml_layout,
    report_layout,
    search_layout,
    cut_layout,
    welcome_layout,
)

# --- UTILIDADES ---
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
    return table_rows

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
                style={"color": "#666"}
            ),
        ]
    )

def _build_report_view(record: dict):
    # Genera la vista detallada del reporte (reutilizada del código original)
    quality = build_quality_summary(record)
    metadata_pairs = build_metadata_pairs(record)
    metrics = build_dataset_metrics(record)
    agent = build_agent_analysis(record)

    header = html.Div([
        html.H3(record.get("name", "Sin título"), style={"color": "#3366CC"}),
        html.P(record.get("Descripción") or "Sin descripción.", style={"color": "#333"}),
        html.Div([
            html.Span(f"UID: {record.get('UID', 'N/D')}", style={"marginRight": "15px", "fontWeight": "bold"}),
            html.Span(f"Entidad: {record.get('entidad', 'N/A')}"),
        ], style={"fontSize": "0.9rem", "color": "#555", "marginBottom": "15px"})
    ])

    # Métricas en tarjetas
    quality_cards = html.Div([
        html.Div([
            html.Div(item["label"], className="metric-title"),
            html.Div(item["value"], className="metric-value"),
            html.Small(item["detail"], style={"color": "#777"})
        ], className="content-card") for item in quality
    ], style=STYLES["metric_grid"])

    # Tabla de métricas
    metrics_table = dash_table.DataTable(
        data=metrics,
        columns=[{"name": col, "id": col} for col in ["Métrica", "Categoría", "Puntaje", "Definición"]],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#3366CC", "color": "white", "fontWeight": "bold"},
        style_cell={"textAlign": "left", "padding": "10px", "border": "1px solid #eee"}
    )

    return html.Div([header, html.Hr(), quality_cards, html.H4("Evaluación Detallada"), metrics_table])

# --- CALLBACKS DE NAVEGACIÓN (EL QUE FALTABA) ---
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

# --- CALLBACKS DE BÚSQUEDA ---
def register_search_callbacks(app, df):
    @app.callback(
        [Output("search-table", "data"), Output("search-results-store", "data"), 
         Output("search-table", "selected_rows"), Output("search-report", "children"), 
         Output("btn-download-report", "style")],
        [Input("search-button", "n_clicks"), Input("search-table", "selected_rows")],
        [State("search-query", "value"), State("search-results-store", "data")],
    )
    def _run_search(n_clicks, selected_rows, query, stored_results):
        triggered = {t["prop_id"] for t in callback_context.triggered}
        btn_style_visible = {"marginTop": "10px", "padding": "10px 20px", "backgroundColor": "#3366CC", "color": "white", "border": "none", "borderRadius": "5px", "cursor": "pointer", "display": "inline-block"}
        btn_style_hidden = {"display": "none"}

        if not n_clicks and not triggered:
            return [], None, [], "Describe un dataset para ver recomendaciones.", btn_style_hidden

        if "search-button.n_clicks" in triggered:
            try:
                results = search_inventory(query, df, top_k=8)
                if results.empty:
                    return [], None, [], f"No se encontraron activos para '{query}'.", btn_style_hidden
                
                display_rows = _format_table_rows(results).to_dict("records")
                report = build_search_report(query, results, row_index=0)
                return display_rows, results.to_dict("records"), [0], report, btn_style_visible
            except Exception as e:
                return [], None, [], f"Error: {e}", btn_style_hidden

        if stored_results and selected_rows:
            results = pd.DataFrame(stored_results)
            report = build_search_report(query or "", results, row_index=selected_rows[0])
            return no_update, no_update, no_update, report, btn_style_visible

        return no_update, no_update, no_update, no_update, btn_style_hidden

    @app.callback(
        Output("download-component", "data"),
        Input("btn-download-report", "n_clicks"),
        [State("search-table", "selected_rows"), State("search-results-store", "data")],
        prevent_initial_call=True,
    )
    def _download_aspa_report(n_clicks, selected_rows, stored_data):
        if not stored_data or not selected_rows: return None
        try:
            idx = selected_rows[0]
            dataset = stored_data[idx]
            entity_name = dataset.get("entidad", dataset.get("owner", "Entidad Desconocida"))
            docx_buffer = create_aspa_report(dataset, entity_name)
            return dcc.send_bytes(docx_buffer.getvalue(), f"Informe_ASPA_{dataset.get('UID')}.docx")
        except Exception: return None

    return app

# --- CALLBACKS DE REPORTE ---
def register_report_callbacks(app, df):
    @app.callback(
        [Output("report-content", "children"), Output("report-record-store", "data"), Output("report-download-button", "disabled")],
        Input("report-search-button", "n_clicks"),
        State("report-uid-input", "value"),
    )
    def _render_report(n_clicks, uid):
        if not n_clicks: return _report_placeholder(), None, True
        matches = get_dataset_by_uid(uid, df)
        if matches.empty:
            return html.Div([html.H4("UID no encontrado"), html.P("Verifica el identificador.")]), None, True
        
        record = _normalize_record(matches.iloc[0])
        return _build_report_view(record), {"record": record}, False

    @app.callback(
        Output("report-download", "data"),
        Input("report-download-button", "n_clicks"),
        State("report-record-store", "data"),
        prevent_initial_call=True,
    )
    def _download_pdf(n_clicks, stored):
        if not stored: return no_update
        record = stored["record"]
        try:
            pdf_bytes = build_pdf_document(record, build_quality_summary(record), build_metadata_pairs(record), build_dataset_metrics(record))
            return dcc.send_bytes(lambda buf: buf.write(pdf_bytes), f"reporte_{record.get('UID')}.pdf")
        except Exception: return no_update
    return app

# --- CALLBACKS DE CORTE ---
def register_cut_callbacks(app, df):
    @app.callback(
        [Output("cut-download", "data"), Output("cut-status", "children")],
        Input("cut-generate-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def _generate_cut(n_clicks):
        if not n_clicks: return no_update, no_update
        scoped = get_public_scope(df)
        csv_text = build_cut_csv(scoped)
        return dcc.send_string(csv_text, f"corte_{datetime.now():%Y%m%d}.csv"), html.Div("Descarga iniciada.")
    return app

def register_chat_callbacks(app, orbi_agent):
    
    # Callback 1: Abrir/Cerrar Ventana
    @app.callback(
        Output("orbit-chat-window", "style"),
        [Input("orbit-toggle-btn", "n_clicks"),
         Input("orbit-close-btn", "n_clicks")],
        State("orbit-chat-window", "style"),
        prevent_initial_call=True
    )
    def toggle_chat(n_open, n_close, current_style):
        ctx = callback_context
        if not ctx.triggered: return no_update
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Estilo para mostrar la ventana (fija abajo derecha)
        show_style = {
            "position": "fixed",
            "bottom": "110px",
            "right": "30px",
            "width": "350px",
            "zIndex": "1001",
            "display": "block"
        }

        if button_id == "orbit-toggle-btn":
            if current_style and current_style.get("display") == "block":
                return {"display": "none"}
            return show_style
        elif button_id == "orbit-close-btn":
            return {"display": "none"}
            
        return no_update

    # Callback 2: Conversación
    @app.callback(
        [Output("orbit-chat-history", "children"),
         Output("orbit-user-input", "value"),
         Output("orbit-conversation-store", "data"),
         Output("orbit-loading-output", "children")],
        [Input("orbit-send-btn", "n_clicks"),
         Input("orbit-user-input", "n_submit"),
         Input("orbit-conversation-store", "data")],
        [State("orbit-user-input", "value"),
         State("orbit-conversation-store", "data")],
    )
    def update_conversation(n_clicks, n_submit, init_trigger, user_text, history):
        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "init"

        # Carga inicial (mostrar mensaje de bienvenida del store)
        if triggered_id == "orbit-conversation-store" or triggered_id == "init":
            history = history or []
            return _render_messages(history), no_update, no_update, ""

        if not user_text:
            return no_update, no_update, no_update, no_update

        # Nuevo mensaje usuario
        history = history or []
        history.append({"role": "user", "content": user_text})
        
        # Generar respuesta
        response_text = "Lo siento, Manaba está durmiendo."
        if orbi_agent:
            try:
                response_text = orbi_agent.chat(user_text, history)
            except Exception as e:
                response_text = f"Error interno: {str(e)}"
        
        history.append({"role": "orbit", "content": response_text})

        return _render_messages(history), "", history, ""

    # Callback 3: Control de Burbuja (15 seg)
    @app.callback(
        [Output("manaba-bubble-container", "style"),
         Output("bubble-timer", "disabled"),
         Output("bubble-timer", "n_intervals")],
        [Input("url", "pathname"),           # Al navegar
         Input("bubble-timer", "n_intervals"), # Al pasar tiempo
         Input("orbit-toggle-btn", "n_clicks")], # Al abrir chat
        State("manaba-bubble-container", "style")
    )
    def control_bubble(pathname, n_intervals, n_clicks_toggle, current_style):
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "init"

        # Si abre el chat -> Ocultar burbuja
        if trigger_id == "orbit-toggle-btn":
            return {"display": "none"}, True, 0

        # Si carga página -> Mostrar burbuja, iniciar timer
        if trigger_id == "url" or trigger_id == "init":
            return {"display": "block"}, False, 0

        # Si timer termina -> Ocultar burbuja
        if trigger_id == "bubble-timer" and n_intervals >= 1:
            return {"display": "none"}, True, 0

        return no_update, no_update, no_update

    # Renderizador de mensajes con Avatar
    def _render_messages(history):
        messages_html = []
        for msg in history:
            is_user = msg["role"] == "user"
            content_list = []
            
            # Avatar Manaba a la izquierda
            if not is_user:
                content_list.append(html.Img(src="assets/images/manaba_bot.png", className="chat-avatar-img"))
            
            # Burbuja
            bubble_class = "chat-bubble-user" if is_user else "chat-bubble-orbit"
            content_list.append(html.Div(dcc.Markdown(msg["content"]), className=bubble_class))
            
            # Contenedor flex
            container_class = "msg-container user-side" if is_user else "msg-container orbit-side"
            messages_html.append(html.Div(content_list, className=container_class))
            
        return messages_html

    return app