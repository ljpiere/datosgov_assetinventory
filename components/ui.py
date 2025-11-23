from __future__ import annotations

from dash import dash_table, dcc, html

STYLES = {
    "container": {"padding": "2rem", "fontFamily": "Inter, Arial, sans-serif"},
    "metric_grid": {
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
        "gap": "1rem",
        "marginBottom": "1rem",
    },
    "metric_card": {
        "border": "1px solid #e0e0e0",
        "borderRadius": "10px",
        "padding": "1rem",
        "backgroundColor": "#f8f9fb",
    },
    "graph_grid": {
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
        "gap": "1.5rem",
    },
    "agent_grid": {
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
        "gap": "1rem",
        "marginBottom": "1.5rem",
    },
    "search_box": {
        "display": "grid",
        "gridTemplateColumns": "1fr auto",
        "gap": "0.75rem",
        "alignItems": "end",
        "marginBottom": "1rem",
    },
    "search_card": {
        "border": "1px solid #dfe3eb",
        "borderRadius": "10px",
        "padding": "1rem",
        "backgroundColor": "#ffffff",
    },
    "nav": {
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
        "padding": "0.75rem 1rem",
        "borderBottom": "1px solid #e0e0e0",
        "marginBottom": "1rem",
        "backgroundColor": "#fdfdfd",
        "position": "sticky",
        "top": 0,
        "zIndex": 5,
    },
    "nav_links": {
        "display": "flex",
        "gap": "1rem",
    },
}


def metric_card(title: str, value: str, detail: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div(value, className="metric-value"),
            html.Small(detail, className="metric-detail"),
        ],
        style=STYLES["metric_card"],
    )


def agent_step_card(step: dict) -> html.Div:
    return html.Div(
        [
            html.Div(step["title"], className="agent-title"),
            html.Div(step["status"], className="agent-status"),
            html.Small(step["detail"], className="agent-detail"),
        ],
        style={
            "border": "1px solid #dfe3eb",
            "borderRadius": "8px",
            "padding": "0.75rem",
            "backgroundColor": "#ffffff",
        },
    )


def build_search_table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="search-table",
        columns=[
            {"name": "Similitud", "id": "similarity", "type": "numeric"},
            {"name": "Título", "id": "name"},
            {"name": "Entidad", "id": "entidad"},
            {"name": "Sector", "id": "sector"},
            {"name": "Tema", "id": "theme_group"},
            {"name": "Completitud (%)", "id": "metadata_completeness", "type": "numeric"},
            {"name": "Días sin actualizar", "id": "days_since_update", "type": "numeric"},
            {"name": "Frecuencia declarada", "id": "update_frequency_norm"},
        ],
        data=[],
        row_selectable="single",
        selected_rows=[],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "0.5rem"},
        style_data_conditional=[
            {"if": {"row_index": 0}, "backgroundColor": "#f0f8ff"},
            {"if": {"column_id": "similarity"}, "textAlign": "center"},
            {"if": {"column_id": "metadata_completeness"}, "textAlign": "center"},
        ],
    )



def build_search_section(search_table: dash_table.DataTable) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Describe lo que buscas", style={"fontWeight": "bold"}),
                            dcc.Input(
                                id="search-query",
                                placeholder="Ej: series históricas de calidad del aire en Bogotá",
                                style={"width": "100%", "padding": "10px", "marginTop": "5px"},
                                type="text"
                            ),
                        ],
                        style={"flex": "1"}
                    ),
                    html.Button(
                        "Buscar dataset",
                        id="search-button",
                        n_clicks=0,
                        style={
                            "height": "42px", 
                            "padding": "0 1.5rem", 
                            "backgroundColor": "#0056b3", 
                            "color": "white", 
                            "border": "none",
                            "borderRadius": "5px",
                            "cursor": "pointer"
                        },
                    ),
                ],
                style=STYLES["search_box"],
            ),
            html.Div(
                [
                    dcc.Markdown(
                        "Escribe una oración y obtén coincidencias aproximadas "
                        "por título, descripción, etiquetas o temas.",
                        style={"marginBottom": "0.5rem", "color": "#666"},
                    ),
                    search_table,
                    
                    # --- NUEVO: Contenedor para reporte y descarga ---
                    html.Div([
                        dcc.Markdown(id="search-report", style={"padding": "10px", "backgroundColor": "#f1f1f1", "flex": "1"}),
                        html.Div([
                            html.Button(
                                "Generar Informe ASPA 2025",
                                id="btn-download-report",
                                style={
                                    "marginTop": "10px",
                                    "padding": "10px 20px",
                                    "backgroundColor": "#28a745", # Verde
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "5px",
                                    "cursor": "pointer",
                                    "fontWeight": "bold",
                                    "display": "none" # Oculto por defecto hasta que se seleccione algo
                                }
                            ),
                            dcc.Download(id="download-component")
                        ], style={"marginLeft": "20px", "alignSelf": "start"})
                    ], style={"display": "flex", "marginTop": "1rem", "alignItems": "flex-start"}),
                    # ------------------------------------------------
                ],
                style=STYLES["search_card"],
            ),
        ]
    )


def navbar(active_path: str = "/") -> html.Nav:
    nav_links = [
        ("Bienvenida", "/"),
        ("Búsqueda", "/search"),
        ("Métricas", "/metrics"),
        ("Brechas", "/gaps"),
        ("Modelo ML", "/ml"),
    ]

    def link(label, href):
        is_active = href == active_path
        return html.A(
            label,
            href=href,
            style={
                "padding": "0.4rem 0.8rem",
                "borderRadius": "6px",
                "backgroundColor": "#e8eef7" if is_active else "transparent",
                "textDecoration": "none",
                "color": "#1f2937",
                "fontWeight": 600 if is_active else 500,
            },
        )

    return html.Nav(
        [
            html.Div("Datos Gov · Inventario Inteligente", style={"fontWeight": 700}),
            html.Div([link(label, href) for label, href in nav_links], style=STYLES["nav_links"]),
        ],
        style=STYLES["nav"],
    )
