from __future__ import annotations
from dash import dcc, html
from analysis import compute_summary
from components.ui import STYLES, build_search_table, render_manaba_page
from pages.helpers import build_metric_cards

CUSTOM_STYLES = {
    "glass_container": {
        "backgroundColor": "rgba(2, 43, 58, 0.6)", 
        "backdropFilter": "blur(12px)",
        "border": "1px solid rgba(30, 201, 189, 0.3)", 
        "borderRadius": "20px",
        "padding": "2rem",
        "marginBottom": "2rem",
        "boxShadow": "0 8px 32px 0 rgba(0, 0, 0, 0.3)",
        "display": "flex",
        "flexDirection": "column",
        "gap": "1.5rem"
    },
    "label": {
        "color": "#1EC9BD", 
        "fontWeight": "bold",
        "fontSize": "1.1rem",
        "marginBottom": "10px",
        "display": "block",
        "fontFamily": "inherit"
    },
    "input_area": {
        "width": "100%",
        "padding": "14px 18px",
        "borderRadius": "12px",
        "backgroundColor": "rgba(0, 0, 0, 0.2)",
        "border": "1px solid rgba(30, 201, 189, 0.5)",
        "color": "white",
        "outline": "none",
        "fontSize": "1rem",
        "fontFamily": "Inter, sans-serif",
        "minHeight": "80px", 
        "resize": "vertical"
    },
    "button_primary": {
        "width": "100%",
        "padding": "15px",
        "borderRadius": "12px",
        "backgroundColor": "#1EC9BD",
        "color": "#01121E", 
        "border": "none",
        "fontWeight": "800",
        "fontSize": "1.1rem",
        "cursor": "pointer",
        "boxShadow": "0 4px 15px rgba(30, 201, 189, 0.3)",
        "transition": "transform 0.2s, boxShadow 0.2s"
    },
    "button_secondary": {
        "padding": "10px 20px",
        "borderRadius": "8px",
        "backgroundColor": "transparent",
        "color": "#1EC9BD",
        "border": "1px solid #1EC9BD",
        "fontWeight": "600",
        "cursor": "pointer",
        "marginTop": "10px"
    },
    "result_card": {
        "backgroundColor": "rgba(255, 255, 255, 0.95)",
        "borderRadius": "16px",
        "padding": "2rem",
        "color": "#333",
        "marginTop": "1rem",
        "boxShadow": "0 10px 40px rgba(0,0,0,0.2)"
    }
}

def layout(df):
    summary = compute_summary(df)
    cards = build_metric_cards(summary)
    search_table = build_search_table()
    
    content = html.Div([
        html.H1("Búsqueda Inteligente", style={"marginBottom": "10px"}),
        html.P(
            "Manaba te ayuda a encontrar datasets escondidos en el bosque de datos. "
            "Describe lo que necesitas y obtén recomendaciones automáticas.", 
            className="subtitle",
            style={"color": "#A8E6CF", "fontSize": "1.1rem", "marginBottom": "2rem"}
        ),
        
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Describe tu necesidad de información", style=CUSTOM_STYLES["label"]),
                        dcc.Textarea(
                            id="search-query",
                            placeholder="Ej: Necesito datos históricos sobre la calidad del aire en Bogotá...",
                            style=CUSTOM_STYLES["input_area"]
                        ),
                    ],
                    style={"flex": "3"} 
                ),
                
                html.Div(
                    [
                        html.Label("Acción", style={**CUSTOM_STYLES["label"], "visibility": "hidden"}), # Spacer
                        html.Button(
                            "Rastrear Dataset",
                            id="search-button",
                            n_clicks=0,
                            style=CUSTOM_STYLES["button_primary"],
                        ),
                    ],
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "justifyContent": "flex-end"}
                ),
            ],
            style={
                **CUSTOM_STYLES["glass_container"], 
                "flexDirection": "row", 
                "flexWrap": "wrap",
                "alignItems": "stretch"
            }
        ),
        
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Coincidencias encontradas", style={"fontSize": "1.2rem", "marginBottom": "15px"}),
                        search_table
                    ],
                    className="glass-card",
                    style={"padding": "1.5rem", "overflowX": "auto"}
                ),

                html.Div(
                    dcc.Markdown(id="search-report"),
                    style=CUSTOM_STYLES["result_card"]
                ),
                
                html.Div([
                    html.Button("Descargar Informe ASPA (Word)", id="btn-download-report", style={"display": "none"}),
                    dcc.Download(id="download-component")
                ], style={"marginTop": "20px", "textAlign": "center"}),
            ]
        ),
        
        dcc.Store(id="search-results-store"),

        html.Br(),
        html.H3("Estado del Inventario", style={"marginTop": "2rem", "marginBottom": "1rem"}),
        html.Div(cards, style=STYLES["metric_grid"]),

    ])
    
    return render_manaba_page(content, active_path="/search")
