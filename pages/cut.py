from __future__ import annotations
from dash import dcc, html
from components.ui import STYLES, render_manaba_page

def layout(df):
    content = html.Div(
        [
            html.H1("Corte de métricas"),
            html.P(
                "Genera un corte masivo en CSV con las métricas de la Guía 2025, metadatos clave, "
                "fecha de corte y mes/año de generación para todos los UID públicos.",
                className="subtitle",
                style={"color": "#A8E6CF", "fontSize": "1.1rem", "marginBottom": "2rem"}
            ),
            
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                "Generar corte CSV",
                                id="cut-generate-button",
                                n_clicks=0,
                                style={
                                    "padding": "15px 30px",
                                    "borderRadius": "12px",
                                    "backgroundColor": "#1EC9BD",
                                    "color": "#01121E",
                                    "border": "none",
                                    "fontWeight": "800",
                                    "fontSize": "1.1rem",
                                    "cursor": "pointer",
                                    "boxShadow": "0 4px 15px rgba(30, 201, 189, 0.4)",
                                    "transition": "transform 0.2s"
                                },
                            ),
                        ],
                        style={"marginBottom": "2rem", "textAlign": "center"}
                    ),
                    
                    html.Div(
                        id="cut-status",
                        children=html.Div(
                            "Pulsa \"Generar corte CSV\" para descargar el archivo con todos los datasets.",
                            style={"textAlign": "center", "color": "#333"}
                        ),
                        style=STYLES["search_card"]
                    ),
                ],
                className="glass-card", 
                style={"padding": "3rem"}
            ),
            dcc.Download(id="cut-download"),
        ]
    )
    
    return render_manaba_page(content, active_path="/cut")
