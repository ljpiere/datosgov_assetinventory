from __future__ import annotations
from dash import dcc, html
from components.ui import render_manaba_page
from pages.helpers import build_cluster_outputs

def layout(df):
    cluster_summary, cluster_table, cluster_keywords, cluster_message = build_cluster_outputs(df)
    cluster_available = cluster_summary.available
    
    content = html.Div(
        [
            html.H1("Modelo ML básico (Clustering)"),
            
            # Tarjeta de Estado
            html.Div(
                [
                    html.H4("Estado del Modelo", style={"marginBottom": "10px", "color": "#022B3A"}),
                    dcc.Markdown(cluster_message, style={"color": "#333"}),
                ],
                style={
                    "backgroundColor": "rgba(255, 255, 255, 0.9)",
                    "borderRadius": "16px",
                    "padding": "1.5rem",
                    "marginBottom": "2rem",
                    "boxShadow": "0 4px 20px rgba(0,0,0,0.1)"
                }
            ),

            # Resultados en Grid
            html.Div(
                [
                    # Columna Izquierda: Palabras Clave
                    html.Div(
                        [
                            html.H3("Palabras Clave por Cluster", style={"fontSize": "1.2rem"}),
                            dcc.Markdown(cluster_keywords, style={"color": "#E0F7FA", "lineHeight": "1.8"}),
                        ],
                        className="glass-card",
                        style={"flex": "1"}
                    ),
                    
                    # Columna Derecha: Tabla
                    html.Div(
                        [
                            html.H3("Estadísticas de Grupos", style={"fontSize": "1.2rem", "marginBottom": "15px"}),
                            cluster_table
                            if cluster_table is not None
                            else (
                                html.Div(
                                    "Aún no hay resultados de clustering para mostrar.", 
                                    style={"color": "#aaa", "fontStyle": "italic"}
                                )
                            ),
                        ],
                        className="glass-card",
                        style={"flex": "2", "overflowX": "auto"}
                    )
                ],
                style={"display": "flex", "gap": "2rem", "flexWrap": "wrap"}
            )
        ]
    )
    
    return render_manaba_page(content, active_path="/ml")
