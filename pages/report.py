from __future__ import annotations

from dash import dcc, html
from components.ui import STYLES, navbar, render_manaba_page

# --- ESTILOS PERSONALIZADOS (SOLO PARA REPORTE) ---
# Estos estilos replican el diseño de la Imagen 2 (Dark Glass + Turquesa)
CUSTOM_STYLES = {
    "glass_container": {
        "backgroundColor": "rgba(2, 43, 58, 0.6)", # Fondo oscuro translúcido
        "backdropFilter": "blur(12px)",
        "border": "1px solid rgba(30, 201, 189, 0.3)", # Borde turquesa suave
        "borderRadius": "20px",
        "padding": "2rem",
        "marginBottom": "2rem",
        "boxShadow": "0 8px 32px 0 rgba(0, 0, 0, 0.3)",
        "display": "flex",
        "flexDirection": "column",
        "gap": "1.5rem"
    },
    "label": {
        "color": "#1EC9BD", # Turquesa Manaba
        "fontWeight": "bold",
        "fontSize": "1.1rem",
        "marginBottom": "8px",
        "display": "block",
        "fontFamily": "inherit"
    },
    "input": {
        "width": "100%",
        "padding": "14px 18px",
        "borderRadius": "12px",
        "backgroundColor": "rgba(0, 0, 0, 0.2)", # Fondo input oscuro
        "border": "1px solid rgba(30, 201, 189, 0.5)",
        "color": "white",
        "outline": "none",
        "fontSize": "1rem",
        "fontFamily": "inherit"
    },
    "button_primary": {
        "width": "100%",
        "padding": "12px",
        "borderRadius": "10px",
        "backgroundColor": "#1EC9BD", # Botón vibrante
        "color": "#01121E", # Texto oscuro
        "border": "none",
        "fontWeight": "800",
        "fontSize": "1rem",
        "cursor": "pointer",
        "boxShadow": "0 4px 15px rgba(30, 201, 189, 0.3)",
        "transition": "transform 0.2s"
    },
    "button_secondary": {
        "width": "100%",
        "padding": "12px",
        "borderRadius": "10px",
        "backgroundColor": "transparent",
        "color": "#1EC9BD",
        "border": "2px solid #1EC9BD",
        "fontWeight": "700",
        "fontSize": "1rem",
        "cursor": "pointer",
        "marginTop": "10px"
    },
    "result_card": {
        "backgroundColor": "rgba(255, 255, 255, 0.95)", # Blanco para el reporte final (legibilidad)
        "borderRadius": "16px",
        "padding": "2rem",
        "color": "#333",
        "marginTop": "2rem",
        "boxShadow": "0 10px 40px rgba(0,0,0,0.2)"
    }
}

def _placeholder():
    return html.Div(
        [
            html.H4("Genera un reporte por UID", style={"color": "#022B3A"}),
            html.P(
                "Ingresa el UID exacto del dataset. Si se encuentra, verás un resumen visual "
                "alineado a la Guía de Calidad e Interoperabilidad 2025.",
                className="subtitle",
                style={"color": "#555"}
            ),
        ],
        style={"textAlign": "center", "padding": "1rem"}
    )

def layout(df):
    # Contenido principal
    content = html.Div(
        [
            html.H1("Reporte por UID", style={"marginBottom": "10px"}),
            html.P(
                "Consulta rápida de un dataset específico y exporta el reporte en PDF.",
                className="subtitle",
                style={"color": "#A8E6CF", "fontSize": "1.1rem", "marginBottom": "2rem"}
            ),
            
            # --- CAJA DE BÚSQUEDA PERSONALIZADA (Estilo Imagen 2) ---
            html.Div(
                [
                    # Columna 1: Input
                    html.Div(
                        [
                            html.Label("UID del dataset", style=CUSTOM_STYLES["label"]),
                            dcc.Input(
                                id="report-uid-input",
                                placeholder="Ej: xkc6-akp5",
                                type="text",
                                style=CUSTOM_STYLES["input"],
                                autoComplete="off"
                            ),
                        ],
                        style={"flex": "2"} # Ocupa más espacio
                    ),
                    
                    # Columna 2: Botones
                    html.Div(
                        [
                            html.Button(
                                "Buscar UID",
                                id="report-search-button",
                                n_clicks=0,
                                style=CUSTOM_STYLES["button_primary"],
                            ),
                            html.Button(
                                "Exportar PDF",
                                id="report-download-button",
                                n_clicks=0,
                                disabled=True,
                                style=CUSTOM_STYLES["button_secondary"],
                            ),
                        ],
                        style={"flex": "1", "display": "flex", "flexDirection": "column"}
                    ),
                ],
                style={
                    **CUSTOM_STYLES["glass_container"], 
                    "flexDirection": "row", 
                    "alignItems": "flex-start",
                    "gap": "20px",
                    "flexWrap": "wrap" # Responsivo
                }
            ),
            
            # --- RESULTADOS ---
            html.Div(
                id="report-content", 
                children=_placeholder(), 
                style=CUSTOM_STYLES["result_card"]
            ),
            
            dcc.Store(id="report-record-store"),
            dcc.Download(id="report-download"),
        ]
    )

    # Usamos render_manaba_page si está disponible en ui.py para mantener el fondo de estrellas
    # Si no, usa un div simple.
    try:
        return render_manaba_page(content, active_path="/report")
    except NameError:
        return html.Div([navbar(active_path="/report"), content], style=STYLES["container"])
