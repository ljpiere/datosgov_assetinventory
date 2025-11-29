from __future__ import annotations
import random
import urllib.parse
from dash import dash_table, dcc, html

# --- COLORES MANABA VIBRANTE ---
COLORS = {
    "night_deep": "#01121E",
    "night_mid": "#022B3A",
    "turquoise": "#1EC9BD",
    "mint": "#A8E6CF",
    "gov_blue": "#3366CC", # Solo para Navbar
}

# --- ESTILOS (SOLUCIÓN KEYERRORS) ---
# Este diccionario es vital para que pages/report.py y pages/gaps.py funcionen
STYLES = {
    "container": {
        "padding": "2rem",
        "maxWidth": "1400px",
        "margin": "0 auto",
        "position": "relative",
        "zIndex": "10",
        "minHeight": "80vh"
    },
    "metric_grid": {
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
        "gap": "1.5rem",
        "marginBottom": "2rem",
    },
    "metric_card": { # Se usará como fallback si no se usa la clase CSS
        "backgroundColor": "rgba(255, 255, 255, 0.08)",
        "backdropFilter": "blur(12px)",
        "border": "1px solid rgba(255, 255, 255, 0.1)",
        "borderRadius": "16px",
        "padding": "1.5rem",
        "color": "white"
    },
    "graph_grid": {
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(500px, 1fr))",
        "gap": "2rem",
        "marginTop": "2rem",
    },
    "agent_grid": {
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
        "gap": "1rem",
        "marginBottom": "1.5rem",
    },
    # Estilos específicos para Buscador y Reporte
    "search_box": {
        "display": "flex",
        "gap": "1rem",
        "alignItems": "flex-end",
        "marginBottom": "1.5rem",
        "backgroundColor": "rgba(30, 201, 189, 0.1)", # Turquesa muy suave
        "padding": "1.5rem",
        "borderRadius": "16px",
        "border": "1px solid rgba(30, 201, 189, 0.2)"
    },
    "search_card": {
        "backgroundColor": "rgba(255, 255, 255, 0.95)", # Casi blanco para legibilidad de texto
        "border": "none",
        "borderRadius": "16px",
        "padding": "2rem",
        "color": "#333", # Texto oscuro dentro de la carta
        "boxShadow": "0 10px 30px rgba(0,0,0,0.2)"
    },
    "nav": {"display": "none"}, # Fallback
    "nav_links": {"display": "none"}
}

# --- GENERADORES VISUALES (Restaurados) ---
def generate_stars(num_stars=60):
    """Estrellas brillantes para fondo oscuro."""
    stars = []
    for _ in range(num_stars):
        size = random.choice([1, 2, 3])
        style = {
            "position": "absolute",
            "left": f"{random.randint(0, 100)}%",
            "top": f"{random.randint(0, 100)}%",
            "width": f"{size}px",
            "height": f"{size}px",
            "backgroundColor": "#FFF",
            "borderRadius": "50%",
            "opacity": random.uniform(0.1, 0.8),
            "animation": f"twinkle {random.uniform(3, 8)}s infinite alternate ease-in-out",
            "boxShadow": f"0 0 {size*2}px rgba(255, 255, 255, 0.8)"
        }
        stars.append(html.Div(style=style))
    return stars

def generate_watercolor_blobs():
    """Manchas de color vibrante (Turquesa/Azul) sobre fondo oscuro."""
    blobs = []
    # Mancha Turquesa Brillante
    blobs.append(html.Div(style={
        "position": "absolute", "top": "10%", "right": "-10%",
        "width": "600px", "height": "600px",
        "background": "radial-gradient(circle, rgba(30, 201, 189, 0.15) 0%, rgba(0,0,0,0) 70%)",
        "borderRadius": "50%", "filter": "blur(80px)", "zIndex": "0"
    }))
    # Mancha Azul Profundo
    blobs.append(html.Div(style={
        "position": "absolute", "bottom": "-10%", "left": "-5%",
        "width": "800px", "height": "500px",
        "background": "radial-gradient(circle, rgba(2, 43, 58, 0.4) 0%, rgba(0,0,0,0) 70%)",
        "borderRadius": "50%", "filter": "blur(100px)", "zIndex": "0"
    }))
    return blobs

def get_mountain_svg_src():
    # Montaña Turquesa Vibrante (Estilo Manaba Original)
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320" preserveAspectRatio="none">
        <path fill="#022B3A" fill-opacity="0.6" d="M0,256L60,245.3C120,235,240,213,360,213.3C480,213,600,235,720,229.3C840,224,960,192,1080,197.3C1200,203,1320,245,1380,266.7L1440,288L1440,320L1380,320C1320,320,1200,320,1080,320C960,320,840,320,720,320C600,320,480,320,360,320C240,320,120,320,60,320L0,320Z"></path>
        <path fill="{COLORS['turquoise']}" fill-opacity="1" d="M0,192L80,181.3C160,171,320,149,480,160C640,171,800,213,960,224C1120,235,1280,213,1360,202.7L1440,192L1440,320L1360,320C1280,320,1120,320,960,320C800,320,640,320,480,320C320,320,160,320,80,320L0,320Z"></path>
    </svg>
    """
    return "data:image/svg+xml;charset=utf-8," + urllib.parse.quote(svg)

# --- WRAPPER PRINCIPAL ---
def render_manaba_page(content, active_path="/"):
    """Layout Oscuro con Navbar Azul."""
    return html.Div(
        style={
            "position": "relative",
            "width": "100%",
            "minHeight": "100vh",
            "background": f"radial-gradient(circle at 50% 0%, {COLORS['night_mid']} 0%, {COLORS['night_deep']} 100%)",
            "color": "white",
            "overflowX": "hidden",
        },
        children=[
            *generate_stars(50),
            *generate_watercolor_blobs(),
            navbar(active_path), # Navbar Institucional
            
            # Contenido
            html.Div(
                content,
                style={
                    "position": "relative",
                    "zIndex": "10",
                    "padding": "2rem",
                    "maxWidth": "1300px",
                    "margin": "0 auto"
                }
            ),
            
            # Montaña Footer
            html.Img(
                src=get_mountain_svg_src(),
                style={
                    "position": "fixed", "bottom": "0", "left": "0",
                    "width": "100%", "height": "auto", "maxHeight": "12vh",
                    "zIndex": "1", "pointerEvents": "none"
                }
            )
        ]
    )

# --- COMPONENTES ---

def metric_card(title: str, value: str, detail: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-title", style={"color": COLORS["turquoise"]}),
            html.Div(value, className="metric-value"),
            html.Small(detail, className="metric-detail"),
        ],
        className="glass-card" # Usa CSS glass-card
    )

def agent_step_card(step: dict) -> html.Div:
    return html.Div(
        [
            html.Div(step["title"], className="agent-title", style={"fontSize": "1.1rem", "marginBottom": "5px"}),
            html.Div(step["status"], style={"fontSize": "1.2rem", "fontWeight": "bold", "color": "white"}),
            html.Small(step["detail"], style={"color": "#B0BEC5"}),
        ],
        className="glass-card"
    )

def navbar(active_path: str = "/") -> html.Nav:
    nav_links = [
        ("Bienvenida", "/"),
        ("Búsqueda", "/search"),
        ("Reporte", "/report"),
        ("Corte", "/cut"),
        ("Métricas", "/metrics"),
        ("Brechas", "/gaps"),
        ("Modelo ML", "/ml"),
    ]

    def link(label, href):
        is_active = href == active_path
        class_name = "gov-nav-link active" if is_active else "gov-nav-link"
        return html.A(label, href=href, className=class_name)

    return html.Nav(
        [
            html.Div(
                [
                    html.Span("Datos Gov · Inventario Inteligente", style={"fontWeight": "700", "fontSize": "1.1rem", "color": "white", "letterSpacing": "0.5px"})
                ],
                className="gov-brand"
            ),
            html.Div(
                [link(label, href) for label, href in nav_links],
                style={"display": "flex", "gap": "5px", "flexWrap": "wrap"}
            ),
        ],
        className="gov-navbar" # Azul Institucional
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
        ],
        data=[],
        row_selectable="single",
        selected_rows=[],
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        # Estilos Dark Mode para tabla
        style_header={
            'backgroundColor': 'rgba(30, 201, 189, 0.1)',
            'color': '#1EC9BD',
            'fontWeight': 'bold',
            'border': 'none',
            'padding': '12px'
        },
        style_cell={
            'backgroundColor': 'rgba(255, 255, 255, 0.05)',
            'color': 'white',
            'borderBottom': '1px solid rgba(255, 255, 255, 0.1)',
            'textAlign': 'left',
            'padding': '10px'
        },
        style_data_conditional=[
            {'if': {'state': 'selected'}, 'backgroundColor': 'rgba(30, 201, 189, 0.2)', 'border': '1px solid #1EC9BD'}
        ]
    )

def build_search_section(search_table: dash_table.DataTable) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Describe lo que buscas", style={"fontWeight": "bold", "color": COLORS["turquoise"], "fontSize": "1.1rem"}),
                    dcc.Textarea(
                        id="search-query",
                        placeholder="Ej: datos de calidad del aire en Bogotá...",
                        style={
                            "width": "100%", "minHeight": "70px", "marginTop": "10px", "padding": "12px",
                            "backgroundColor": "rgba(0,0,0,0.2)", "border": "1px solid rgba(30, 201, 189, 0.5)",
                            "borderRadius": "12px", "color": "white", "fontFamily": "Inter"
                        },
                    ),
                    html.Button(
                        "Rastrear Dataset",
                        id="search-button",
                        n_clicks=0,
                        style={
                            "marginTop": "15px", "width": "100%", "padding": "12px",
                            "background": f"linear-gradient(135deg, {COLORS['turquoise']} 0%, #009185 100%)",
                            "color": "white", "border": "none", "borderRadius": "12px",
                            "fontWeight": "bold", "cursor": "pointer", "fontSize": "1rem",
                            "boxShadow": "0 4px 15px rgba(30, 201, 189, 0.4)"
                        }
                    ),
                ],
                className="glass-card" # Tarjeta translúcida oscura
            ),
            html.Div(
                [
                    dcc.Markdown("**Resultados del rastro:**", style={"marginBottom": "15px", "color": COLORS["mint"]}),
                    search_table,
                    # Reporte en tarjeta blanca para legibilidad
                    html.Div(
                        dcc.Markdown(id="search-report"),
                        style=STYLES["search_card"] # Fondo blanco
                    ),
                ]
            ),
        ]
    )
