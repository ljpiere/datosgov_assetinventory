from __future__ import annotations
import random
import urllib.parse
from dash import html
from analysis import compute_summary
from components.ui import navbar, metric_card, STYLES

# --- PALETA DE COLORES MANABA ---
COLORS = {
    "night_deep": "#01121E",    # Azul casi negro
    "night_mid": "#022B3A",     # Azul noche medio
    "turquoise_mid": "#1EC9BD", # Turquesa cuerpo (el color de la montaña)
    "watercolor_1": "#A8E6CF",  # Verde menta suave
    "white": "#FFFFFF",
}

# --- GENERADOR DE ESTRELLAS ---
def generate_stars(num_stars=80):
    """Genera estrellas estáticas y parpadeantes usando CSS."""
    stars = []
    for _ in range(num_stars):
        size = random.choice([1, 1.5, 2, 2.5])
        style = {
            "position": "absolute",
            "left": f"{random.randint(0, 100)}%",
            "top": f"{random.randint(0, 90)}%",
            "width": f"{size}px",
            "height": f"{size}px",
            "backgroundColor": "white",
            "borderRadius": "50%",
            "opacity": random.uniform(0.2, 0.9),
            "boxShadow": f"0 0 {size*2}px rgba(255, 255, 255, 0.9)",
            "animation": f"twinkle {random.uniform(3, 7)}s infinite alternate ease-in-out"
        }
        stars.append(html.Div(style=style))
    return stars

def generate_shooting_stars(num=2):
    """Añade estrellas fugaces."""
    stars = []
    for i in range(num):
        style = {
            "position": "absolute",
            "top": f"{random.randint(10, 40)}%",
            "left": f"{random.randint(0, 80)}%",
            "width": "150px",
            "height": "2px",
            "background": "linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 50%, rgba(255,255,255,0) 100%)",
            "transform": "rotate(-45deg)",
            "opacity": "0",
            "animation": f"shooting-star 6s infinite ease-in-out {i*3}s"
        }
        stars.append(html.Div(style=style))
    return stars

def generate_watercolor_background():
    """Crea 'manchas' de color desenfocadas para simular acuarela."""
    blobs = []
    blobs.append(html.Div(style={
        "position": "absolute", "top": "10%", "left": "-10%",
        "width": "600px", "height": "600px",
        "background": COLORS["watercolor_1"],
        "borderRadius": "50%", "filter": "blur(100px)", "opacity": "0.4", "zIndex": "0"
    }))
    blobs.append(html.Div(style={
        "position": "absolute", "bottom": "10%", "right": "-5%",
        "width": "500px", "height": "500px",
        "background": COLORS["turquoise_mid"],
        "borderRadius": "50%", "filter": "blur(120px)", "opacity": "0.2", "zIndex": "0"
    }))
    return blobs

def get_mountain_svg_src():
    """
    Retorna el código SVG de la montaña codificado para usar en html.Img.
    Esto soluciona el error 'AttributeError: module dash.html has no attribute Svg'.
    """
    # Definimos el SVG como string
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320" preserveAspectRatio="none">
        <path fill="#009185" fill-opacity="0.4" d="M0,256L60,245.3C120,235,240,213,360,213.3C480,213,600,235,720,229.3C840,224,960,192,1080,197.3C1200,203,1320,245,1380,266.7L1440,288L1440,320L1380,320C1320,320,1200,320,1080,320C960,320,840,320,720,320C600,320,480,320,360,320C240,320,120,320,60,320L0,320Z"></path>
        <path fill="{COLORS['turquoise_mid']}" fill-opacity="1" d="M0,192L80,181.3C160,171,320,149,480,160C640,171,800,213,960,224C1120,235,1280,213,1360,202.7L1440,192L1440,320L1360,320C1280,320,1120,320,960,320C800,320,640,320,480,320C320,320,160,320,80,320L0,320Z"></path>
    </svg>
    """
    # Codificamos para URL
    return "data:image/svg+xml;charset=utf-8," + urllib.parse.quote(svg)

def layout(df):
    summary = compute_summary(df)
    
    # Estilo para las tarjetas tipo "vidrio"
    glass_card_style = {
        "backgroundColor": "rgba(255, 255, 255, 0.65)",
        "backdropFilter": "blur(12px)",
        "border": "1px solid rgba(255, 255, 255, 0.8)",
        "borderRadius": "16px",
        "padding": "1.5rem",
        "boxShadow": "0 8px 32px 0 rgba(31, 38, 135, 0.07)",
        "transition": "transform 0.3s ease",
        "cursor": "default"
    }
    
    # Construcción manual de las tarjetas para aplicar el estilo visual específico
    cards = []
    card_data = [
        ("Activos analizados", f"{summary['total_assets']:,}", "Total en inventario"),
        ("Completitud promedio", f"{summary['avg_metadata_completeness']:.1%}", "Campos críticos"),
        ("Coherencia", f"{summary['coherence_ratio']:.1%}", "Público vs Acceso"),
        ("Activos recientes", f"{summary['recent_updates']:,}", "Actualizados ≤90 días"),
    ]
    
    for title, value, detail in card_data:
        cards.append(html.Div([
            html.Div(title, className="metric-title", style={"color": "#014550", "fontSize": "0.9rem", "fontWeight": "bold", "textTransform": "uppercase", "letterSpacing": "1px"}),
            html.Div(value, className="metric-value", style={"color": "#021B2B", "fontSize": "2.2rem", "fontWeight": "800", "margin": "0.5rem 0"}),
            html.Small(detail, className="metric-detail", style={"color": "#546E7A", "fontFamily": "Inter"})
        ], style=glass_card_style))

    return html.Div(
        [
            navbar(active_path="/"),

            # ==========================================
            # SECCIÓN SUPERIOR: NOCHE ESTRELLADA
            # ==========================================
            html.Div(
                style={
                    "position": "relative",
                    "width": "100%",
                    "minHeight": "65vh",
                    "background": f"radial-gradient(circle at 50% 10%, {COLORS['night_mid']} 0%, {COLORS['night_deep']} 80%)",
                    "color": COLORS["white"],
                    "textAlign": "center",
                    "paddingTop": "8vh",
                    "overflow": "hidden",
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center"
                },
                children=[
                    # Fondo animado
                    *generate_stars(90),
                    *generate_shooting_stars(3),

                    # Título
                    html.H1(
                        "Manaba",
                        style={
                            "fontFamily": "'Berlin Sans FB', 'Trebuchet MS', sans-serif",
                            "fontSize": "clamp(4rem, 10vw, 8rem)",
                            "marginBottom": "0.5rem",
                            "zIndex": "10",
                            "position": "relative",
                            "color": COLORS["white"],
                            "textShadow": "0 0 40px rgba(0, 229, 204, 0.4)",
                            "letterSpacing": "-2px"
                        }
                    ),

                    # Subtítulo
                    html.P(
                        "Explora, busca y diagnostica los activos de datos",
                        style={
                            "fontFamily": "'Inter', sans-serif",
                            "fontSize": "clamp(1rem, 2vw, 1.5rem)",
                            "fontWeight": "300",
                            "color": "#B2EBF2",
                            "zIndex": "10",
                            "maxWidth": "600px",
                            "margin": "0 auto",
                            "position": "relative",
                            "letterSpacing": "0.5px"
                        }
                    ),

                    # VECTOR SVG (Corregido: Usamos html.Img en lugar de html.Svg)
                    html.Img(
                        src=get_mountain_svg_src(),
                        style={
                            "position": "absolute",
                            "bottom": "-2px", # Solapa ligeramente para evitar líneas blancas
                            "left": "0",
                            "width": "100%",
                            "height": "auto",
                            "minHeight": "120px",
                            "zIndex": "5",
                            "display": "block",
                            "pointerEvents": "none"
                        }
                    )
                ]
            ),

            # ==========================================
            # SECCIÓN INFERIOR: ACUARELA Y CONTENIDO
            # ==========================================
            html.Div(
                style={
                    "position": "relative",
                    "width": "100%",
                    "backgroundColor": COLORS["turquoise_mid"], 
                    "minHeight": "50vh",
                    "marginTop": "-2px" 
                },
                children=[
                    html.Div(
                        style={
                            "background": f"linear-gradient(to bottom, {COLORS['turquoise_mid']} 0%, #F0FBFA 20%, #FFFFFF 100%)",
                            "padding": "4rem 2rem 6rem 2rem",
                            "position": "relative",
                            "zIndex": "10",
                            "borderTop": "none"
                        },
                        children=[
                            *generate_watercolor_background(),

                            html.Div(
                                [
                                    html.H3(
                                        "Usa el panel para navegar y el agente de búsqueda\npara llegar rápido al dataset que necesitas.",
                                        style={
                                            "fontFamily": "'Berlin Sans FB', sans-serif",
                                            "fontSize": "clamp(1.5rem, 3vw, 2.2rem)",
                                            "textAlign": "center",
                                            "lineHeight": "1.3",
                                            "color": COLORS["night_deep"],
                                            "whiteSpace": "pre-wrap",
                                            "marginBottom": "3rem",
                                            "position": "relative",
                                            "zIndex": "2"
                                        }
                                    ),
                                    
                                    html.Div(
                                        [
                                            html.P("1. Búsqueda en lenguaje natural para hallar datasets.", style={"marginBottom": "8px"}),
                                            html.P("2. Métricas rápidas de completitud, coherencia y frescura.", style={"marginBottom": "8px"}),
                                            html.P("3. Visualizaciones listas para priorizar aperturas y mejoras.", style={"marginBottom": "8px"}),
                                        ],
                                        style={
                                            "fontFamily": "'Inter', sans-serif",
                                            "fontSize": "1.1rem",
                                            "textAlign": "center",
                                            "color": "#004D40",
                                            "marginBottom": "4rem",
                                            "fontWeight": "500",
                                            "position": "relative",
                                            "zIndex": "2"
                                        }
                                    ),
                                ],
                                style={"maxWidth": "900px", "margin": "0 auto"}
                            ),

                            # Contenedor de Métricas
                            html.Div(
                                children=[
                                    html.Div(
                                        "Estado rápido del inventario",
                                        style={
                                            "textAlign": "center", 
                                            "marginBottom": "2rem",
                                            "fontFamily": "'Berlin Sans FB', sans-serif",
                                            "fontSize": "1.5rem",
                                            "color": COLORS["night_deep"]
                                        }
                                    ),
                                    html.Div(
                                        cards, 
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
                                            "gap": "1.5rem",
                                        }
                                    ),
                                ],
                                style={
                                    "backgroundColor": "rgba(255, 255, 255, 0.5)",
                                    "borderRadius": "24px",
                                    "padding": "3rem",
                                    "boxShadow": "0 20px 50px rgba(0, 150, 136, 0.1)",
                                    "border": "1px solid rgba(255,255,255, 0.6)",
                                    "maxWidth": "1200px",
                                    "margin": "0 auto",
                                    "position": "relative",
                                    "zIndex": "5"
                                }
                            ),

                            # Caja "¿Qué puedes hacer?"
                            html.Div(
                                [
                                    html.H4("¿Qué puedes hacer?", style={"fontWeight": "bold", "marginBottom": "1rem", "fontFamily": "'Berlin Sans FB'"}),
                                    html.Ul(
                                        [
                                            html.Li("Ir a Métricas para ver visualizaciones y flujo de agente.", style={"marginBottom": "0.5rem"}),
                                            html.Li("Probar el Agente de búsqueda con frases libres.", style={"marginBottom": "0.5rem"}),
                                            html.Li("Revisar Brechas de metadatos o el Modelo ML básico.", style={"marginBottom": "0.5rem"}),
                                        ],
                                        style={"paddingLeft": "1.5rem", "fontFamily": "'Inter'"}
                                    ),
                                ],
                                style={
                                    "backgroundColor": "white",
                                    "borderRadius": "16px",
                                    "padding": "2rem",
                                    "maxWidth": "1200px",
                                    "margin": "2rem auto 0 auto",
                                    "boxShadow": "0 4px 20px rgba(0,0,0,0.03)",
                                    "position": "relative",
                                    "zIndex": "5"
                                }
                            )
                        ]
                    )
                ]
            )
        ],
        style={"padding": "0", "margin": "0", "width": "100%", "overflowX": "hidden", "backgroundColor": "#FFFFFF"}
    )
