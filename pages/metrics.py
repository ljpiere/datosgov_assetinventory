from __future__ import annotations
from dash import dcc, html
from analysis import agent_flow, compute_summary
from components.ui import STYLES, agent_step_card, render_manaba_page
from pages.helpers import build_diagnostic, build_figures, build_metric_cards

def layout(df):
    summary = compute_summary(df)
    cards = build_metric_cards(summary)
    diagnostic_md = build_diagnostic(summary, df)
    agent_steps = agent_flow(summary)
    figures = build_figures(df)

    # Actualizar layout de las figuras para que se vean bien en fondo blanco
    for key, fig in figures.items():
        fig.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#333"),
            title_font=dict(color="#3366CC", size=16)
        )

    content = html.Div([
        html.H2("Métricas Claves (OE1 y OE2)", style={"marginBottom": "1.5rem"}),
        
        # Tarjetas de resumen
        html.Div(cards, style=STYLES["metric_grid"]),
        
        html.Br(),
        
        # Diagnóstico
        html.Div(
            [
                dcc.Markdown(diagnostic_md)
            ],
            className="content-card",
            style={"backgroundColor": "#ffffff00"}
        ),
        
        html.H3("Flujo del Agente", style={"marginTop": "2rem"}),
        html.Div([agent_step_card(step) for step in agent_steps], style=STYLES["agent_grid"]),
        
        html.H3("Visualizaciones del Inventario", style={"marginTop": "2rem"}),
        
        # Grillas de gráficas (Cada una en su tarjeta)
        html.Div(
            [
                html.Div(dcc.Graph(figure=figures["hist"]), className="content-card"),
                html.Div(dcc.Graph(figure=figures["entity"]), className="content-card"),
                html.Div(dcc.Graph(figure=figures["freshness"]), className="content-card"),
                html.Div(dcc.Graph(figure=figures["theme"]), className="content-card"),
                html.Div(dcc.Graph(figure=figures["scatter"]), className="content-card", style={"gridColumn": "1 / -1"}), # Scatter ancho completo
            ],
            style={
                    **STYLES["graph_grid"],
                    "backgroundColor": "rgb(121,197,196)",
                    "padding": "2rem",
                    "border-radius":"20px",
                    "backdrop-filter":"blur(16px)",
                    "block-size":"unset",
                    "margin-bottom":"6rem",
                    "box-shadow":"0 8px 32px 0 rgba(0,0,0,0.3)",
                    "gap":"6rem",
                    "margin-top":"5rem",
                    "border":"12rem",
                    }
        ),
    ])

    return render_manaba_page(content, active_path="/metrics")