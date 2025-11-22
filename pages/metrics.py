from __future__ import annotations

from dash import dcc, html

from analysis import agent_flow, compute_summary
from components.ui import STYLES, agent_step_card, navbar
from pages.helpers import build_diagnostic, build_figures, build_metric_cards


def layout(df):
    summary = compute_summary(df)
    cards = build_metric_cards(summary)
    diagnostic_md = build_diagnostic(summary, df)
    agent_steps = agent_flow(summary)
    figures = build_figures(df)

    return html.Div(
        [
            navbar(active_path="/metrics"),
            html.H1("Métricas claves (OE1 y OE2)"),
            html.Div(cards, style=STYLES["metric_grid"]),
            dcc.Markdown(
                diagnostic_md,
                style={
                    "border": "1px solid #dfe3eb",
                    "borderRadius": "10px",
                    "padding": "1rem",
                    "backgroundColor": "#ffffff",
                },
            ),
            html.H2("Flujo de agente"),
            html.Div([agent_step_card(step) for step in agent_steps], style=STYLES["agent_grid"]),
            html.Div(
                [
                    dcc.Graph(figure=figures["hist"]),
                    dcc.Graph(figure=figures["entity"]),
                    dcc.Graph(figure=figures["freshness"]),
                    dcc.Graph(figure=figures["theme"]),
                    dcc.Graph(figure=figures["scatter"]),
                ],
                style=STYLES["graph_grid"],
            ),
        ],
        style=STYLES["container"],
    )
