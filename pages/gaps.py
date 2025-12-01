from __future__ import annotations
from dash import html
from components.ui import render_manaba_page
from pages.helpers import build_gap_table

def layout(df):
    gap_table = build_gap_table(df)
    
    content = html.Div(
        [
            html.H1("Activos con brechas de metadatos"),
            html.P(
                "Prioriza la mejora de metadatos en los activos con menor completitud.",
                className="subtitle",
                style={"color": "#A8E6CF", "fontSize": "1.1rem", "marginBottom": "2rem"}
            ),
            
            html.Div(
                gap_table,
                className="glass-card",
                style={"padding": "1.5rem", "overflowX": "auto"}
            )
        ]
    )
    
    return render_manaba_page(content, active_path="/gaps")
