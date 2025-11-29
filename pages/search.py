from dash import dcc, html
from analysis import compute_summary
from components.ui import STYLES, build_search_section, build_search_table, render_manaba_page
from pages.helpers import build_metric_cards

def layout(df):
    summary = compute_summary(df)
    cards = build_metric_cards(summary)
    search_table = build_search_table()
    
    content = html.Div([
        html.P("Manaba te ayuda a encontrar datasets escondidos en el bosque de datos. Describe lo que necesitas y obtén datasets recomendados con un mini-reporte automático.", className="subtitle"),
        
        html.Br(),
        build_search_section(search_table),
        
        html.Div([
            html.Button("Descargar Informe ASPA (Word)", id="btn-download-report", style={"display": "none"}),
            dcc.Download(id="download-component")
        ], style={"marginTop": "20px", "textAlign": "center"}),
        
        dcc.Store(id="search-results-store"),

        html.Div(cards, style=STYLES["metric_grid"]),

        ])
    
    return render_manaba_page(content, active_path="/search")
