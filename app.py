from __future__ import annotations

from dash import Dash, dcc, html
from analysis import load_inventory, OrbiAssistant  
from components.chat import render_chat_interface 
from callbacks import (
    register_page_routing,
    register_report_callbacks,
    register_search_callbacks,
    register_cut_callbacks,
    register_chat_callbacks
)
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Advertencia: variable de entorno HF_TOKEN no encontrada; se omite el login a Hugging Face.")

orbi_agent = None
if os.environ.get("LOAD_ORBIT", "True") == "True": # Flag opcional para desarrollo
    try:
        print("--- Despertando a MANABA ---")
        orbi_agent = OrbiAssistant() 
        print("--- MANABA ha despertado ---")
    except Exception as e:
        print(f"Advertencia: MANABA Esta ocupado. El chat no funcionará. Error: {e}")

def create_app() -> Dash:
    """Inicializa la app Dash, layout y callbacks."""
    df = load_inventory()
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            html.Div(id="page-container"),
            render_chat_interface()
        ]
    )
    register_page_routing(app, df)
    register_search_callbacks(app, df)
    register_report_callbacks(app, df)
    register_cut_callbacks(app, df)
    register_chat_callbacks(app,orbi_agent)
    return app


if __name__ == "__main__":
    app = create_app()
    # lo pase a false para que no cargue 2 veces el modelo
    app.run(debug=False, host="0.0.0.0", port=8050)
