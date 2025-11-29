from dash import html, dcc

def render_chat_interface():
    initial_history = [
        {
            "role": "orbit",
            "content": "¡Hola! Soy **Manaba**, tu oso de anteojos explorador de datos 🐻.\n\nPuedo ayudarte a encontrar datasets, revisar su calidad o explicarte conceptos. \n\n¿Qué quieres rastrear hoy?"
        }
    ]

    return html.Div([
        html.Div(
            id="manaba-bubble-container",
            className="manaba-floating-bubble", # Clase CSS para el diseño
            style={"display": "none"}, # Oculto por defecto, el callback lo muestra
            children=[
                # Texto editable
                html.Span("¿Tienes dudas? Estoy aquí para ayudarte", id="manaba-bubble-text"),
                # Triangulito (Cola de la burbuja) creado con CSS
                html.Div(className="bubble-tail")
            ]
        ),

        dcc.Interval(
            id="bubble-timer",
            interval=15000, # 15 segundos (en milisegundos)
            n_intervals=0,
            disabled=True # Apagado por defecto
        ),

        html.Button(
            "", 
            id="orbit-toggle-btn",
            style={
                "position": "fixed",
                "bottom": "30px",
                "right": "30px",
                "zIndex": "1000",
            }
        ),

        html.Div(
            id="orbit-chat-window",
            style={"display": "none"}, 
            children=[
                html.Div(
                    children=[
                        html.Span("Manaba 🐻", style={"fontSize": "1.1rem"}),
                        html.Span("✕", id="orbit-close-btn", style={"cursor": "pointer", "fontSize": "1.2rem", "fontWeight": "bold"})
                    ],
                    className="chat-header"
                ),
                
                html.Div(
                    id="orbit-chat-history",
                    style={
                        "height": "350px",
                        "overflowY": "auto",
                        "display": "flex",
                        "flexDirection": "column",
                    }
                ),

                dcc.Loading(id="orbit-loading", type="dot", color="#3366CC", children=html.Div(id="orbit-loading-output")),

                html.Div(
                    children=[
                        dcc.Input(
                            id="orbit-user-input",
                            type="text",
                            placeholder="Escribe tu mensaje...",
                            autoComplete="off",
                            className="chat-input-pill"
                        ),
                        html.Button(
                            "🐾", 
                            id="orbit-send-btn",
                            className="chat-send-btn"
                        )
                    ],
                    className="chat-input-area"
                )
            ]
        ),
        
        dcc.Store(id="orbit-conversation-store", data=initial_history)
    ])