from dash import html, dcc

def render_chat_interface():
    return html.Div([
        # Botón flotante ¿
        html.Button(
            "Orbi",
            id="orbi-toggle-btn",
            style={
                "position": "fixed",
                "bottom": "20px",
                "right": "20px",
                "zIndex": "1000",
                "borderRadius": "50px",
                "width": "60px",
                "height": "60px",
                "backgroundColor": "#003366", 
                "color": "white",
                "border": "none",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.2)",
                "cursor": "pointer",
                "fontWeight": "bold",
                "fontSize": "1.2rem"
            }
        ),

        # Ventana del Chat 
        html.Div(
            id="orbi-chat-window",
            style={"display": "none"}, 
            children=[
                # Encabezado
                html.Div(
                    children=[
                        html.Span("Asistente Orbi", style={"fontWeight": "bold"}),
                        html.Span("✕", id="orbi-close-btn", style={"cursor": "pointer", "float": "right", "fontWeight": "bold"})
                    ],
                    style={
                        "backgroundColor": "#003366",
                        "color": "white",
                        "padding": "10px 15px",
                        "borderTopLeftRadius": "10px",
                        "borderTopRightRadius": "10px",
                    }
                ),
                
                # Cuerpo 
                html.Div(
                    id="orbi-chat-history",
                    style={
                        "height": "300px",
                        "overflowY": "auto",
                        "padding": "15px",
                        "backgroundColor": "#f9f9f9",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "10px"
                    }
                ),

                # Spinner de carga
                dcc.Loading(
                    id="orbi-loading",
                    type="dot",
                    children=html.Div(id="orbi-loading-output")
                ),

                # Pie
                html.Div(
                    children=[
                        dcc.Input(
                            id="orbi-user-input",
                            type="text",
                            placeholder="Pregúntame sobre los datos...",
                            style={
                                "width": "75%",
                                "padding": "8px",
                                "borderRadius": "5px",
                                "border": "1px solid #ccc",
                                "marginRight": "5px"
                            }
                        ),
                        html.Button(
                            "->",
                            id="orbi-send-btn",
                            style={
                                "width": "20%",
                                "padding": "8px",
                                "backgroundColor": "#003366",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "5px",
                                "cursor": "pointer"
                            }
                        )
                    ],
                    style={
                        "padding": "10px",
                        "borderTop": "1px solid #eee",
                        "display": "flex",
                        "justifyContent": "space-between"
                    }
                )
            ]
        ),
        
        # Almacén de estado para mantener la conversación
        dcc.Store(id="orbi-conversation-store", data=[])
    ])