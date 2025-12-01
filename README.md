# MANABA

Manaba es un sistema diseñado para automatizar la inspección, evaluación y documentación técnica de activos de datos provenientes de portales de datos abiertos. El proyecto integra procesos de consulta, análisis estructural, validación de metadatos, generación de reportes estandarizados, permitiendo consolidar información crítica sobre la calidad, actualización, formato y disponibilidad de los recursos publicados por entidades públicas.

El sistema utiliza componentes programáticos, un agente conversacional como interfaz de apoyo para facilitar la interpretación de resultados, la guía en procesos técnicos y la consulta asistida de documentación. Esta arquitectura híbrida mejora la experiencia de uso al ofrecer una interacción más fluida, explicativa, accesible para diferentes perfiles de usuario, desde equipos institucionales hasta investigadores y desarrolladores.

Manaba contribuye al fortalecimiento de la gobernanza de datos mediante la automatización de inventarios, la reducción de tareas manuales, la trazabilidad de los recursos, la estandarización de criterios técnicos. Esto promueve mejores prácticas en el manejo del ciclo de vida del dato, facilitando la reutilización de información pública con fines analíticos, cívicos y académicos.


<img width="1127" height="633" alt="manaba" src="https://github.com/user-attachments/assets/4591632d-c54a-4b74-a6f1-226dab3f1d12" />

## Datos usados 

[Asset Inventory - Public](https://www.datos.gov.co/Ciencia-Tecnolog-a-e-Innovaci-n/Asset-Inventory-Public/uzcf-b9dh/about_data)


## Métodos de inteligencia artificial

- [Agente MANABA](https://huggingface.co/google/gemma-2-2b)
- Agente de búsqueda NL.

## Requisitos

- Python 3.10+
- 4GBVRAM NVIDIA
- Dependencias listadas en `requirements.txt`

Instala el entorno (idealmente dentro de un `venv`):

```bash
pip install -r requirements.txt
```

## Cómo ejecutar

```bash
python app.py
```

El servidor se levanta en `http://127.0.0.1:8050/`.

## ¿Qué incluye el panel?

- **OE1. Coherencia y cobertura:** tarjetas de métricas, flujo del agente y tablas para entidades con brechas.
- **OE2. Métricas de completitud/frecuencia:** histogramas, treemap temático, heatmap de frecuencia declarada vs. frescura observada.
- **OE3. Informe y visualizaciones:** markdown dinámico de diagnóstico y módulo ML sencillo (clustering TF-IDF + KMeans) para detectar grupos temáticos.
- **Agente de búsqueda NL:** ingresa una frase (ej. "series históricas de calidad del aire en Bogotá") y obtén datasets recomendados con reporte resumido.
- **Agente MANABA:** Modelo gemma 2 2b personalizado para respuestas controlados con bajo nivel de creatividad para un control de respuestas exacto.

La fuente de datos es una api sota, que se carga automáticamente al iniciar la app.

## Estructura modular

- `analysis.py`: limpieza/enriquecimiento y lógica analítica (métricas, clustering, búsqueda).
- `components/`: estilos y componentes reutilizables (tarjetas, tablas, navbar).
- `pages/`: layouts por sección (bienvenida, búsqueda, métricas, brechas, modelo ML).
- `callbacks.py`: callbacks de interacción (búsqueda aproximada) y ruteo de páginas.
- `app.py`: punto de entrada; arma la app, layout raíz y registra callbacks.

## Navegación

- `/` Bienvenida con estado rápido.
- `/search` Agente de búsqueda y reportería.
- `/metrics` Métricas OE1/OE2 y visualizaciones.
- `/gaps` Tabla de brechas de metadatos.
- `/ml` Resumen de clustering TF-IDF + KMeans.

## Autores

- Leyton Jean Piere Castro Clavijo
- Nicolas Andres Rodriguez Alarcon
- María José Henao Vargas
