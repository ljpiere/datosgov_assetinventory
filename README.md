# datosgov_assetinventory

Dashboard para optimizar el uso de activos de datos abiertos en la planificación sectorial.  
La aplicación combina métricas de diagnóstico, visualizaciones con Plotly/Dash y un flujo básico de agente que resume el avance sobre los objetivos OE1-OE3.

## Requisitos

- Python 3.10+
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

La fuente de datos es `datasets/Asset_Inventory_Public_20251119.csv`, que se carga automáticamente al iniciar la app.

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
