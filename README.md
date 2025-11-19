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

La fuente de datos es `datasets/Asset_Inventory_Public_20251119.csv`, que se carga automáticamente al iniciar la app.
