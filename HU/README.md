# Historias de usuario · Dashboard de activos abiertos

El producto prioriza la toma de decisiones institucionales a partir del inventario de activos abiertos, por lo que se definieron perfiles clave y sus historias de usuario relacionadas.

## Perfiles

1. **Planner Sectorial (PS):** Analista responsable de la planeación de programas sectoriales que necesita señales ejecutivas para priorizar acciones.
2. **Gestor de Datos Abiertos (GDA):** Encargado del inventario y la calidad de datos en la entidad, con foco en coherencia, cumplimiento y actualización.
3. **Comunicador Institucional (CI):** Prepara reportes y presentaciones para la alta dirección, incluyendo visualizaciones e insumos de narrativa.

## Historias de usuario

| ID | Perfil | Historia | Criterios de aceptación |
|----|--------|----------|-------------------------|
| HU-01 | PS | Como planner sectorial quiero ver tarjetas ejecutivas con total de activos, coherencia y completitud promedio para saber si la entidad está cumpliendo los lineamientos OE1. | El panel muestra métricas calculadas automáticamente a partir del CSV y las colorea según umbrales (<70%, 70-90%, >90%). |
| HU-02 | PS | Como planner sectorial necesito visualizar la cobertura temática por sector para identificar brechas en la planificación y orientar nuevas aperturas. | Se renderiza un treemap que agrupa `sector` y `theme_group` y permite hacer drill-down; los valores se actualizan al recargar el dataset. |
| HU-03 | GDA | Como gestor de datos abiertos quiero identificar activos con completitud crítica (<50%) para priorizar acciones de corrección según OE2. | Se listan los 15 activos con menor `metadata_completeness` incluyendo entidad responsable, contacto y campos faltantes. |
| HU-04 | GDA | Como gestor de datos abiertos necesito un resumen de coherencia entre el campo “Público” y “Public Access Level” para corregir inconsistencias. | La app calcula `coherence_flag`, muestra el porcentaje de casos consistentes y permite exportar (copiar) la tabla de inconsistentes. |
| HU-05 | GDA | Como gestor de datos abiertos deseo conocer la frecuencia de actualización declarada vs. la frescura real para planear mantenimientos. | Se despliega un gráfico que cruza `update_frequency_norm` con `freshness_bucket` y destaca activos con más de 365 días sin actualización. |
| HU-06 | CI | Como comunicador institucional quiero un bloque de diagnóstico en Markdown listo para integrarlo en informes. | Desde la app se genera el texto con insights destacados (inventario total, coherencia, completitud, temas predominantes). |
| HU-07 | CI | Como comunicador institucional necesito visualizar la relación entre vistas y antigüedad para construir narrativas de impacto. | El scatter `days_since_update` vs `views` permite filtrar hover data con título, entidad y completitud. |
| HU-08 | PS/GDA | Como planner/gestor deseo un flujo de agente que indique el estado de avance de OE1-OE3 para dar seguimiento rápido. | Se muestran tarjetas de estado (“Completado”, “En ejecución”, etc.) basadas en las métricas calculadas y actualizadas con el dataset. |
| HU-09 | PS | Como planner sectorial quiero detectar clústeres temáticos mediante ML básico para identificar duplicidades u oportunidades. | Si scikit-learn está instalado, el módulo KMeans genera grupos con palabras clave y tabla de métricas por clúster. |
| HU-10 | GDA | Como gestor de datos abiertos necesito poder cargar cambios en el CSV sin modificar la app para mantener el panel actualizado. | La app lee automáticamente `datasets/Asset_Inventory_Public_20251119.csv` al iniciar, sin configuraciones adicionales. |

## Mapeo actividades → historias

- **Actividad PS:** Revisar cobertura sectorial → HU-01, HU-02, HU-08, HU-09.
- **Actividad GDA:** Monitorear calidad y actualización → HU-03, HU-04, HU-05, HU-10.
- **Actividad CI:** Elaborar informes → HU-06, HU-07.

Este set garantiza trazabilidad entre las necesidades institucionales (OE1-OE3) y los componentes implementados en la aplicación.
