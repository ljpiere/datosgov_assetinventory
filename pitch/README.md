# Pitch de 10 minutos – Inventario de Activos de Datos (MinTIC)

Base para la presentación en 10 minutos. La demo dura 4 minutos; los otros 6 minutos cubren contexto, valor y próximos pasos. Tono cercano y concreto.

## Slide 1 — Portada (0:30)
- Título, logo MinTIC, nombres del equipo.
- Idea fuerza: “Un inventario vivo y confiable de datos públicos para decidir más rápido”.

## Slide 2 — Qué pidió el Ministerio TIC (0:50)
- Petición directa: consolidar y evaluar los activos de datos que publican las entidades.
- Necesidad humana: “Queremos ver, en un solo lugar, qué tenemos, qué sirve y qué hay que corregir”.
- Condición de producto: operable hoy, seguro, trazable y listo para crecer.

## Slide 3 — Complejidad del reto (1:00)
- Fuentes desordenadas: distintos formatos, frecuencias y calidades.
- Mucha variación: múltiples entidades y dominios con reglas propias.
- Producto exigente: métricas comparables, transparencia y experiencia simple para perfiles no técnicos.

## Slide 4 — Alcance y conjuntos de datos (0:50)
- Entregables de esta fase: inventario central, evaluación de calidad y tablero interactivo.
- Conjuntos incluidos: lista breve con nombre/fuente/periodicidad (reemplazar con reales).
- Delimitación: qué quedó fuera y pasa a backlog para no inflar expectativas.

## Slide 5 — Mejora de la calidad de datos (1:00)
- Pipeline de limpieza: validaciones de esquema, duplicados, tipos y consistencia temporal.
- Reglas configurables y bitácora de cambios para auditar ajustes.
- Métricas visibles en el tablero: completitud, unicidad, validez y actualidad.

## Slide 6 — Volumen y variables (0:50)
- Número de variables trabajadas: 57
- Volumen de registros procesados y frecuencia: 49,2K por consulta a la API.
- Número de fuentes consultadas: Datos propios abiertos proporcionados por el MINTIC.
- Viñetas con entidades y formatos (CSV/JSON/API) para dar tangibilidad.

## Slide 7 — Arquitectura y flujo (0:50)
- Camino claro: ingesta → validación/limpieza → normalización → almacenamiento → visualización.
- Tecnologías clave (ej.: Python, pandas, Dash/Plotly) y cómo se automatiza (jobs programados).
- Mensaje humano: “Lo dejamos listo para operar y para que cualquier equipo pueda continuarlo”.

## Slide 8 — Ventajas para MinTIC (0:50)
- Visibilidad unificada con trazabilidad y métricas accionables.
- Menos tiempo en limpiar, más tiempo en decidir qué publicar o priorizar.
- Base para priorizar políticas y detectar brechas de información.
- Extensible: agregar nuevas fuentes o reglas sin rehacer el pipeline.

## Slide 9 — Demo (4:00)
- Guion sugerido:
  1) Apertura (10s): “Aquí está el tablero que junta los activos de datos y su salud”.
  2) Vista general (40s): KPIs de datasets, calidad promedio, fuentes activas.
  3) Catálogo (40s): filtrar por entidad/dominio; ver metadatos y estado de calidad.
  4) Calidad de un dataset (60s): validaciones aplicadas, % de celdas válidas, duplicados, fechas faltantes.
  5) Línea de tiempo (40s): última actualización y programación de nuevas corridas.
  6) Exportar/descargar (30s): dataset limpio o reporte de calidad.
  7) Cierre (20s): “En minutos sabemos qué datos sirven, cuáles corregir y quién es responsable”.

## Slide 10 — Cierre y próximos pasos (0:30)
- Lecciones aprendidas y riesgos mitigados.
- Próximos incrementos: nuevas fuentes, más reglas, alertas automáticas.
- Llamado a acción: definir ciclo de adopción con equipos de datos de las entidades.

## Discurso sugerido (para cerrar la demo)
- “El Ministerio nos pidió algo muy concreto: saber qué datos hay, qué tan buenos son y cómo usarlos sin perder tiempo. Hoy lo pueden ver en un solo tablero”.
- “Automatizamos la ingesta y la validación. Eso recorta horas de limpieza manual y reduce el margen de error humano”.
- “Cada dataset tiene dueño y estado de salud visible. Sabemos qué funciona, qué corregir y en qué orden priorizar”.
- “El sistema es vivo: podemos sumar nuevas fuentes y reglas sin volver a empezar. Solo se enchufa y corre”.
- “Lo que ganamos hoy es transparencia y confianza; lo que ganamos mañana es velocidad para lanzar políticas basadas en datos confiables”.
