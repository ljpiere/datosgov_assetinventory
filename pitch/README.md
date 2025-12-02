# Pitch de 10 minutos – Inventario de Activos de Datos (MinTIC)

Estructura sugerida para las diapositivas y notas para guiar la presentación de 10 minutos. La demo dura 4 minutos; el resto se reparte en los primeros 6 minutos.

## Slide 1 — Portada (0:30)
- Título del proyecto, logo MinTIC y equipo.
- Frase de valor: inventario vivo y confiable de datos públicos para acelerar decisiones.

## Slide 2 — Qué pidió el Ministerio TIC (0:50)
- Reto: consolidar y evaluar activos de datos de entidades públicas.
- Necesidad: visibilidad centralizada, trazabilidad de fuentes y calidad antes de publicar.
- Expectativa: herramienta operativa, reproducible y segura que soporte actualizaciones continuas.

## Slide 3 — Complejidad del reto (1:00)
- Heterogeneidad de fuentes (formatos, frecuencia, calidad).
- Volumen y variabilidad: múltiples entidades y dominios.
- Requerimientos de producto: transparencia, métricas comparables y experiencia sencilla para usuarios no técnicos.

## Slide 4 — Alcance y conjuntos de datos (0:50)
- Qué cubrimos en esta fase: inventario, evaluación de calidad, panel interactivo.
- Conjuntos de datos incluidos: lista con nombre/fuente/periodicidad (reemplazar con los reales).
- Lo que quedó fuera (para backlog) para gestionar expectativas.

## Slide 5 — Mejora de la calidad de datos (1:00)
- Pipeline de limpieza: validaciones de esquema, duplicados, tipos y consistencia temporal.
- Reglas de calidad configurables y bitácora de cambios.
- Métricas de salud que se reflejan en el tablero (completitud, unicidad, validez, actualidad).

## Slide 6 — Volumen y variables (0:50)
- Número de variables trabajadas: `<completar>`.
- Volumen de registros procesados: `<completar>` y frecuencia de actualización.
- Número de fuentes de datos consultadas: `<completar>`.
- Breve tabla/viñetas con las principales entidades y formatos (CSV/JSON/API).

## Slide 7 — Arquitectura y flujo (0:50)
- Ingesta → validación/limpieza → normalización → almacenamiento → visualización.
- Tecnologías clave del proyecto (ej.: Python, pandas, Dash/Plotly, etc.).
- Cómo se automatiza la actualización (jobs programados, scripts reproducibles).

## Slide 8 — Ventajas para MinTIC (0:50)
- Visibilidad unificada con trazabilidad y métricas de calidad.
- Reducción de tiempo para auditar y preparar datos antes de publicarlos.
- Base para priorizar políticas de datos y detectar brechas de información.
- Extensible: agregar nuevas fuentes o reglas sin reescribir el pipeline.

## Slide 9 — Demo (4:00)
- Video de 4 minutos. Guion sugerido:
  1) Apertura (10s): “Mostraremos el tablero que consolida los activos de datos y su salud”.
  2) Vista general (40s): panel principal con KPIs de cantidad de datasets, calidad promedio, fuentes activas.
  3) Navegación por catálogo (40s): filtrar por entidad y dominio; ver metadatos y estado de calidad.
  4) Calidad a nivel de dataset (60s): abrir un dataset, mostrar validaciones aplicadas, porcentaje de celdas válidas, duplicados, fechas faltantes.
  5) Línea de tiempo/actualización (40s): mostrar cuándo se actualizó por última vez y cómo se programan nuevas corridas.
  6) Exportar/descargar (30s): enseñar cómo obtener el dataset limpio o el reporte de calidad.
  7) Cierre (20s): “En minutos se entiende qué datos sirven, cuáles necesitan corrección y qué entidad es responsable”.

## Slide 10 — Cierre y próximos pasos (0:30)
- Lecciones aprendidas y principales riesgos mitigados.
- Próximos incrementos: nuevas fuentes, más reglas de calidad, alertas automáticas.
- Llamado a acción: definir ciclo de adopción con equipos de datos de las entidades.

## Notas para el discurso final (post-demo)
- “Este proyecto entrega al MinTIC un inventario vivo de activos de datos, con trazabilidad y métricas de calidad, permitiendo decidir rápido qué datasets publicar o priorizar”.
- “Automatizamos la ingesta y validación para reducir el tiempo que hoy se invierte en limpieza manual”.
- “El tablero deja claro quién provee qué datos, su estado de salud y cómo evolucionan en el tiempo”.
- “Es una base extensible: podemos sumar nuevas fuentes y reglas sin rehacer el pipeline”.
- “El valor inmediato es transparencia y confianza; el valor a mediano plazo es acelerar políticas públicas basadas en datos confiables”.
