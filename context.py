"""
Contexto y definiciones globales para el inventario de datos gubernamentales.
Contiene acrónimos comunes y sus significados completos.
Contiene el diccionario GOV_ACRONYMS.
Contiene la guía de calidad para datos abiertos.

"""

GUIA_CALIDAD_TEXT = """
TITULO: Guía de estándares de CALIDAD E INTEROPERABILIDAD de los Datos Abiertos (MinTIC Colombia)

INTRODUCCION:
Para que los datos abiertos generen valor social y económico, deben cumplir criterios de calidad e interoperabilidad.
La norma base es ISO/IEC 25012. Dimensiones clave: Precisión, Integridad, Validez, Consistencia, Unicidad, Actualidad.

CRITERIOS DE CALIDAD (17 Criterios):
1. Accesibilidad: Sin requisitos de registro, contraseñas o software especial.
2. Actualidad: Los datos reflejan el estado más reciente según su periodicidad.
3. Completitud: Todos los campos obligatorios están diligenciados (ningún campo crítico en blanco).
4. Comprensibilidad: Datos interpretables, encabezados claros, glosarios.
5. Conformidad: Cumplimiento de estándares (plantillas MinTIC).
6. Confidencialidad: No publicar datos personales o sensibles sin anonimizar (Ley 1581 de 2012).
7. Consistencia: Datos coherentes y sin contradicción (misma codificación).
8. Credibilidad: Fuente oficial declarada y responsable visible.
9. Disponibilidad: Datos en línea 24/7 sin caídas.
10. Eficiencia: Plataforma permite descargas con buen rendimiento.
11. Exactitud: Datos diligenciados correctamente (sintáctica y semántica).
12. Portabilidad: Formatos abiertos (CSV, JSON) sin bloqueos.
13. Precisión: Nivel de desagregación adecuado al original.
14. Recuperabilidad: Copias de seguridad y control de versiones.
15. Relevancia: Datos útiles para la toma de decisiones.
16. Trazabilidad: Histórico de cambios y fechas documentadas.
17. Unicidad: Sin registros duplicados (filas o columnas).

ERRORES COMUNES Y CÓDIGOS (Validación):
- ERR001 (Metadata errada): Título o descripción mal nombrados o poco claros.
- ERR002 (Metadata vacía): Falta completar campos obligatorios en la metadata.
- ERR003 (Entidad): El nombre de usuario no está vinculado a la entidad oficial.
- ERR004 (Sin filas): El conjunto de datos está vacío (0 registros).
- ERR005 (Pocas filas): Menos de 50 registros (afecta reutilización). Excepción: listados completos pequeños.
- ERR005_01 (Agregado): Datos con pocas filas y además agregados (totales), no desagregados.
- ERR007 (Filas vacías): Conjunto con campos vacíos o basura.
- ERR008 (Columna única): El dataset tiene una sola columna.
- ERR008_1 (Pocas columnas): Menos de 3 columnas.
- ERR008_2 (Columnas mal nombradas): Nombres como "Unnamed Column", "Column1".
- ERR009 (Geolocalización): Falta latitud/longitud cuando hay direcciones.
- ERR010 (Enlace inválido): Link externo roto o apunta a un PDF/DOC (no estructurado).
- ERR011 (Clasificación): Datos divididos por periodos que deberían estar unidos.
- ERR012 (Subconjunto maestro): Publicar fragmentos de datos que ya existen en un dataset nacional (ej: SECOP, ICFES).
- ERR013 (Mal cargado): Error técnico en la carga.
- ERR015 (Desactualizado): La fecha actual supera la frecuencia de actualización prometida.
- ERR017 (Enlace roto): URL externa no funciona.
- ERR018 (Agregaciones): El dataset presenta totales en lugar de datos crudos.
- ITA_1, ITA_2, ITA_3: Errores relacionados con la Ley de Transparencia (activos de información, esquemas de publicación).

SELLOS DE CALIDAD:
- Sello 0: No cumple requisitos mínimos.
- Sello 1: Cumple criterios básicos (Actualidad 10, Consistencia/Completitud > 8, Licencia válida).
- Sello 2: Cumple Sello 1 + Formatos estándar + Incentivo de uso (vistas/descargas).
- Sello 3 (Máximo): Documentación completa (URL), Incentivo de uso avanzado, Mejora continua.

POTENCIAL DE USO (CTR):
Se mide con el CTR (Click Through Rate) = Número de Descargas / Número de Vistas.
Ayuda a identificar qué tan atractiva es la información.

MARCO DE INTEROPERABILIDAD:
Dominios: Político-Legal, Organizacional, Semántico (Lenguaje común), Técnico.
"""

GOV_ACRONYMS = {
    "pot": "Plan de Ordenamiento Territorial",
    "eot": "Esquema de Ordenamiento Territorial",
    "pbot": "Plan Básico de Ordenamiento Territorial",
    "secop": "Contratación Pública y Compras",
    "sicep": "Sistema de Información de Empleo Público",
    "sigep": "Sistema de Información y Gestión del Empleo Público",
    "pae": "Programa de Alimentación Escolar",
    "sisben": "Sistema de Identificación de Potenciales Beneficiarios de Programas Sociales",
    "dane": "Departamento Administrativo Nacional de Estadística",
    "icbf": "Instituto Colombiano de Bienestar Familiar",
    "pib": "Producto Interno Bruto",
    "trm": "Tasa Representativa del Mercado",
    "sena": "Servicio Nacional de Aprendizaje",
    "icetex": "Crédito Educativo",

}