"""
Módulo para la generación de reportes en formato .docx según el estándar ASPA 2025.

NOTA: Este módulo genera documentos estructurados con secciones y tablas.
No utiliza plantillas externas para mantener todo el contenido generado dentro del código.
No utiliza servicios externos para garantizar privacidad y control total sobre el contenido.
No utiliza archivos de configuración para simplificar la implementación y despliegue del módulo.
No se utilizan modelos de lenguaje para asegurar consistencia y evitar sesgos.

- Esta nota no queda en la version final:
No se utilizan modelos porque esta cara la api o porque al correr local mi pc hace boom.

- TAREAS PENDIENTES:
CORREGIR ERROR CRÍTICO: 
1.
Hay campos que al generar el informe no encuentra como 
Descripción o dominio.
Intente solucionar creando la funcion _smart_get con nombres de columnas alternativas 
por si no existen esos campos, ya que la api a ratos cambia los nombres (: NO FUNCIONO.

2.
Si se puede añadir mas criterios de cumplimiento de calidad del PDF ASPA 2025 mejor.
son 19 criterios en total, pero por ahora solo se implementan 8 a medias ajajjaja.
"""

from datetime import datetime
from io import BytesIO
import math
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ==========================================
# UTILIDADES DE EXTRACCIÓN Y LÓGICA
# ==========================================

def _smart_get(data: dict, keys: list, default="N/A") -> str:
    """Busca valor en lista de llaves. Retorna el primero válido."""
    for key in keys:
        val = data.get(key)
        if isinstance(val, dict):
            val = val.get('name', val.get('url', str(val)))
        if val is not None:
            str_val = str(val).strip()
            if str_val and str_val.lower() not in ["", "nan", "none", "null", "n/a"]:
                return str_val
    return default

def _calc_actualidad(row: pd.Series) -> tuple:
    days_diff = row.get("days_since_update", 9999)
    try: days_diff = int(days_diff)
    except: days_diff = 9999
    freq_norm = str(row.get("update_frequency_norm", "")).lower()
    
    limit_days = 365
    if "mensual" in freq_norm: limit_days = 45
    if "trimestral" in freq_norm: limit_days = 100
    if "semestral" in freq_norm: limit_days = 190
    
    if days_diff <= limit_days: return 10, f"Vigente (Hace {days_diff} días)"
    else: return 0, f"Vencido (Hace {days_diff} días)"

def _calc_comprensibilidad(row: pd.Series) -> tuple:
    description = _smart_get(row, ["description", "notes", "about", "descripcion"], "")
    length = len(description)
    if length < 5: return 0, "Crítico: Descripción ausente."
    score = 10 * (1 - math.exp(-0.05 * length))
    score = min(10, max(0, score))
    obs = "Descripción detallada." if score >= 8 else ("Descripción aceptable." if score >= 5 else "Descripción muy breve.")
    return round(score, 1), obs

def _calc_exactitud_sintactica(row: pd.Series) -> tuple:
    name = _smart_get(row, ["name", "title"], "")
    desc = _smart_get(row, ["description", "notes"], "")
    text_corpus = (name + " " + desc).lower()
    syntactic_errors = ["&amp;", "&nbsp;", "&lt;", "&gt;", "ã³", "ã±", "ã", "Ã³", "Ã±", "Ã", "<div>", "<span>", "<br>", "<p>", "ï¿½"]
    detected = [err for err in syntactic_errors if err in text_corpus]
    count = len(detected)
    score = max(0, 10 - (count * 2))
    obs = "Sintaxis limpia." if count == 0 else f"Errores de codificación/HTML: {', '.join(detected[:2])}..."
    return score, obs

def _calc_exactitud_semantica(row: pd.Series) -> tuple:
    title = str(row.get("name", "")).lower()
    desc = str(row.get("description", "")).lower()
    tags = str(row.get("tags", "")).lower()
    category = str(row.get("category", "")).lower()
    
    fillers = ["prueba", "test", "borrar", "sin descripcion", "no disponible", "nan", "null"]
    if any(f in title for f in fillers) or (len(desc) < 20 and any(f in desc for f in fillers)):
        return 0, "Crítico: Dato de prueba/relleno."

    score = 10
    obs_parts = []
    if len(category) > 3 and category not in ["no aplica", "otros", "nan", "n/a"]:
        cat_tokens = set(category.replace(",", "").split())
        text_corpus = title + " " + desc + " " + tags
        match = any(t in text_corpus for t in cat_tokens)
        if not match:
            score -= 3
            obs_parts.append("Posible inconsistencia Categoría vs Contenido.")
    if len(tags) < 3 or tags == "nan": 
        score -= 2
        obs_parts.append("Faltan etiquetas.")
    
    return max(0, score), "Coherencia alta." if score == 10 else (" ".join(obs_parts) if obs_parts else "Metadatos básicos aceptables.")

def calculate_pdf_criteria(row: pd.Series) -> dict:
    sc_act, obs_act = _calc_actualidad(row)
    raw_comp = row.get("metadata_completeness", 0)
    sc_comp = round(raw_comp * 10, 1)
    is_public = str(row.get("public_access_level", "")).lower() == "public"
    sc_acc = 10 if is_public else 0
    
    contact = _smart_get(row, ["commoncore_contactemail", "contact_email", "email"])
    has_email = "@" in contact and "example" not in contact
    license_val = _smart_get(row, ["license", "commoncore_license", "licencia"])
    has_license = len(license_val) > 5 and license_val.lower() != "no especificada"
    sc_cred = 5 + (2.5 if has_email else 0) + (2.5 if has_license else 0)

    sc_conf = 10 
    sc_compr, obs_compr = _calc_comprensibilidad(row)
    sc_exact_sin, obs_exact_sin = _calc_exactitud_sintactica(row)
    sc_exact_sem, obs_exact_sem = _calc_exactitud_semantica(row)

    return {
        "Actualidad": sc_act, "Obs_Actualidad": obs_act,
        "Completitud": sc_comp,
        "Accesibilidad": sc_acc,
        "Credibilidad": int(sc_cred),
        "Conformidad": sc_conf,
        "Comprensibilidad": sc_compr, "Obs_Comprensibilidad": obs_compr,
        "Exactitud_Sintactica": sc_exact_sin, "Obs_Sintactica": obs_exact_sin,
        "Exactitud_Semantica": sc_exact_sem, "Obs_Semantica": obs_exact_sem
    }

# GENERACIÓN DE TEXTO AUTOMÁTICO:
# Generación de secciones narrativas para el informe ASPA 2025.
# son textos predefinidos y análisis automáticos basados en los puntajes calculados.
# se realiza de esta manera para estandarizar y agilizar la creación de informes técnicos.
# no se utilizan modelos de lenguaje para asegurar consistencia y evitar sesgos.
# no se utilizan plantillas externas para mantener todo el contenido generado dentro del código.
# no se utilizan servicios externos para garantizar privacidad y control total sobre el contenido.
# no se utilizan archivos de configuración para simplificar la implementación y despliegue del módulo.


def _generar_motivo_estudio(entity_name: str, dataset_name: str) -> str:
    """Genera el texto del Motivo de Estudio."""
    return (
        f"El presente informe técnico tiene como objetivo realizar la auditoría de calidad del activo de información digital "
        f"denominado '{dataset_name}', bajo la custodia de la entidad '{entity_name}'.\n\n"
        "Esta evaluación se realiza en el marco de la estrategia de Gobierno Digital y la normativa ASPA 2025, "
        "cuyo propósito es garantizar que los datos abiertos del Estado cumplan con los principios de calidad (Norma ISO 25012), "
        "interoperabilidad técnica y semántica, así como la usabilidad necesaria para generar valor público. "
        "El análisis busca identificar brechas en la documentación, estructura y actualización del recurso para "
        "elevar su nivel de madurez."
    )

def _generar_analisis_resultados(metrics: dict, avg_score: float) -> str:
    """
    Genera un análisis narrativo basado en los puntajes obtenidos.
    Identifica fortalezas y debilidades automáticamente.
    """
    # Clasificación General
    if avg_score >= 8.0:
        conclusion = "El activo presenta un nivel de calidad SOBRESALIENTE, cumpliendo con la mayoría de los estándares nacionales."
    elif avg_score >= 5.0:
        conclusion = "El activo presenta un nivel de calidad MEDIO. Es funcional pero requiere acciones correctivas en metadatos específicos."
    else:
        conclusion = "El activo presenta un nivel de calidad CRÍTICO. No cumple con los estándares mínimos para su reutilización efectiva."

    # Identificar criterios bajos y altos
    scores = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    # Filtramos solo los criterios base, no observaciones
    criterios_relevantes = {
        "Actualidad": scores.get("Actualidad", 0),
        "Completitud": scores.get("Completitud", 0),
        "Credibilidad": scores.get("Credibilidad", 0),
        "Accesibilidad": scores.get("Accesibilidad", 0),
        "Comprensibilidad": scores.get("Comprensibilidad", 0),
        "Exactitud": (scores.get("Exactitud_Sintactica", 0) + scores.get("Exactitud_Semantica", 0)) / 2
    }
    
    sorted_criteria = sorted(criterios_relevantes.items(), key=lambda item: item[1])
    peor_criterio = sorted_criteria[0]
    mejor_criterio = sorted_criteria[-1]

    analisis = (
        f"{conclusion}\n\n"
        f"Al analizar los componentes específicos, se destaca un desempeño sólido en {mejor_criterio[0]} "
        f"(Puntaje: {mejor_criterio[1]:.1f}/10). Sin embargo, la principal brecha se encuentra en el criterio de "
        f"{peor_criterio[0]} (Puntaje: {peor_criterio[1]:.1f}/10). "
    )

    # Recomendación específica basada en el peor criterio
    recomendaciones = {
        "Actualidad": "Se recomienda actualizar el conjunto de datos inmediatamente o revisar la frecuencia declarada.",
        "Completitud": "Es necesario diligenciar los campos de metadatos vacíos en el inventario.",
        "Credibilidad": "Se sugiere añadir una licencia clara y un correo de contacto institucional.",
        "Accesibilidad": "Verifique que el activo esté marcado como 'Público' en la plataforma.",
        "Comprensibilidad": "Se debe ampliar la descripción del activo para dar contexto al usuario.",
        "Exactitud": "Revise la codificación de caracteres y la coherencia de las etiquetas."
    }
    
    analisis += recomendaciones.get(peor_criterio[0], "Se recomienda revisar la guía de calidad.")
    
    return analisis

# Generación del reporte en formato .docx
# genera un documento estructurado con secciones y tablas según el estándar ASPA 2025.
# no utiliza plantillas externas para mantener todo el contenido generado dentro del código.
# no utiliza servicios externos para garantizar privacidad y control total sobre el contenido.
# no utiliza archivos de configuración para simplificar la implementación y despliegue del módulo.

def create_aspa_report(dataset_data: dict, entity_name: str) -> BytesIO:
    doc = Document()
    
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(10)

    # --- HEADER ---
    header = doc.sections[0].header
    hp = header.paragraphs[0]
    # cambiar por el logo
    hp.text = "MINISTERIO TIC"
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Titulo
    months = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    now = datetime.now()
    fecha_str = f"Bogotá D.C, {now.day} de {months[now.month-1]} de {now.year}"
    
    p = doc.add_paragraph()
    p.add_run(fecha_str).bold = True
    
    p_tit = doc.add_paragraph("INFORME TÉCNICO DEL ACTIVO DIGITAL - ASPA 2025")
    p_tit.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_tit.runs[0].bold = True
    p_tit.runs[0].font.size = Pt(14)
    p_tit.runs[0].font.color.rgb = RGBColor(0, 51, 102)

    doc.add_paragraph() 

    # calculos
    row_series = pd.Series(dataset_data)
    metrics = calculate_pdf_criteria(row_series)
    
    # Motivos de estudio
    h_motivo = doc.add_paragraph("1. MOTIVO DE ESTUDIO")
    h_motivo.runs[0].bold = True
    h_motivo.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    
    dataset_name = _smart_get(dataset_data, ["name", "Titulo"], "Sin nombre")
    texto_motivo = _generar_motivo_estudio(entity_name, dataset_name)
    doc.add_paragraph(texto_motivo)
    
    doc.add_paragraph()

    # Tabla de estudio
    url_recurso = dataset_data.get("url")
    uid = dataset_data.get("uid", "N/A")
    final_url = "No disponible"
    if url_recurso and len(url_recurso) > 5:
        final_url = url_recurso
    elif uid != "N/A": final_url = f"https://www.datos.gov.co/d/{uid}"

    derived_text = "Sí (Vista/Mapa derivado)" if str(dataset_data.get("derived_view", "false")).lower() in ['true', '1', 'yes'] else "No (Conjunto base)"
    lic_text = _smart_get(dataset_data, ["license", "Licencia", "La licencia del recurso"], "No especificada")
    type_res = _smart_get(dataset_data, ["type", "Tipo"], "N/A")
    domain = _smart_get(dataset_data, ["domain", "Dominio","El sitio que posee el recurso"], "www.datos.gov.co")
    coverage = _smart_get(dataset_data, ["informacindedatos_coberturageogrfica", "Información de Datos: Cobertura Geográfica"], "No registrada")
    desc_full = _smart_get(dataset_data, ["description", "Descripción"], "Sin descripción")
    desc_visual = desc_full[:400] + "..." if len(desc_full) > 400 else desc_full
    api_endpoint = dataset_data.get("api_endpoint")

    filas_reporte = [
        ("INFORMACIÓN GENERAL DEL RECURSO", "", True),
        ("Nombre del Activo:", dataset_name, False),
        ("Descripción:", desc_visual, False),
        ("Entidad Propietaria:", entity_name, False),
        ("UID (Identificador):", str(uid), False),
        ("URL del Recurso:", final_url, False),
        
        ("DETALLES TÉCNICOS Y METADATOS", "", True),
        ("Tipo de Recurso:", type_res, False),
        ("Dominio:", domain, False),
        ("Cobertura Geográfica:", coverage, False),
        ("Licencia:", lic_text, False),
        ("¿Es Recurso Derivado?:", derived_text, False),
    ]
    
    if api_endpoint: filas_reporte.append(("API Endpoint:", str(api_endpoint), False))

    filas_reporte.extend([
        ("EVALUACIÓN DE CALIDAD (GUÍA NACIONAL)", "", True),
        ("Actualidad:", f"{metrics['Obs_Actualidad']} (Puntaje: {metrics['Actualidad']}/10)", False),
        ("Completitud Metadatos:", f"{metrics['Completitud']} / 10.0", False),
        ("Credibilidad:", f"{metrics['Credibilidad']} / 10.0", False),
        ("Conformidad Técnica:", f"{metrics['Conformidad']} / 10.0", False),
        ("Comprensibilidad:", f"{metrics['Obs_Comprensibilidad']} (Puntaje: {metrics['Comprensibilidad']}/10)", False),
        ("Exactitud Sintáctica:", f"{metrics['Obs_Sintactica']} (Puntaje: {metrics['Exactitud_Sintactica']}/10)", False),
        ("Exactitud Semántica:", f"{metrics['Obs_Semantica']} (Puntaje: {metrics['Exactitud_Semantica']}/10)", False),
        ("Accesibilidad Pública:", "Cumple" if metrics['Accesibilidad'] == 10 else "Restringido", False),
    ])

    # Genera tabla
    table = doc.add_table(rows=len(filas_reporte) + 1, cols=2)
    table.style = 'Table Grid'
    table.autofit = False 
    for row in table.rows:
        row.cells[0].width = Inches(2.3)
        row.cells[1].width = Inches(4.4)

    for idx, (label, value, is_section) in enumerate(filas_reporte):
        row = table.rows[idx]
        if is_section:
            cell = row.cells[0]
            cell.merge(row.cells[1])
            cell.text = label
            _style_section_header(cell)
        else:
            p = row.cells[0].paragraphs[0]
            p.add_run(label).bold = True
            row.cells[1].text = str(value)

    # Promedio Global
    avg_score = (metrics['Actualidad'] + metrics['Completitud'] + metrics['Credibilidad'] + 
                 metrics['Accesibilidad'] + metrics['Conformidad'] + metrics['Comprensibilidad'] + 
                 metrics['Exactitud_Sintactica'] + metrics['Exactitud_Semantica']) / 8
    
    last_row = table.rows[-1].cells
    last_row[0].text = "ÍNDICE GLOBAL DE CALIDAD ASPA:"
    last_row[0].paragraphs[0].runs[0].bold = True
    last_row[1].text = f"{avg_score:.2f} / 10.0"
    last_row[1].paragraphs[0].runs[0].bold = True
    _style_score_cell(last_row[1], avg_score)

    doc.add_paragraph() # Espacio

    # Seccion de análisis de resultados
    h_resultados = doc.add_paragraph("2. ANÁLISIS DE RESULTADOS")
    h_resultados.runs[0].bold = True
    h_resultados.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    
    texto_analisis = _generar_analisis_resultados(metrics, avg_score)
    doc.add_paragraph(texto_analisis)

    doc.add_paragraph()

    # Sección de conclusiones y recomendaciones
    h_conclusiones = doc.add_paragraph("3. CONCLUSIONES Y RECOMENDACIONES")
    h_conclusiones.runs[0].bold = True
    h_conclusiones.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    
    doc.add_paragraph("Próximamente: Esta sección incluirá recomendaciones basadas en análisis avanzado de patrones y normativa específica del sector.")

    # Nota pie de página
    doc.add_paragraph()
    p_note = doc.add_paragraph("Informe generado automáticamente por el sistema de auditoría de activos digitales.")
    p_note.runs[0].font.size = Pt(8)
    p_note.runs[0].font.italic = True

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Estilos de tabla específicos para el reporte ASPA 2025.
# mejorar presentación visual de la tabla en el documento.

def _style_section_header(cell):
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.runs[0]
    run.bold = True
    run.font.color.rgb = RGBColor(255, 255, 255)
    _set_cell_background(cell, "1F4E78")

def _style_score_cell(cell, score):
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if score >= 8: _set_cell_background(cell, "E2EFDA")
    elif score >= 6: _set_cell_background(cell, "FFF2CC")
    else: _set_cell_background(cell, "FCE4D6")

def _set_cell_background(cell, color_hex):
    tcPr = cell._element.tcPr
    try: shd = tcPr.xpath('w:shd')[0]
    except IndexError: shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), color_hex)
    tcPr.append(shd)