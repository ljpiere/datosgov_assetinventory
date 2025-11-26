from __future__ import annotations

import math
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import csv
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except Exception:  # pragma: no cover - dependencia opcional
    REPORTLAB_AVAILABLE = False

REPORT_METADATA_FIELDS: Sequence[str] = (
    "UID",
    "Titulo",
    "Descripción",
    "entidad",
    "sector",
    "theme_group",
    "Etiqueta",
    "Categoría",
    "Público",
    "Common Core: Public Access Level",
    "Common Core: Contact Email",
    "Common Core: License",
    "Información de Datos: Cobertura Geográfica",
    "Información de Datos: Idioma",
    "Información de Datos: Frecuencia de Actualización",
    "update_frequency_norm",
)


def _clean_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        import pandas as pd  # import local para evitar dependencia dura al importar módulo
    except Exception:
        pd = None

    if pd is not None and isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _to_numeric(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _frequency_days(freq: str) -> int:
    freq = (freq or "").lower().strip()
    mapping = {
        "diaria": 1,
        "diario": 1,
        "semanal": 7,
        "quincenal": 15,
        "mensual": 30,
        "bimestral": 60,
        "trimestral": 90,
        "cuatrimestral": 120,
        "semestral": 180,
        "anual": 365,
        "anualidad": 365,
    }
    return mapping.get(freq, 90)


def _score_actualidad(days_since_update: Optional[float], freq: str) -> float:
    if days_since_update is None:
        return 5.0
    threshold = _frequency_days(freq)
    days = float(days_since_update)
    if days <= threshold:
        return 10.0
    ratio = min(days / threshold, 3.0)
    return max(0.0, 10.0 - (ratio - 1) * 5.0)


def _score_accesibilidad(record: Dict[str, Any]) -> float:
    public = str(record.get("Público", "") or "").lower() == "public"
    url = bool(record.get("url"))
    api = bool(record.get("API"))
    if public and (api or url):
        return 10.0
    if public and (api or url):
        return 8.0
    if public:
        return 6.0
    return 3.0


def _score_completitud(record: Dict[str, Any]) -> float:
    completeness = _to_numeric(record.get("metadata_completeness"))
    return round(max(0.0, min(10.0, completeness * 10)), 1)


def _score_comprensibilidad(record: Dict[str, Any]) -> float:
    desc_len = len(str(record.get("Descripción") or ""))
    has_tags = bool(record.get("Etiqueta")) or bool(record.get("Categoría"))
    has_theme = bool(record.get("theme_group"))
    score = 5.0
    if desc_len > 200:
        score += 2
    if has_tags:
        score += 1.5
    if has_theme:
        score += 1.5
    return min(10.0, round(score, 1))


def _score_credibilidad(record: Dict[str, Any]) -> float:
    completeness = _to_numeric(record.get("metadata_completeness"))
    coherence_flag_raw = record.get("coherence_flag")
    coherence_flag = False
    if isinstance(coherence_flag_raw, str):
        coherence_flag = coherence_flag_raw.lower() == "true"
    else:
        coherence_flag = bool(coherence_flag_raw)
    entidad = record.get("entidad") or ""
    score = 5.0
    if coherence_flag:
        score += 2.0
    if completeness >= 0.7:
        score += 2.0
    if entidad and entidad.lower() != "entidad sin registro":
        score += 1.0
    return min(10.0, round(score, 1))


def _score_relevancia(record: Dict[str, Any]) -> float:
    rows = _to_numeric(record.get("Número de Filas"))
    views = _to_numeric(record.get("Vistas"))
    downloads = _to_numeric(record.get("Descargas"))
    base = 5.0
    if rows > 50:
        base += 2.0
    if views > 500 or downloads > 100:
        base += 2.0
    if views > 2000 or downloads > 500:
        base += 1.0
    return min(10.0, round(base, 1))


def _score_portabilidad(record: Dict[str, Any]) -> float:
    url = str(record.get("url") or "")
    if any(url.lower().endswith(ext) for ext in (".csv", ".json", ".geojson", ".xlsx")):
        return 10.0
    if url:
        return 6.0
    return 4.0


def _score_disponibilidad(record: Dict[str, Any]) -> float:
    url = bool(record.get("url"))
    api = bool(record.get("API"))
    if url and api:
        return 10.0
    if url or api:
        return 8.0
    return 4.0


def _score_trazabilidad(record: Dict[str, Any]) -> float:
    issued = record.get("Common Core: Issued")
    last_update = record.get("Common Core: Last Update")
    contact = record.get("Common Core: Contact Email") or record.get("Correo Electrónico de Contacto")
    has_issued = bool(issued)
    has_update = bool(last_update)
    has_contact = bool(contact)
    score = 5.0
    if has_issued:
        score += 2.0
    if has_update:
        score += 2.0
    if has_contact:
        score += 1.0
    return min(10.0, round(score, 1))


def load_metric_catalog(path: Optional[Path] = None) -> List[Dict[str, str]]:
    csv_path = path or (Path(__file__).resolve().parent / "docs" / "metrics.csv")
    if not csv_path.exists():
        return []

    import csv

    catalog: List[Dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            catalog.append(row)
    return catalog


def build_dataset_metrics(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Calcula puntajes heurísticos para las métricas de la guía."""
    catalog = load_metric_catalog()
    scores = {
        "Accesibilidad": _score_accesibilidad(record),
        "Actualidad": _score_actualidad(
            record.get("days_since_update"), record.get("update_frequency_norm")
        ),
        "Completitud": _score_completitud(record),
        "Comprensibilidad": _score_comprensibilidad(record),
        "Credibilidad": _score_credibilidad(record),
        "Disponibilidad": _score_disponibilidad(record),
        "Portabilidad": _score_portabilidad(record),
        "Relevancia": _score_relevancia(record),
        "Trazabilidad": _score_trazabilidad(record),
    }

    metrics: List[Dict[str, Any]] = []
    for item in catalog:
        name = item.get("métrica")
        definition = item.get("definición", "")
        category = item.get("categoría", "")
        calc_note = item.get("cálculo", "")
        score = scores.get(name)
        if score is None:
            metrics.append(
                {
                    "Métrica": name,
                    "Categoría": category,
                    "Definición": definition,
                    "Puntaje": "No evaluado",
                    "Detalle": calc_note,
                }
            )
        else:
            metrics.append(
                {
                    "Métrica": name,
                    "Categoría": category,
                    "Definición": definition,
                    "Puntaje": f"{score:.1f}/10",
                    "Detalle": calc_note,
                }
            )
    return metrics


def build_quality_summary(record: Dict[str, Any]) -> List[Dict[str, str]]:
    completeness = _to_numeric(record.get("metadata_completeness"))
    days_since_update = record.get("days_since_update")
    days_since_update_num = None
    if days_since_update is not None:
        days_since_update_num = int(_to_numeric(days_since_update))
    views = _to_numeric(record.get("Vistas"))
    downloads = _to_numeric(record.get("Descargas"))
    freq = record.get("update_frequency_norm") or record.get(
        "Información de Datos: Frecuencia de Actualización", "sin registro"
    )
    coherence_flag_raw = record.get("coherence_flag")
    coherence_flag = False
    if isinstance(coherence_flag_raw, str):
        coherence_flag = coherence_flag_raw.lower() == "true"
    else:
        coherence_flag = bool(coherence_flag_raw)
    access_level = record.get("Common Core: Public Access Level") or "desconocido"

    return [
        {
            "label": "Completitud de metadatos",
            "value": f"{float(completeness) * 100:.1f}%",
            "detail": "Porcentaje de campos críticos diligenciados.",
        },
        {
            "label": "Coherencia de acceso",
            "value": "Consistente" if coherence_flag else "Revisar",
            "detail": f"Público vs. Public Access Level ({access_level}).",
        },
        {
            "label": "Frescura de datos",
            "value": f"{days_since_update_num if days_since_update_num is not None else 'N/D'} días",
            "detail": f"Frecuencia declarada: {freq}.",
        },
        {
            "label": "Consumo",
            "value": f"{int(views):,} vistas / {int(downloads):,} descargas",
            "detail": "Uso reciente del activo.",
        },
    ]


def build_metadata_pairs(
    record: Dict[str, Any], fields: Sequence[str] = REPORT_METADATA_FIELDS
) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for field in fields:
        value = _clean_value(record.get(field))
        if value in ("", None):
            continue
        pairs.append({"Campo": field, "Valor": str(value)})
    return pairs


def build_pdf_document(
    record: Dict[str, Any],
    quality: List[Dict[str, str]],
    metadata_pairs: List[Dict[str, str]],
    metric_scores: List[Dict[str, Any]],
) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab no está instalado.")

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin

    title = f"Reporte de dataset · UID {record.get('UID', '')}"
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title)
    y -= 18

    c.setFont("Helvetica", 11)
    subtitle = record.get("Titulo") or "Sin título"
    for line in textwrap.wrap(subtitle, 95):
        c.drawString(margin, y, line)
        y -= 14

    y -= 6
    c.setFont("Helvetica", 10)
    description = record.get("Descripción") or "Sin descripción disponible."
    for line in textwrap.wrap(description, 95):
        c.drawString(margin, y, line)
        y -= 12

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Métricas clave (Guía 2025)")
    y -= 16
    c.setFont("Helvetica", 10)
    for item in quality:
        c.drawString(margin, y, f"- {item['label']}: {item['value']} ({item['detail']})")
        y -= 12

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Métricas de la guía")
    y -= 16
    c.setFont("Helvetica", 10)
    for metric in metric_scores:
        if y < margin + 60:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)
        name = metric.get("Métrica", "Métrica")
        score = metric.get("Puntaje", "N/D")
        definition = metric.get("Definición", "")
        c.drawString(margin, y, f"{name}: {score}")
        y -= 12
        for line in textwrap.wrap(definition, 95):
            c.drawString(margin + 12, y, line)
            y -= 12
        detail = metric.get("Detalle", "")
        if detail:
            for line in textwrap.wrap(f"Nota: {detail}", 95):
                c.drawString(margin + 12, y, line)
                y -= 12
        y -= 6

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Metadatos")
    y -= 16
    c.setFont("Helvetica", 10)
    for pair in metadata_pairs:
        if y < margin + 40:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)

        wrapped = textwrap.wrap(f"{pair['Campo']}: {pair['Valor']}", 100)
        for line in wrapped:
            c.drawString(margin, y, line)
            y -= 12

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def _extract_score(score_label: str) -> str:
    if not score_label:
        return ""
    if isinstance(score_label, (int, float)):
        return f"{float(score_label):.1f}"
    text = str(score_label)
    if text.lower().startswith("no evaluado"):
        return ""
    if "/" in text:
        try:
            return f"{float(text.split('/')[0]):.1f}"
        except Exception:
            return text
    try:
        return f"{float(text):.1f}"
    except Exception:
        return text


def build_cut_csv(df) -> str:
    """Genera CSV plano con métricas y metadatos para todos los datasets públicos."""
    now = datetime.utcnow()
    catalog = load_metric_catalog()
    metric_names = [m.get("métrica") for m in catalog]

    base_fields = [
        "UID",
        "Titulo",
        "entidad",
        "sector",
        "theme_group",
        "metadata_completeness",
        "days_since_update",
        "update_frequency_norm",
        "Público",
        "Common Core: Public Access Level",
        "Fecha de última actualización de datos (UTC)",
    ]
    meta_fields = [f for f in REPORT_METADATA_FIELDS if f not in base_fields]
    fields = base_fields + meta_fields + metric_names + [
        "fecha_corte",
        "mes_reporte",
        "anio_reporte",
    ]

    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=";")
    writer.writeheader()

    for _, row in df.iterrows():
        record = {k: _clean_value(v) for k, v in row.items()}
        metric_scores = build_dataset_metrics(record)
        metric_map = {m["Métrica"]: _extract_score(m.get("Puntaje")) for m in metric_scores}

        last_update = record.get("Fecha de última actualización de datos (UTC)") or record.get(
            "Common Core: Last Update"
        )
        if isinstance(last_update, str):
            last_update_str = last_update
        else:
            last_update_str = _clean_value(last_update)

        row_dict = {
            "fecha_corte": now.date().isoformat(),
            "mes_reporte": now.strftime("%Y-%m"),
            "anio_reporte": now.year,
        }
        for field in base_fields + meta_fields:
            row_dict[field] = record.get(field, "")
        row_dict["Fecha de última actualización de datos (UTC)"] = last_update_str
        row_dict.update(metric_map)

        writer.writerow(row_dict)

    return buffer.getvalue()
