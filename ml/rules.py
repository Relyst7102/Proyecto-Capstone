# ml/rules.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Etiquetas unificadas (5 niveles)
EUSTRES = "Eustrés"
AGUDO = "Estrés agudo"
AGUDO_EP = "Estrés agudo episódico"
DISTRES = "Distrés"
CRONICO = "Estrés crónico"

UnifiedLabel = str

@dataclass
class SubscalePerc:
    by_section: Dict[str, float]   # % por sección (clave = code de Section)
    total: float                   # % total (0..100)

@dataclass
class ClassifiedResult:
    native_label: str                  # etiqueta según el test
    unified_label: Optional[str]       # etiqueta unificada 5-niveles (si se solicita)
    percents: SubscalePerc
    raw_scores: Dict[str, int]         # puntajes crudos por sección + total
    max_scores: Dict[str, int]         # máximos por sección + total
    total_percent_0_100: float         # total normalizado 0..100
    percentile_in_sample: Optional[float] = None  # 0..100 si se calcula fuera

# ---------- utilidades ----------
def _invert_if_needed(val: int, likert_min: int, likert_max: int, is_reversed: bool) -> int:
    if not is_reversed:
        return val
    # Ej.: Likert 1..5 -> invertido = (6 - val)
    return likert_min + (likert_max - val)

def _to_percent(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0

# ---------- mapeo unificado por percentil ----------
def to_unified_5_levels_from_percentile(pctl: float) -> UnifiedLabel:
    """
    Mapea el percentil (0..100) del puntaje total normalizado a 5 niveles.
    Cortes por defecto: 20/40/60/80 (puedes reemplazarlos con cortes empíricos guardados en meta).
    """
    if pctl <= 20:
        return EUSTRES
    if pctl <= 40:
        return AGUDO
    if pctl <= 60:
        return AGUDO_EP
    if pctl <= 80:
        return DISTRES
    return CRONICO

def to_unified_from_cutpoints(total_percent: float, cut20: float, cut40: float, cut60: float, cut80: float) -> UnifiedLabel:
    """
    Variante basada en cortes en la escala 0..100 (no percentil). Útil si guardas
    cortes empíricos en meta ya transformados a 0..100.
    """
    if total_percent <= cut20:
        return EUSTRES
    if total_percent <= cut40:
        return AGUDO
    if total_percent <= cut60:
        return AGUDO_EP
    if total_percent <= cut80:
        return DISTRES
    return CRONICO

# ---------- reglas por test ----------
def classify_burnout(section_scores: Dict[str, int], section_max: Dict[str, int]) -> Tuple[str, SubscalePerc]:
    """
    Burnout académico con 4 subescalas: personal, estudios, compañeros, profesores.
    Reproduce tu lógica previa con decisiones por % de subescalas y % total.
    """
    required = ["personal", "estudios", "compañeros", "profesores"]
    for k in required:
        section_scores.setdefault(k, 0)
        section_max.setdefault(k, 0)

    total = sum(section_scores.values())
    total_max = sum(section_max.values())
    psecs = {k: _to_percent(section_scores[k], section_max[k]) for k in required}
    ptotal = _to_percent(total, total_max)
    subs = [psecs["personal"], psecs["estudios"], psecs["compañeros"], psecs["profesores"]]

    # Misma lógica en orden 1→5
    if ptotal <= 33 and sum(1 for x in subs if x <= 40) >= 3:
        label = EUSTRES
    elif ptotal <= 66 and max(subs) >= 75 and sum(1 for x in subs if x < 60) >= 3:
        label = AGUDO
    elif ptotal <= 66 and sum(1 for x in subs if x >= 60) >= 2 and ptotal < 67:
        label = AGUDO_EP
    elif 34 <= ptotal <= 66 and sum(1 for x in subs if x >= 50) >= 2:
        label = DISTRES
    elif ptotal >= 67 or sum(1 for x in subs if x >= 70) >= 3:
        label = CRONICO
    else:
        label = EUSTRES

    return label, SubscalePerc(by_section=psecs, total=ptotal)

def classify_pss10(all_values_1_5: List[Tuple[int, bool]], likert_min: int, likert_max: int) -> Tuple[str, SubscalePerc]:
    """
    PSS-10 nativo: 0–40 (Likert 0..4 con ítems invertidos 4,5,7,8).
    0–13 bajo (Eustrés), 14–26 moderado (Distrés), 27–40 alto (Estrés crónico).
    """
    conv = []
    for val, is_rev in all_values_1_5:
        v = _invert_if_needed(val, likert_min, likert_max, is_rev)
        conv.append(v - likert_min)  # 1..5 -> 0..4
    raw_total_0_40 = int(sum(conv))  # 10 ítems

    if raw_total_0_40 <= 13:
        label = EUSTRES
    elif raw_total_0_40 <= 26:
        label = DISTRES
    else:
        label = CRONICO

    ptotal = _to_percent(raw_total_0_40, 40)
    return label, SubscalePerc(by_section={"total": ptotal}, total=ptotal)

def classify_sisco(section_scores: Dict[str, int], section_max: Dict[str, int]) -> Tuple[str, SubscalePerc]:
    """
    SISCO SV-21 (29 ítems) — dimensiones: estresores, reacciones, afrontamiento.
    Bandas por % total (ajústalas si tienes normas locales):
      <=33%  -> Eustrés
      34-50% -> Estrés agudo
      51-66% -> Estrés agudo episódico
      67-79% -> Distrés
      >=80%  -> Estrés crónico
    """
    total = sum(section_scores.values())
    total_max = sum(section_max.values())
    psecs = {k: _to_percent(section_scores[k], section_max[k]) for k in section_scores}
    ptotal = _to_percent(total, total_max)

    if ptotal <= 33:
        label = EUSTRES
    elif ptotal <= 50:
        label = AGUDO
    elif ptotal <= 66:
        label = AGUDO_EP
    elif ptotal <= 79:
        label = DISTRES
    else:
        label = CRONICO

    return label, SubscalePerc(by_section=psecs, total=ptotal)

# ---------- orquestador ----------
def classify_session(
    *,
    test_code: str,
    values_1_5: List[int],
    reversed_flags: List[bool],
    section_codes: List[str],
    likert_min: int,
    likert_max: int,
    # Opcional para etiqueta unificada:
    use_unified_5_levels: bool = False,
    unified_cutpoints_0_100: Optional[Tuple[float, float, float, float]] = None,
    # Si ya tienes el percentil (0..100) del total normalizado en tu muestra, pásalo aquí:
    known_percentile: Optional[float] = None,
) -> ClassifiedResult:
    """
    Calcula puntajes por sección y clasifica nativamente; opcionalmente devuelve
    etiqueta unificada 5-niveles según percentiles/cortes empíricos.
    """
    # 1) Puntajes por sección con inversión cuando aplique
    sec_scores: Dict[str, int] = {}
    sec_max:    Dict[str, int] = {}
    for v, is_rev, sec in zip(values_1_5, reversed_flags, section_codes):
        v_adj = _invert_if_needed(v, likert_min, likert_max, is_rev)
        sec_scores[sec] = sec_scores.get(sec, 0) + int(v_adj)
        sec_max[sec]    = sec_max.get(sec, 0)    + likert_max

    # 2) Clasificación nativa
    if test_code == "burnout":
        native, perc = classify_burnout(sec_scores, sec_max)
    elif test_code == "pss10":
        native, perc = classify_pss10(
            all_values_1_5=[(v, r) for v, r in zip(values_1_5, reversed_flags)],
            likert_min=likert_min,
            likert_max=likert_max,
        )
    elif test_code == "sisco":
        native, perc = classify_sisco(sec_scores, sec_max)
    else:
        # Genérico tipo SISCO por % total
        total = sum(sec_scores.values())
        total_max = sum(sec_max.values())
        ptotal = _to_percent(total, total_max)
        if ptotal <= 33:
            native = EUSTRES
        elif ptotal <= 50:
            native = AGUDO
        elif ptotal <= 66:
            native = AGUDO_EP
        elif ptotal <= 79:
            native = DISTRES
        else:
            native = CRONICO
        perc = SubscalePerc(by_section={k: _to_percent(sec_scores[k], sec_max[k]) for k in sec_scores}, total=ptotal)

    total_raw = sum(sec_scores.values())
    total_max = sum(sec_max.values())
    total_pct = _to_percent(total_raw, total_max)

    # 3) Etiqueta unificada (opcional)
    unified: Optional[str] = None
    used_percentile: Optional[float] = None
    if use_unified_5_levels:
        if unified_cutpoints_0_100:
            c20, c40, c60, c80 = unified_cutpoints_0_100
            unified = to_unified_from_cutpoints(total_pct, c20, c40, c60, c80)
        elif known_percentile is not None:
            used_percentile = float(known_percentile)
            unified = to_unified_5_levels_from_percentile(used_percentile)
        else:
            # Sin cortes ni percentil: usa cortes por defecto 20/40/60/80 en percentil teórico
            unified = to_unified_5_levels_from_percentile(total_pct)

    raw = dict(sec_scores)
    raw["total"] = total_raw
    mx  = dict(sec_max)
    mx["total"]  = total_max

    return ClassifiedResult(
        native_label=native,
        unified_label=unified,
        percents=perc,
        raw_scores=raw,
        max_scores=mx,
        total_percent_0_100=total_pct,
        percentile_in_sample=used_percentile,
    )
