# ml/rules.py
from dataclasses import dataclass

MAX_PERSONAL   = 30  # 6*5
MAX_ESTUDIOS   = 35  # 7*5
MAX_COMPANEROS = 30  # 6*5
MAX_PROFESORES = 30  # 6*5
MAX_TOTAL      = 125 # 25*5

@dataclass
class SubscalePerc:
    personal: float
    estudios: float
    companeros: float
    profesores: float
    total: float

def to_percents(score_personal: int, score_estudios: int, score_comp: int, score_prof: int, score_total: int) -> SubscalePerc:
    return SubscalePerc(
        personal   = 100.0 * score_personal / MAX_PERSONAL   if MAX_PERSONAL   else 0.0,
        estudios   = 100.0 * score_estudios / MAX_ESTUDIOS   if MAX_ESTUDIOS   else 0.0,
        companeros = 100.0 * score_comp     / MAX_COMPANEROS if MAX_COMPANEROS else 0.0,
        profesores = 100.0 * score_prof     / MAX_PROFESORES if MAX_PROFESORES else 0.0,
        total      = 100.0 * score_total    / MAX_TOTAL      if MAX_TOTAL      else 0.0,
    )

def classify_from_perc(p: SubscalePerc) -> str:
    """
    Orden lógico 1→5. Devuelve uno de:
      'Eustrés', 'Estrés agudo', 'Estrés agudo episódico', 'Distrés', 'Estrés crónico'
    """
    subs = [p.personal, p.estudios, p.companeros, p.profesores]

    # 1) Eustrés
    if p.total <= 33 and sum(1 for x in subs if x <= 40) >= 3:
        return "Eustrés"

    # 2) Estrés agudo
    if p.total <= 66 and max(subs) >= 75 and sum(1 for x in subs if x < 60) >= 3:
        return "Estrés agudo"

    # 3) Estrés agudo episódico
    if p.total <= 66 and sum(1 for x in subs if x >= 60) >= 2 and p.total < 67:
        return "Estrés agudo episódico"

    # 4) Distrés
    if 34 <= p.total <= 66 and sum(1 for x in subs if x >= 50) >= 2:
        return "Distrés"

    # 5) Estrés crónico
    if p.total >= 67 or sum(1 for x in subs if x >= 70) >= 3:
        return "Estrés crónico"

    # Si no encaja en ninguna (poco probable), cae al caso más benigno
    return "Eustrés"

def classify_from_scores(score_personal: int, score_estudios: int, score_comp: int, score_prof: int, score_total: int) -> tuple[str, SubscalePerc]:
    perc = to_percents(score_personal, score_estudios, score_comp, score_prof, score_total)
    return classify_from_perc(perc), perc
