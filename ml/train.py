# ml/train.py
from __future__ import annotations
import os
import json
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from ml.rules import classify_session

load_dotenv()
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:712002@127.0.0.1:3306/proyectocapstone?charset=utf8mb4",
)

# Estrategia de etiquetas: 'native' (por test) o 'unified' (5 niveles por percentil)
LABEL_STRATEGY = os.getenv("LABEL_STRATEGY", "native").strip().lower()  # 'native' | 'unified'

# Qué tests entrenar (por defecto los 3)
TRAIN_TESTS = os.getenv("TRAIN_TESTS", "burnout,pss10,sisco")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _posix(p: Path | str) -> str:
    return Path(p).as_posix()

# ========== DATA ==========
def _fetch_test_meta(eng, test_code: str):
    qtest = pd.read_sql(
        "SELECT id, code, name, likert_min, likert_max FROM tests WHERE code=%s LIMIT 1",
        eng,
        params=[test_code],
    )
    if qtest.empty:
        raise ValueError(f"No existe test code='{test_code}'")
    test_id = int(qtest.iloc[0]["id"])
    likert_min = int(qtest.iloc[0]["likert_min"])
    likert_max = int(qtest.iloc[0]["likert_max"])

    q = pd.read_sql(
        """
        SELECT q.id AS question_id, q.number, q.reversed, COALESCE(s.code, 'total') AS section_code
        FROM questions q
        LEFT JOIN sections s ON s.id = q.section_id
        WHERE q.test_id=%s
        ORDER BY q.number
        """,
        eng,
        params=[test_id],
    )
    if q.empty:
        raise ValueError(f"Test '{test_code}' no tiene preguntas")
    numbers = q["number"].astype(int).tolist()
    reversed_flags = q["reversed"].astype(bool).tolist()
    section_codes = q["section_code"].astype(str).tolist()
    qid_by_number = dict(zip(numbers, q["question_id"].tolist()))
    return {
        "test_id": test_id,
        "likert_min": likert_min,
        "likert_max": likert_max,
        "numbers": numbers,
        "reversed_flags": reversed_flags,
        "section_codes": section_codes,
        "qid_by_number": qid_by_number,
    }

def fetch_dataset(test_code: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Devuelve:
      X: matriz (n, Npreg) con valores 1..5 en orden q1..qN
      y_native: etiquetas nativas (str) por test
      total_pct_vec: vector con total normalizado 0..100 para cada sesión
      info: dict con feature_order y meta del test
    """
    eng = create_engine(DATABASE_URL)
    meta = _fetch_test_meta(eng, test_code)

    df = pd.read_sql(
        """
        SELECT s.id AS session_id, r.question_id, r.value
        FROM test_sessions s
        JOIN responses r ON r.session_id = s.id
        WHERE s.test_id=%s AND s.completed_at IS NOT NULL
        ORDER BY s.id, r.question_id
        """,
        eng,
        params=[meta["test_id"]],
    )
    if df.empty:
        raise ValueError(f"No hay respuestas para el test '{test_code}'")

    pv = (
        df.pivot_table(index="session_id", columns="question_id", values="value")
          .sort_index(axis=1)
    )
    needed_qids = [meta["qid_by_number"][n] for n in meta["numbers"]]
    pv = pv.reindex(columns=needed_qids)
    pv = pv.dropna(axis=0)  # exige todas las respuestas

    X_df = pv.copy()
    X_df.columns = [f"q{n}" for n in meta["numbers"]]
    values_matrix = X_df.values.astype(int)

    y_native: List[str] = []
    total_pct_vec: List[float] = []
    for row in values_matrix:
        res = classify_session(
            test_code=test_code,
            values_1_5=row.tolist(),
            reversed_flags=meta["reversed_flags"],
            section_codes=meta["section_codes"],
            likert_min=meta["likert_min"],
            likert_max=meta["likert_max"],
            use_unified_5_levels=False,
        )
        y_native.append(res.native_label)
        total_pct_vec.append(res.total_percent_0_100)

    cnt = Counter(y_native)
    print(f"[{test_code}] Distribución de clases (nativas):", dict(cnt))
    print(f"[{test_code}] Total sesiones usadas:", len(X_df))

    return (
        X_df.values,
        np.array(y_native, dtype=str),
        np.array(total_pct_vec, dtype=float),
        {"feature_order": X_df.columns.tolist(), "meta": meta},
    )

# ========== MODELOS ==========
def build_models(cv_svm: int) -> Dict[str, object]:
    """
    12 modelos. Calibramos clasificadores sin probas para permitir Stacking/Voting 'soft'.
    """
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="auto", random_state=42)),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=42),
                cv=cv_svm
            )),
        ]),
        "rf": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)),
        ]),
        "extratrees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42),
        "adaboost": AdaBoostClassifier(random_state=42),
        "gaussnb": Pipeline([("clf", GaussianNB())]),
        "lda": Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LinearDiscriminantAnalysis()),
        ]),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                estimator=RidgeClassifier(random_state=42),
                cv=cv_svm
            )),
        ]),
        "dtree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    }
    return models

# ========== STACKING / VOTING ==========
def _compute_unified_cuts_from_percentiles(total_pct_vec: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calcula cortes empíricos en la escala 0..100 equivalentes a los percentiles
    20/40/60/80 de tu muestra.
    """
    c20 = float(np.percentile(total_pct_vec, 20))
    c40 = float(np.percentile(total_pct_vec, 40))
    c60 = float(np.percentile(total_pct_vec, 60))
    c80 = float(np.percentile(total_pct_vec, 80))
    return c20, c40, c60, c80

def _labels_unified_from_cuts(total_pct_vec: np.ndarray, cuts: Tuple[float, float, float, float]) -> List[str]:
    c20, c40, c60, c80 = cuts
    out = []
    for v in total_pct_vec:
        if v <= c20:
            out.append("Eustrés")
        elif v <= c40:
            out.append("Estrés agudo")
        elif v <= c60:
            out.append("Estrés agudo episódico")
        elif v <= c80:
            out.append("Distrés")
        else:
            out.append("Estrés crónico")
    return out

# ========== TRAIN ==========
def train_one(test_code: str):
    X, y_native, total_pct_vec, info = fetch_dataset(test_code)
    stamp = time.strftime("%Y%m%d")
    outdir = MODELS_DIR / test_code
    outdir.mkdir(parents=True, exist_ok=True)

    if len(X) < 40:
        print(f"⚠ [{test_code}] Dataset pequeño ({len(X)}). Entrenando igual…")

    # Etiquetas a usar
    if LABEL_STRATEGY == "unified":
        cuts = _compute_unified_cuts_from_percentiles(total_pct_vec)
        y_labels = np.array(_labels_unified_from_cuts(total_pct_vec, cuts), dtype=str)
        label_mode = "unified"
        unified_meta = {
            "unified_cutpoints_0_100": {
                "p20": cuts[0], "p40": cuts[1], "p60": cuts[2], "p80": cuts[3]
            }
        }
        print(f"[{test_code}] Cortes unificados (0..100):", unified_meta["unified_cutpoints_0_100"])
    else:
        y_labels = y_native
        label_mode = "native"
        unified_meta = {
            "unified_cutpoints_0_100": None
        }

    le = LabelEncoder().fit(y_labels)
    y_enc = le.transform(y_labels)
    present = np.unique(y_enc)

    # Caso 1 sola clase
    if len(present) < 2:
        only = le.inverse_transform([present[0]])[0]
        print(f"ℹ [{test_code}] Solo hay una clase ('{only}'). Guardando DummyClassifier.")
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X, y_enc)

        path = outdir / f"{stamp}_dummy.joblib"
        joblib.dump(dummy, path, compress=3)
        meta = {
            "test_code": test_code,
            "version": stamp,
            "label_strategy": label_mode,
            "classes": le.classes_.tolist(),
            "feature_order": info["feature_order"],
            "models": [_posix(path)],
            "val_results": [{"name": "dummy", "acc": 1.0, "f1_macro": 1.0}],
            "class_counts": {cls: int(np.sum(y_labels == cls)) for cls in le.classes_},
            "train_size": int(len(X)),
            "val_size": 0,
            **unified_meta
        }
        joblib.dump(le, outdir / f"{stamp}_label_encoder.joblib", compress=3)
        with open(outdir / f"{stamp}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"✔ [{test_code}] DummyClassifier guardado en {outdir}/")
        return

    # split
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y_enc, test_size=0.25, stratify=y_enc, random_state=42
        )
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y_enc, test_size=0.25, random_state=42
        )

    # cv seguro para calibración
    counts = np.bincount(y_tr)
    min_per_class = int(counts[counts > 0].min())
    cv_svm = max(2, min(5, min_per_class))

    models = build_models(cv_svm)

    results, saved = [], []
    for name, model in models.items():
        print(f"\n[{test_code}] Entrenando {name}…")
        model.fit(X_tr, y_tr)
        p = model.predict(X_val)
        acc = accuracy_score(y_val, p)
        f1m = f1_score(y_val, p, average="macro")
        print(classification_report(y_val, p, target_names=le.classes_, zero_division=0))
        print(f"[{test_code}] {name}  ACC={acc:.3f}  F1(macro)={f1m:.3f}")

        path = outdir / f"{stamp}_{name}.joblib"
        joblib.dump(model, path, compress=3)
        results.append({"name": name, "acc": acc, "f1_macro": f1m})
        saved.append(_posix(path))

    # Ensamble Voting (opcional, útil para comparar)
    estimators_voting = []
    for r, pth in zip(results, saved):
        mdl = joblib.load(Path(pth))
        if hasattr(mdl, "predict_proba"):
            estimators_voting.append((r["name"], mdl))

    if len(estimators_voting) >= 2:
        vc = VotingClassifier(estimators=estimators_voting, voting="soft")
        print(f"\n[{test_code}] Entrenando VOTING (soft) con {len(estimators_voting)} modelos…")
        vc.fit(X_tr, y_tr)
        p = vc.predict(X_val)
        acc = accuracy_score(y_val, p)
        f1m = f1_score(y_val, p, average="macro")
        print(classification_report(y_val, p, target_names=le.classes_, zero_division=0))
        print(f"[{test_code}] voting  ACC={acc:.3f}  F1(macro)={f1m:.3f}")

        ens_path = outdir / f"{stamp}_voting.joblib"
        joblib.dump(vc, ens_path, compress=3)
        results.append({"name": "voting", "acc": acc, "f1_macro": f1m})
        saved.append(_posix(ens_path))
    else:
        print(f"ℹ [{test_code}] No hay suficientes modelos con predict_proba para voting.")

    # Ensamble STACKING (meta-modelo) — alineado con tu artículo
    base_estimators = []
    for r, pth in zip(results, saved):
        mdl = joblib.load(Path(pth))
        if hasattr(mdl, "predict_proba"):
            base_estimators.append((r["name"], mdl))

    if len(base_estimators) >= 2:
        stack = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(
                max_iter=2000, class_weight="balanced", multi_class="auto", random_state=42
            ),
            stack_method="predict_proba",
            passthrough=True,
            n_jobs=None
        )
        print(f"\n[{test_code}] Entrenando STACKING con {len(base_estimators)} modelos base…")
        stack.fit(X_tr, y_tr)
        p = stack.predict(X_val)
        acc = accuracy_score(y_val, p)
        f1m = f1_score(y_val, p, average="macro")
        print(classification_report(y_val, p, target_names=le.classes_, zero_division=0))
        print(f"[{test_code}] stacking  ACC={acc:.3f}  F1(macro)={f1m:.3f}")

        stack_path = outdir / f"{stamp}_stacking.joblib"
        joblib.dump(stack, stack_path, compress=3)
        results.append({"name": "stacking", "acc": acc, "f1_macro": f1m})
        saved.append(_posix(stack_path))
    else:
        print(f"ℹ [{test_code}] No hay suficientes modelos con predict_proba para stacking.")

    meta = {
        "test_code": test_code,
        "version": stamp,
        "label_strategy": label_mode,
        "classes": le.classes_.tolist(),
        "feature_order": info["feature_order"],
        "models": saved,
        "val_results": results,
        "class_counts": {cls: int(np.sum(y_labels == cls)) for cls in le.classes_},
        "train_size": int(len(X_tr)),
        "val_size": int(len(X_val)),
        **unified_meta
    }
    joblib.dump(le, outdir / f"{stamp}_label_encoder.joblib", compress=3)
    with open(outdir / f"{stamp}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✔ [{test_code}] Modelos guardados en {outdir}/")

def train():
    codes = [t.strip() for t in TRAIN_TESTS.split(",") if t.strip()]
    for code in codes:
        print(f"\n================== {code} ==================")
        try:
            train_one(code)
        except Exception as e:
            print(f"✖ Error entrenando {code}: {e}")

if __name__ == "__main__":
    train()
