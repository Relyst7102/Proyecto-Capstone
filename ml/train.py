# ml/train.py
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

from ml.rules import classify_from_scores

load_dotenv()
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:712002@127.0.0.1:3306/test_burnout?charset=utf8mb4",
)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _posix(p: Path | str) -> str:
    """Asegura rutas tipo POSIX para despliegues en Linux/Render."""
    return Path(p).as_posix()

def fetch_dataset():
    """
    Lee respuestas y subpuntuaciones desde la BD y construye:
      X: matriz (n_sesiones, 25) con q1..q25
      y: etiquetas de clase según reglas (string)
    Solo usa sesiones completadas y con TODAS las 25 respuestas.
    """
    eng = create_engine(DATABASE_URL)
    sql = """
    SELECT s.id AS session_id, r.question_id, r.value,
           s.score_personal, s.score_studies, s.score_peers, s.score_teachers, s.score_total
    FROM test_sessions s
    JOIN responses r ON r.session_id = s.id
    WHERE s.completed_at IS NOT NULL
    ORDER BY s.id, r.question_id
    """
    df = pd.read_sql(sql, eng)

    # X: q1..q25
    X = (
        df.pivot_table(index="session_id", columns="question_id", values="value")
          .rename(columns=lambda q: f"q{int(q)}")
          .dropna(axis=0)    # exige las 25 respuestas
          .sort_index()
    )

    # Subpuntuaciones por sesión (alineadas a X.index)
    meta = (
        df.drop_duplicates("session_id")[
            ["session_id", "score_personal", "score_studies", "score_peers", "score_teachers", "score_total"]
        ]
        .set_index("session_id")
        .loc[X.index]
    )

    # y (etiqueta) con reglas
    y = []
    for _, row in meta.iterrows():
        label, _ = classify_from_scores(
            int(row["score_personal"]),
            int(row["score_studies"]),
            int(row["score_peers"]),
            int(row["score_teachers"]),
            int(row["score_total"]),
        )
        y.append(label)

    y = pd.Series(y, index=X.index, name="label")

    # Info útil en consola
    cnt = Counter(y.values.tolist())
    print("Distribución de clases:", dict(cnt))
    print("Total sesiones usadas:", len(X))
    return X.values, y.values

def build_models(cv_svm: int):
    """Modelos base. SVM calibrada para obtener predict_proba y permitir Voting 'soft'."""
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                multi_class="auto",
                random_state=42
            )),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=42),
                cv=cv_svm
            )),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=42
        ),
        "gb": GradientBoostingClassifier(random_state=42),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)),
        ]),
    }

def train():
    X, y = fetch_dataset()

    if len(X) < 40:
        print(f"⚠ Dataset pequeño ({len(X)}). Entrenando igual…")

    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    present = np.unique(y_enc)

    # ---- CASO 1 SOLA CLASE: guardamos DummyClassifier y salimos ----
    if len(present) < 2:
        only = le.inverse_transform([present[0]])[0]
        print(f"ℹ Solo hay una clase en los datos: '{only}'. "
              "Se guardará un DummyClassifier (predice la clase más frecuente).")

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X, y_enc)

        stamp = time.strftime("%Y%m%d")
        path = MODELS_DIR / f"{stamp}_dummy.joblib"
        joblib.dump(dummy, path, compress=3)

        meta = {
            "version": stamp,
            "classes": le.classes_.tolist(),
            "feature_order": [f"q{i}" for i in range(1, 26)],
            "models": [_posix(path)],
            "val_results": [{"name": "dummy", "acc": 1.0, "f1_macro": 1.0}],
            "class_counts": {cls: int((y == cls).sum()) for cls in le.classes_},
            "train_size": int(len(X)),
            "val_size": 0
        }
        joblib.dump(le, MODELS_DIR / f"{stamp}_label_encoder.joblib", compress=3)
        with open(MODELS_DIR / f"{stamp}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print("✔ DummyClassifier guardado en /models. Repite el entrenamiento cuando tengas más variedad de clases.")
        return
    # ----------------------------------------------------------------

    # split estratificado si es posible
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y_enc, test_size=0.25, stratify=y_enc, random_state=42
        )
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y_enc, test_size=0.25, random_state=42
        )

    # cv seguro para la SVM calibrada (no mayor que el mínimo por clase)
    counts = np.bincount(y_tr)
    min_per_class = int(counts[counts > 0].min())
    cv_svm = max(2, min(5, min_per_class))

    models = build_models(cv_svm)

    results, saved = [], []
    stamp = time.strftime("%Y%m%d")

    for name, model in models.items():
        print(f"\nEntrenando {name}…")
        model.fit(X_tr, y_tr)
        p = model.predict(X_val)
        acc = accuracy_score(y_val, p)
        f1m = f1_score(y_val, p, average="macro")
        print(classification_report(y_val, p, target_names=le.classes_, zero_division=0))
        print(f"ACC={acc:.3f}  F1(macro)={f1m:.3f}")

        path = MODELS_DIR / f"{stamp}_{name}.joblib"
        joblib.dump(model, path, compress=3)
        results.append({"name": name, "acc": acc, "f1_macro": f1m})
        saved.append(_posix(path))

    # Ensamble (soft voting necesita predict_proba en todos)
    estimators = [(r["name"], joblib.load(Path(p))) for r, p in zip(results, saved)]
    vc = VotingClassifier(estimators=estimators, voting="soft")
    print("\nEntrenando ensamble…")
    vc.fit(X_tr, y_tr)
    p = vc.predict(X_val)
    acc = accuracy_score(y_val, p)
    f1m = f1_score(y_val, p, average="macro")
    print(classification_report(y_val, p, target_names=le.classes_, zero_division=0))
    print(f"ACC(ensamble)={acc:.3f}  F1(macro)={f1m:.3f}")

    ens_path = MODELS_DIR / f"{stamp}_voting.joblib"
    joblib.dump(vc, ens_path, compress=3)
    results.append({"name": "voting", "acc": acc, "f1_macro": f1m})
    saved.append(_posix(ens_path))

    meta = {
        "version": stamp,
        "classes": le.classes_.tolist(),
        "feature_order": [f"q{i}" for i in range(1, 26)],
        "models": saved,  # rutas POSIX compatibles con Linux/Render
        "val_results": results,
        "class_counts": {cls: int((y == cls).sum()) for cls in le.classes_},
        "train_size": int(len(X_tr)),
        "val_size": int(len(X_val)),
    }
    joblib.dump(le, MODELS_DIR / f"{stamp}_label_encoder.joblib", compress=3)
    with open(MODELS_DIR / f"{stamp}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✔ Modelos guardados en /models")

if __name__ == "__main__":
    train()
