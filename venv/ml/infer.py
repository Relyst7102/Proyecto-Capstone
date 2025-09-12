# ml/infer.py
from __future__ import annotations
import json
import numpy as np
import joblib
from pathlib import Path

class MLModelPredictor:
    """
    Carga el último modelo entrenado (por fecha) y hace predicción con q1..q25.
    Busca en /models:
      - <YYYYMMDD>_meta.json
      - <YYYYMMDD>_label_encoder.joblib
      - modelos *.joblib (voting si existe)
    """
    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.label_encoder = None
        self.feature_order = [f"q{i}" for i in range(1, 26)]
        self.version = None
        self.model_name = None
        self._load_latest()

    def _load_latest(self):
        metas = sorted(self.models_dir.glob("*_meta.json"), reverse=True)
        if not metas:
            return
        meta_path = metas[0]
        self.version = meta_path.stem.replace("_meta", "")
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        # Encoder y orden de features
        self.label_encoder = joblib.load(self.models_dir / f"{self.version}_label_encoder.joblib")
        self.feature_order = meta.get("feature_order", self.feature_order)

        # Modelo: si hay voting úsalo; si no, el primero
        model_path = None
        for p in meta.get("models", []):
            if "voting" in p:
                model_path = p
                break
        if not model_path:
            model_path = meta.get("models", [None])[0]
        if model_path:
            self.model = joblib.load(model_path)
            self.model_name = Path(model_path).stem

    def is_ready(self) -> bool:
        return self.model is not None and self.label_encoder is not None

    def predict(self, answers: dict[str, float]):
        """
        answers: dict con claves 'q1'..'q25' (valores 1..5)
        """
        if not self.is_ready():
            return {"label": None, "proba": {}, "version": self.version, "model": None}

        x = np.array([[answers.get(f, 0) for f in self.feature_order]], dtype=float)
        proba = {}
        label_idx = None

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)[0]
            classes = list(self.label_encoder.classes_)
            for c, p in zip(classes, probs):
                proba[str(c)] = float(p)
            label_idx = int(np.argmax(probs))
        else:
            label_idx = int(self.model.predict(x)[0])

        label = str(self.label_encoder.inverse_transform([label_idx])[0])
        return {
            "label": label,
            "proba": proba,
            "version": self.version,
            "model": self.model_name or self.model.__class__.__name__,
        }
