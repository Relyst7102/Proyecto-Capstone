# ml/infer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np


class MLModelPredictor:
    """
    Carga el último modelo entrenado (por fecha) desde la carpeta `models`
    y realiza predicción con features q1..q25.

    Espera encontrar (idealmente):
      - <YYYYMMDD>_meta.json
      - <YYYYMMDD>_label_encoder.joblib
      - Uno o más modelos *.joblib (preferencia por 'voting' si existe)

    Esta versión es tolerante:
      - Si el meta tiene rutas con backslashes (Windows) o relativas,
        las normaliza.
      - Si falta algo, NO revienta la app; simplemente deja el predictor
        en modo "no listo" para que la UI use las reglas.
    """

    def __init__(self, models_dir: str | Path = "models") -> None:
        self.models_dir: Path = Path(models_dir)
        self.model: Any = None
        self.label_encoder: Any = None
        self.feature_order = [f"q{i}" for i in range(1, 26)]
        self.version: Optional[str] = None
        self.model_name: Optional[str] = None
        self._load_latest()

    # ---------- utilidades internas ----------

    def _norm_path(self, p: str | Path) -> Path:
        """
        Normaliza:
          - backslashes -> slashes
          - rutas relativas -> intenta resolver dentro de self.models_dir
        """
        p = Path(str(p).replace("\\", "/"))
        if p.is_absolute():
            return p
        # Si viene "models/xxx.joblib" o "xxx.joblib", probamos dentro de la carpeta models
        cand_name = self.models_dir / p.name
        if cand_name.exists():
            return cand_name
        cand_full = self.models_dir / p
        return cand_full

    def _load_latest(self) -> None:
        # 1) Carpeta models
        if not self.models_dir.exists():
            print(f"[WARN] models dir not found: {self.models_dir}")
            return

        # 2) Localiza el meta más reciente
        metas = sorted(self.models_dir.glob("*_meta.json"), reverse=True)
        if not metas:
            print(f"[WARN] no *_meta.json found in {self.models_dir}")
            # fallback: si hubiera un .joblib suelto, lo intentamos
            self._fallback_load_any_joblib()
            return

        meta_path = metas[0]
        self.version = meta_path.stem.replace("_meta", "")

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] cannot read meta {meta_path}: {e}")
            self._fallback_load_any_joblib()
            return

        # 3) feature_order opcional desde meta
        if isinstance(meta, dict) and "feature_order" in meta:
            fo = meta.get("feature_order")
            if isinstance(fo, list) and len(fo) == 25:
                self.feature_order = fo

        # 4) label encoder
        le_path = self.models_dir / f"{self.version}_label_encoder.joblib"
        try:
            self.label_encoder = joblib.load(le_path)
        except Exception as e:
            print(f"[WARN] label encoder not loaded ({le_path}): {e}")
            self.label_encoder = None

        # 5) el/los modelos (prefiere 'voting' si existe)
        model_path = None
        models_in_meta = meta.get("models", []) if isinstance(meta, dict) else []
        if isinstance(models_in_meta, list):
            # prefer voting
            voting = [m for m in models_in_meta if "voting" in str(m).lower()]
            if voting:
                model_path = voting[0]
            elif models_in_meta:
                model_path = models_in_meta[0]

        if model_path is not None:
            mp = self._norm_path(model_path)
            try:
                self.model = joblib.load(mp)
                self.model_name = mp.stem
                print(f"[INFO] Loaded model: {mp}")
                return
            except Exception as e:
                print(f"[WARN] model not loaded ({mp}): {e}")

        # 6) fallback si el meta no trae ruta válida
        self._fallback_load_any_joblib()

    def _fallback_load_any_joblib(self) -> None:
        """Si el meta no sirve, intenta cargar el *.joblib más reciente del directorio."""
        candidates = sorted(self.models_dir.glob("*.joblib"))
        if not candidates:
            print(f"[WARN] no .joblib files found in {self.models_dir}")
            return
        latest = candidates[-1]
        try:
            self.model = joblib.load(latest)
            self.model_name = latest.stem
            if self.version is None:
                self.version = latest.stem
            print(f"[INFO] Fallback loaded model: {latest}")
        except Exception as e:
            print(f"[WARN] fallback model not loaded ({latest}): {e}")
            self.model = None
            self.model_name = None

    # ---------- API pública ----------

    def is_ready(self) -> bool:
        """
        Listo para predecir si hay modelo.
        (El label_encoder es opcional: si falta, el label será el que
         devuelva el modelo sin decodificar.)
        """
        return self.model is not None

    def predict(self, answers: Dict[str, float]) -> Dict[str, Any]:
        """
        answers: dict con claves 'q1'..'q25' (valores 1..5)
        Devuelve:
          {
            "label": str | None,
            "proba": dict[str,float],
            "version": str | None,
            "model": str | None,
          }
        """
        if not self.is_ready():
            return {"label": None, "proba": {}, "version": self.version, "model": None}

        x = np.array([[answers.get(f, 0) for f in self.feature_order]], dtype=float)

        # Probabilidades (si el modelo las soporta)
        proba: Dict[str, float] = {}
        label_value: Optional[Any] = None

        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(x)[0]
                if self.label_encoder is not None and hasattr(self.label_encoder, "classes_"):
                    classes = list(self.label_encoder.classes_)
                else:
                    # sin encoder: usar índices 0..n-1 como clases
                    classes = list(range(len(probs)))
                for c, p in zip(classes, probs):
                    proba[str(c)] = float(p)
                # label por máxima prob
                label_idx = int(np.argmax(probs))
                if self.label_encoder is not None:
                    label_value = self.label_encoder.inverse_transform([label_idx])[0]
                else:
                    label_value = label_idx
            except Exception as e:
                print(f"[WARN] predict_proba failed: {e}")

        # Si no se pudo proba o el modelo no tiene predict_proba
        if label_value is None:
            try:
                pred = self.model.predict(x)[0]
                if self.label_encoder is not None:
                    # si el modelo devuelve índice, decodifica; si ya devuelve clase, inverse_transform lo maneja
                    label_value = self.label_encoder.inverse_transform([pred])[0]
                else:
                    label_value = pred
            except Exception as e:
                print(f"[WARN] predict failed: {e}")
                return {"label": None, "proba": {}, "version": self.version, "model": self.model_name}

        return {
            "label": str(label_value) if label_value is not None else None,
            "proba": proba,
            "version": self.version,
            "model": self.model_name or (self.model.__class__.__name__ if self.model is not None else None),
        }
