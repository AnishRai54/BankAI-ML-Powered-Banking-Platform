from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.repositories import PredictionRepository


class ModelUnavailableError(RuntimeError):
    pass


class ModelRegistry:
    """Loads versioned joblib artifacts lazily; models are never trained during requests."""
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root, self._loaded = artifact_root, {}

    def metadata(self, name: str) -> dict[str, Any]:
        path = self.artifact_root / f"{name}.json"
        if not path.exists():
            raise ModelUnavailableError(f"Model '{name}' is not trained. Run: python -m ml.trainers {name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def model(self, name: str) -> Any:
        if name not in self._loaded:
            path = self.artifact_root / f"{name}.joblib"
            if not path.exists():
                raise ModelUnavailableError(f"Model artifact '{name}' is unavailable. Train it before serving predictions.")
            self._loaded[name] = joblib.load(path)
        return self._loaded[name]

    def all_metadata(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.artifact_root.glob("*.json"))]


class PredictionService:
    def __init__(self, registry: ModelRegistry, repository: PredictionRepository) -> None:
        self.registry, self.repository = registry, repository

    def predict(self, model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = self.registry.metadata(model_name)
        features = metadata["features"]
        missing = [column for column in features if column not in payload]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        frame = pd.DataFrame([{column: payload[column] for column in features}])
        started = time.perf_counter()
        model = self.registry.model(model_name)
        result: dict[str, Any] = {"model": model_name, "model_version": metadata["version"], "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        if model_name in {"credit_risk", "fraud_detection", "sms_security"}:
            values = payload["message"] if model_name == "sms_security" else frame
            probability = float(model.predict_proba([values] if model_name == "sms_security" else values)[0, 1])
            result.update({"risk_probability": probability, "risk_category": "high" if probability >= 0.75 else "medium" if probability >= 0.4 else "low"})
            if model_name == "sms_security":
                result["advisory"] = "This is a probabilistic spam assessment, not a guarantee that a message is safe."
        elif model_name == "loan_amount":
            estimate = max(0.0, float(model.predict(frame)[0]))
            result.update({"estimated_amount": estimate, "advisory": "Indicative estimate only; it is not a bank-approved offer."})
        elif model_name == "customer_segmentation":
            cluster = int(model.predict(frame)[0])
            result.update({"segment": f"Cluster {cluster + 1}", "cluster_id": cluster, "advisory": "Descriptive cluster only; not a credit or eligibility decision."})
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        prediction_id = self.repository.record_prediction(model_name, metadata["version"], result)
        result["prediction_id"] = prediction_id
        if model_name == "fraud_detection" and result["risk_probability"] >= 0.75:
            result["alert_id"] = self.repository.add_fraud_alert(prediction_id, "High")
        return result

