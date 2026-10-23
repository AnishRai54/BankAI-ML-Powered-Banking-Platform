from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "dataset"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
MODEL_ROOT = ARTIFACT_ROOT / "models"
VALIDATION_ROOT = ARTIFACT_ROOT / "data_validation"

