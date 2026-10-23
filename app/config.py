from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("BANKAI_ENV", "development")
    database_url: str = os.getenv("BANKAI_DATABASE_URL", "sqlite:///./bankai.db")
    api_key: str | None = os.getenv("BANKAI_API_KEY")
    artifact_root: Path = Path(__file__).resolve().parents[1] / "artifacts" / "models"


settings = Settings()

