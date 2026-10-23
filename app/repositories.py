"""Small local-development persistence layer; PostgreSQL can replace this via configuration."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator
from uuid import uuid4


class PredictionRepository:
    def __init__(self, database_url: str) -> None:
        if not database_url.startswith("sqlite:///"):
            raise ValueError("This starter repository supports SQLite locally; configure a SQLAlchemy/PostgreSQL repository for deployment.")
        self.path = Path(database_url.removeprefix("sqlite:///"))

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY, module TEXT NOT NULL, model_version TEXT NOT NULL,
                    result_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
                CREATE TABLE IF NOT EXISTS fraud_alerts (
                    id TEXT PRIMARY KEY, prediction_id TEXT NOT NULL, severity TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
                );
            """)

    def record_prediction(self, module: str, model_version: str, result: dict[str, object]) -> str:
        identifier, now = str(uuid4()), datetime.now(UTC).isoformat()
        with self.connection() as connection:
            connection.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)", (identifier, module, model_version, json.dumps(result), now))
        return identifier

    def add_fraud_alert(self, prediction_id: str, severity: str) -> str:
        identifier, now = str(uuid4()), datetime.now(UTC).isoformat()
        with self.connection() as connection:
            connection.execute("INSERT INTO fraud_alerts VALUES (?, ?, ?, 'New', ?, ?)", (identifier, prediction_id, severity, now, now))
        return identifier

    def history(self, limit: int = 25) -> list[dict[str, object]]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(row), "result": json.loads(row["result_json"])} for row in rows]

    def alerts(self, limit: int = 25) -> list[dict[str, object]]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM fraud_alerts ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def update_alert(self, alert_id: str, status: str) -> dict[str, object] | None:
        now = datetime.now(UTC).isoformat()
        with self.connection() as connection:
            cursor = connection.execute("UPDATE fraud_alerts SET status=?, updated_at=? WHERE id=?", (status, now, alert_id))
            if not cursor.rowcount:
                return None
            row = connection.execute("SELECT * FROM fraud_alerts WHERE id=?", (alert_id,)).fetchone()
        return dict(row) if row else None

    def summary(self) -> dict[str, int]:
        with self.connection() as connection:
            prediction_count = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            alert_count = connection.execute("SELECT COUNT(*) FROM fraud_alerts WHERE status IN ('New', 'Under Review', 'Escalated')").fetchone()[0]
        return {"prediction_count": prediction_count, "open_fraud_alerts": alert_count}

