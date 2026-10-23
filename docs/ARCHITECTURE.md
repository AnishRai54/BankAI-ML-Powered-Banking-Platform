# Architecture

`ml.data_catalog.DataCatalog` profiles source files as read-only streams. `BaseModelTrainer` defines the training contract; module trainers create and persist an sklearn pipeline plus versioned JSON metadata. `ModelRegistry` lazily loads artifacts, while `PredictionService` validates required columns, serves a model, and delegates persistence to `PredictionRepository`. FastAPI routes are deliberately thin.

The local starter uses SQLite (`PredictionRepository`) only for developer convenience. It is not the stated deployment database. A production adapter should implement the same repository contract with SQLAlchemy, PostgreSQL migrations, database roles, managed secrets, audit retention, and separately authorized identity/role providers.

