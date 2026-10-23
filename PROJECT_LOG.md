# Project log

## 2026-10-23 — Initial working milestone

- Inspected the pre-existing `dataset/` directory. No source dataset was moved, overwritten, or downloaded.
- Added dataset catalog/profiling code, model trainers, FastAPI application, local SQLite history/alerts, dashboard, and API tests.
- Implemented only data-supported modules: credit risk, indicative loan amount, fraud, SMS spam, and descriptive segmentation.
- Full validation completed and saved to `artifacts/data_validation/dataset_validation.json`.
- Trained artifacts saved to `artifacts/models/`. Held-out metrics: credit risk ROC-AUC 0.6711 / PR-AUC 0.1604; fraud ROC-AUC 0.9721 / PR-AUC 0.7190; SMS ROC-AUC 0.9910 / PR-AUC 0.9745; loan MAE 266,852.67, RMSE 366,137.83, R² 0.1623; segmentation silhouette 0.3566.
- `python -m pytest -q` passed: 3 tests (one third-party TestClient deprecation warning).

