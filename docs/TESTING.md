# Testing

Run `python -m pytest -q`. Tests cover public health, API-key enforcement, and request validation without requiring model artifacts or a banking database. After training, perform a manual smoke test through `/docs` with the configured `X-API-Key`; confirm history and alert status changes in a separate local database.

