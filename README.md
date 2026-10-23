# BankAI

BankAI is a local-development banking intelligence dashboard built from the datasets already present in `dataset/`. It is decision support software: it does not approve loans, block transactions, or certify messages as safe.

## What is implemented

- FastAPI backend with versioned health, readiness, prediction, model, history and fraud-alert endpoints.
- Persistent local prediction history and fraud-review states in SQLite.
- API-key guard for sensitive endpoints. Set a real local key before use; no key is embedded in source.
- A responsive light/dark dashboard served by FastAPI.
- Reproducible object-oriented model trainers and non-destructive data validation.
- Real-data modules: Home Credit default risk and loan amount, credit-card fraud, SMS spam, and Bank Marketing customer segmentation.

## Run locally in VS Code / PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
$env:BANKAI_API_KEY = "choose-a-long-random-local-key"
python -m ml.data_catalog --dataset home_credit_applications_train
python -m ml.trainers all
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`, enter the same API key, and use the dashboard. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

On Windows, `python -m ml.trainers all` may take several minutes; it creates artifacts under `artifacts/models/` and never writes into `dataset/`.

## Test

```powershell
python -m pytest -q
```

## Dataset caveats

`application_train.csv` has the observed `TARGET` label for default-risk classification and `AMT_CREDIT` for an indicative regression target. The amount model deliberately does not consume contemporaneous amount-derived fields. `creditcard.csv` is an extremely imbalanced anonymized fraud dataset. `spam.csv` is a spam/ham corpus, not a bank-specific phishing corpus. Bank Marketing has no customer-segmentation ground truth: segmentation is descriptive, and call duration is excluded because it is post-contact.

Read [the validation summary](docs/DATA_VALIDATION_SUMMARY.md), [the data dictionary](docs/DATA_DICTIONARY.md), [architecture](docs/ARCHITECTURE.md), [security notes](docs/SECURITY.md), and [model governance](docs/MODEL_GOVERNANCE.md) before any non-demo use.

