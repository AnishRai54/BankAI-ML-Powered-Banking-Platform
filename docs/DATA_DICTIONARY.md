# Data inventory and supported modules

Run `python -m ml.data_catalog` to produce `artifacts/data_validation/dataset_validation.json`, containing actual row/column counts, dtypes, missing values, samples, target distributions, duplicate scan status, and warnings for every discovered logical dataset.

| Dataset | Target examined | Supported use |
|---|---|---|
| Home Credit `application_train.csv` | `TARGET`, `AMT_CREDIT` | Credit-risk classification; indicative loan amount regression |
| Home Credit test/relational tables | None | Scoring/feature-engineering candidates; not independently trainable |
| `creditcard.csv` | `Class` | Fraud classification |
| `spam.csv` | `v1` (ham/spam), `v2` (text) | SMS spam classifier |
| Bank Marketing archives | `y` (term-deposit result) | Descriptive segmentation only; it is not a credit target |

Local source/license provenance is not present for most files. The catalog records that limitation rather than asserting an unverified license. Consult the original provider before redistribution or commercial use.

