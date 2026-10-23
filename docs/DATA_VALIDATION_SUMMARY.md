# Actual local data validation summary

The complete, machine-readable validation report is [dataset_validation.json](../artifacts/data_validation/dataset_validation.json). It contains each filename, all columns and dtypes, per-column missing counts, three sample records, target distribution, duplicate-check result, and data-quality warnings. Source datasets were only read.

| File / archive member | Rows × columns | Label examined | Duplicate result |
|---|---:|---|---:|
| `application_train.csv` | 307,511 × 122 | `TARGET`: 24,825 positive / 282,686 negative | 0 |
| `application_test.csv` | 48,744 × 121 | no label | 0 |
| `bureau.csv` | 1,716,428 × 17 | none | not computed at database scale |
| `bureau_balance.csv` | 27,299,925 × 3 | none | not computed at database scale |
| `credit_card_balance.csv` | 3,840,312 × 23 | none | not computed at database scale |
| `installments_payments.csv` | 13,605,401 × 8 | none | not computed at database scale |
| `POS_CASH_balance.csv` | 10,001,358 × 8 | none | not computed at database scale |
| `previous_application.csv` | 1,670,214 × 37 | none | not computed at database scale |
| `creditcard.csv` | 284,807 × 31 | `Class`: 492 fraud / 284,315 non-fraud | 1,081 |
| `spam.csv` | 5,572 × 5 | `v1`: 747 spam / 4,825 ham | 403 |
| `bank.zip:bank-full.csv` | 45,211 × 17 | `y`: 5,289 yes / 39,922 no | 0 |
| `bank-additional.zip:bank-additional-full.csv` | 41,188 × 21 | `y`: 4,640 yes / 36,548 no | 12 |

The Home Credit column-description and sample-submission files were also profiled. `sample_submission.csv` has placeholder `TARGET=0.5` throughout and was not treated as labelled data. Exact duplicate detection is intentionally bounded at one million rows in the starter to avoid unsafe memory growth; all rows were still read for row counts, schemas, and missing-value totals.

