"""Dataset discovery and validation that never mutates source data."""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from ml.config import DATA_ROOT, VALIDATION_ROOT


@dataclass(frozen=True)
class DatasetDefinition:
    key: str
    relative_path: str
    source: str
    license: str
    task: str
    target: str | None
    delimiter: str = ","
    encoding: str = "utf-8"
    zip_member: str | None = None


DATASETS = (
    DatasetDefinition("home_credit_applications_train", "home-credit-default-risk/application_train.csv", "Home Credit Default Risk (identified from folder name; local provenance file not supplied)", "Not documented in this workspace", "credit-risk classification and loan-amount regression", "TARGET"),
    DatasetDefinition("home_credit_applications_test", "home-credit-default-risk/application_test.csv", "Home Credit Default Risk (identified from folder name; local provenance file not supplied)", "Not documented in this workspace", "unlabelled scoring data", None),
    DatasetDefinition("home_credit_bureau", "home-credit-default-risk/bureau.csv", "Home Credit Default Risk", "Not documented in this workspace", "relational credit-history features", None),
    DatasetDefinition("home_credit_bureau_balance", "home-credit-default-risk/bureau_balance.csv", "Home Credit Default Risk", "Not documented in this workspace", "relational credit-history features", None),
    DatasetDefinition("home_credit_credit_card_balance", "home-credit-default-risk/credit_card_balance.csv", "Home Credit Default Risk", "Not documented in this workspace", "relational credit-history features", None),
    DatasetDefinition("home_credit_installments", "home-credit-default-risk/installments_payments.csv", "Home Credit Default Risk", "Not documented in this workspace", "relational repayment-history features", None),
    DatasetDefinition("home_credit_pos_cash", "home-credit-default-risk/POS_CASH_balance.csv", "Home Credit Default Risk", "Not documented in this workspace", "relational repayment-history features", None),
    DatasetDefinition("home_credit_previous_application", "home-credit-default-risk/previous_application.csv", "Home Credit Default Risk", "Not documented in this workspace", "relational application-history features", None),
    DatasetDefinition("home_credit_column_description", "home-credit-default-risk/HomeCredit_columns_description.csv", "Home Credit Default Risk", "Not documented in this workspace", "data dictionary", None, encoding="latin-1"),
    DatasetDefinition("home_credit_sample_submission", "home-credit-default-risk/sample_submission.csv", "Home Credit Default Risk", "Not documented in this workspace", "submission template", "TARGET"),
    DatasetDefinition("credit_card_fraud", "archive/creditcard.csv", "Credit card fraud dataset (identified from filename; local provenance file not supplied)", "Not documented in this workspace", "fraud classification", "Class"),
    DatasetDefinition("sms_spam", "archive (1)/spam.csv", "SMS Spam Collection (identified from filename; local provenance file not supplied)", "Not documented in this workspace", "SMS spam/phishing-style classification", "v1", encoding="latin-1"),
    DatasetDefinition("bank_marketing_full", "bank+marketing/bank.zip", "UCI Bank Marketing (identified from archive documentation)", "Check original UCI terms; no license file is stored locally", "customer segmentation; marketing classification is out of scope", "y", delimiter=";", zip_member="bank-full.csv"),
    DatasetDefinition("bank_marketing_additional_full", "bank+marketing/bank-additional.zip", "UCI Bank Marketing (identified from archive documentation)", "Check original UCI terms; no license file is stored locally", "customer segmentation; marketing classification is out of scope", "y", delimiter=";", zip_member="bank-additional/bank-additional-full.csv"),
)


class DataCatalog:
    """Streams source files to produce auditable, non-destructive profiles."""

    def __init__(self, data_root: Path = DATA_ROOT, chunk_size: int = 100_000) -> None:
        self.data_root, self.chunk_size = data_root, chunk_size

    def definitions(self) -> tuple[DatasetDefinition, ...]:
        return DATASETS

    def _reader(self, definition: DatasetDefinition, **kwargs: object) -> Iterator[pd.DataFrame]:
        path = self.data_root / definition.relative_path
        options = {"sep": definition.delimiter, "encoding": definition.encoding, "chunksize": self.chunk_size, "low_memory": False, **kwargs}
        if definition.zip_member:
            with zipfile.ZipFile(path) as archive:
                with archive.open(definition.zip_member) as member:
                    yield from pd.read_csv(member, **options)
        else:
            yield from pd.read_csv(path, **options)

    def profile(self, definition: DatasetDefinition, duplicate_limit: int = 1_000_000) -> dict[str, object]:
        """Compute all basic validation facts; duplicate scan is exact up to a stated cap."""
        rows = 0
        missing: Counter[str] = Counter()
        dtypes: dict[str, str] = {}
        target_counts: Counter[str] = Counter()
        sample: list[dict[str, object]] = []
        duplicate_hashes: set[int] | None = set()
        duplicates = 0
        duplicate_status = "exact"
        for chunk in self._reader(definition):
            if not dtypes:
                dtypes = {column: str(dtype) for column, dtype in chunk.dtypes.items()}
            rows += len(chunk)
            missing.update(chunk.isna().sum().astype(int).to_dict())
            if len(sample) < 3:
                sample.extend(chunk.head(3 - len(sample)).where(pd.notna(chunk.head(3 - len(sample))), None).to_dict("records"))
            if definition.target and definition.target in chunk:
                target_counts.update(chunk[definition.target].dropna().astype(str).tolist())
            if duplicate_hashes is not None:
                if rows > duplicate_limit:
                    duplicate_hashes = None
                    duplicate_status = f"not computed: dataset exceeds {duplicate_limit:,} rows; use a database-scale duplicate job"
                else:
                    for value in pd.util.hash_pandas_object(chunk, index=False).astype("uint64"):
                        hashed = int(value)
                        if hashed in duplicate_hashes:
                            duplicates += 1
                        else:
                            duplicate_hashes.add(hashed)
        warnings: list[str] = []
        if definition.key.startswith("bank_marketing"):
            warnings.append("duration occurs after a marketing call begins and must not be used for pre-contact inference; excluded from segmentation.")
        if definition.key == "home_credit_sample_submission":
            warnings.append("TARGET values are placeholder probabilities, not observed labels; this is not training data.")
        if definition.key == "home_credit_applications_test":
            warnings.append("TARGET is absent, so this file cannot train or evaluate a supervised model.")
        if definition.key == "credit_card_fraud":
            warnings.append("Class imbalance must be handled with stratification and class weighting; V1–V28 are anonymized, limiting interpretability.")
        if definition.key == "sms_spam":
            warnings.append("Columns beyond v1/v2 are empty export artifacts and are excluded from modelling.")
        return {
            "dataset": asdict(definition), "path_exists": (self.data_root / definition.relative_path).exists(),
            "row_count": rows, "column_count": len(dtypes), "columns": list(dtypes), "dtypes": dtypes,
            "missing_values": dict(missing), "duplicate_records": duplicates if duplicate_hashes is not None else None,
            "duplicate_check": duplicate_status, "target_distribution": dict(target_counts), "sample_records": sample,
            "potential_leakage_or_limitations": warnings,
        }

    def validate_all(self) -> dict[str, object]:
        return {definition.key: self.profile(definition) for definition in self.definitions()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile existing datasets without modifying them.")
    parser.add_argument("--dataset", choices=[item.key for item in DATASETS])
    args = parser.parse_args()
    catalog = DataCatalog()
    definitions = [item for item in DATASETS if not args.dataset or item.key == args.dataset]
    result = {item.key: catalog.profile(item) for item in definitions}
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    destination = VALIDATION_ROOT / (f"{args.dataset}.json" if args.dataset else "dataset_validation.json")
    destination.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
