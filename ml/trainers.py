"""OOP model trainers using only the datasets already in this repository."""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, mean_absolute_error, precision_recall_fscore_support,
    r2_score, roc_auc_score, root_mean_squared_error, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import DATA_ROOT, MODEL_ROOT


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    version: str
    dataset: str
    rows_used: int
    features: list[str]
    metrics: dict[str, float]
    leakage_controls: list[str]
    trained_at: str


class BaseModelTrainer(ABC):
    model_name: str

    @abstractmethod
    def train(self) -> TrainingResult: ...

    def save(self, pipeline: Any, result: TrainingResult) -> TrainingResult:
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_ROOT / f"{self.model_name}.joblib")
        (MODEL_ROOT / f"{self.model_name}.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return result

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat()


class CreditRiskTrainer(BaseModelTrainer):
    model_name = "credit_risk"
    features = ["CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED", "REGION_RATING_CLIENT", "EXT_SOURCE_2"]

    def train(self) -> TrainingResult:
        data = pd.read_csv(DATA_ROOT / "home-credit-default-risk/application_train.csv", usecols=self.features + ["TARGET"])
        x, y = data[self.features], data["TARGET"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        pipeline = Pipeline([("preprocess", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])), ("model", LogisticRegression(max_iter=1_000, class_weight="balanced", n_jobs=None))])
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, prediction, average="binary", zero_division=0)
        return self.save(pipeline, TrainingResult(self.model_name, "1.0.0", "application_train.csv", len(data), self.features, {"precision": float(precision), "recall": float(recall), "f1": float(f1), "roc_auc": float(roc_auc_score(y_test, probability)), "pr_auc": float(average_precision_score(y_test, probability))}, ["Stratified holdout split", "TARGET excluded from features", "Preprocessing fitted on training partition only", "No relational tables joined because temporal alignment must be validated first"], self.now()))


class LoanAmountTrainer(BaseModelTrainer):
    model_name = "loan_amount"
    features = ["CNT_CHILDREN", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED", "REGION_RATING_CLIENT", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]

    def train(self) -> TrainingResult:
        data = pd.read_csv(DATA_ROOT / "home-credit-default-risk/application_train.csv", usecols=self.features + ["AMT_CREDIT"])
        x, y = data[self.features], data["AMT_CREDIT"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        numeric = [name for name in self.features if pd.api.types.is_numeric_dtype(x[name])]
        categorical = [name for name in self.features if name not in numeric]
        preprocess = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric), ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)], sparse_threshold=0)
        pipeline = Pipeline([("preprocess", preprocess), ("model", TransformedTargetRegressor(regressor=HistGradientBoostingRegressor(max_iter=250, l2_regularization=2.0, random_state=42), func=np.log1p, inverse_func=np.expm1))])
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        return self.save(pipeline, TrainingResult(self.model_name, "1.0.0", "application_train.csv (AMT_CREDIT)", len(data), self.features, {"mae": float(mean_absolute_error(y_test, predicted)), "rmse": float(root_mean_squared_error(y_test, predicted)), "r2": float(r2_score(y_test, predicted))}, ["Holdout split before preprocessing", "AMT_CREDIT is the target", "AMT_ANNUITY and AMT_GOODS_PRICE deliberately excluded as likely contemporaneous/derived amount fields"], self.now()))


class FraudDetectionTrainer(BaseModelTrainer):
    model_name = "fraud_detection"

    def train(self) -> TrainingResult:
        data = pd.read_csv(DATA_ROOT / "archive/creditcard.csv")
        features = [column for column in data.columns if column != "Class"]
        x_train, x_test, y_train, y_test = train_test_split(data[features], data["Class"], test_size=0.2, random_state=42, stratify=data["Class"])
        pipeline = Pipeline([("preprocess", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])), ("model", LogisticRegression(max_iter=1_000, class_weight="balanced"))])
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, prediction, average="binary", zero_division=0)
        return self.save(pipeline, TrainingResult(self.model_name, "1.0.0", "creditcard.csv", len(data), features, {"precision": float(precision), "recall": float(recall), "f1": float(f1), "roc_auc": float(roc_auc_score(y_test, probability)), "pr_auc": float(average_precision_score(y_test, probability))}, ["Stratified holdout split", "Class excluded from features", "Class-weighted baseline for extreme imbalance", "This dataset is anonymized; V1–V28 cannot produce customer-readable reasons"], self.now()))


class SMSClassifierTrainer(BaseModelTrainer):
    model_name = "sms_security"

    def train(self) -> TrainingResult:
        data = pd.read_csv(DATA_ROOT / "archive (1)/spam.csv", encoding="latin-1", usecols=["v1", "v2"]).dropna()
        labels = (data["v1"].str.lower() == "spam").astype(int)
        x_train, x_test, y_train, y_test = train_test_split(data["v2"], labels, test_size=0.2, random_state=42, stratify=labels)
        pipeline = Pipeline([("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)), ("model", LogisticRegression(max_iter=1_000, class_weight="balanced"))])
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, prediction, average="binary", zero_division=0)
        return self.save(pipeline, TrainingResult(self.model_name, "1.0.0", "spam.csv", len(data), ["v2"], {"precision": float(precision), "recall": float(recall), "f1": float(f1), "roc_auc": float(roc_auc_score(y_test, probability)), "pr_auc": float(average_precision_score(y_test, probability))}, ["Stratified holdout split", "Only label v1 and message v2 used", "Empty export columns excluded", "Spam probability is decision support, not a safety guarantee"], self.now()))


class CustomerSegmentationTrainer(BaseModelTrainer):
    model_name = "customer_segmentation"
    features = ["age", "balance", "campaign", "pdays", "previous"]

    def train(self) -> TrainingResult:
        archive = DATA_ROOT / "bank+marketing/bank.zip"
        with zipfile.ZipFile(archive) as bundle, bundle.open("bank-full.csv") as member:
            data = pd.read_csv(member, sep=";", usecols=self.features)
        pipeline = Pipeline([("preprocess", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])), ("model", KMeans(n_clusters=4, random_state=42, n_init=20))])
        labels = pipeline.fit_predict(data)
        transformed = pipeline.named_steps["preprocess"].transform(data)
        return self.save(pipeline, TrainingResult(self.model_name, "1.0.0", "bank.zip:bank-full.csv", len(data), self.features, {"silhouette_score": float(silhouette_score(transformed, labels, sample_size=min(10_000, len(data)), random_state=42))}, ["Unsupervised clustering: y not used", "duration excluded because it is measured during the call and unsuitable for a pre-contact profile", "Clusters are descriptive groups, not risk categories or offers"], self.now()))


TRAINERS: dict[str, type[BaseModelTrainer]] = {item.model_name: item for item in (CreditRiskTrainer, LoanAmountTrainer, FraudDetectionTrainer, SMSClassifierTrainer, CustomerSegmentationTrainer)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BankAI models from existing local datasets.")
    parser.add_argument("model", choices=[*TRAINERS, "all"])
    args = parser.parse_args()
    names = TRAINERS if args.model == "all" else {args.model: TRAINERS[args.model]}
    started = time.monotonic()
    for name, trainer_class in names.items():
        result = trainer_class().train()
        print(json.dumps(asdict(result), indent=2))
    print(f"Completed in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
