# Model governance

Every artifact is accompanied by metadata: version, dataset identifier, training timestamp, feature list, evaluation metrics, and leakage controls. Prediction records retain the artifact version. Training is an explicit CLI action—normal API requests never retrain or replace a model.

Artifacts are experimental decision-support baselines. Validate temporal stability, fairness, explainability, operational thresholds, privacy, human-review workflow, and regulatory obligations before any authorized use. Fraud alerts are review items, not confirmed fraud; SMS scores are not safety guarantees.

