# OOP guide

- `DataCatalog` owns safe discovery and streaming validation; `DatasetDefinition` is immutable metadata.
- `BaseModelTrainer` is the shared training abstraction. `CreditRiskTrainer`, `LoanAmountTrainer`, `FraudDetectionTrainer`, `SMSClassifierTrainer`, and `CustomerSegmentationTrainer` implement one responsibility each.
- `ModelRegistry` owns artifact loading and version metadata; `PredictionService` composes it with `PredictionRepository` through dependency injection.
- FastAPI endpoint functions only validate request schemas and call the service. They do not contain preprocessing or database logic.

This separation makes the local SQLite repository replaceable by a PostgreSQL/SQLAlchemy implementation without coupling model classes to persistence.

