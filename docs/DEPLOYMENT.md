# Deployment notes

## Local development

Use the commands in the README. The default database is SQLite and the dashboard runs from the same FastAPI process.

## Internal staging

Use a managed PostgreSQL instance behind an internal network boundary; implement a SQLAlchemy repository and Alembic migrations before switching `BANKAI_DATABASE_URL`. Supply `BANKAI_API_KEY` or, preferably, integrate the approved identity provider. Terminate TLS at an approved reverse proxy, set restrictive CORS origins, rotate secrets, and run migrations explicitly.

## Bank-managed production

No production deployment is configured or authorized by this repository. It needs formal model validation, security/privacy review, retention policy, approved audit controls, operational monitoring, backups and restore exercises, a human review process, and a formal release approval.

