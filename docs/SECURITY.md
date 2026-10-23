# Security boundaries

Prediction APIs require `X-API-Key` from `BANKAI_API_KEY`; no key is committed. Health/readiness remain public for local orchestration. Input schemas constrain types and ranges, error responses do not expose stack traces, and prediction history stores result metadata rather than the submitted financial profile.

This is a development baseline, not a compliance claim. Before bank deployment: replace API-key access with an approved OIDC identity provider and server-side RBAC; use TLS, PostgreSQL access controls and migrations, managed secrets, rate limiting, structured/redacted audit logging, retention rules, security testing, and an approved incident-response process.

