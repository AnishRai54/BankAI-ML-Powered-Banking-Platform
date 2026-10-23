from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.repositories import PredictionRepository
from app.services import ModelRegistry, ModelUnavailableError, PredictionService


repository = PredictionRepository(settings.database_url)
registry = ModelRegistry(settings.artifact_root)
prediction_service = PredictionService(registry, repository)


@asynccontextmanager
async def lifespan(_: FastAPI):
    repository.initialize()
    yield


app = FastAPI(title="BankAI", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if not settings.api_key:
        raise HTTPException(503, "Authentication is not configured. Set BANKAI_API_KEY in .env before using prediction APIs.")
    if x_api_key != settings.api_key:
        raise HTTPException(401, "Invalid or missing API key.")


def get_service() -> PredictionService:
    return prediction_service


class CreditRiskRequest(BaseModel):
    cnt_children: int = Field(ge=0, le=30)
    income_total: float = Field(gt=0)
    credit_amount: float = Field(gt=0)
    annuity_amount: float = Field(gt=0)
    age_years: int = Field(ge=18, le=120)
    employment_years: float = Field(ge=0, le=80)
    region_rating_client: int = Field(ge=1, le=3)
    ext_source_2: float | None = Field(default=None, ge=0, le=1)

    def payload(self) -> dict[str, Any]:
        return {"CNT_CHILDREN": self.cnt_children, "AMT_INCOME_TOTAL": self.income_total, "AMT_CREDIT": self.credit_amount, "AMT_ANNUITY": self.annuity_amount, "DAYS_BIRTH": -self.age_years * 365, "DAYS_EMPLOYED": -round(self.employment_years * 365), "REGION_RATING_CLIENT": self.region_rating_client, "EXT_SOURCE_2": self.ext_source_2}


class LoanAmountRequest(BaseModel):
    cnt_children: int = Field(ge=0, le=30)
    income_total: float = Field(gt=0)
    age_years: int = Field(ge=18, le=120)
    employment_years: float = Field(ge=0, le=80)
    region_rating_client: int = Field(ge=1, le=3)
    income_type: str = Field(min_length=1, max_length=100)
    education_type: str = Field(min_length=1, max_length=100)
    family_status: str = Field(min_length=1, max_length=100)
    housing_type: str = Field(min_length=1, max_length=100)

    def payload(self) -> dict[str, Any]:
        return {"CNT_CHILDREN": self.cnt_children, "AMT_INCOME_TOTAL": self.income_total, "DAYS_BIRTH": -self.age_years * 365, "DAYS_EMPLOYED": -round(self.employment_years * 365), "REGION_RATING_CLIENT": self.region_rating_client, "NAME_INCOME_TYPE": self.income_type, "NAME_EDUCATION_TYPE": self.education_type, "NAME_FAMILY_STATUS": self.family_status, "NAME_HOUSING_TYPE": self.housing_type}


class FraudRequest(BaseModel):
    features: dict[str, float]


class SMSRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3_000)


class SegmentRequest(BaseModel):
    age: int = Field(ge=18, le=120)
    balance: float
    campaign: int = Field(ge=0, le=100)
    pdays: int = Field(ge=-1, le=999)
    previous: int = Field(ge=0, le=100)


class AlertUpdateRequest(BaseModel):
    status: Literal["New", "Under Review", "Escalated", "Resolved", "False Positive"]


@app.middleware("http")
async def request_id(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", str(uuid4()))
    return response


@app.exception_handler(ModelUnavailableError)
async def unavailable_model(_: Request, error: ModelUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse("frontend/index.html")


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}


@app.get("/api/v1/readiness")
async def readiness() -> dict[str, Any]:
    return {"status": "ready", "models_available": [item["model_name"] for item in registry.all_metadata()]}


@app.post("/api/v1/predictions/credit-risk", dependencies=[Depends(require_api_key)])
async def credit_risk(body: CreditRiskRequest, service: PredictionService = Depends(get_service)) -> dict[str, Any]:
    return service.predict("credit_risk", body.payload())


@app.post("/api/v1/predictions/loan-amount", dependencies=[Depends(require_api_key)])
async def loan_amount(body: LoanAmountRequest, service: PredictionService = Depends(get_service)) -> dict[str, Any]:
    return service.predict("loan_amount", body.payload())


@app.post("/api/v1/predictions/fraud", dependencies=[Depends(require_api_key)])
async def fraud(body: FraudRequest, service: PredictionService = Depends(get_service)) -> dict[str, Any]:
    return service.predict("fraud_detection", body.features)


@app.post("/api/v1/predictions/sms", dependencies=[Depends(require_api_key)])
async def sms(body: SMSRequest, service: PredictionService = Depends(get_service)) -> dict[str, Any]:
    return service.predict("sms_security", {"v2": body.message, "message": body.message})


@app.post("/api/v1/predictions/segmentation", dependencies=[Depends(require_api_key)])
async def segmentation(body: SegmentRequest, service: PredictionService = Depends(get_service)) -> dict[str, Any]:
    return service.predict("customer_segmentation", body.model_dump())


@app.get("/api/v1/predictions/history", dependencies=[Depends(require_api_key)])
async def history(limit: int = 25) -> dict[str, Any]:
    return {"items": repository.history(max(1, min(limit, 100)))}


@app.get("/api/v1/fraud/alerts", dependencies=[Depends(require_api_key)])
async def alerts(limit: int = 25) -> dict[str, Any]:
    return {"items": repository.alerts(max(1, min(limit, 100)))}


@app.patch("/api/v1/fraud/alerts/{alert_id}", dependencies=[Depends(require_api_key)])
async def update_alert(alert_id: str, body: AlertUpdateRequest) -> dict[str, Any]:
    alert = repository.update_alert(alert_id, body.status)
    if not alert:
        raise HTTPException(404, "Alert not found")
    return alert


@app.get("/api/v1/models", dependencies=[Depends(require_api_key)])
async def models() -> dict[str, Any]:
    return {"items": registry.all_metadata()}


@app.get("/api/v1/dashboard/summary", dependencies=[Depends(require_api_key)])
async def summary() -> dict[str, Any]:
    return {**repository.summary(), "models": registry.all_metadata(), "environment": settings.environment}
