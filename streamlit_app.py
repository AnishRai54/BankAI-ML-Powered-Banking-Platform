"""Public Streamlit interface for the BankAI demonstration models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from app.repositories import PredictionRepository
from app.services import ModelRegistry, ModelUnavailableError, PredictionService


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "artifacts" / "models"
DATABASE_PATH = ROOT / "streamlit_bankai.db"

st.set_page_config(page_title="BankAI", page_icon="🏦", layout="wide")


@st.cache_resource
def get_service() -> PredictionService:
    """Create one lazily-loaded model service per Streamlit process."""
    repository = PredictionRepository(f"sqlite:///{DATABASE_PATH.as_posix()}")
    repository.initialize()
    return PredictionService(ModelRegistry(MODEL_ROOT), repository)


def predict(model_name: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return get_service().predict(model_name, payload)
    except (ModelUnavailableError, ValueError) as error:
        st.error(str(error))
    except Exception:
        st.error("The prediction could not be completed. Please check the supplied values and try again.")
    return None


def show_result(result: dict[str, Any]) -> None:
    if "risk_probability" in result:
        probability = result["risk_probability"]
        st.metric("Estimated risk probability", f"{probability:.1%}", result["risk_category"].title())
        st.progress(probability)
    elif "estimated_amount" in result:
        st.metric("Indicative amount", f"{result['estimated_amount']:,.0f}")
    else:
        st.success(f"{result['segment']} (ID {result['cluster_id']})")
    if advisory := result.get("advisory"):
        st.caption(advisory)


def credit_risk_page() -> None:
    st.header("Credit-risk assessment")
    st.caption("Demo decision support only. It must not be used to approve, decline, or price credit.")
    with st.form("credit_risk"):
        first, second = st.columns(2)
        with first:
            children = st.number_input("Number of children", 0, 30, 0)
            income = st.number_input("Annual income", min_value=1.0, value=180000.0, step=1000.0)
            credit = st.number_input("Requested credit amount", min_value=1.0, value=500000.0, step=1000.0)
            annuity = st.number_input("Annual annuity amount", min_value=1.0, value=25000.0, step=1000.0)
        with second:
            age = st.number_input("Age (years)", 18, 120, 35)
            employed = st.number_input("Employment length (years)", 0.0, 80.0, 5.0, step=0.5)
            region = st.selectbox("Region rating", [1, 2, 3], index=1)
            external = st.number_input("External source score (optional)", 0.0, 1.0, 0.5, step=0.01)
        submitted = st.form_submit_button("Assess risk", type="primary")
    if submitted:
        result = predict("credit_risk", {
            "CNT_CHILDREN": children, "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity, "DAYS_BIRTH": -age * 365,
            "DAYS_EMPLOYED": -round(employed * 365), "REGION_RATING_CLIENT": region,
            "EXT_SOURCE_2": external,
        })
        if result:
            show_result(result)


def loan_amount_page() -> None:
    st.header("Indicative loan amount")
    st.caption("An illustrative regression estimate, not a bank offer or lending decision.")
    with st.form("loan_amount"):
        first, second = st.columns(2)
        with first:
            children = st.number_input("Number of children", 0, 30, 0, key="loan_children")
            income = st.number_input("Annual income", min_value=1.0, value=180000.0, step=1000.0, key="loan_income")
            age = st.number_input("Age (years)", 18, 120, 35, key="loan_age")
            employed = st.number_input("Employment length (years)", 0.0, 80.0, 5.0, step=0.5, key="loan_employment")
            region = st.selectbox("Region rating", [1, 2, 3], index=1, key="loan_region")
        with second:
            income_type = st.selectbox("Income type", ["Working", "Commercial associate", "Pensioner", "State servant"])
            education = st.selectbox("Education", ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"])
            family = st.selectbox("Family status", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])
            housing = st.selectbox("Housing type", ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"])
        submitted = st.form_submit_button("Estimate amount", type="primary")
    if submitted:
        result = predict("loan_amount", {
            "CNT_CHILDREN": children, "AMT_INCOME_TOTAL": income, "DAYS_BIRTH": -age * 365,
            "DAYS_EMPLOYED": -round(employed * 365), "REGION_RATING_CLIENT": region,
            "NAME_INCOME_TYPE": income_type, "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": family, "NAME_HOUSING_TYPE": housing,
        })
        if result:
            show_result(result)


def sms_page() -> None:
    st.header("SMS security check")
    st.caption("This classifies messages as likely spam or ham. It does not certify a message as safe.")
    with st.form("sms"):
        message = st.text_area("SMS message", placeholder="Paste an SMS message to assess", max_chars=3000)
        submitted = st.form_submit_button("Check message", type="primary")
    if submitted:
        if not message.strip():
            st.warning("Enter a message first.")
            return
        result = predict("sms_security", {"v2": message.strip(), "message": message.strip()})
        if result:
            show_result(result)


def segmentation_page() -> None:
    st.header("Customer segmentation")
    st.caption("Clusters are descriptive only and must not be used for eligibility or credit decisions.")
    with st.form("segmentation"):
        first, second = st.columns(2)
        with first:
            age = st.number_input("Age", 18, 120, 35, key="segment_age")
            balance = st.number_input("Balance", value=1000.0, step=100.0)
            campaign = st.number_input("Campaign contacts", 0, 100, 1)
        with second:
            pdays = st.number_input("Days since previous contact (-1 if none)", -1, 999, -1)
            previous = st.number_input("Previous contacts", 0, 100, 0)
        submitted = st.form_submit_button("Assign descriptive cluster", type="primary")
    if submitted:
        result = predict("customer_segmentation", {"age": age, "balance": balance, "campaign": campaign, "pdays": pdays, "previous": previous})
        if result:
            show_result(result)


def fraud_page() -> None:
    st.header("Fraud-screening model")
    st.warning("Use only de-identified, authorised test data. This baseline needs all 30 anonymised dataset features.")
    features = get_service().registry.metadata("fraud_detection")["features"]
    example = {feature: 0.0 for feature in features}
    example["Amount"] = 100.0
    raw = st.text_area("Transaction features (JSON)", value=json.dumps(example, indent=2), height=360)
    if st.button("Screen transaction", type="primary"):
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("The JSON must be an object.")
            missing = [feature for feature in features if feature not in payload]
            if missing:
                raise ValueError(f"Missing features: {', '.join(missing)}")
            result = predict("fraud_detection", {feature: float(payload[feature]) for feature in features})
            if result:
                show_result(result)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            st.error(f"Invalid transaction JSON: {error}")


def activity_page() -> None:
    st.header("Session activity")
    service = get_service()
    summary = service.repository.summary()
    left, right = st.columns(2)
    left.metric("Predictions recorded", summary["prediction_count"])
    right.metric("Open fraud alerts", summary["open_fraud_alerts"])
    history = service.repository.history()
    if history:
        st.dataframe(history, width="stretch", hide_index=True)
    else:
        st.info("Predictions made during this app instance will appear here.")


st.title("🏦 BankAI")
st.caption("A Streamlit demonstration of locally trained banking analytics models.")

pages = {
    "Overview": None,
    "Credit risk": credit_risk_page,
    "Loan amount": loan_amount_page,
    "SMS security": sms_page,
    "Customer segments": segmentation_page,
    "Fraud screening": fraud_page,
    "Activity": activity_page,
}
page_name = st.sidebar.radio("Module", list(pages))
st.sidebar.divider()
st.sidebar.caption("Predictions are demonstrative. Do not enter personal, financial, or production data.")

if page_name == "Overview":
    st.subheader("Deployment-ready demonstration")
    st.write("Choose a module from the sidebar to run one of the included pretrained models.")
    try:
        metadata = get_service().registry.all_metadata()
        st.dataframe([
            {"Model": item["model_name"], "Version": item["version"], "Training data": item["dataset"], "Rows": item["rows_used"]}
            for item in metadata
        ], width="stretch", hide_index=True)
    except ModelUnavailableError as error:
        st.error(str(error))
else:
    pages[page_name]()
