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


def apply_design() -> None:
    """Bring the Streamlit shell in line with the original BankAI dashboard."""
    st.markdown("""
    <style>
      :root { --navy:#0b1220; --blue:#2563eb; --teal:#14b8a6; --paper:#f8fafc;
              --line:#e2e8f0; --muted:#64748b; --ink:#162033; --card:#fff; }
      .stApp { background:var(--paper); color:var(--ink); }
      header[data-testid="stHeader"] { background:transparent; }
      #MainMenu, footer { visibility:hidden; }
      .block-container { max-width:1500px; padding:2.6rem clamp(1.5rem,5vw,4.5rem) 3rem; }
      section[data-testid="stSidebar"] { background:var(--navy); width:250px !important; }
      section[data-testid="stSidebar"] > div { background:var(--navy); padding:1.55rem 1rem; }
      section[data-testid="stSidebar"] * { color:#afbdd1; }
      .bankai-brand { color:#fff; font-size:25px; font-weight:750; letter-spacing:-.03em; margin:0 0 .15rem; }
      .bankai-brand b { color:var(--teal); }
      .platform-label { color:#91a4c2 !important; font-size:10px; font-weight:800; letter-spacing:.12em; margin:0 0 1.5rem; }
      section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius:8px; color:#afbdd1; padding:.6rem .7rem; margin:0 0 .25rem;
      }
      section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:#21304a; color:#fff; }
      section[data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size:.9rem; font-weight:600; }
      section[data-testid="stSidebar"] [data-testid="stRadio"] input { display:none; }
      h1, h2, h3 { color:var(--ink); letter-spacing:-.035em; }
      h1 { font-size:clamp(1.75rem,3vw,2.5rem) !important; margin:.3rem 0 .25rem !important; }
      h2 { font-size:1.25rem !important; }
      .eyebrow { color:var(--muted); font-size:.68rem; font-weight:800; letter-spacing:.12em; margin:0; }
      .subhead { color:var(--muted); max-width:700px; margin:0 0 1.4rem; line-height:1.55; }
      .notice { display:flex; gap:.8rem; align-items:center; background:#e9f7f5; border:1px solid #bfe8e2;
                color:#17665d; padding:.8rem 1rem; border-radius:10px; margin:0 0 1.4rem; }
      .notice b { white-space:nowrap; }
      div[data-testid="stMetric"] { background:var(--card); border:1px solid var(--line); border-radius:12px;
                                    box-shadow:0 4px 15px #17255408; padding:1.15rem; }
      div[data-testid="stMetricLabel"] { color:var(--muted); font-size:.8rem; }
      div[data-testid="stMetricValue"] { font-size:1.7rem; }
      div[data-testid="stForm"] { background:var(--card); border:1px solid var(--line); border-radius:12px;
                                    box-shadow:0 4px 15px #17255408; padding:1.45rem; }
      div[data-testid="stForm"] [data-testid="stNumberInput"],
      div[data-testid="stForm"] [data-testid="stSelectbox"],
      div[data-testid="stForm"] [data-testid="stTextArea"] { margin-bottom:.3rem; }
      label, [data-testid="stWidgetLabel"] p { font-size:.78rem !important; font-weight:700 !important; }
      input, textarea, [data-baseweb="select"] > div { border-color:var(--line) !important; border-radius:7px !important; }
      .stButton button, .stFormSubmitButton button { background:var(--blue) !important; border:0 !important;
         border-radius:7px !important; color:#fff !important; font-weight:750 !important; padding:.58rem .9rem !important; }
      .module-heading { position:relative; background:var(--card); border:1px solid var(--line); border-radius:12px 12px 0 0;
                        padding:1.35rem 1.45rem .2rem; margin-bottom:-.25rem; }
      .module-heading h2 { margin:.25rem 0 .35rem; }
      .module-heading p:last-child { color:var(--muted); max-width:720px; margin:0 0 .8rem; }
      .module-mark { position:absolute; top:1.25rem; right:1.35rem; width:34px; height:34px; display:grid; place-items:center;
                     border-radius:8px; font-size:.68rem; font-weight:850; }
      .blue { background:#dbeafe; color:#1d4ed8; } .teal { background:#ccfbf1; color:#0f766e; }
      .amber { background:#fef3c7; color:#b45309; } .violet { background:#ede9fe; color:#6d28d9; }
      .red { background:#fee2e2; color:#b91c1c; }
      .result-card { margin-top:1rem; border:1px solid var(--line); border-radius:10px; padding:1.25rem; background:#f8fafc; }
      .risk-hero { text-align:center; background:linear-gradient(145deg,#fbfdff,#f3f7fc); border-color:#dbe5f0; }
      .score-orbit { --score:0%; --risk-accent:#2563eb; width:150px; aspect-ratio:1; margin:1rem auto;
          border-radius:50%; display:grid; place-items:center; background:conic-gradient(var(--risk-accent) var(--score),#e4eaf1 0); }
      .score-core { width:124px; height:124px; border-radius:50%; display:grid; place-content:center; background:#fff;
                    box-shadow:inset 0 0 0 1px var(--line); }
      .score-core strong { font-size:2.2rem; letter-spacing:-.07em; line-height:1; }
      .score-core span, .result-caption, .result-meta { color:var(--muted); font-size:.75rem; }
      .risk-low { --risk-accent:#16a34a; } .risk-medium { --risk-accent:#f59e0b; } .risk-high { --risk-accent:#dc2626; }
      .risk-low h3 { color:#15803d; } .risk-medium h3 { color:#b45309; } .risk-high h3 { color:#b91c1c; }
      .result-label { color:var(--muted); font-size:.65rem; font-weight:800; letter-spacing:.11em; }
      .result-value { display:block; font-size:2.1rem; letter-spacing:-.05em; margin:.6rem 0 0; }
      .result-note { color:#475569; font-size:.84rem; margin:.7rem 0 0; }
      .estimate { background:#effcf9; border-color:#bfe8e2; } .segment { background:#f5f3ff; border-color:#ddd6fe; }
      @media(max-width: 900px) { section[data-testid="stSidebar"] { width:64px !important; }
        .block-container { padding:1.6rem 1rem; } .bankai-brand, .platform-label, section[data-testid="stSidebar"] p { font-size:0 !important; }
        .bankai-brand b { font-size:1.4rem !important; } }
    </style>
    """, unsafe_allow_html=True)


def module_heading(mark: str, color: str, eyebrow: str, title: str, copy: str) -> None:
    st.markdown(f"""
    <section class="module-heading"><span class="module-mark {color}">{mark}</span>
      <p class="eyebrow">{eyebrow}</p><h2>{title}</h2><p>{copy}</p>
    </section>
    """, unsafe_allow_html=True)


apply_design()


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
        category = result["risk_category"]
        st.markdown(f"""
        <section class="result-card risk-hero risk-{category}">
          <span class="result-label">MODEL RISK ASSESSMENT</span>
          <div class="score-orbit" style="--score:{probability * 100:.1f}%"><div class="score-core">
             <strong>{probability:.1%}</strong><span>risk score</span></div></div>
          <h3>{category.upper()} RISK</h3>
          <p class="result-note">Estimated model probability. Review is required before taking any action.</p>
          <p class="result-meta">Model {result['model_version']} · {result['latency_ms']} ms</p>
        </section>
        """, unsafe_allow_html=True)
    elif "estimated_amount" in result:
        st.markdown(f"""
        <section class="result-card estimate"><span class="result-label">INDICATIVE ESTIMATE</span>
          <strong class="result-value">{result['estimated_amount']:,.0f}</strong>
          <span class="result-caption">Estimated credit amount in dataset units</span>
          <p class="result-note">Model {result['model_version']} · {result['latency_ms']} ms</p>
        </section>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <section class="result-card segment"><span class="result-label">DESCRIPTIVE GROUP</span>
          <strong class="result-value">{result['segment']}</strong>
          <span class="result-caption">Behavior-pattern grouping · Cluster ID {result['cluster_id']}</span>
        </section>
        """, unsafe_allow_html=True)
    if advisory := result.get("advisory"):
        st.caption(advisory)


def credit_risk_page() -> None:
    module_heading("CR", "blue", "CREDIT INTELLIGENCE", "Credit-risk check", "Enter an authorized application profile. Values use the training dataset’s units, not a live bank credit score.")
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
    module_heading("LN", "teal", "LOAN ANALYTICS", "Indicative loan estimate", "This is a research estimate from the available dataset — not a bank offer.")
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
    module_heading("SM", "amber", "SMS SECURITY", "Check a message", "Paste an SMS to estimate spam likelihood. A low score does not mean the message is safe.")
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
    module_heading("CI", "violet", "CUSTOMER INTELLIGENCE", "Find a descriptive segment", "Groups behavior patterns only. It does not determine credit eligibility.")
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
    module_heading("FR", "red", "FRAUD INTELLIGENCE", "Analyst-only fraud scoring", "This baseline uses anonymized transaction inputs. Use only de-identified, authorised test data.")
    st.warning("The model needs all 30 dataset features; scores support review and do not block transactions.")
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
    module_heading("AC", "blue", "MODEL MONITORING", "Session activity", "Prediction records are stored for this Streamlit instance only.")
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


pages = {
    "Overview": None,
    "Credit risk": credit_risk_page,
    "Loan amount": loan_amount_page,
    "SMS security": sms_page,
    "Customer segments": segmentation_page,
    "Fraud screening": fraud_page,
    "Activity": activity_page,
}
st.sidebar.markdown('<div class="bankai-brand"><b>◇</b> BankAI</div><p class="platform-label">INTELLIGENCE PLATFORM</p>', unsafe_allow_html=True)
page_name = st.sidebar.radio("Module", list(pages))
st.sidebar.divider()
st.sidebar.markdown("<p class='platform-label'>DEVELOPMENT</p>", unsafe_allow_html=True)
st.sidebar.caption("Decision support only")

if page_name == "Overview":
    st.markdown("<p class='eyebrow'>BANKAI / WORKSPACE</p><h1>One place for your banking intelligence checks.</h1><p class='subhead'>Use sample or authorized information only. Every result is a model estimate, never an automatic decision.</p>", unsafe_allow_html=True)
    st.markdown("<section class='notice'><b>Before you begin</b><span>Choose a module from the sidebar. Source datasets remain unchanged and the included artifacts load locally.</span></section>", unsafe_allow_html=True)
    service = get_service()
    summary = service.repository.summary()
    metadata = service.registry.all_metadata()
    first, second, third = st.columns(3)
    first.metric("Stored predictions", summary["prediction_count"])
    second.metric("Open fraud alerts", summary["open_fraud_alerts"])
    third.metric("Trained models", len(metadata))
    st.markdown("<br><p class='eyebrow'>MODEL MONITORING</p><h2>Actual artifact registry</h2>", unsafe_allow_html=True)
    try:
        st.dataframe([
            {"Model": item["model_name"], "Version": item["version"], "Training data": item["dataset"], "Rows": item["rows_used"]}
            for item in metadata
        ], width="stretch", hide_index=True)
    except ModelUnavailableError as error:
        st.error(str(error))
else:
    pages[page_name]()
