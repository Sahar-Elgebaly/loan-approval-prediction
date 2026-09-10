import os
import joblib
import pandas as pd
import streamlit as st
from Data_prep import add_engineered_features

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💳", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #eef2f9 0%, #e6ecf5 100%);
}

/* Force readable dark text everywhere by default (hero/button override below win due to higher specificity) */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stCaptionContainer"] p,
div[data-testid="stCaptionContainer"] span {
    color: #0f172a;
}

/* Hero title band */
.hero {
    background: linear-gradient(135deg, #3730a3 0%, #4f46e5 60%, #6366f1 100%);
    padding: 28px 30px;
    border-radius: 18px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(67,56,202,0.25);
}

.hero h1 {
    color: #ffffff !important;
    font-size: 28px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #e0e7ff !important;
    font-size: 14px;
    margin: 6px 0 0 0;
}

div[data-testid="stForm"] {
    background-color: #ffffff;
    border-radius: 18px;
    border: 1px solid #dbe2ef;
    padding: 14px 24px 22px 24px;
    box-shadow: 0 6px 20px rgba(30,41,59,0.08);
}

h3 {
    color: #1e1b4b !important;
    font-weight: 800 !important;
    font-size: 18px !important;
    border-left: 5px solid #4f46e5;
    padding-left: 10px;
    margin-top: 18px !important;
}

input[type="number"] {
    background-color: #ffffff;
    font-weight: 700;
    color: #0f172a !important;
    caret-color: #4f46e5;
    padding: 10px 8px;
    height: 42px;
    font-size: 15px;
}

div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    border: 1.5px solid #a5b4cf;
    border-radius: 8px;
    min-height: 44px;
    background-color: #ffffff;
}

div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border: 1.5px solid #4f46e5;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.15);
}

[data-testid="stWidgetLabel"] p {
    font-weight: 700 !important;
    font-size: 14px !important;
    color: #1e293b !important;
}

div[data-baseweb="select"] > div {
    background-color: #ffffff;
    border: 1.5px solid #a5b4cf !important;
    color: #0f172a;
    font-weight: 600;
}

div[data-baseweb="popover"] ul {
    background-color: #ffffff;
}

li[role="option"] {
    color: #1e293b;
    font-weight: 600;
}

li[role="option"]:hover {
    background-color: #eef2ff;
}

/* dropdown arrow icon was invisible (white-on-white) — force it dark */
div[data-baseweb="select"] svg {
    fill: #1e293b !important;
}

.result {
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    margin-top: 15px;
    font-size: 22px;
    font-weight: 800;
}

.approved {
    background-color: #dcfce7;
    color: #15803d;
    border: 2px solid #4ade80;
}

.rejected {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 2px solid #f87171;
}

div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-size: 18px;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(79,70,229,0.35);
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #4338ca, #4f46e5);
    color: white;
    border: none;
}

/* probability gauge */
.gauge-wrap {
    display: flex;
    justify-content: center;
    margin: 10px 0 4px 0;
}

.gauge-label {
    text-align: center;
    font-weight: 800;
    font-size: 15px;
    color: #1e293b;
    margin-top: -6px;
}
</style>

<div class="hero">
    <h1>💳 Loan Approval Predictor</h1>
</div>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "loan_pipeline.pkl")


@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)


def build_gauge_svg(probability, prediction):
    """Circular gauge showing the approval probability, colored by outcome."""
    radius = 60
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - probability)
    color = "#22c55e" if prediction == 1 else "#ef4444"
    pct_text = f"{probability:.0%}"

    return (
        f'<svg width="150" height="150" viewBox="0 0 140 140">'
        f'<circle cx="70" cy="70" r="{radius}" stroke="#e2e8f0" stroke-width="14" fill="none" />'
        f'<circle cx="70" cy="70" r="{radius}" stroke="{color}" stroke-width="14" fill="none" '
        f'stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}" '
        f'stroke-linecap="round" transform="rotate(-90 70 70)" />'
        f'<text x="70" y="66" text-anchor="middle" font-size="26" font-weight="800" fill="#1e293b">{pct_text}</text>'
        f'<text x="70" y="86" text-anchor="middle" font-size="11" font-weight="600" fill="#64748b">approval chance</text>'
        f'</svg>'
    )


pipeline = load_pipeline()


with st.form("loan_form"):

    st.markdown("### Personal Information")
    col1, col2 , col3= st.columns(3)

    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        person_gender = st.selectbox("Gender", ["male", "female"])
        person_education = st.selectbox(
            "Education Level",
            ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
        )

    with col2:
        person_income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
        person_emp_exp = st.number_input("Years of Work Experience", min_value=0, max_value=60, value=5)
        
        person_home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT", "MORTGAGE", "OWN", "OTHER"]
        )
    with col3:
        cb_person_cred_hist_length = st.number_input(
                    "Credit History Length (Years)", min_value=0, value=5
                )
        credit_score = st.number_input(
                    "Credit Score", min_value=300, max_value=850, value=650
                )
        previous_loan_defaults_on_file = st.selectbox(
                    "Previous Loan Defaults", ["No", "Yes"]
                )

    st.markdown("### Loan Information")
    col1, col2 = st.columns(2)
    with col1:
        loan_amnt = st.number_input(
            "Loan Amount Requested ($)", min_value=500, value=10000, step=500)
        loan_intent = st.selectbox(
            "Loan Purpose",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
             "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )

    with col2:
        loan_int_rate = st.number_input(
            "Interest Rate (%)", min_value=0.0, max_value=40.0, value=11.0, step=0.1)
       
    submitted = st.form_submit_button("Predict Loan Status")

if submitted:
    raw_input = pd.DataFrame([{
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_amnt / person_income if person_income > 0 else 0,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }])

    input_engineered = add_engineered_features(raw_input)
    prediction = pipeline.predict(input_engineered)[0]
    probability = pipeline.predict_proba(input_engineered)[0][1]

    st.divider()

    if prediction == 1:
        st.markdown(
            f'<div class="result approved">✅ Loan Approved<br>'
            f'<small>Approval Probability: {probability:.1%}</small></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result rejected">❌ Loan Rejected<br>'
            f'<small>Approval Probability: {probability:.1%}</small></div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="gauge-wrap">' + build_gauge_svg(probability, prediction) + '</div>',
        unsafe_allow_html=True
    )

    st.caption("This is an automated estimate and not a final banking decision.")
