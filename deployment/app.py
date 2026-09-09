import os
import joblib
import pandas as pd
import streamlit as st
from Data_prep import add_engineered_features

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #52796f;
    }
div[data-testid="stForm"] {
    background-color: #84a98c;
    border-radius: 20px;
    border: 2px solid #2f3e46;
    }
input[type="number"] {
    padding-bottom: 20px;
    margin:20px,
    margin-bottom: 10px;
    background-color: #52796f;
    font-weight: bold;  
}

[data-testid="stWidgetLabel"] p {
    font-weight: bold;
    font-size: 15px;
}

div[data-baseweb="select"] > div {
    background-color: #52796f;
    border-color: #4a90d9;
    font-weight: bold;

}

div[data-baseweb="popover"] ul {
    background-color: #52796f;
}
li[role="option"] {
    color: white;
}
li[role="option"]:hover {
    background-color: #354f52;
}
.result {
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    margin-top: 15px;
    font-size: 20px;
    font-weight: 600;
}
.approved {background-color: #ecfdf3; color: #027a48;}
.rejected {background-color: #fef3f2; color: #b42318;}
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    background-color: #52796f;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px;
    font-size: 20px;
    font-weight: bold;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #cad2c5;
    font-weight: bold;
    color: white;
    border: 2px solid #52796f;
}

</style>
""", unsafe_allow_html=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "loan_pipeline.pkl")

@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)

pipeline = load_pipeline()

st.title("Loan Approval Predictor")

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
            f'<div class="result approved">Loan Approved<br>'
            f'<small>Approval Probability: {probability:.1%}</small></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result rejected">Loan Rejected<br>'
            f'<small>Approval Probability: {probability:.1%}</small></div>',
            unsafe_allow_html=True
        )

    st.progress(float(probability))
    st.caption(
        "This prediction is generated by a machine learning model "
        "and is not a final banking decision."
    )
