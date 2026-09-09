# loan-approval-prediction
# 💳 Loan Approval Prediction

An end-to-end Machine Learning project that predicts whether a bank loan application will be
**Approved** or **Rejected**, based on the applicant's financial and credit information — from
exploratory data analysis and feature engineering, through training and comparing multiple
models, to deploying the final model as an interactive web app.

## 📊 Dataset Overview

The dataset contains **45,000 rows**, with information about each loan applicant:
age, gender, education level, annual income, years of employment experience, home ownership,
requested loan amount, loan intent, interest rate, loan-to-income ratio, credit history length,
credit score, and whether the applicant has previous loan defaults on file.

**Target:** `loan_status` — `1` = Approved, `0` = Rejected

## 🗂️ Project Structure

```
loan-approval-prediction/
├── Project_loan.ipynb          # Exploratory Data Analysis (EDA) + Feature Engineering + Feature Selection
├── Modeling_Evaluation.ipynb   # Imbalance handling + training & comparing 9 models + final evaluation
├── data_prep.py                # Shared module for feature engineering and preprocessing pipelines
│                                #   (used identically at both training and inference time)
├── outputs/                    # Plots and results (correlation heatmap, class imbalance, ROC curves, confusion matrix...)
└── deployment/
    ├── app.py                  # Streamlit app for interactive predictions
    ├── loan_pipeline.pkl       # Final trained pipeline (preprocessing + model)
    ├── data_prep.py            # Copy of the feature engineering module (required at inference time)
    ├── requirements.txt
    └── README.md
```
## 🚀 How to Run

```bash
git clone https://github.com/Sahar-Elgebaly/loan-approval-prediction.git
cd loan-approval-prediction/deployment
pip install -r requirements.txt
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

## 🛠️ Tech Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · imbalanced-learn (SMOTE) ·
Matplotlib / Seaborn · Streamlit

## ⚠️ Disclaimer

Predictions are estimates from a machine learning model trained on historical data, not a final
credit decision — human review is required before any real-world action.