import numpy as np
import pandas as pd
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

selected_numeric_engineered = ['loan_burden', 'total_interest_cost', 'income_per_exp_year']

log_cols = [
    "person_income",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
] + selected_numeric_engineered 

passthrough_cols = ["loan_int_rate", "credit_score"]

categorical_cols = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]
 
 
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['loan_burden'] = (df['loan_percent_income'] * df['loan_int_rate'])
    df['total_interest_cost'] = df['loan_amnt'] * (df['loan_int_rate'] / 100)
    df['income_per_exp_year'] = df['person_income'] / (df['person_emp_exp'] + 1)
    return df
 
 
def build_preprocessors():
    linear_preprocessor = ColumnTransformer(transformers=[
    ("age", FunctionTransformer(partial(np.clip, a_min=None, a_max=90)), ["person_age"]),
    ("emp_exp", Pipeline([
        ("cap", FunctionTransformer(partial(np.clip, a_min=None, a_max=60))),
        ("log", FunctionTransformer(np.log1p)),
    ]), ["person_emp_exp"]),
    ("log", FunctionTransformer(np.log1p), log_cols),
    ("scale_pass", StandardScaler(), passthrough_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ])
 
    tree_preprocessor = ColumnTransformer(transformers=[
    ("age", FunctionTransformer(partial(np.clip, a_min=None, a_max=90)), ["person_age"]),
    ("emp_exp", FunctionTransformer(partial(np.clip, a_min=None, a_max=60)), ["person_emp_exp"]),
    ("passthrough_num", "passthrough", log_cols + passthrough_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ])
 
    return linear_preprocessor, tree_preprocessor
 
 
def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = add_engineered_features(df)
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]
    return X, y
 
