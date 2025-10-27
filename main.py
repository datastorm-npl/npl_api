from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pipelines import DateAndDomainFeatures
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="NPL Prediction API",
    description="API care prezice probabilitatea de Non-Performing Loan pentru un client nou",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading preprocessing artifacts & ensemble model...")

artifacts = joblib.load("preprocessing_artifacts.pkl")
transformer = artifacts['feature_transformer']

ensemble = joblib.load("stacking_ensemble.pkl")
base_models = ensemble['base_models']
meta_model = ensemble['meta_model']
threshold = ensemble['threshold']
model_names = ensemble['model_names']

print("Models loaded successfully!")

class CreditRequest(BaseModel):
    origination_date: str
    maturity_date: str
    loan_amount_mdl: float
    monthly_income_mdl: float
    interest_rate_annual: float
    credit_bureau_score: float
    pti: float
    dti: float
    tenor_months: int
    loan_amount_ccy: float
    currency: str
    segment: str
    region: str
    urban: int
    loan_type: str
    collateral_type: str
    ltv: float
    salary_project: int
    employer_sector: str
    remittances_share_income: float
    guarantor_present: int
    num_loans_bureau: int
    previous_dpd_30: int
    previous_dpd_60: int
    installment_mdl: float
    fx_mismatch: int
    inflation_at_origination: float
    policy_rate_at_origination: float
    shock_2022_2023: int

@app.post("/predict_npl")
def predict_npl(request: CreditRequest):
    df = pd.DataFrame([request.dict()])

    df_transformed = transformer.transform(df)

    predictii_individuale = {}
    meta_features = []

    for model, name in zip(base_models, model_names):
        proba = model.predict_proba(df_transformed)[:, 1][0]
        predictii_individuale[name] = float(proba)
        meta_features.append(proba)

    X_meta = np.column_stack(meta_features).reshape(1, -1)
    proba_npl = float(meta_model.predict_proba(X_meta)[:, 1][0])
    predictie_clasa = int(proba_npl >= threshold)

    if proba_npl < 0.25:
        categorie = "Scăzut"
        decizie = "APROBAT - Risc minim"
    elif proba_npl < 0.40:
        categorie = "Mediu-Scăzut"
        decizie = "APROBAT - Monitorizare standard"
    elif proba_npl < 0.55:
        categorie = "Mediu"
        decizie = "APROBAT CONDIȚIONAT - Necesită garanții suplimentare"
    elif proba_npl < 0.70:
        categorie = "Ridicat"
        decizie = "REVIZUIRE - Reducere sumă sau refuz"
    else:
        categorie = "Foarte Ridicat"
        decizie = "RESPINS - Risc prea mare"

    rezultat = {
        'probabilitate_npl': proba_npl,
        'predictie_clasa': predictie_clasa,
        'categorie_risc': categorie,
        'decizie_recomandata': decizie,
        'threshold_folosit': threshold,
        'predictii_individuale': predictii_individuale
    }

    return rezultat

@app.get("/")
def root():
    return {"message": "NPL Prediction API - Ready to use!"}
