# 🌐 NPL Prediction API

API pentru predicția Non-Performing Loans (NPL), bazat pe un model de tip ensemble stacking (Logistic Regression, Random Forest, XGBoost).

# 📁 Structura Proiectului
```
├── api_npl.py                  # Server FastAPI pentru predicții NPL
├── stacking_ensemble.pkl       # Model ensemble antrenat
├── preprocessing_artifacts.pkl # Pipeline de preprocesare
├── ensemble.py                 # Logica de combinare a modelelor
├── pipelines.py                # Transformeri și feature engineering
├── prepare_data.py             # Pregătirea datelor
└── README.md                   # Acest fișier
```
# 🔧 Instalare Dependențe
```bash
pip install fastapi uvicorn joblib scikit-learn xgboost pandas numpy
```


(opțional, pentru interfața web /docs)
```
pip install fastapi[all]
```
## 🏃 Rulare API

Pornește serverul local:
```
uvicorn api_npl:app --reload
```

Serverul va fi disponibil la:

Swagger UI → http://127.0.0.1:8000/docs

Redoc UI → http://127.0.0.1:8000/redoc

## 📤 Endpoint Principal
POST /predict_npl

Primește un JSON cu detaliile unui credit și returnează:

- probabilitatea NPL

- clasa prezisă (0 = bun, 1 = neperformant)

- categoria de risc

- decizia recomandată

## Exemplu input:
```
{
    "loan_amount_mdl": 50000,
    "monthly_income_mdl": 12000,
    "interest_rate_annual": 12.5,
    "credit_bureau_score": 680,
    "pti": 0.18,
    "dti": 0.28,
    "tenor_months": 36,
    "segment": "Retail",
    "urban": 1,
    "salary_project": 1,
    "num_loans_bureau": 2,
    "previous_dpd_30": 0,
    "previous_dpd_60": 0
}
```

## Exemplu output:
```
{
    "probabilitate_npl": 0.1845,
    "predictie_clasa": 0,
    "categorie_risc": "Scăzut",
    "decizie_recomandata": "APROBAT - Risc minim",
    "threshold_folosit": 0.35,
    "predictii_individuale": {
        "LogReg": 0.201,
        "RF": 0.172,
        "XGB": 0.180
    }
}
```
## 🧠 Categorii de Risc și Decizii
Interval Probabilitate	Categorie	Decizie Recomandată
- < 0.25	Scăzut	✅ Aprobat - Risc minim
- < 0.40	Mediu-Scăzut	✅ Aprobat - Monitorizare standard
- < 0.55	Mediu	⚠️ Aprobat condiționat
- < 0.70	Ridicat	⚠️ Revizuire necesară
- ≥ 0.70	Foarte Ridicat	❌ Respins - Risc prea mare
## 🧾 Testare Rapidă
🔹 Cu curl
```
curl -X POST "http://127.0.0.1:8000/predict_npl" ^
-H "Content-Type: application/json" ^
-d "{\"loan_amount_mdl\":50000, \"monthly_income_mdl\":12000, \"interest_rate_annual\":12.5, \"credit_bureau_score\":680, \"pti\":0.18, \"dti\":0.28, \"tenor_months\":36}"
```
🔹 Cu Postman

1. Deschide Postman

2. Alege metoda POST

3. Adaugă URL-ul: http://127.0.0.1:8000/predict_npl

4. În tabul Body → selectează raw → JSON

5. Inserează exemplul de input de mai sus

6. Apasă Send

# 📦 Structura Finală
```
npl_api/
├── api_npl.py
├── stacking_ensemble.pkl
├── preprocessing_artifacts.pkl
├── pipelines.py
├── ensemble.py
└── README.md
```
# 🎯 Status

- ✅ API funcțional local
- ✅ Integrare completă cu modelele antrenate
- ✅ Compatibil cu Postman și Swagger UI
