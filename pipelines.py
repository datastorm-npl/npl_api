# pipelines.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer


class DateAndDomainFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, date_cols=('origination_date', 'maturity_date'), 
                 reference_date='2025-09-30'):
        self.date_cols = date_cols
        self.reference_date = pd.to_datetime(reference_date)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        X[self.date_cols[0]] = pd.to_datetime(X[self.date_cols[0]], errors='coerce')
        X[self.date_cols[1]] = pd.to_datetime(X[self.date_cols[1]], errors='coerce')

        X['year_origination'] = X[self.date_cols[0]].dt.year
        X['month_origination'] = X[self.date_cols[0]].dt.month
        X['quarter_origination'] = X[self.date_cols[0]].dt.quarter
        
        X['year_maturity'] = X[self.date_cols[1]].dt.year
        X['month_maturity'] = X[self.date_cols[1]].dt.month

        day = X[self.date_cols[0]].dt.day.fillna(15)
        X['day_orig_sin'] = np.sin(2 * np.pi * day / 31)
        X['day_orig_cos'] = np.cos(2 * np.pi * day / 31)

        month = X[self.date_cols[0]].dt.month
        X['month_orig_sin'] = np.sin(2 * np.pi * month / 12)
        X['month_orig_cos'] = np.cos(2 * np.pi * month / 12)

        X['tenor_days'] = (X[self.date_cols[1]] - X[self.date_cols[0]]).dt.days
        
        X['loan_age_days'] = (self.reference_date - X[self.date_cols[0]]).dt.days
        
        X['days_to_maturity'] = (X[self.date_cols[1]] - self.reference_date).dt.days
        X['is_matured'] = (X['days_to_maturity'] < 0).astype(int)

        if 'loan_amount_mdl' in X.columns and 'monthly_income_mdl' in X.columns:
            X['loan_to_income'] = X['loan_amount_mdl'] / (X['monthly_income_mdl'] + 1e-9)
            X['income_to_installment'] = X['monthly_income_mdl'] / (X['installment_mdl'] + 1e-9)
        
        if 'pti' in X.columns and 'dti' in X.columns:
            X['total_debt_ratio'] = X['pti'] + X['dti']
            X['high_debt_burden'] = ((X['dti'] > 0.4) | (X['pti'] > 0.3)).astype(int)
        
        if 'ltv' in X.columns:
            X['high_ltv'] = (X['ltv'] > 0.8).astype(int)

        if 'credit_bureau_score' in X.columns:
            X['low_credit_score'] = (X['credit_bureau_score'] < 600).astype(int)
            X['high_credit_score'] = (X['credit_bureau_score'] > 700).astype(int)
        
        if 'interest_rate_annual' in X.columns:
            X['high_interest'] = (X['interest_rate_annual'] > 15).astype(int)

        cols_to_drop = ['loan_id', 'borrower_id', self.date_cols[0], self.date_cols[1]]
        for col in cols_to_drop:
            if col in X.columns:
                X.drop(columns=col, inplace=True)
        
        return X


def build_column_lists(df, target='npl_target'):
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if target in num_cols:
        num_cols.remove(target)
    if target in cat_cols:
        cat_cols.remove(target)
    
    return num_cols, cat_cols


def preproc_logistic(df, target='npl_target'):
    num_cols, cat_cols = build_column_lists(df, target)
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='drop'
    )


def preproc_rf(df, target='npl_target'):

    num_cols, cat_cols = build_column_lists(df, target)
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            min_frequency=0.01,
            max_categories=50
        ))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='drop'
    )


def preproc_xgb(df, target='npl_target'):
    return preproc_rf(df, target)

