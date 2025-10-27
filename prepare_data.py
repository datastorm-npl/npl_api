# prepare_data.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import joblib
from pipelines import DateAndDomainFeatures
import warnings
warnings.filterwarnings('ignore')


def validate_data(df, stage="Initial"):
    print(f"\n{'='*60}")
    print(f"VALIDARE DATE - {stage}")
    print(f"{'='*60}")
    
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values:")
        for col, count in missing[missing > 0].items():
            pct = 100 * count / len(df)
            print(f"   {col:30s}: {count:5d} ({pct:5.2f}%)")
    else:
        print("No missing values")
    
    dups = df.duplicated().sum()
    if dups > 0:
        print(f"Duplicate rows: {dups}")
    else:
        print("No duplicates")
    
    print(f"\nNumeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")


def check_leakage(X, y, feature_names=None):
    print(f"\n{'='*60}")
    print("LEAKAGE CHECK - Top correlații cu target")
    print(f"{'='*60}")
    
    if feature_names is None:
        feature_names = X.columns if hasattr(X, 'columns') else [f"feat_{i}" for i in range(X.shape[1])]
    
    if hasattr(X, 'columns'):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            corr = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
            print("\nTop 10 features cu cea mai mare corelație:")
            for i, (feat, val) in enumerate(corr.head(10).items(), 1):
                warning = " SUSPICION!" if val > 0.7 else ""
                print(f"   {i:2d}. {feat:30s}: {val:.4f}{warning}")
            
            if corr.max() > 0.7:
                print("\nWARNING: Corelații > 0.7 detectate! Verifică pentru leakage!")


def prepare_data(file_path, sheet_name="NPL_Data", test_size=0.2, random_state=42):

    print("PREGĂTIRE DATE PENTRU ENSEMBLE")
    print("="*60)
    
    print(f"\nÎncărcare date din {file_path}...")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    validate_data(df, "Initial")
    
    print(f"\nAplicare feature engineering...")
    reference_date = df['origination_date'].max()
    print(f"   Reference date: {reference_date}")
    
    transformer = DateAndDomainFeatures(reference_date=reference_date)
    df_transformed = transformer.fit_transform(df)
    validate_data(df_transformed, "After Feature Engineering")
    
    print(f"\nSplit train/test (test_size={test_size}, stratified)...")
    target = 'npl_target'
    X = df_transformed.drop(columns=[target])
    y = df_transformed[target]
    
    check_leakage(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    print(f"\nRezultate split:")
    print(f"   Train: {X_train.shape}, NPL rate: {y_train.mean():.2%}")
    print(f"   Test:  {X_test.shape}, NPL rate: {y_test.mean():.2%}")
    
    train_npl_rate = y_train.mean()
    test_npl_rate = y_test.mean()
    diff = abs(train_npl_rate - test_npl_rate)
    if diff > 0.02:
        print(f"   WARNING: NPL rate difference > 2%: {diff:.2%}")
    else:
        print(f"   NPL rate difference OK: {diff:.2%}")
    
    print(f"\nFeature info:")
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns
    print(f"   Numeric features: {len(num_cols)}")
    print(f"   Categorical features: {len(cat_cols)}")
    
    if len(cat_cols) > 0:
        print(f"\n   Categorical features:")
        for col in cat_cols:
            n_unique = X_train[col].nunique()
            print(f"      {col:25s}: {n_unique:3d} unique values")
    
    print(f"\nSalvare artifacts...")
    
    joblib.dump((X_train, X_test, y_train, y_test), "train_test.pkl")
    print("   train_test.pkl")
    
    joblib.dump({
        'feature_transformer': transformer,
        'feature_names': list(X_train.columns),
        'categorical_features': list(cat_cols),
        'numeric_features': list(num_cols),
        'target_name': target,
        'reference_date': reference_date,
        'train_npl_rate': train_npl_rate,
        'test_npl_rate': test_npl_rate
    }, "preprocessing_artifacts.pkl")
    print("   preprocessing_artifacts.pkl")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'transformer': transformer,
        'feature_names': list(X_train.columns),
        'cat_cols': list(cat_cols),
        'num_cols': list(num_cols)
    }


if __name__ == "__main__":
    data = prepare_data(
        file_path="moldova_npl.xlsx",
        sheet_name="NPL_Data",
        test_size=0.2,
        random_state=42
    )
    
    print("\n" + "="*60)
    print("PREGĂTIRE COMPLETĂ!")
    print("="*60)
    print("\nFișiere generate:")
    print("   train_test.pkl    - Date pentru training")
    print("   preprocessing_artifacts.pkl     - Artifacts pentru producție")
    print("\nNext steps:")
    print("   1. python optimize_logistic.py")
    print("   2. python optimize_rf.py")
    print("   3. python optimize_xgb.py")
    print("   4. python ensemble.py")
