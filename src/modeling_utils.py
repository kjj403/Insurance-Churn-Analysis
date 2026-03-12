# modeling_utils.py
# 공통 feature engineering 함수 — 04_modeling.py, 04b_modeling_xgb.py에서 공유

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def build_features(df: pd.DataFrame):
    """
    Encode all categorical variables for modeling.
    Returns feature matrix X (DataFrame) and target y (Series).
    """
    df = df.copy()

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                   'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in binary_cols:
        df[col + '_enc'] = (df[col] == 'Yes').astype(int)

    df['gender_enc'] = (df['gender'] == 'Male').astype(int)

    contract_dummies = pd.get_dummies(df['Contract'],        prefix='Contract',  drop_first=True)
    internet_dummies = pd.get_dummies(df['InternetService'], prefix='Internet',  drop_first=True)
    payment_dummies  = pd.get_dummies(df['PaymentMethod'],   prefix='Payment',   drop_first=True)

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    enc_cols = [c for c in df.columns if c.endswith('_enc')]

    X = pd.concat([df[num_cols], df[enc_cols],
                   contract_dummies, internet_dummies, payment_dummies], axis=1)
    y = df['Churn_binary']
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split."""
    # numpy array와 DataFrame 모두 처리
    if hasattr(X, 'values'):
        X_arr = X.values
    else:
        X_arr = X

    y_arr = y.values if hasattr(y, 'values') else y

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )
    print(f"  Train: {len(X_train):,} samples  |  Churn rate: {y_train.mean():.1%}")
    print(f"  Test : {len(X_test):,} samples  |  Churn rate: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test