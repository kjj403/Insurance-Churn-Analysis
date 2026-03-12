# load_data_utils.py
# 공통 데이터 로드 함수 — 모든 src 파일에서 import해서 사용

import pandas as pd

def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)
    df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)
    return df