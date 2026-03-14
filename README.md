# 📊 보험 앱 사용자 행동 데이터 기반 고객 세분화 및 이탈 예측
> **통계/데이터사이언스 최종 과제** > 통신사 구독 데이터를 보험 도메인으로 재해석하여, 고객 세분화(Clustering)부터 이탈 예측 모델링(Predictive Modeling), 그리고 데이터 기반 리텐션 전략 기획까지 수행한 데이터 분석 프로젝트입니다.

<br/>

## 프로젝트 개요 
- **배경:** 디지털 보험 플랫폼의 경쟁 심화로 고객 이탈(Churn) 방지가 핵심 비즈니스 과제로 대두됨. 실제 보험사 로그 데이터의 대외비 특성을 고려하여, 구조가 동일한 **통신사 구독 데이터(IBM Telco Churn)를 보험 도메인으로 매핑**하여 분석을 수행함.
- **목표:** 1. 행동 패턴 기반의 고객 세분화를 통한 고위험 핵심 타겟 도출
  2. 선형/비선형 머신러닝 모델을 활용한 이탈 요인 파악
  3. SHAP 기법을 적용한 초개인화 타겟팅 근거 마련
  4. 데이터 기반의 실무적 맞춤형 리텐션 전략 도출

<br/>

## 주요 비즈니스 임팩트
분석 결과를 바탕으로 제안한 **3대 리텐션 마케팅 전략**을 실행하여 목표 KPI 달성 시, 다음과 같은 수익 방어 효과를 추산함.
- **예상 유지 고객:** 연간 **573명**의 이탈 추가 방어
- **수익 방어 효과:** 연간 **약 47만 6천 달러 (한화 약 6억 3천만 원)** 규모의 재무적 가치 창출 기대
- **핵심 전략:** 1. 이탈 고위험군(군집 2) 타겟 30일 집중 온보딩 프로그램
  2. 월납 고객 대상 2년 장기 약정 전환 프로모션
  3. 디지털 부가 서비스(온라인 보안 등) 무료 체험을 통한 락인

<br/>

## 기술 스택 
- **Language:** Python 3
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainable AI (XAI):** SHAP
- **Data Visualization:** Matplotlib, Seaborn

<br/>

## 프로젝트 구조
```text
insurance-churn-analysis/
├── data/
│   ├── Telco-Customer-Churn.csv       # 원본 데이터 (Kaggle)
│   └── telco_clustered.csv            # 군집화 완료 데이터 (중간 산출물)
├── models/
│   ├── kmeans_model.pkl               # K-Means 군집화 모델
│   ├── logistic_regression.pkl        # 최종 예측 모델 (LR, L2 Ridge)
│   └── xgboost_model.pkl              # 성능 비교용 비선형 모델 (XGB)
├── reports/figures/                   # 시각화 산출물 (EDA, 모델 평가, SHAP, 기획안 등)
├── src/
│   ├── 01_load_data.py                # 데이터 로드 및 결측치/타입 전처리
│   ├── 02_eda.py                      # 탐색적 데이터 분석 (EDA)
│   ├── 03_clustering.py               # K-Means 고객 세분화 및 프로파일링
│   ├── 04_modeling.py                 # 로지스틱 회귀 모델 학습 및 계수 분석
│   ├── 04b_modeling_xgb.py            # XGBoost 학습 및 SHAP 요인 교차 검증
│   ├── 05_interpretation.py           # 분석 결과 종합 및 기획안/기대효과 시각화
│   ├── load_data_utils.py             # 공통 데이터 로드 유틸리티
│   └── modeling_utils.py              # 공통 특성 공학 및 스플릿 유틸리티
├── README.md
└── requirements.txt