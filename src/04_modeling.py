# =============================================================================
# 04_modeling.py
# 로지스틱 회귀를 활용한 이탈 예측 모델링
#
# [분석 목적 및 구성 이유]
# 어떤 고객이 이탈할지 예측하기 위해 로지스틱 회귀 모델을 구축한다.
# 예측 성능만 본다면 앙상블 모델(XGBoost, Random Forest 등)이 유리할 수 있으나,
# 서비스 기획 직무의 특성상 "누가 이탈하는가"보다 "왜 이탈하는가"를 설명하는 것이 중요하다.
# 따라서 회귀 계수(Log-Odds)를 통해 각 변수의 영향력을 정확한 수치로 해석할 수 있는
# 로지스틱 회귀를 최종 모델로 선정하였다.
#
# [주요 분석 단계]
#   1. 변수 전처리: 범주형 변수의 수치화 (원핫 인코딩 등)
#   2. 데이터 분할: 이탈률 비율을 유지하는 계층적 분할 적용
#   3. 모델 학습: 기본 모델과 과적합 방지를 위한 L2 규제(Ridge) 모델 비교
#   4. 성능 평가: 정확도, 정밀도, 재현율, F1 점수, AUC-ROC 확인
#   5. 결과 해석: 회귀 계수를 기반으로 한 이탈 위험 요인 도출 및 기획안 연결
#   6. 군집별 위험도: 앞선 군집화 결과와 예측 모델의 일치성 검증
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import joblib

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, roc_curve,
                                     confusion_matrix, classification_report)
from sklearn.pipeline        import Pipeline

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
sns.set_style('whitegrid')

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'Telco-Customer-Churn.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

sys.path.append(os.path.join(BASE_DIR, 'src'))
from load_data_utils import load_data


# =============================================================================
# STEP 1. 군집화 및 예측을 위한 변수 전처리 (Feature Engineering)
# =============================================================================

def build_features(df: pd.DataFrame):
    """
    모든 범주형 변수를 모델이 학습할 수 있는 수치형 형태로 변환한다.

    - 이진 범주형(Yes/No): 1과 0으로 직관적으로 매핑한다.
    - 다중 범주형(납입 주기 등): 원핫 인코딩(One-Hot Encoding)을 적용하되,
      다중공선성(더미 변수 함정)을 방지하기 위해 첫 번째 범주를 제거(drop_first=True)한다.
    - 고객 ID와 문자열 타겟 변수는 학습에서 제외한다.
    """
    df = df.copy()

    # 이진 범주형 변수 처리 (Yes=1, No=0)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                   'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in binary_cols:
        # '인터넷 서비스 없음' 등의 값도 논리상 'No(0)'로 일괄 처리
        df[col + '_enc'] = (df[col] == 'Yes').astype(int)

    # 성별 처리 (Male=1, Female=0)
    df['gender_enc'] = (df['gender'] == 'Male').astype(int)

    # 다중 범주형 변수 원핫 인코딩 (drop_first=True 적용)
    contract_dummies  = pd.get_dummies(df['Contract'],        prefix='Contract',  drop_first=True)
    internet_dummies  = pd.get_dummies(df['InternetService'], prefix='Internet',  drop_first=True)
    payment_dummies   = pd.get_dummies(df['PaymentMethod'],   prefix='Payment',   drop_first=True)

    # 최종 특성 행렬(Feature Matrix) 구성
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    enc_cols = [c for c in df.columns if c.endswith('_enc')]

    X = pd.concat([df[num_cols], df[enc_cols],
                   contract_dummies, internet_dummies, payment_dummies], axis=1)
    y = df['Churn_binary']

    return X, y


# =============================================================================
# STEP 2. 학습 및 테스트 데이터 분할
# =============================================================================

def split_data(X, y, test_size=0.2, random_state=42):
    """
    전체 데이터를 학습용(80%)과 테스트용(20%)으로 분할한다.

    이탈 고객의 비율이 전체의 약 26.6%로 불균형하므로, 무작위로 분할할 경우
    학습/테스트 세트 간 이탈률 차이가 발생하여 평가가 왜곡될 수 있다.
    이를 방지하기 위해 계층적 분할(stratify=y)을 적용하여 원본 데이터의
    클래스 비율을 동일하게 유지한다.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  학습 데이터: {len(X_train):,}명  |  이탈률: {y_train.mean():.1%}")
    print(f"  평가 데이터: {len(X_test):,}명  |  이탈률: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 3. 예측 모델 학습
# =============================================================================

def train_models(X_train, y_train):
    """
    성능 비교를 위해 기본 로지스틱 회귀 모델과 L2 규제가 적용된 모델을 각각 학습한다.

    - 모델 A (기본): 훈련 데이터에 최대한 맞추어 학습한다. 상관관계가 높은 변수들로 인해
      과적합(Overfitting)될 위험이 존재한다.
    - 모델 B (L2 규제 - Ridge): 회귀 계수가 너무 커지는 것을 페널티(C=0.1)로 제어하여,
      새로운 데이터에 대한 일반화 성능을 높인다. 변수의 영향력을 모두 살리면서 크기만
      조절하는 L2 규제가 본 분석 목적에 더 부합한다.
    - 클래스 가중치(class_weight='balanced'): 소수 클래스인 이탈 고객(26.6%)의
      예측 오류에 더 큰 가중치를 부여하여 클래스 불균형 문제를 해결한다.
    """
    # 모델 A: 규제 없는 기본 모델 (C를 무한대에 가깝게 설정)
    pipe_baseline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1e9, class_weight='balanced',
                                   max_iter=1000, random_state=42))
    ])

    # 모델 B: L2 규제 적용 모델 (메인 모델)
    pipe_ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, penalty='l2', class_weight='balanced',
                                   max_iter=1000, random_state=42))
    ])

    pipe_baseline.fit(X_train, y_train)
    pipe_ridge.fit(X_train, y_train)

    print("  모델 A (기본 로지스틱 회귀): 학습 완료")
    print("  모델 B (L2 규제 적용 모델) : 학습 완료")

    return pipe_baseline, pipe_ridge


# =============================================================================
# STEP 4. 모델 성능 평가
# =============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str):
    """
    다양한 평가지표를 활용하여 모델의 예측 성능을 다각도로 검증한다.

    데이터 불균형 상태에서는 모든 고객이 이탈하지 않는다고 예측해도 정확도(Accuracy)가 
    높게 나오는 함정이 존재한다. 따라서 '실제 이탈자를 얼마나 잘 찾아냈는가'를 의미하는
    재현율(Recall)과, 분류 임계값에 구애받지 않는 전반적인 성능 지표인 AUC-ROC를
    핵심 평가지표로 활용한다. 또한 5-Fold 교차 검증을 통해 모델의 안정성을 확인한다.
    """
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        '정확도 (Accuracy)' : accuracy_score(y_test, y_pred),
        '정밀도 (Precision)': precision_score(y_test, y_pred),
        '재현율 (Recall)'   : recall_score(y_test, y_pred),
        'F1 점수 (F1)'      : f1_score(y_test, y_pred),
        'AUC-ROC'           : roc_auc_score(y_test, y_pred_prob),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_train, y_train,
                              cv=cv, scoring='roc_auc')

    print(f"\n  [{model_name}]")
    for k, v in metrics.items():
        print(f"    {k:<15}: {v:.4f}")
    print(f"    교차검증 AUC (5-Fold): {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    return metrics, y_pred, y_pred_prob


# =============================================================================
# STEP 5. 결과 시각화 및 해석
# =============================================================================

def plot_evaluation(y_test, results: dict, X_test):
    """
    혼동 행렬(Confusion Matrix)과 ROC 곡선을 시각화하여 모델 간 성능을 비교한다.
    """
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    colors = {'모델 A (기본)': '#FF9800', '모델 B (L2 규제)': '#2196F3'}

    # 혼동 행렬 시각화
    for i, (name, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, i])
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Stay', 'Churn'],
                    yticklabels=['Stay', 'Churn'],
                    annot_kws={'size': 12})
        # 타이틀 영문 유지
        ax.set_title(f'Confusion Matrix\n{name}\nAUC={res["metrics"]["AUC-ROC"]:.4f}',
                     fontsize=10)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    # ROC 곡선 시각화
    ax_roc = fig.add_subplot(gs[0, 2])
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.50)')

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_pred_prob'])
        auc = res['metrics']['AUC-ROC']
        ax_roc.plot(fpr, tpr, linewidth=2,
                    color=colors[name], label=f'{name} (AUC={auc:.4f})')

    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate (Recall)')
    ax_roc.set_title('ROC Curve Comparison\n(Higher AUC = Better Discrimination)')
    ax_roc.legend(fontsize=8)
    ax_roc.spines['top'].set_visible(False)
    ax_roc.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '09_model_evaluation.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


def plot_coefficients(model, feature_names: list, top_n: int = 20):
    """
    로지스틱 회귀 모델의 계수(Coefficient)를 추출하여 이탈에 영향을 미치는 핵심 요인을 파악한다.

    본 연구의 핵심 목적은 분석 결과를 서비스 기획으로 연결하는 것이다.
    양(+)의 계수는 이탈 위험을 높이는 요인이므로 개선의 대상이 되고,
    음(-)의 계수는 이탈을 방어하는 요인이므로 혜택을 통해 강화해야 할 대상이 된다.
    표준화된 데이터를 사용했으므로 계수의 절대 크기로 영향력의 우선순위를 정할 수 있다.
    """
    clf    = model.named_steps['clf']
    coefs  = clf.coef_[0]

    coef_df = pd.DataFrame({
        'Feature'    : feature_names,
        'Coefficient': coefs
    }).sort_values('Coefficient', key=abs, ascending=False).head(top_n)

    coef_df = coef_df.sort_values('Coefficient')

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#F44336' if c > 0 else '#2196F3' for c in coef_df['Coefficient']]
    bars   = ax.barh(coef_df['Feature'], coef_df['Coefficient'],
                     color=colors, edgecolor='white', linewidth=0.8)

    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Coefficient (Log-Odds)\nRed = Increases Churn Risk  |  Blue = Decreases Churn Risk')
    ax.set_title(f'Top {top_n} Features by Logistic Regression Coefficient\n'
                 f'(Model B — L2 Regularized, Standardized Features)',
                 fontsize=12, pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '10_coefficients.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")

    return coef_df


def plot_cluster_churn_risk(df_clustered: pd.DataFrame, model, X, feature_names):
    """
    앞선 단계에서 나눈 군집별로 모델이 예측한 평균 이탈 확률을 산출하여 실제 이탈률과 비교한다.

    고객 세분화(군집화) 결과와 예측 모델링 결과를 결합하여 비즈니스 정합성을 검증한다.
    실제 이탈률이 높은 고위험 군집에서 예측 확률도 동일하게 높게 나타난다면,
    해당 군집을 타겟으로 한 서비스 기획안의 논리적 타당성이 확보된다.
    """
    df_clustered = df_clustered.copy()

    X_full, y_full = build_features(df_clustered)
    X_full = X_full[feature_names]

    df_clustered['churn_prob'] = model.predict_proba(X_full)[:, 1]

    cluster_risk = df_clustered.groupby('Cluster').agg(
        actual_churn_rate =('Churn_binary', 'mean'),
        predicted_churn_prob=('churn_prob', 'mean'),
        count=('Cluster', 'count')
    ).round(3)
    cluster_risk['actual_pct']    = (cluster_risk['actual_churn_rate'] * 100).round(1)
    cluster_risk['predicted_pct'] = (cluster_risk['predicted_churn_prob'] * 100).round(1)

    print("\n[ 군집별 실제 이탈률 vs 예측 이탈 확률 검증 ]")
    print(cluster_risk[['count', 'actual_pct', 'predicted_pct']].to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    x       = np.arange(len(cluster_risk))
    width   = 0.35

    bars1 = ax.bar(x - width/2, cluster_risk['actual_pct'],    width,
                   label='Actual Churn Rate (%)',    color='#F44336', alpha=0.8)
    bars2 = ax.bar(x + width/2, cluster_risk['predicted_pct'], width,
                   label='Predicted Churn Prob (%)', color='#2196F3', alpha=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Cluster {i}' for i in cluster_risk.index])
    ax.set_ylabel('Churn Rate / Predicted Probability (%)')
    ax.set_title('Actual vs Predicted Churn by Cluster\n'
                 '(Validates model consistency with clustering results)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '11_cluster_churn_risk.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


# =============================================================================
# STEP 6. 분석 요약 결과 출력
# =============================================================================

def print_model_summary(results: dict, coef_df: pd.DataFrame):
    """터미널에 최종 모델 성능 및 회귀 계수 해석 결과를 국문으로 요약 출력한다."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("[ 모델 성능 최종 요약 ]")
    print(sep)

    for name, res in results.items():
        m = res['metrics']
        print(f"\n  {name}")
        print(f"    정확도(Accuracy) : {m['정확도 (Accuracy)']:.4f}")
        print(f"    정밀도(Precision): {m['정밀도 (Precision)']:.4f} (이탈 예측 고객 중 실제 이탈 비율)")
        print(f"    재현율(Recall)   : {m['재현율 (Recall)']:.4f} (실제 이탈 고객 중 모델이 잡아낸 비율)")
        print(f"    F1 점수          : {m['F1 점수 (F1)']:.4f}")
        print(f"    AUC-ROC          : {m['AUC-ROC']:.4f}")

    print(f"\n{sep}")
    print("[ 핵심 이탈 위험 요인 (모델 B 회귀 계수 기준) ]")
    print(sep)

    top_pos = coef_df[coef_df['Coefficient'] > 0].tail(5)[::-1]
    top_neg = coef_df[coef_df['Coefficient'] < 0].head(5)

    print("\n  🔴 이탈 위험을 상승시키는 주요 요인 (해결 과제):")
    for _, row in top_pos.iterrows():
        print(f"    {row['Feature']:<35} 영향력(계수)={row['Coefficient']:+.4f}")

    print("\n  🔵 이탈 위험을 하락시키는 주요 요인 (강화 과제):")
    for _, row in top_neg.iterrows():
        print(f"    {row['Feature']:<35} 영향력(계수)={row['Coefficient']:+.4f}")

    print(f"\n{sep}")
    print("[ 분석 결과 기반 서비스 기획 방향성 제안 ]")
    print(sep)
    print("  1. '2년 약정(Contract_Two year)' 변수가 이탈 방어력이 가장 강함.")
    print("     → 신규 가입 온보딩 시, 장기 계약 전환을 유도하는 혜택/프로모션 전면 배치")
    print("  2. '유지 기간(tenure)'이 짧을수록 이탈 위험이 급증함.")
    print("     → 가입 후 1년 내 고객 이탈을 막는 초기 푸시 알림 및 집중 케어 서비스 필수")
    print("  3. '고액 납입금(TotalCharges)' 및 특정 상품 이용 고객의 이탈률이 높음.")
    print("     → 프리미엄(우수) 고객을 위한 전용 멤버십 가치 제공 필요")
    print("  4. 기술 지원(TechSupport) 및 부가 특약(OnlineSecurity) 이용 시 이탈 방어 효과 발생.")
    print("     → 앱 내 고객센터 접근성 상향 및 부가 보장 서비스 무료 체험 프로모션 기획")


# =============================================================================
# 실행
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  보험 디지털 플랫폼 고객 이탈 분석")
    print("  STEP 4: 이탈 예측 모델링 (로지스틱 회귀)")
    print("=" * 60)

    df = load_data(DATA_PATH)

    print("\n[1] 예측 변수 생성 및 인코딩 중...")
    X, y = build_features(df)
    feature_names = X.columns.tolist()
    print(f"  변환 완료: {X.shape[0]:,}명 고객 × {X.shape[1]}개 특성 변수")

    print("\n[2] 학습 및 평가 데이터 분할 중...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n[3] 모델 학습 중 (기본 모델 vs L2 규제 모델)...")
    model_a, model_b = train_models(X_train, y_train)

    print("\n[4] 모델 성능 평가 진행...")
    metrics_a, y_pred_a, y_prob_a = evaluate_model(
        model_a, X_train, X_test, y_train, y_test, '모델 A (기본)')
    metrics_b, y_pred_b, y_prob_b = evaluate_model(
        model_b, X_train, X_test, y_train, y_test, '모델 B (L2 규제)')

    results = {
        '모델 A (기본)' : {'metrics': metrics_a, 'y_pred': y_pred_a, 'y_pred_prob': y_prob_a},
        '모델 B (L2 규제)': {'metrics': metrics_b, 'y_pred': y_pred_b, 'y_pred_prob': y_prob_b},
    }

    print("\n[5] 평가 지표 및 회귀 계수 시각화 생성...")
    plot_evaluation(y_test, results, X_test)
    coef_df = plot_coefficients(model_b, feature_names)

    clustered_path = os.path.join(BASE_DIR, 'data', 'telco_clustered.csv')
    if os.path.exists(clustered_path):
        print("\n[6] 군집별 예측 위험도 결합 분석...")
        df_clustered = pd.read_csv(clustered_path)
        df_clustered['TotalCharges'] = pd.to_numeric(
            df_clustered['TotalCharges'], errors='coerce')
        df_clustered = df_clustered.dropna(subset=['TotalCharges'])
        plot_cluster_churn_risk(df_clustered, model_b, X, feature_names)

    print_model_summary(results, coef_df)

    model_path = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
    joblib.dump(model_b, model_path)
    print(f"\n  최종 모델 저장 완료: {model_path}")

    print("\n" + "=" * 60)
    print("✅ 04_modeling.py 실행 완료")
    print("   → 다음 단계: python src/05_interpretation.py")
    print("=" * 60)