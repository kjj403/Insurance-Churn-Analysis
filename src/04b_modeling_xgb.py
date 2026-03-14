# =============================================================================
# 04b_modeling_xgb.py
# 비선형 모델(XGBoost) 구축 및 SHAP 분석을 통한 이탈 요인 교차 검증
#
# [분석 목적 및 구성 이유]
# 앞서 로지스틱 회귀(LR)는 변수 간의 관계를 '선형적'으로 가정하였다.
# 하지만 현실의 고객 행동은 "월 납입액이 높을 때, '월납' 고객만 이탈이 급증한다"와 
# 같은 비선형적 상호작용(Interaction)을 포함할 수 있다. 
# 이를 포착하기 위해 트리 기반의 앙상블 모델인 XGBoost를 추가로 학습한다.
# 
# 또한, XGBoost와 같은 블랙박스 모델의 해석력을 보완하기 위해 
# 게임 이론 기반의 SHAP(SHapley Additive exPlanations) 기법을 도입한다.
# 이를 통해 전체 변수의 중요도(Global)뿐만 아니라, 특정 고객 단 한 명의 
# 이탈 원인(Local)까지 시각적으로 도출하여 초개인화된 서비스 기획의 근거를 마련한다.
#
# [주요 분석 단계]
#   1. XGBoost 모델 학습 (클래스 불균형 가중치 적용)
#   2. 로지스틱 회귀 모델과의 성능 비교 (ROC 곡선 및 주요 지표)
#   3. SHAP 전체 변수 중요도 시각화 (막대그래프 및 비즈웜 플롯)
#   4. SHAP Waterfall 시각화 (개별 고객의 이탈 요인 상세 분석)
#   5. 분석 요약 및 서비스 기획 시사점 도출
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, roc_curve)

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
from load_data_utils  import load_data
from modeling_utils   import build_features, split_data


# =============================================================================
# STEP 1. XGBoost 모델 학습
# =============================================================================

def train_xgboost(X_train, y_train):
    """
    하이퍼파라미터 튜닝이 적용된 XGBoost 분류기를 학습한다.

    - max_depth=4: 정형(Tabular) 데이터에서의 과적합을 방지하기 위해 트리의 깊이를 얕게 설정한다.
    - scale_pos_weight: 유지(73%)와 이탈(27%)의 클래스 불균형을 해소하기 위해 
      소수 클래스(이탈)에 약 2.76배의 가중치를 부여한다.
    """
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  클래스 가중치(scale_pos_weight) = {scale_pos:.2f} (불균형 데이터 보정)")

    model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos,
        eval_metric       = 'auc',
        use_label_encoder = False,
        random_state      = 42,
        n_jobs            = -1,
    )
    model.fit(X_train, y_train, verbose=False)
    print(f"  학습 완료: 트리 개수={model.n_estimators}, 최대 깊이={model.max_depth}")
    return model


# =============================================================================
# STEP 2. 모델 평가 및 로지스틱 회귀와의 성능 비교
# =============================================================================

def evaluate_xgb(model, X_train, X_test, y_train, y_test):
    """XGBoost 모델의 예측 성능을 측정하고 5-Fold 교차 검증 AUC를 산출한다."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy' : accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall'   : recall_score(y_test, y_pred),
        'F1'       : f1_score(y_test, y_pred),
        'AUC-ROC'  : roc_auc_score(y_test, y_prob),
    }

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

    print(f"\n  [XGBoost 성능 지표]")
    for k, v in metrics.items():
        print(f"    {k:<15}: {v:.4f}")
    print(f"    교차 검증 AUC (5-Fold): {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    return metrics, y_pred, y_prob


def plot_model_comparison(y_test, lr_prob, xgb_prob, lr_metrics, xgb_metrics):
    """
    로지스틱 회귀와 XGBoost의 성능을 ROC 곡선과 막대그래프로 비교 시각화한다.

    두 모델 간의 성능 차이(Δ AUC)가 미미하다면(<0.02), 해석의 투명성이 높은 
    로지스틱 회귀를 실무 서비스 기획의 최종 모델로 선택하는 것이 타당함을 입증하기 위함이다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC 곡선
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.50)')
    for name, prob, color in [
        ('Logistic Regression', lr_prob,  '#2196F3'),
        ('XGBoost',             xgb_prob, '#F44336'),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, linewidth=2, color=color, label=f'{name} (AUC={auc:.4f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curve: LR vs XGBoost')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 주요 지표 비교 막대그래프
    ax = axes[1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
    lr_vals  = [lr_metrics[m]  for m in metric_names]
    xgb_vals = [xgb_metrics[m] for m in metric_names]
    x, w = np.arange(len(metric_names)), 0.35

    bars1 = ax.bar(x - w/2, lr_vals,  w, label='Logistic Regression',
                   color='#2196F3', alpha=0.8, edgecolor='white')
    bars2 = ax.bar(x + w/2, xgb_vals, w, label='XGBoost',
                   color='#F44336', alpha=0.8, edgecolor='white')
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison\nLogistic Regression vs XGBoost')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '12_model_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


# =============================================================================
# STEP 3. SHAP 전체 변수 중요도 분석 (Global Importance)
# =============================================================================

def run_shap(model, X_test, feature_names):
    """트리 기반 모델 전용 TreeExplainer를 활용하여 SHAP 값을 산출한다."""
    print("  SHAP 값을 계산 중입니다...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


def plot_shap_bar(shap_values, X_test, feature_names):
    """특성별 평균 절대 SHAP 값을 막대그래프로 시각화하여 전역적 중요도를 확인한다."""
    X_df = pd.DataFrame(X_test, columns=feature_names)
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_df, plot_type='bar',
                      show=False, max_display=15)
    plt.title('SHAP Feature Importance (Mean |SHAP Value|)\n'
              'Average impact magnitude on churn prediction',
              fontsize=11, pad=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '13_shap_importance.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


def plot_shap_beeswarm(shap_values, X_test, feature_names):
    """
    각 특성이 이탈 예측에 미치는 '방향'과 '크기'를 동시에 보여주는 비즈웜(Beeswarm) 플롯을 시각화한다.

    빨간색 점(특성값이 높음)이 오른쪽(+)에 위치하면 해당 특성이 이탈을 유발함을 의미한다.
    이를 통해 로지스틱 회귀에서 파악한 회귀 계수의 방향성과 비선형 모델의 예측 논리가 
    일치하는지 상호 교차 검증할 수 있다.
    """
    X_df = pd.DataFrame(X_test, columns=feature_names)
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_df, show=False, max_display=15)
    plt.title('SHAP Beeswarm Plot\n'
              'Red = high feature value  |  Blue = low feature value\n'
              'Right (+) = raises churn risk  |  Left (−) = lowers churn risk',
              fontsize=10, pad=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '14_shap_beeswarm.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


# =============================================================================
# STEP 4. SHAP Waterfall 시각화 (개별 고객 설명)
# =============================================================================

def plot_shap_waterfall(model, shap_values, X_test, feature_names):
    """
    특정 고객의 예측 결과가 평균 기대치로부터 어떻게 도출되었는지 단계별로 보여준다.

    초고위험 고객(이탈 확률 80% 이상)과 초저위험 고객(20% 미만)의 사례를 비교하여,
    "상담원이 특정 고객을 응대할 때 어떤 맞춤형 제안(리텐션 오퍼)을 해야 하는가?"라는
    초개인화 서비스 기획의 구체적인 예시를 제공한다.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    high_idx = np.where(y_prob > 0.8)[0]
    low_idx  = np.where(y_prob < 0.2)[0]
    if len(high_idx) == 0: high_idx = [np.argmax(y_prob)]
    if len(low_idx)  == 0: low_idx  = [np.argmin(y_prob)]

    for idx, label, title in [
        (high_idx[0], 'high', f'HIGH-RISK Customer (Predicted Churn: {y_prob[high_idx[0]]:.1%})'),
        (low_idx[0],  'low',  f'LOW-RISK Customer  (Predicted Churn: {y_prob[low_idx[0]]:.1%})'),
    ]:
        customer_shap = shap_values[idx]
        top_idx  = np.argsort(np.abs(customer_shap))[-12:][::-1]
        top_feat = [feature_names[i] for i in top_idx]
        top_shap = [customer_shap[i]  for i in top_idx]
        top_val  = [X_test[idx, i]    for i in top_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#F44336' if s > 0 else '#2196F3' for s in top_shap]
        y_pos  = range(len(top_feat))

        ax.barh(list(y_pos), top_shap[::-1], color=colors[::-1], edgecolor='white')
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(
            [f'{f}  (value={v:.1f})' for f, v in zip(top_feat[::-1], top_val[::-1])],
            fontsize=9)
        ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('SHAP Value  (+) Increases churn risk  |  (−) Decreases churn risk')
        ax.set_title(f'Individual Prediction Explanation\n{title}', fontsize=11, pad=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f'15_shap_waterfall_{label}_risk.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"  저장 완료: {path}")


# =============================================================================
# STEP 5. 모델 비교 및 분석 요약 출력
# =============================================================================

def print_summary(lr_metrics, xgb_metrics, shap_values, feature_names):
    """
    모델 간 성능 차이 및 SHAP 변수 중요도의 비즈니스 해석을 터미널에 출력한다.
    """
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)

    sep = "=" * 60
    print(f"\n{sep}")
    print("[ 모델 성능 비교 요약 (Logistic Regression vs XGBoost) ]")
    print(sep)
    print(f"  {'평가지표':<12} {'로지스틱':>10} {'XGBoost':>10} {'차이(Δ)':>8}")
    print("-" * 45)
    for m in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']:
        delta = xgb_metrics[m] - lr_metrics[m]
        print(f"  {m:<12} {lr_metrics[m]:>10.4f} {xgb_metrics[m]:>10.4f} {delta:>+8.4f}")

    diff = xgb_metrics['AUC-ROC'] - lr_metrics['AUC-ROC']
    print(f"\n  AUC 차이: {diff:+.4f}")
    if abs(diff) < 0.02:
        print("  → 해석: Δ AUC가 0.02 미만으로, 데이터 내 비선형적 패턴이 크지 않음을 시사함.")
        print("          따라서 설명력(해석의 투명성)이 높은 로지스틱 회귀를 최종 서비스 기획 모델로 채택함.")
    else:
        print("  → 해석: Δ AUC가 0.02 이상으로, XGBoost가 유의미한 비선형 패턴을 포착함.")
        print("          예측은 XGBoost로, 요인 해석은 SHAP을 활용하는 전략 채택.")

    print(f"\n{sep}")
    print("[ SHAP 상위 10개 핵심 요인 — 보험 도메인 해석 ]")
    print(sep)
    interp_map = {
        'tenure'              : '계약 유지 기간 → 길수록 이탈 방어력 극대화',
        'Contract_Two year'   : '2년 약정 가입 → 이탈을 막는 가장 강력한 락인(Lock-in) 장치',
        'Contract_One year'   : '1년 약정 가입 → 중간 수준의 이탈 방어 효과',
        'MonthlyCharges'      : '월 보험료 수준 → 높을수록 이탈 위험 증가',
        'TotalCharges'        : '누적 납입 보험료 (유지 기간과 강한 상관관계)',
        'Internet_Fiber optic': '프리미엄 디지털 서비스 이용군 → 전환 비용이 낮아 쉽게 이탈',
        'Internet_No'         : '디지털 채널 미이용 → 오히려 안정적인 유지 고객',
        'OnlineSecurity_enc'  : '부가 보장 특약 가입 여부 → 이탈 억제 효과 존재',
        'TechSupport_enc'     : '고객센터/챗봇 이용 → 불만 해소를 통한 유지 유도',
        'PaperlessBilling_enc': '전자 고지서 수신 (디지털 친화) → 타사 비교 탐색이 활발해 이탈 소폭 증가',
    }
    for feat, val in mean_abs.head(10).items():
        interp = interp_map.get(feat, '')
        print(f"  {feat:<28} |SHAP|={val:.4f}  {interp}")


# =============================================================================
# 실행
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  보험 디지털 플랫폼 고객 이탈 분석")
    print("  STEP 4b: XGBoost + SHAP 분석을 통한 심층 요인 검증")
    print("=" * 60)

    df = load_data(DATA_PATH)
    X, y = build_features(df)
    feature_names = X.columns.tolist()
    X_arr = X.values

    print("\n[1] 학습 및 평가 데이터 분할 중...")
    X_train, X_test, y_train, y_test = split_data(X_arr, y)

    print("\n[2] XGBoost 모델 학습 중...")
    xgb_model = train_xgboost(X_train, y_train)

    print("\n[3] XGBoost 모델 성능 평가...")
    xgb_metrics, xgb_pred, xgb_prob = evaluate_xgb(
        xgb_model, X_train, X_test, y_train, y_test)

    # 이전에 학습된 로지스틱 회귀 모델 로드 및 비교
    lr_model_path = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
    if os.path.exists(lr_model_path):
        print("\n[4] 로지스틱 회귀 모델과의 비교 분석 진행...")
        lr_model  = joblib.load(lr_model_path)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        lr_prob   = lr_model.predict_proba(X_test_df)[:, 1]
        lr_pred   = lr_model.predict(X_test_df)
        lr_metrics = {
            'Accuracy' : accuracy_score(y_test, lr_pred),
            'Precision': precision_score(y_test, lr_pred),
            'Recall'   : recall_score(y_test, lr_pred),
            'F1'       : f1_score(y_test, lr_pred),
            'AUC-ROC'  : roc_auc_score(y_test, lr_prob),
        }
        plot_model_comparison(y_test, lr_prob, xgb_prob, lr_metrics, xgb_metrics)
    else:
        print("  경고: 로지스틱 회귀 모델을 찾을 수 없습니다. 04_modeling.py를 먼저 실행하세요.")
        lr_metrics = None

    print("\n[5] SHAP 기반 심층 요인 분석 진행...")
    explainer, shap_values = run_shap(xgb_model, X_test, feature_names)

    print("\n[6] SHAP 시각화 이미지 생성 중...")
    plot_shap_bar(shap_values, X_test, feature_names)
    plot_shap_beeswarm(shap_values, X_test, feature_names)
    plot_shap_waterfall(xgb_model, shap_values, X_test, feature_names)

    if lr_metrics:
        print_summary(lr_metrics, xgb_metrics, shap_values, feature_names)

    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    print(f"\n  최종 모델 저장 완료: models/xgboost_model.pkl")

    print("\n" + "=" * 60)
    print("✅ 04b_modeling_xgb.py 실행 완료")
    print("   → 다음 단계: python src/05_interpretation.py")
    print("=" * 60)