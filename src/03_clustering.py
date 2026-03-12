# =============================================================================
# 03_clustering.py
# K-Means 알고리즘을 활용한 고객 군집화
#
# [분석 목적 및 구성 이유]
# 이탈을 예측하기에 앞서, 전체 고객을 행동 패턴에 따라 몇 개의 의미 있는 그룹으로 묶는다.
# 이는 "고객이 누구인지 파악한 후, 그에 맞는 서비스를 기획한다"는 실제 디지털 서비스 기획의
# 업무 흐름을 그대로 반영하기 위함이다.
#
# [보험 도메인 해석 예시]
#   - 고위험 군집: 가입 기간이 짧고 월 보험료가 높은 고객 -> 온보딩 및 초기 케어 대상
#   - 저위험 군집: 장기 유지 중이며 납입액이 안정적인 고객 -> 우수 고객 혜택 제공 대상
#
# [분석 방법]
#   - 활용 변수: 계약 유지 기간, 월 납입액, 누적 납입액 및 주요 범주형 변수
#   - 최적 군집 수(K) 도출: 엘보우 방법 및 실루엣 점수 교차 검증
#   - 알고리즘: K-Means (거리 기반 군집화)
#   - 시각화: 주성분 분석(PCA)을 통한 2차원 산점도 및 군집별 프로파일 막대그래프
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

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
# STEP 1. 군집화를 위한 파생 변수 생성 및 전처리
# =============================================================================

def prepare_features(df: pd.DataFrame):
    """
    K-Means 군집화에 사용할 핵심 변수를 선택하고 알맞은 형태로 인코딩한다.

    [구성 이유]
    K-Means 알고리즘은 데이터 포인트 간의 '유클리드 거리'를 기반으로 작동한다.
    따라서 월 납입액(수십 단위)과 누적 납입액(수천 단위)처럼 스케일이 다른 변수들을
    그대로 사용하면 결과가 왜곡되므로, 평균이 0이고 표준편차가 1이 되도록
    표준화(StandardScaler)를 반드시 수행해야 한다. 범주형 변수는 원핫 인코딩을 적용한다.
    """
    df = df.copy()

    # 이진 범주형 변수 인코딩 (Yes=1, No=0)
    binary_cols = ['PaperlessBilling', 'PhoneService', 'Partner', 'Dependents']
    for col in binary_cols:
        df[col + '_enc'] = (df[col] == 'Yes').astype(int)

    # 다중 범주형 변수 원핫 인코딩
    contract_dummies = pd.get_dummies(df['Contract'], prefix='Contract')
    internet_dummies = pd.get_dummies(df['InternetService'], prefix='Internet')

    # 군집화에 활용할 최종 독립 변수 목록
    feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                    'SeniorCitizen',
                    'PaperlessBilling_enc', 'PhoneService_enc',
                    'Partner_enc', 'Dependents_enc']

    X = pd.concat([df[feature_cols], contract_dummies, internet_dummies], axis=1)

    # 거리 기반 알고리즘을 위한 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, X.columns.tolist(), scaler


# =============================================================================
# STEP 2. 최적의 군집 수(K) 탐색
# =============================================================================

def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 9)):
    """
    엘보우 방법과 실루엣 점수를 결합하여 객관적인 최적의 군집 수(K)를 찾는다.

    [구성 이유]
    엘보우 방법은 군집 수가 늘어날수록 관성(Inertia)이 무조건 감소하므로 최적점을
    주관적으로 판단해야 하는 한계가 있다. 이를 보완하기 위해, 군집 내 응집도와 
    군집 간 분리도를 동시에 평가하는 실루엣 점수(Silhouette Score)를 함께 산출하여 
    가장 높은 점수를 기록하는 K값을 최종 군집 수로 선정한다.
    """
    inertias    = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        print(f"  K={k}: 관성(Inertia)={km.inertia_:.1f}, 실루엣 점수={silhouette_score(X_scaled, labels):.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 엘보우 방법 시각화
    ax1.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=7)
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)')
    ax1.set_title('Elbow Method\n(Look for the bend point)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 실루엣 점수 시각화
    best_k = list(k_range)[np.argmax(silhouettes)]
    ax2.plot(list(k_range), silhouettes, 'rs-', linewidth=2, markersize=7)
    ax2.axvline(best_k, color='red', linestyle='--', alpha=0.5,
                label=f'Best K={best_k}')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score\n(Higher is better)')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Optimal K Selection: Elbow + Silhouette', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '06_optimal_k.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  저장 완료: {path}")
    print(f"  실루엣 점수 기준 최적의 K: {best_k}")

    return best_k, inertias, silhouettes


# =============================================================================
# STEP 3. K-Means 모델 최종 학습
# =============================================================================

def fit_kmeans(X_scaled: np.ndarray, k: int):
    """
    선정된 최적의 K값을 바탕으로 최종 군집화 모델을 학습한다.

    [구성 이유]
    초기 중심점 위치에 따라 결과가 달라지는 K-Means의 단점을 보완하기 위해,
    n_init=10 옵션을 주어 10번 반복 실행한 후 가장 우수한 결과를 선택하도록 한다.
    학습된 모델은 추후 새로운 데이터가 유입되었을 때 군집을 판별할 수 있도록 저장한다.
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    model_path = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
    joblib.dump(km, model_path)
    print(f"  모델 저장 완료: {model_path}")

    return km, labels


# =============================================================================
# STEP 4. 군집화 결과 시각화 (PCA 2차원 축소)
# =============================================================================

def plot_clusters_pca(X_scaled: np.ndarray, labels: np.ndarray, k: int):
    """
    고차원의 데이터를 2차원으로 축소하여 군집의 분포를 시각적으로 확인한다.

    [구성 이유]
    14개의 변수로 이루어진 데이터를 한눈에 파악하기 위해 주성분 분석(PCA)을 활용한다.
    이는 데이터를 2개의 축으로 압축하여 시각화할 뿐, 실제 군집화 결과 자체를
    왜곡하거나 변경하지는 않는다. 분산 설명력을 통해 축소의 타당성을 함께 검증한다.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    print(f"  PCA 분산 설명력: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, "
          f"총합={sum(explained):.1%}")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 0.8, k))

    for i in range(k):
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[i]], label=f'Cluster {i+1}',
                   alpha=0.5, s=15, edgecolors='none')

    ax.set_xlabel(f'PC1 ({explained[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({explained[1]:.1%} variance)')
    ax.set_title(f'Customer Segments — K-means (K={k})\n'
                 f'Visualized via PCA (Total variance explained: {sum(explained):.1%})')
    ax.legend(markerscale=2, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '07_clusters_pca.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


# =============================================================================
# STEP 5. 군집 특성 프로파일링
# =============================================================================

def profile_clusters(df: pd.DataFrame, labels: np.ndarray, k: int):
    """
    각 군집의 통계적 특성을 분석하여 비즈니스적 의미를 부여한다.

    [구성 이유]
    서비스 기획 관점에서 가장 중요한 단계이다. 단순히 '1번 군집', '2번 군집'으로 
    나누는 것에 그치지 않고, 각 군집의 이탈률, 평균 계약 기간, 평균 보험료 등을 
    산출하여 어떤 특성을 가진 고객들이 모여 있는지 정량적으로 확인한다.
    """
    df = df.copy()
    df['Cluster'] = labels + 1 

    # 군집별 주요 수치형 변수 요약
    numeric_profile = df.groupby('Cluster').agg(
        Count       =('Cluster', 'count'),
        Tenure_mean =('tenure', 'mean'),
        Monthly_mean=('MonthlyCharges', 'mean'),
        Total_mean  =('TotalCharges', 'mean'),
        Senior_pct  =('SeniorCitizen', 'mean'),
        Churn_rate  =('Churn_binary', 'mean'),
    ).round(2)
    numeric_profile['Churn_rate'] = (numeric_profile['Churn_rate'] * 100).round(1)
    numeric_profile['Senior_pct'] = (numeric_profile['Senior_pct'] * 100).round(1)

    print("\n[ 군집별 주요 수치 특성 요약 ]")
    print(numeric_profile.to_string())

    # 군집별 납입 주기 분포 확인
    contract_profile = (df.groupby(['Cluster', 'Contract'])
                          .size()
                          .unstack(fill_value=0))
    contract_pct = contract_profile.div(contract_profile.sum(axis=1), axis=0) * 100
    print("\n[ 군집별 납입 주기 분포 비율 (%) ]")
    print(contract_pct.round(1).to_string())

    # 군집 특성 비교 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.Set1(np.linspace(0, 0.8, k))

    # 이탈률 비교
    ax = axes[0]
    bars = ax.bar(numeric_profile.index.astype(str),
                  numeric_profile['Churn_rate'],
                  color=colors[:k], edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, numeric_profile['Churn_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.axhline(df['Churn_binary'].mean()*100, color='gray',
               linestyle='--', alpha=0.6, label=f'Avg {df["Churn_binary"].mean()*100:.1f}%')
    ax.set_title('Churn Rate by Cluster\n(Insurance: Cancellation Rate)')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Churn Rate (%)')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 평균 계약 기간 비교
    ax = axes[1]
    bars = ax.bar(numeric_profile.index.astype(str),
                  numeric_profile['Tenure_mean'],
                  color=colors[:k], edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, numeric_profile['Tenure_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}mo', ha='center', va='bottom', fontsize=10)
    ax.set_title('Avg Tenure by Cluster\n(Insurance: Avg Contract Duration)')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Avg Tenure (months)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 평균 보험료 비교
    ax = axes[2]
    bars = ax.bar(numeric_profile.index.astype(str),
                  numeric_profile['Monthly_mean'],
                  color=colors[:k], edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, numeric_profile['Monthly_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'${val:.1f}', ha='center', va='bottom', fontsize=10)
    ax.set_title('Avg Monthly Charges by Cluster\n(Insurance: Premium Level)')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Avg Monthly Charges ($)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Cluster Profiles: Churn Rate, Tenure, Monthly Charges',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '08_cluster_profiles.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  저장 완료: {path}")

    return df, numeric_profile


# =============================================================================
# STEP 6. 비즈니스 페르소나 및 액션 플랜 부여
# =============================================================================

def assign_personas(numeric_profile: pd.DataFrame):
    """
    통계적으로 분류된 군집에 비즈니스 페르소나를 부여하고 맞춤형 기획안을 도출한다.
    
    [구성 이유]
    데이터 분석 결과를 현업 부서(서비스 기획팀, 마케팅팀 등)에서 즉시 활용할 수 있는
    실행 가능한 언어(Actionable Insight)로 번역하는 핵심 과정이다.
    """
    print("\n[ 군집별 고객 페르소나 및 서비스 기획 방안 ]")
    print("=" * 60)

    for cluster_id, row in numeric_profile.iterrows():
        churn  = row['Churn_rate']
        tenure = row['Tenure_mean']
        charge = row['Monthly_mean']
        count  = int(row['Count'])

        if churn > 40:
            persona = "고위험군 (High Risk) — 이탈 가능성이 매우 높은 초기 고객"
            action  = "최우선 과제: 앱 온보딩 강화 및 가입 초기 집중 케어 서비스 안내"
        elif churn > 20:
            persona = "중위험군 (Medium Risk) — 혜택에 민감한 중기 이탈 예상 고객"
            action  = "우선 과제: 연납 전환 시 보험료 할인 프로모션 및 리텐션 오퍼 제공"
        else:
            persona = "저위험군 (Low Risk) — 장기 유지 중인 충성 고객"
            action  = "유지 과제: VIP 로열티 프로그램 제공 및 맞춤형 특약 상품 업셀링"

        print(f"\n  군집 {cluster_id}: {persona}")
        print(f"    이탈률   : {churn:.1f}%")
        print(f"    평균 기간: {tenure:.1f}개월")
        print(f"    평균 납입: 월 {charge:.1f}달러")
        print(f"    고객 수  : {count:,}명")
        print(f"    기획 방향: {action}")


# =============================================================================
# 실행
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  보험 디지털 플랫폼 고객 이탈 분석")
    print("  STEP 3: 고객 세분화 (K-Means Clustering)")
    print("=" * 60)

    df = load_data(DATA_PATH)

    print("\n[1] 군집화 변수 전처리 중...")
    X_scaled, feature_names, scaler = prepare_features(df)
    print(f"  변환 완료: {X_scaled.shape[0]}명 고객 × {X_scaled.shape[1]}개 특성 변수")

    print("\n[2] 최적의 군집 수(K) 탐색 중...")
    best_k, inertias, silhouettes = find_optimal_k(X_scaled)

    print(f"\n[3] 도출된 최적값(K={best_k})으로 K-Means 모델 학습 중...")
    km, labels = fit_kmeans(X_scaled, best_k)

    print("\n[4] 군집화 결과 시각화 (PCA 축소)...")
    plot_clusters_pca(X_scaled, labels, best_k)

    print("\n[5] 군집별 특성 프로파일링 분석...")
    df_clustered, numeric_profile = profile_clusters(df, labels, best_k)

    # 6. 페르소나 부여
    assign_personas(numeric_profile)

    # 다음 단계(로지스틱 회귀 모델링)를 위해 군집 정보가 추가된 데이터 저장
    out_path = os.path.join(BASE_DIR, 'data', 'telco_clustered.csv')
    df_clustered.to_csv(out_path, index=False)
    print(f"\n  군집화 데이터 저장 완료: {out_path}")

    print("\n" + "=" * 60)
    print("✅ 03_clustering.py 실행 완료")
    print("   → 다음 단계: python src/04_modeling.py")
    print("=" * 60)