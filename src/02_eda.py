# =============================================================================
# 02_eda.py
# 탐색적 데이터 분석 (EDA: Exploratory Data Analysis)
#
# [분석 목적 및 구성 이유]
# 본 코드는 고객 이탈과 관련된 주요 패턴을 시각적으로 파악하고,
# 향후 군집화(Clustering) 및 로지스틱 회귀 모델링에 활용할 핵심 변수를 선정하기 위해 작성되었다.
# 단순한 통계량 확인을 넘어, 도메인 지식(보험업)을 결합하여 서비스 기획의 근거를 마련한다.
#
# [주요 분석 단계]
#   1. 수치형 변수 분포 비교 (이탈 vs 유지): 커널 밀도 추정(KDE) 활용
#   2. 범주형 변수별 이탈률 확인: 막대 그래프를 통한 직관적 비교
#   3. 수치형 변수 간 상관관계 분석: 선형 모델의 다중공선성(VIF) 사전 진단
#   4. 계약 기간(Tenure) 그룹별 이탈 패턴: 비즈니스 인사이트 도출의 핵심
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 깔끔한 출력을 위해 불필요한 경고 메시지 숨김 처리
warnings.filterwarnings('ignore')

# -- 폰트 및 시각화 환경 설정 --
# 한글 폰트 깨짐 방지 및 OS 독립성을 위해 그래프 라벨은 영문을 기본으로 사용하되, 
# 주석과 결과 해석은 국문으로 작성하여 가독성을 높인다.
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
sns.set_style('whitegrid')

# 데이터 및 이미지 저장 경로 설정 (상대 경로 활용으로 이식성 확보)
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'Telco-Customer-Churn.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# src 폴더 내의 모듈(01_load_data.py 등)을 불러오기 위한 경로 추가
sys.path.append(os.path.join(BASE_DIR, 'src'))
from load_data_utils import load_data


# =============================================================================
# EDA 1. 이탈 여부에 따른 수치형 변수 분포 시각화
# =============================================================================

def plot_numeric_distributions(df: pd.DataFrame):
    """
    이탈 고객과 유지 고객 간의 수치형 변수(계약 기간, 월 납입액 등) 분포 차이를 비교한다.
    
    [구성 이유] 
    히스토그램보다 분포의 모양을 부드럽게 보여주는 커널 밀도 추정(KDE) 플롯을 사용하여,
    이탈이 집중되는 특정 구간(예: 가입 초기, 고액 납입 구간)을 시각적으로 명확히 식별한다.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    col_labels = {
        'tenure'        : 'Tenure (months) / Contract Duration',
        'MonthlyCharges': 'Monthly Charges ($) / Monthly Premium',
        'TotalCharges'  : 'Total Charges ($) / Cumulative Premium',
    }
    colors = {'No': '#2196F3', 'Yes': '#F44336'}
    legend_labels = {'No': 'Retained', 'Yes': 'Churned'}

    for ax, (col, label) in zip(axes, col_labels.items()):
        for churn_val, color in colors.items():
            subset = df[df['Churn'] == churn_val][col]
            # 밀도 추정 플롯 생성 및 평균값 수직선(점선) 추가
            subset.plot.kde(ax=ax, label=legend_labels[churn_val],
                            color=color, linewidth=2)
            ax.axvline(subset.mean(), color=color, linestyle='--',
                       alpha=0.6, linewidth=1)
        
        ax.set_title(label, fontsize=10, pad=10)
        ax.set_xlabel('')
        ax.legend(fontsize=9)
        # 시각적 피로도를 줄이기 위해 그래프 상/우측 테두리 제거
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Numeric Variable Distributions: Churned vs Retained\n'
                 '(Insurance: Contract Duration & Premium)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '02_numeric_distributions.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


# =============================================================================
# EDA 2. 주요 범주형 변수별 이탈률 분석
# =============================================================================

def plot_categorical_churn_rates(df: pd.DataFrame):
    """
    납입 주기, 부가서비스 가입 여부 등 범주형 변수의 각 그룹별 이탈률을 계산하여 시각화한다.
    
    [구성 이유]
    각 범주가 이탈에 미치는 영향을 직관적으로 비교하기 위해 막대 그래프를 사용하며,
    전체 평균 이탈률(Baseline)을 점선으로 추가하여 각 그룹의 위험도를 객관적으로 평가한다.
    """
    cat_cols = [
        ('Contract',         'Contract Type / Payment Frequency'),
        ('InternetService',  'Internet Service / Digital Channel'),
        ('PaymentMethod',    'Payment Method'),
        ('PaperlessBilling', 'Paperless Billing / Digital Notice'),
        ('TechSupport',      'Tech Support / Customer Service'),
        ('OnlineSecurity',   'Online Security / Add-on Coverage'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    avg_churn = df['Churn_binary'].mean() * 100 # 전체 평균 이탈률 기준선

    for ax, (col, label) in zip(axes, cat_cols):
        # 그룹별 이탈률 계산 후 내림차순 정렬
        churn_rate = (df.groupby(col)['Churn_binary']
                        .mean()
                        .sort_values(ascending=False) * 100)

        # 위험도(30% 초과)에 따라 막대 색상을 다르게 적용하여 가시성 확보
        bars = ax.bar(range(len(churn_rate)), churn_rate.values,
                      color=['#F44336' if v > 30 else '#2196F3'
                             for v in churn_rate.values],
                      edgecolor='white', linewidth=1.2)

        # 막대 위에 정확한 수치 텍스트 추가
        for bar, val in zip(bars, churn_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5)

        ax.set_xticks(range(len(churn_rate)))
        ax.set_xticklabels(churn_rate.index, rotation=20, ha='right', fontsize=8)
        ax.set_title(label, fontsize=10, pad=8)
        ax.set_ylabel('Churn Rate (%)')
        ax.set_ylim(0, churn_rate.max() * 1.3)
        ax.axhline(avg_churn, color='gray', linestyle='--',
                   alpha=0.5, linewidth=1, label=f'Avg {avg_churn:.1f}%')
        ax.legend(fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Churn Rate by Categorical Variables\n'
                 '(Insurance: Payment Frequency, Channel, Add-on Coverage)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '03_categorical_churn_rates.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


# =============================================================================
# EDA 3. 수치형 변수 간 상관관계 히트맵 도출
# =============================================================================

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    수치형 독립 변수들 간의 피어슨 상관계수를 히트맵으로 시각화한다.
    
    [구성 이유]
    추후 모델링 단계에서 로지스틱 회귀를 사용할 예정이므로, 독립 변수 간의 강한 상관관계로 인한
    다중공선성(Multicollinearity) 문제를 사전에 진단하기 위해 필수적인 과정이다.
    (예: tenure와 TotalCharges 간의 높은 상관관계 확인)
    """
    numeric_df = df[['tenure', 'MonthlyCharges', 'TotalCharges',
                     'SeniorCitizen', 'Churn_binary']]

    rename_map = {
        'tenure'        : 'Tenure',
        'MonthlyCharges': 'MonthlyCharge',
        'TotalCharges'  : 'TotalCharge',
        'SeniorCitizen' : 'Senior',
        'Churn_binary'  : 'Churn',
    }
    corr = numeric_df.rename(columns=rename_map).corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax,
                annot_kws={'size': 10})

    ax.set_title('Correlation Heatmap\n(Numeric Variables)', fontsize=12, pad=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '04_correlation_heatmap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


# =============================================================================
# EDA 4. 계약 유지 기간(Tenure) 그룹별 이탈 패턴 시각화 (핵심 분석)
# =============================================================================

def plot_tenure_churn_pattern(df: pd.DataFrame):
    """
    연속형 변수인 계약 기간(Tenure)을 연 단위(12개월) 범주형 변수로 구간화(Binning)하여 이탈률을 확인한다.
    
    [구성 이유]
    연속된 숫자 자체보다는 '가입 1년 차', '3년 차' 등 비즈니스 단위로 고객을 묶어 분석해야
    '1년 차 신규 고객 온보딩 강화'와 같은 실제 서비스 기획 인사이트를 도출할 수 있기 때문이다.
    """
    bins   = [0, 12, 24, 36, 48, 60, 72]
    labels = ['1-12mo\n(New)', '13-24mo\n(Early)', '25-36mo\n(Mid)',
              '37-48mo\n(Stable)', '49-60mo\n(Long)', '61-72mo\n(Loyal)']

    df = df.copy()
    # pd.cut을 활용하여 연속형 변수를 지정된 구간(Bins)으로 분할
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    # 그룹별 이탈률 및 고객 수 집계 (최신 pandas 문법에 맞춰 observed=True 적용)
    group_stats = df.groupby('tenure_group', observed=True).agg(
        churn_rate=('Churn_binary', lambda x: x.mean() * 100),
        count=('Churn_binary', 'count')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx() # 동일한 X축을 공유하는 두 번째 Y축 생성 (고객 수 표기용)

    # 이탈률 막대 그래프 (위험도에 따른 색상 차등 적용)
    bars = ax1.bar(group_stats['tenure_group'], group_stats['churn_rate'],
                   color=['#F44336' if v > 30 else '#FF9800' if v > 20 else '#2196F3'
                          for v in group_stats['churn_rate']],
                   alpha=0.8, edgecolor='white', linewidth=1.2, zorder=2)

    # 고객 수 꺾은선 그래프
    ax2.plot(group_stats['tenure_group'], group_stats['count'],
             color='gray', marker='o', linewidth=1.5,
             linestyle='--', alpha=0.6, label='Customer Count')

    for bar, val in zip(bars, group_stats['churn_rate']):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    ax1.set_xlabel('Tenure Group (Insurance: Contract Duration)', fontsize=11)
    ax1.set_ylabel('Churn Rate (%)', fontsize=11)
    ax2.set_ylabel('Customer Count', fontsize=11, color='gray')
    ax1.set_title('Churn Rate by Tenure Group\n'
                  '→ Basis for New Customer Retention Service Design',
                  fontsize=12, pad=12)
    ax1.axhline(df['Churn_binary'].mean() * 100, color='black',
                linestyle=':', alpha=0.4, label='Overall Average')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '05_tenure_churn_pattern.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


# =============================================================================
# EDA 요약 결과 출력
# =============================================================================

def print_eda_summary(df: pd.DataFrame):
    """
    시각화 결과를 바탕으로 도메인 지식이 결합된 비즈니스 인사이트를 터미널에 출력한다.
    """
    sep = "=" * 60

    print(f"\n{sep}")
    print("[ 탐색적 데이터 분석(EDA) 주요 요약 — 보험 도메인 해석 ]")
    print(sep)

    bins   = [0, 12, 24, 36, 72]
    labels = ['신규(1~12개월)', '초기(13~24개월)', '중기(25~36개월)', '장기(37개월+)']
    df2 = df.copy()
    df2['tenure_group'] = pd.cut(df2['tenure'], bins=bins, labels=labels)
    tenure_churn = df2.groupby('tenure_group', observed=True)['Churn_binary'].mean() * 100
    print("\n[1] 계약 기간별 이탈률 (Tenure)")
    for g, r in tenure_churn.items():
        print(f"    {g}: {r:.1f}%")

    contract_churn = df.groupby('Contract')['Churn_binary'].mean() * 100
    print("\n[2] 납입 주기별 이탈률 (Contract Type)")
    for c, r in contract_churn.sort_values(ascending=False).items():
        print(f"    {c}: {r:.1f}%")

    pb_churn = df.groupby('PaperlessBilling')['Churn_binary'].mean() * 100
    print("\n[3] 디지털 채널 이용 여부별 이탈률 (Paperless Billing)")
    for c, r in pb_churn.items():
        print(f"    {'이용 (Yes)' if c=='Yes' else '미이용 (No)'}: {r:.1f}%")

    print(f"\n{sep}")
    print("[ 서비스 기획 시사점 (Service Planning Implications) ]")
    print(sep)
    print("  1. 가입 1년 이내 신규 고객의 이탈률이 약 48%로 가장 치명적임.")
    print("     → 초기 정착을 위한 앱 온보딩 강화 및 1년 차 특별 케어 서비스 기획 필요")
    print("  2. 월납 고객의 이탈률(42.7%)이 2년납(2.8%) 대비 압도적으로 높음.")
    print("     → 장기 납입 전환 유도를 위한 플랫폼 내 리워드/할인 프로모션 필요")


# =============================================================================
# 실행
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  보험 디지털 플랫폼 고객 이탈 분석")
    print("  STEP 2: 탐색적 데이터 분석 (EDA)")
    print("=" * 60)

    # 앞서 구현한 모듈을 통해 데이터 로드
    df = load_data(DATA_PATH)

    print("\n시각화 이미지를 생성 중입니다...")
    plot_numeric_distributions(df)
    plot_categorical_churn_rates(df)
    plot_correlation_heatmap(df)
    plot_tenure_churn_pattern(df)

    print_eda_summary(df)

    print("\n" + "=" * 60)
    print("✅ 02_eda.py 실행 완료")
    print("   저장된 이미지 경로: reports/figures/")
    print("   → 다음 단계: python src/03_clustering.py")
    print("=" * 60)