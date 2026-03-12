# =============================================================================
# 02_eda.py
# 탐색적 데이터 분석 (Exploratory Data Analysis)
#
# [분석 목적]
# 본 파일에서는 고객 이탈과 관련된 주요 패턴을 시각화하고,
# 이후 군집 분석 및 모델링에 활용할 핵심 변수를 선별한다.
#
# [분석 구성]
#   1. 수치형 변수 분포 및 이탈 여부에 따른 차이
#   2. 범주형 변수별 이탈률 비교
#   3. 상관관계 분석
#   4. 보험 도메인 관점의 인사이트 도출
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
sns.set_style('whitegrid')

# 경로 설정
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'Telco-Customer-Churn.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# 01_load_data의 load_data 함수 재사용
sys.path.append(os.path.join(BASE_DIR, 'src'))
from load_data_utils import load_data


# =============================================================================
# EDA 1. 수치형 변수 분포 — 이탈 여부에 따른 비교
# =============================================================================

def plot_numeric_distributions(df: pd.DataFrame):
    """
    tenure, MonthlyCharges, TotalCharges 세 변수의 분포를
    이탈 여부(Churn)에 따라 KDE 플롯으로 비교한다.

    보험 도메인 해석:
    - tenure(계약 기간)이 짧을수록 이탈률이 높다면
      → 신규 계약자 조기 이탈 방지 서비스가 필요함을 시사
    - MonthlyCharges(월 보험료)가 높을수록 이탈률이 높다면
      → 고보험료 고객 대상 맞춤 혜택 제공 필요
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    numeric_cols = {
        'tenure'        : '계약 기간 (월) / 보험 계약 기간',
        'MonthlyCharges': '월 납입액 ($) / 월 보험료',
        'TotalCharges'  : '총 납입액 ($) / 누적 보험료',
    }

    colors = {'No': '#2196F3', 'Yes': '#F44336'}

    for ax, (col, label) in zip(axes, numeric_cols.items()):
        for churn_val, color in colors.items():
            subset = df[df['Churn'] == churn_val][col]
            subset.plot.kde(ax=ax, label=f"{'유지' if churn_val=='No' else '이탈'}",
                           color=color, linewidth=2)
            ax.axvline(subset.mean(), color=color, linestyle='--', alpha=0.6, linewidth=1)

        ax.set_title(label, fontsize=11, pad=10)
        ax.set_xlabel('')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('수치형 변수 분포: 이탈 vs 유지 고객 비교\n(보험 도메인: 계약기간·보험료 관점)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '02_numeric_distributions.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")


# =============================================================================
# EDA 2. 범주형 변수별 이탈률 비교
# =============================================================================

def plot_categorical_churn_rates(df: pd.DataFrame):
    """
    주요 범주형 변수별 이탈률을 막대 그래프로 시각화한다.

    보험 도메인 해석:
    - Contract(납입 주기): 월납 고객의 이탈률이 높다면
      → 연납 전환 유도 프로모션이 효과적일 수 있음
    - PaperlessBilling(전자고지): 디지털 채널 이용 고객의 이탈 패턴 파악
    - InternetService: 부가 서비스 이용 여부에 따른 이탈률 차이
    """
    # 분석할 범주형 변수 선택 (도메인 관련성 높은 변수 우선)
    cat_cols = [
        ('Contract',        '계약 유형 / 납입 주기'),
        ('InternetService', '인터넷 서비스 / 디지털 채널'),
        ('PaymentMethod',   '결제 수단'),
        ('PaperlessBilling','전자고지 / 디지털 고지 수신'),
        ('TechSupport',     '기술 지원 / 고객센터 이용'),
        ('OnlineSecurity',  '온라인 보안 / 부가 특약'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, cat_cols):
        # 범주별 이탈률 계산
        churn_rate = (df.groupby(col)['Churn_binary']
                       .mean()
                       .sort_values(ascending=False) * 100)

        bars = ax.bar(range(len(churn_rate)), churn_rate.values,
                      color=['#F44336' if v > 30 else '#2196F3'
                             for v in churn_rate.values],
                      edgecolor='white', linewidth=1.2)

        # 막대 위에 수치 표시
        for bar, val in zip(bars, churn_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5)

        ax.set_xticks(range(len(churn_rate)))
        ax.set_xticklabels(churn_rate.index, rotation=20, ha='right', fontsize=8)
        ax.set_title(label, fontsize=10, pad=8)
        ax.set_ylabel('이탈률 (%)')
        ax.set_ylim(0, churn_rate.max() * 1.25)
        ax.axhline(df['Churn_binary'].mean() * 100, color='gray',
                   linestyle='--', alpha=0.5, linewidth=1,
                   label=f'전체 평균 {df["Churn_binary"].mean()*100:.1f}%')
        ax.legend(fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('범주형 변수별 이탈률 비교\n(보험 도메인: 납입주기·채널·특약 관점)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '03_categorical_churn_rates.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")


# =============================================================================
# EDA 3. 상관관계 히트맵
# =============================================================================

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    수치형 변수 간 상관관계를 히트맵으로 시각화한다.
    Churn_binary와의 상관관계를 통해 이탈 예측에 중요한 변수를 파악한다.

    - tenure와 Churn의 음의 상관: 장기 고객일수록 이탈 가능성 낮음
    - MonthlyCharges와 Churn의 양의 상관: 고보험료 고객의 이탈 위험
    - 다중공선성 확인: tenure와 TotalCharges는 강한 양의 상관 예상
      → 모델링 시 변수 선택에 활용
    """
    numeric_df = df[['tenure', 'MonthlyCharges', 'TotalCharges',
                     'SeniorCitizen', 'Churn_binary']]

    corr = numeric_df.corr()

    # 변수명을 보험 도메인으로 변환
    rename_map = {
        'tenure'        : '계약기간',
        'MonthlyCharges': '월보험료',
        'TotalCharges'  : '누적보험료',
        'SeniorCitizen' : '고령고객',
        'Churn_binary'  : '이탈여부',
    }
    corr = corr.rename(index=rename_map, columns=rename_map)

    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # 상삼각 마스킹

    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax,
                annot_kws={'size': 10})

    ax.set_title('수치형 변수 간 상관관계\n(보험 도메인 변수명 기준)', fontsize=12, pad=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '04_correlation_heatmap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")


# =============================================================================
# EDA 4. 계약 기간별 이탈 패턴 (핵심 인사이트)
# =============================================================================

def plot_tenure_churn_pattern(df: pd.DataFrame):
    """
    계약 기간(tenure)을 구간으로 나눠 구간별 이탈률을 시각화한다.

    보험 도메인 관점의 핵심 인사이트:
    - 계약 초기(1~12개월) 이탈률이 가장 높다면
      → 신규 가입자 온보딩 서비스, 첫 1년 집중 관리 프로그램 필요
    - 이는 삼성화재 디지털 플랫폼의 '신규 고객 이탈 방지 서비스' 기획으로 직결
    """
    # 계약 기간을 6개 구간으로 분할
    # 보험 도메인: 1~12개월=신규, 13~24=초기, 25~36=중기, 37~48=안정, 49~60=장기, 61+=최장기
    bins   = [0, 12, 24, 36, 48, 60, 72]
    labels = ['1~12개월\n(신규)', '13~24개월\n(초기)', '25~36개월\n(중기)',
              '37~48개월\n(안정)', '49~60개월\n(장기)', '61~72개월\n(최장기)']

    df = df.copy()
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    group_stats = df.groupby('tenure_group', observed=True).agg(
        이탈률=('Churn_binary', lambda x: x.mean() * 100),
        고객수=('Churn_binary', 'count')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()  # 고객 수를 위한 보조 축

    # 이탈률 막대
    bars = ax1.bar(group_stats['tenure_group'], group_stats['이탈률'],
                   color=['#F44336' if v > 30 else '#FF9800' if v > 20 else '#2196F3'
                          for v in group_stats['이탈률']],
                   alpha=0.8, edgecolor='white', linewidth=1.2, zorder=2)

    # 고객 수 선 그래프
    ax2.plot(group_stats['tenure_group'], group_stats['고객수'],
             color='gray', marker='o', linewidth=1.5,
             linestyle='--', alpha=0.6, label='고객 수')

    # 막대 위 수치
    for bar, val in zip(bars, group_stats['이탈률']):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xlabel('계약 유지 기간 (보험 계약 기간)', fontsize=11)
    ax1.set_ylabel('이탈률 (%)', fontsize=11)
    ax2.set_ylabel('고객 수 (명)', fontsize=11, color='gray')
    ax1.set_title('계약 기간별 이탈률 패턴\n→ 신규 고객 조기 이탈 방지 서비스 기획 근거',
                  fontsize=12, pad=12)
    ax1.axhline(df['Churn_binary'].mean() * 100, color='black',
                linestyle=':', alpha=0.4, label=f'전체 평균')
    ax1.spines['top'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '05_tenure_churn_pattern.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")


# =============================================================================
# EDA 요약 출력
# =============================================================================

def print_eda_summary(df: pd.DataFrame):
    """
    EDA에서 발견한 주요 인사이트를 보험 도메인 관점으로 요약 출력한다.
    이 인사이트는 이후 군집 분석 및 서비스 기획의 근거가 된다.
    """
    sep = "=" * 60

    print(f"\n{sep}")
    print("[ EDA 주요 인사이트 — 보험 도메인 해석 ]")
    print(sep)

    # 1. 계약 기간별 이탈률
    bins   = [0, 12, 24, 36, 72]
    labels = ['신규(1~12개월)', '초기(13~24개월)', '중기(25~36개월)', '장기(37개월~)']
    df2 = df.copy()
    df2['tenure_group'] = pd.cut(df2['tenure'], bins=bins, labels=labels)
    tenure_churn = df2.groupby('tenure_group', observed=True)['Churn_binary'].mean() * 100
    print("\n📌 계약 기간별 이탈률 (보험: 계약 기간별 해약률)")
    for group, rate in tenure_churn.items():
        print(f"  {group}: {rate:.1f}%")

    # 2. 납입 주기별 이탈률
    contract_churn = df.groupby('Contract')['Churn_binary'].mean() * 100
    print("\n📌 납입 주기별 이탈률 (보험: 월납 vs 연납)")
    for c, r in contract_churn.sort_values(ascending=False).items():
        print(f"  {c}: {r:.1f}%")

    # 3. 전자고지 이용 여부
    pb_churn = df.groupby('PaperlessBilling')['Churn_binary'].mean() * 100
    print("\n📌 전자고지(디지털 채널) 이용 여부별 이탈률")
    for c, r in pb_churn.items():
        print(f"  {'이용' if c=='Yes' else '미이용'}: {r:.1f}%")

    # 4. 월 보험료 구간별 이탈률
    df2['charge_group'] = pd.cut(df['MonthlyCharges'],
                                  bins=[0, 35, 65, 95, 120],
                                  labels=['저가(~35)', '중가(35~65)',
                                          '고가(65~95)', '초고가(95~)'])
    charge_churn = df2.groupby('charge_group', observed=True)['Churn_binary'].mean() * 100
    print("\n📌 월 보험료 구간별 이탈률")
    for c, r in charge_churn.items():
        print(f"  {c}: {r:.1f}%")

    print(f"\n{sep}")
    print("[ 서비스 기획 시사점 ]")
    print(sep)
    print("  1. 신규 계약자(1~12개월)의 이탈률이 가장 높음")
    print("     → 온보딩 강화, 첫 1년 집중 케어 서비스 필요")
    print("  2. 월납 고객의 이탈률 > 연납 고객")
    print("     → 연납 전환 유도 프로모션 기획 필요")
    print("  3. 고보험료 고객의 이탈 위험 높음")
    print("     → 고가 구간 고객 대상 맞춤 혜택 제공 필요")


# =============================================================================
# 실행
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  보험 디지털 플랫폼 고객 이탈 분석")
    print("  STEP 2: 탐색적 데이터 분석 (EDA)")
    print("=" * 60)

    df = load_data(DATA_PATH)

    print("\n📊 시각화 생성 중...")
    plot_numeric_distributions(df)
    plot_categorical_churn_rates(df)
    plot_correlation_heatmap(df)
    plot_tenure_churn_pattern(df)

    print_eda_summary(df)

    print("\n" + "=" * 60)
    print("✅ 02_eda.py 완료")
    print(f"   시각화 저장 위치: reports/figures/")
    print("   → 다음 단계: python src/03_clustering.py")
    print("=" * 60)