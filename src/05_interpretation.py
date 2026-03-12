# =============================================================================
# 05_interpretation.py
# 분석 결과 종합 및 디지털 서비스 기획안 도출
#
# [분석 목적 및 구성 이유]
# 본 코드는 앞서 진행한 탐색적 데이터 분석(EDA), 고객 세분화(Clustering), 
# 예측 모델링(Modeling)의 결과를 하나로 통합하여 구체적인 '서비스 기획안'으로 발전시킨다.
# "데이터 분석 자체는 목적이 아니며, 서비스 기획을 위한 인풋(Input)이다"라는
# 실무적 관점을 보여주는 본 프로젝트의 가장 핵심적인 단계이다.
#
# [주요 분석 단계]
#   1. 전체 분석 요약 대시보드 시각화 (임원 보고용 요약본 구성)
#   2. 3가지 맞춤형 디지털 서비스 기획안 도출 및 프레임워크 시각화
#   3. 우선순위 매트릭스 (비즈니스 임팩트 vs 실행 가능성) 시각화
#   4. 서비스 도입 시 예상되는 비즈니스 임팩트(기대 수익 방어액) 추정
#   5. 본 연구의 한계점 및 향후 과제 정리
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
sns.set_style('whitegrid')

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'Telco-Customer-Churn.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.append(os.path.join(BASE_DIR, 'src'))
from load_data_utils  import load_data
from modeling_utils   import build_features, split_data


# =============================================================================
# STEP 1. 분석 결과 종합 대시보드 시각화
# =============================================================================

def plot_analysis_summary(df: pd.DataFrame):
    """
    EDA, 군집화, 예측 모델링의 핵심 결과를 한 장의 대시보드로 요약한다.
    
    [구성 이유]
    보고서의 요약본(Executive Summary)에 들어갈 자료로, 
    경영진이나 유관 부서가 전체 분석의 흐름과 핵심 결론을 한눈에 파악할 수 있도록
    가장 중요한 차트 6개만 선별하여 재구성하였다.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        'Insurance Digital Platform — Customer Churn Analysis Summary\n'
        'Key Findings Across EDA, Clustering, and Predictive Modeling',
        fontsize=14, y=1.01
    )

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── 패널 1: 전체 이탈률 ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df['Churn'].value_counts()
    colors = ['#2196F3', '#F44336']
    wedges, texts, autotexts = ax1.pie(
        counts.values, labels=['Retained', 'Churned'],
        colors=colors, autopct='%1.1f%%',
        startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax1.set_title('Overall Churn Rate\n(n=7,032)', fontsize=10)

    # ── 패널 2: 계약 기간별 이탈률 ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bins   = [0, 12, 24, 36, 72]
    labels = ['1-12mo\n(New)', '13-24mo\n(Early)', '25-36mo\n(Mid)', '37mo+\n(Long)']
    df2 = df.copy()
    df2['tg'] = pd.cut(df2['tenure'], bins=bins, labels=labels)
    tg_churn = df2.groupby('tg', observed=True)['Churn_binary'].mean() * 100
    bar_colors = ['#F44336', '#FF9800', '#FF9800', '#2196F3']
    bars = ax2.bar(tg_churn.index, tg_churn.values,
                   color=bar_colors, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, tg_churn.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.axhline(df['Churn_binary'].mean()*100, color='gray',
                linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_title('Churn Rate by Tenure\n(Insurance: Contract Duration)', fontsize=10)
    ax2.set_ylabel('Churn Rate (%)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── 패널 3: 납입 주기별 이탈률 ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ct_churn = df.groupby('Contract')['Churn_binary'].mean() * 100
    ct_churn = ct_churn.sort_values(ascending=False)
    bar_colors3 = ['#F44336', '#FF9800', '#2196F3']
    bars3 = ax3.bar(ct_churn.index, ct_churn.values,
                    color=bar_colors3, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars3, ct_churn.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_title('Churn Rate by Contract Type\n(Insurance: Payment Frequency)', fontsize=10)
    ax3.set_ylabel('Churn Rate (%)')
    ax3.set_xticklabels(ct_churn.index, rotation=10, ha='right', fontsize=8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ── 패널 4: 모델 성능 비교 ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    models  = ['Logistic\nRegression', 'XGBoost']
    auc_vals = [0.8347, 0.8316]
    f1_vals  = [0.6066, 0.6149]
    x, w = np.arange(2), 0.3
    bars4a = ax4.bar(x - w/2, auc_vals, w, label='AUC-ROC',
                     color='#2196F3', alpha=0.85, edgecolor='white')
    bars4b = ax4.bar(x + w/2, f1_vals,  w, label='F1 Score',
                     color='#F44336', alpha=0.85, edgecolor='white')
    for bar in list(bars4a) + list(bars4b):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, fontsize=9)
    ax4.set_ylim(0, 1.05)
    ax4.set_title('Model Performance\n(AUC-ROC & F1)', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # ── 패널 5: SHAP 상위 중요 변수 ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    features  = ['Contract_Two year', 'tenure', 'Internet_Fiber optic',
                 'Contract_One year', 'TotalCharges', 'MonthlyCharges',
                 'Electronic check']
    shap_vals = [0.60, 0.55, 0.35, 0.29, 0.24, 0.22, 0.19]
    colors5   = ['#2196F3', '#2196F3', '#F44336', '#2196F3',
                 '#F44336', '#F44336', '#F44336']
    ax5.barh(features[::-1], shap_vals[::-1], color=colors5[::-1],
             edgecolor='white', linewidth=0.8)
    ax5.set_xlabel('Mean |SHAP Value|', fontsize=9)
    ax5.set_title('Top SHAP Features (XGBoost)\nBlue=reduces churn | Red=raises churn',
                  fontsize=10)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # ── 패널 6: 주요 군집별 위험도 요약 ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    clusters    = ['C2\n(High Risk)', 'C6\n(Med Risk)', 'C8\n(Med Risk)',
                   'C1\n(Watch)', 'C7\n(Loyal)']
    churn_rates = [55.0, 28.0, 25.0, 19.0, 5.0]
    sizes       = [2126, 858, 680, 541, 882]
    bar_colors6 = ['#F44336', '#FF9800', '#FF9800', '#FFC107', '#2196F3']
    bars6 = ax6.bar(clusters, churn_rates, color=bar_colors6,
                    edgecolor='white', linewidth=1.2)
    for bar, val, sz in zip(bars6, churn_rates, sizes):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.0f}%\n(n={sz:,})', ha='center', va='bottom', fontsize=7.5)
    ax6.set_title('Churn Rate by Key Clusters\n(K-means, K=8)', fontsize=10)
    ax6.set_ylabel('Churn Rate (%)')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '16_analysis_summary.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


# =============================================================================
# STEP 2. 서비스 기획안 프레임워크 시각화
# =============================================================================

def plot_service_proposals():
    """
    3가지 핵심 서비스 기획안을 '증거(데이터) - 기획 내용 - 목표(KPI)' 구조로 시각화한다.

    [구성 이유]
    "왜 이 서비스를 기획했는가?"라는 질문에 데이터를 근거로 대답하기 위함이다.
    제안 1: 신규 고객 이탈률(47.7%) + 고위험 군집 2 데이터에 기반함.
    제안 2: 월납 이탈률(42.7%) 및 SHAP 1위 변수(2년 약정 방어력)에 기반함.
    제안 3: 부가 서비스 미이용 시 이탈률 상승 패턴에 기반함.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    proposals = [
        {
            'title'   : 'Proposal 1\nNew Customer\nOnboarding Service',
            'color'   : '#F44336',
            'evidence': [
                'EDA: New customers (1-12mo)',
                'churn rate = 47.7%',
                '',
                'Cluster 2: 2,126 customers',
                'Month-to-month, high premium',
                'churn rate = 55%',
                '',
                'SHAP: tenure is the 2nd most',
                'important churn predictor',
            ],
            'service' : [
                '▸ In-app welcome journey',
                '  (first 90 days)',
                '▸ Personalized push alerts',
                '  at key risk milestones',
                '  (Day 30, 60, 90)',
                '▸ Live chat / chatbot',
                '  proactive outreach',
                '▸ First-year special benefit',
                '  (discount or add-on trial)',
            ],
            'kpi'     : 'Target: Reduce 1-12mo\nchurn rate 47.7% → 35%',
        },
        {
            'title'   : 'Proposal 2\nAnnual Plan\nConversion Campaign',
            'color'   : '#FF9800',
            'evidence': [
                'EDA: Month-to-month churn',
                '= 42.7%',
                'Two-year churn = 2.8%',
                '→ 15x difference',
                '',
                'SHAP: Contract_Two year is',
                'the #1 most important',
                'churn-reducing feature',
                '(|SHAP| = 0.60)',
            ],
            'service' : [
                '▸ Digital nudge campaign',
                '  for monthly payers',
                '▸ App-based calculator:',
                '  "Switch to annual & save X"',
                '▸ Limited-time annual plan',
                '  discount via push/email',
                '▸ Auto-renewal reminder',
                '  with incentive 30 days',
                '  before expiry',
            ],
            'kpi'     : 'Target: 15% of monthly\npayers convert to annual',
        },
        {
            'title'   : 'Proposal 3\nDigital Engagement\n& Add-on Activation',
            'color'   : '#2196F3',
            'evidence': [
                'EDA: TechSupport users',
                'churn 15.2% vs 41.6%',
                '(non-users) → 2.7x diff',
                '',
                'EDA: OnlineSecurity users',
                'churn 14.6% vs 41.8%',
                '',
                'SHAP: OnlineSecurity &',
                'TechSupport both appear',
                'in top 10 features',
            ],
            'service' : [
                '▸ In-app add-on discovery',
                '  (personalized, risk-based)',
                '▸ "Customers like you also',
                '  use..." recommendation',
                '▸ Free 1-month add-on trial',
                '  for at-risk segments',
                '▸ Gamified engagement:',
                '  loyalty points for',
                '  service usage',
            ],
            'kpi'     : 'Target: 20% increase in\nadd-on adoption rate',
        },
    ]

    for ax, prop in zip(axes, proposals):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Title bar
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 8.5), 10, 1.4, boxstyle='round,pad=0.1',
            facecolor=prop['color'], edgecolor='none', alpha=0.9
        ))
        ax.text(5, 9.2, prop['title'], ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

        # Evidence box
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 4.8), 10, 3.5, boxstyle='round,pad=0.1',
            facecolor='#F5F5F5', edgecolor=prop['color'], linewidth=1.5
        ))
        ax.text(5, 8.1, 'Analytical Evidence', ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=prop['color'])
        for i, line in enumerate(prop['evidence']):
            ax.text(0.3, 7.9 - i * 0.37, line, ha='left', va='center',
                    fontsize=7.8, color='#333333')

        # Service design box
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0.9), 10, 3.7, boxstyle='round,pad=0.1',
            facecolor='#E3F2FD', edgecolor=prop['color'], linewidth=1.5
        ))
        ax.text(5, 4.6, 'Service Design', ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=prop['color'])
        for i, line in enumerate(prop['service']):
            ax.text(0.3, 4.4 - i * 0.42, line, ha='left', va='center',
                    fontsize=7.5, color='#333333')

        # KPI box
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0.0), 10, 0.85, boxstyle='round,pad=0.05',
            facecolor=prop['color'], edgecolor='none', alpha=0.15
        ))
        ax.text(5, 0.42, prop['kpi'], ha='center', va='center',
                fontsize=8, color=prop['color'], fontweight='bold')

    fig.suptitle('Digital Service Proposals Based on Churn Analysis\n'
                 'Samsung Fire Insurance — Digital Platform Service Planning',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '17_service_proposals.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


# =============================================================================
# STEP 3. 실행 우선순위 매트릭스 시각화
# =============================================================================

def plot_priority_matrix():
    """
    비즈니스 임팩트와 실행 가능성을 기준으로 기획안의 우선순위를 2x2 매트릭스로 평가한다.

    [구성 이유]
    실제 서비스 기획 부서에서는 한정된 자원(시간, 예산, 개발 인력) 하에서 
    어떤 과제부터 실행할지 결정해야 한다. 모델 개발을 넘어 실무적 의사결정까지 
    제안함으로써 분석의 가치를 높인다.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    initiatives = [
        # (feasibility, impact, label, color, size)
        (7.5, 8.5, 'Annual Plan\nConversion\nCampaign',    '#FF9800', 300),
        (8.0, 9.0, 'New Customer\nOnboarding\n(Push Alert)', '#F44336', 350),
        (6.5, 7.5, 'Add-on\nActivation\nCampaign',         '#2196F3', 280),
        (5.0, 8.0, 'AI Churn\nPrediction\nDashboard',      '#9C27B0', 260),
        (8.5, 5.5, 'Loyalty\nPoints\nProgram',              '#4CAF50', 220),
        (4.0, 7.0, 'Personalized\nPricing\nEngine',         '#607D8B', 200),
    ]

    for fx, fy, label, color, size in initiatives:
        ax.scatter(fx, fy, s=size, color=color, alpha=0.85,
                   edgecolors='white', linewidth=1.5, zorder=3)
        ax.annotate(label, (fx, fy),
                    textcoords='offset points', xytext=(10, 5),
                    fontsize=8, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, alpha=0.8))

    ax.axhline(6.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axvline(6.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)

    ax.text(3.5, 9.3, 'Strategic Bets\n(Plan carefully)',
            ha='center', fontsize=9, color='gray', style='italic')
    ax.text(8.5, 9.3, '✓ Quick Wins\n(Do first)',
            ha='center', fontsize=9, color='#F44336', fontweight='bold')
    ax.text(3.5, 4.0, 'Deprioritize',
            ha='center', fontsize=9, color='lightgray', style='italic')
    ax.text(8.5, 4.0, 'Fill-ins\n(Do if time permits)',
            ha='center', fontsize=9, color='gray', style='italic')

    ax.set_xlim(2, 10)
    ax.set_ylim(3, 10)
    ax.set_xlabel('Implementation Feasibility →', fontsize=11)
    ax.set_ylabel('Business Impact (Churn Reduction) →', fontsize=11)
    ax.set_title('Service Initiative Priority Matrix\n'
                 'Based on Churn Analysis Findings',
                 fontsize=12, pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '18_priority_matrix.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  저장 완료: {path}")


# =============================================================================
# STEP 4. 기대 효과 (비즈니스 임팩트) 추정
# =============================================================================

def estimate_business_impact(df: pd.DataFrame):
    """
    제안한 서비스 기획안이 KPI를 달성했을 때 보전할 수 있는 예상 수익을 산출한다.

    [구성 이유]
    "이 분석으로 회사는 얼마를 벌 수 있는가?"에 답하는 파트이다. 
    투명성을 위해 가정(Assumptions)을 명확히 밝히고 보수적으로 추정하여 논리를 뒷받침한다.
    """
    sep = "=" * 65

    n_total       = len(df)
    n_churners    = df['Churn_binary'].sum()
    avg_monthly   = df['MonthlyCharges'].mean()

    # 군집 2 (최고위험군) 지표 요약
    c2_size       = 2126
    c2_avg_charge = 87.0

    print(f"\n{sep}")
    print("[ 비즈니스 임팩트 (기대 효과) 추정 ]")
    print(sep)
    print(f"\n  [현재 데이터셋 기준선 (Baseline)]")
    print(f"    전체 고객 수       : {n_total:,}명")
    print(f"    현재 이탈 고객 수  : {n_churners:,}명 ({n_churners/n_total:.1%})")
    print(f"    고객 평균 월 납입액: ${avg_monthly:.1f}")

    print(f"\n  [제안 1] 신규 고객 온보딩 집중 케어")
    print(f"  목표: 가입 1년 내 이탈률을 47.7%에서 35.0%로 감축")
    n_new   = df[df['tenure'] <= 12].shape[0]
    current = int(n_new * 0.477)
    target  = int(n_new * 0.350)
    saved   = current - target
    revenue = saved * avg_monthly * 12
    print(f"    대상 신규 고객 수       : {n_new:,}명")
    print(f"    예상 방어(유지) 고객 수 : {saved:,}명")
    print(f"    추정 연간 수익 방어액   : ${revenue:,.0f}")

    print(f"\n  [제안 2] 장기 약정(연납) 전환 캠페인")
    print(f"  목표: 월납 고객의 15%를 장기 약정으로 유도 (이탈률 42.7% → 11.3% 효과)")
    n_monthly  = (df['Contract'] == 'Month-to-month').sum()
    n_convert  = int(n_monthly * 0.15)
    churn_diff = 0.427 - 0.113  
    saved2     = int(n_convert * churn_diff)
    revenue2   = saved2 * avg_monthly * 12
    print(f"    대상 월납 고객 수       : {n_monthly:,}명")
    print(f"    예상 방어(유지) 고객 수 : {saved2:,}명")
    print(f"    추정 연간 수익 방어액   : ${revenue2:,.0f}")

    print(f"\n  [제안 3] 디지털 부가 서비스(특약) 활성화 프로모션")
    print(f"  목표: 고위험 군집(C2) 내 부가서비스 가입률 20% 향상 (이탈률 41.8% → 14.6% 효과)")
    n_atrisk   = c2_size
    n_addons   = int(n_atrisk * 0.20)
    churn_diff3 = 0.418 - 0.146 
    saved3     = int(n_addons * churn_diff3)
    revenue3   = saved3 * c2_avg_charge * 12
    print(f"    대상 고위험 고객 수     : {n_atrisk:,}명")
    print(f"    예상 방어(유지) 고객 수 : {saved3:,}명")
    print(f"    추정 연간 수익 방어액   : ${revenue3:,.0f}")

    total_revenue = revenue + revenue2 + revenue3
    total_saved   = saved + saved2 + saved3
    print(f"\n  ─────────────────────────────────────────────────────")
    print(f"  총합 기대 효과 (3가지 제안 동시 달성 시):")
    print(f"    총 유지 성공 예상 고객 수: {total_saved:,}명")
    print(f"    총 연간 추정 수익 방어액 : ${total_revenue:,.0f}")
    print(f"\n  * 가정: 월 평균 보험료 ${avg_monthly:.1f}, 12개월 유지 기준")
    print(f"  * 한계: 프로모션 실행/개발 비용이 제외된 보수적 추정치이며, 실제 효과는 A/B 테스트를 요함.")


# =============================================================================
# STEP 5. 연구 한계점 및 향후 과제
# =============================================================================

def print_limitations():
    """본 연구가 지닌 한계점과 이를 보완하기 위한 현업에서의 발전 방향을 출력한다."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("[ 분석 한계점 및 향후 과제 (Limitations & Future Work) ]")
    print(sep)

    print("""
  [한계점 (Limitations)]
  1. 데이터 도메인의 한계 (통신 vs 보험)
     - 실제 보험사 앱 로그 데이터의 부재로 인해, 구조가 유사한 통신사 구독 데이터를 대체 활용함.
     - 사고/청구 이력(Claim history), 언더라이팅 위험 스코어, 가입 채널(설계사 vs 다이렉트) 등
       보험업에 특화된 핵심 변수가 누락되어 있음.

  2. 시간적 동태성(Temporal structure) 미반영
     - 이탈은 시간에 따라 변화하는 동태적 사건이나, 본 연구의 분류 모델은 시계열 특성을 배제함.
     - 고객이 정확히 '언제' 이탈할지 예측하는 데는 한계가 존재함.

  3. 클래스 불균형 처리의 한계
     - 가중치 조정(class_weight) 기법을 사용했으나, 실제 비즈니스 환경에서는 
       이탈자를 놓치는 비용(오답)과 비이탈자에게 과도한 혜택을 주는 비용이 다르므로
       비용 행렬(Cost Matrix)에 기반한 최적화가 필요함.

  [향후 과제 (Future Work)]
  1. 보험 특화 변수 통합 분석
     - 실제 회사 내부 데이터웨어하우스(DW)에 연동하여 청구 빈도, 갱신일 도래 시점 등을 추가 학습.

  2. 생존 분석(Survival Analysis) 모델 도입
     - 이탈 여부(Binary)가 아닌 '이탈까지 걸리는 시간(Time-to-churn)'을 예측하는 Cox 모델을 적용하여
       가장 적절한 개입(Intervention) 타이밍을 도출.

  3. A/B 테스트 프레임워크 구축 및 실시간 연동
     - 도출된 기획안을 일괄 적용하지 않고, 통제군/실험군으로 나누어 실제 리텐션 상승 효과를 검증.
     - 최종 모델을 API 형태로 백엔드에 배포하여, 앱 접속 시 실시간으로 이탈 위험도를 스코어링하고
       맞춤형 푸시 알림을 자동 발송하는 파이프라인 구축.
    """)


# =============================================================================
# 실행
# =============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("  보험 디지털 플랫폼 고객 이탈 분석")
    print("  STEP 5: 분석 결과 종합 및 서비스 기획안 도출")
    print("=" * 65)

    df = load_data(DATA_PATH)

    print("\n[1] 전체 분석 결과 요약 대시보드 시각화 중...")
    plot_analysis_summary(df)

    print("\n[2] 서비스 기획안 프레임워크 시각화 중...")
    plot_service_proposals()

    print("\n[3] 실행 우선순위 매트릭스 시각화 중...")
    plot_priority_matrix()

    estimate_business_impact(df)
    print_limitations()

    print("\n" + "=" * 65)
    print("✅ 05_interpretation.py 실행 완료")
    print("   모든 시각화 이미지 저장 완료: reports/figures/")
    print("\n   [최종 생성된 이미지 목록]")
    figures_dir = os.path.join(BASE_DIR, 'reports', 'figures')
    for f in sorted(os.listdir(figures_dir)):
        if f.endswith('.png'):
            print(f"   - {f}")
    print("=" * 65)