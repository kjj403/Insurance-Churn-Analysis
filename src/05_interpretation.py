# =============================================================================
# 05_interpretation.py
# Results Synthesis & Digital Service Planning Proposals
#
# [Purpose]
# This file synthesizes findings from EDA, clustering, and modeling
# into concrete digital service planning proposals.
#
# This is the most important step for Samsung Fire's Service Planning role:
# "Data analysis is not the end goal — it is the input to service design."
#
# [Structure]
#   1. Full analysis summary (EDA + Clustering + Modeling)
#   2. Three actionable digital service proposals
#   3. Priority matrix (impact vs feasibility)
#   4. Expected business impact estimation
#   5. Limitations & future work
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
# SECTION 1. Analysis Summary Dashboard
# =============================================================================

def plot_analysis_summary(df: pd.DataFrame):
    """
    One-page summary dashboard combining key findings from all analysis steps.
    Designed for inclusion in the final report as an executive summary figure.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        'Insurance Digital Platform — Customer Churn Analysis Summary\n'
        'Key Findings Across EDA, Clustering, and Predictive Modeling',
        fontsize=14, y=1.01
    )

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── Panel 1: Overall churn rate ──────────────────────────────────────────
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

    # ── Panel 2: Churn by tenure group ──────────────────────────────────────
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

    # ── Panel 3: Churn by contract type ─────────────────────────────────────
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

    # ── Panel 4: Model performance comparison ───────────────────────────────
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

    # ── Panel 5: SHAP top features ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    # Approximate SHAP values from full run
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

    # ── Panel 6: Cluster risk summary ───────────────────────────────────────
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
    print(f"  Saved: {path}")


# =============================================================================
# SECTION 2. Service Planning Proposals
# =============================================================================

def plot_service_proposals():
    """
    Visualize three digital service proposals as a structured framework.

    Each proposal is grounded in a specific analytical finding:
    Proposal 1 ← New customer churn 47.7% (EDA) + Cluster 2 (clustering)
    Proposal 2 ← Month-to-month churn 42.7% vs Two-year 2.8% (EDA + SHAP)
    Proposal 3 ← TechSupport 41.6% vs 15.2% (EDA) + SHAP top features
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
    print(f"  Saved: {path}")


# =============================================================================
# SECTION 3. Priority Matrix
# =============================================================================

def plot_priority_matrix():
    """
    2x2 priority matrix: Business Impact vs Implementation Feasibility.
    Standard framework used in product/service planning to prioritize initiatives.

    Quadrants:
    - Top Right (High Impact, High Feasibility): Do first — Quick wins
    - Top Left  (High Impact, Low Feasibility) : Plan carefully — Strategic bets
    - Bottom Right (Low Impact, High Feasibility): Do if time permits
    - Bottom Left  (Low Impact, Low Feasibility) : Deprioritize
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

    # Quadrant lines
    ax.axhline(6.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axvline(6.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)

    # Quadrant labels
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
    print(f"  Saved: {path}")


# =============================================================================
# SECTION 4. Business Impact Estimation
# =============================================================================

def estimate_business_impact(df: pd.DataFrame):
    """
    Rough estimation of business impact if churn reduction targets are met.
    Uses conservative assumptions — clearly stated for transparency.

    This section demonstrates the ability to connect analytical findings
    to business value, a core skill for service planning roles.
    """
    sep = "=" * 65

    # Basic stats
    n_total       = len(df)
    n_churners    = df['Churn_binary'].sum()
    avg_monthly   = df['MonthlyCharges'].mean()
    avg_tenure    = df['tenure'].mean()

    # Cluster 2 (highest risk) stats
    # Approximated from clustering results
    c2_size       = 2126
    c2_churn_rate = 0.55
    c2_avg_charge = 87.0

    print(f"\n{sep}")
    print("[ Business Impact Estimation ]")
    print(sep)
    print(f"\n  Dataset baseline:")
    print(f"    Total customers    : {n_total:,}")
    print(f"    Churners           : {n_churners:,} ({n_churners/n_total:.1%})")
    print(f"    Avg monthly charge : ${avg_monthly:.1f}")

    print(f"\n  Proposal 1 — New Customer Onboarding Service")
    print(f"  Target: Reduce 1-12mo churn rate from 47.7% to 35%")
    n_new   = df[df['tenure'] <= 12].shape[0]
    current = int(n_new * 0.477)
    target  = int(n_new * 0.350)
    saved   = current - target
    revenue = saved * avg_monthly * 12
    print(f"    New customers (1-12mo)     : {n_new:,}")
    print(f"    Current churners           : {current:,}")
    print(f"    After intervention         : {target:,}")
    print(f"    Customers saved            : {saved:,}")
    print(f"    Est. annual revenue saved  : ${revenue:,.0f}")

    print(f"\n  Proposal 2 — Annual Plan Conversion Campaign")
    print(f"  Target: Convert 15% of monthly payers to annual plan")
    n_monthly  = (df['Contract'] == 'Month-to-month').sum()
    n_convert  = int(n_monthly * 0.15)
    churn_diff = 0.427 - 0.113  # month-to-month vs one-year churn rate
    saved2     = int(n_convert * churn_diff)
    revenue2   = saved2 * avg_monthly * 12
    print(f"    Monthly-plan customers     : {n_monthly:,}")
    print(f"    Target conversions (15%)   : {n_convert:,}")
    print(f"    Est. additional retentions : {saved2:,}")
    print(f"    Est. annual revenue saved  : ${revenue2:,.0f}")

    print(f"\n  Proposal 3 — Add-on Activation Campaign")
    print(f"  Target: 20% increase in add-on adoption among at-risk segment")
    n_atrisk   = c2_size
    n_addons   = int(n_atrisk * 0.20)
    churn_diff3 = 0.418 - 0.146  # no security vs security user churn rate
    saved3     = int(n_addons * churn_diff3)
    revenue3   = saved3 * c2_avg_charge * 12
    print(f"    At-risk customers (Cluster 2): {n_atrisk:,}")
    print(f"    New add-on adopters (20%)    : {n_addons:,}")
    print(f"    Est. additional retentions   : {saved3:,}")
    print(f"    Est. annual revenue saved    : ${revenue3:,.0f}")

    total_revenue = revenue + revenue2 + revenue3
    total_saved   = saved + saved2 + saved3
    print(f"\n  ─────────────────────────────────────────────────────")
    print(f"  Combined Impact (all 3 proposals):")
    print(f"    Total customers retained   : {total_saved:,}")
    print(f"    Total annual revenue saved : ${total_revenue:,.0f}")
    print(f"\n  * Assumptions: avg monthly charge ${avg_monthly:.1f}, 12-month horizon")
    print(f"  * Conservative estimate — does not account for implementation cost")
    print(f"  * Actual impact subject to A/B testing and operational constraints")


# =============================================================================
# SECTION 5. Limitations & Future Work
# =============================================================================

def print_limitations():
    sep = "=" * 65
    print(f"\n{sep}")
    print("[ Limitations & Future Work ]")
    print(sep)

    print("""
  [Limitations]
  1. Data domain gap
     - Source data is telecom (IBM Telco), not insurance
     - Key insurance-specific variables are absent:
       claim history, underwriting risk score, agent channel,
       product type (life/auto/fire), renewal date proximity
     - Domain mapping (contract → payment frequency) is approximate

  2. Temporal structure not modeled
     - Churn is a time-dependent process
     - Logistic Regression & XGBoost treat each customer as i.i.d.
     - Survival analysis (Cox model) or LSTM would better capture
       the dynamics of "when" a customer churns

  3. Class imbalance handling
     - Used class_weight='balanced' and scale_pos_weight
     - More rigorous approaches: SMOTE, threshold optimization
       based on cost matrix (FN cost >> FP cost in insurance)

  4. Hyperparameter tuning
     - XGBoost parameters were manually set, not grid/random searched
     - Bayesian optimization (Optuna) could improve performance

  [Future Work]
  1. Incorporate insurance-specific features
     - Claim frequency, policy tenure, renewal proximity
     - Agent/channel type (digital vs. offline)

  2. Survival analysis
     - Model "time-to-churn" rather than binary churn label
     - Enables proactive intervention at the right moment

  3. A/B testing framework
     - Deploy churn model as a scoring API
     - Randomly assign at-risk customers to treatment/control
     - Measure actual retention lift from each service proposal

  4. Real-time scoring
     - Integrate model into Samsung Fire app backend
     - Trigger personalized interventions at predicted churn events
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("  Insurance Digital Platform - Customer Churn Analysis")
    print("  STEP 5: Results Synthesis & Service Planning Proposals")
    print("=" * 65)

    df = load_data(DATA_PATH)

    print("\n[1] Generating analysis summary dashboard...")
    plot_analysis_summary(df)

    print("\n[2] Generating service proposals visualization...")
    plot_service_proposals()

    print("\n[3] Generating priority matrix...")
    plot_priority_matrix()

    estimate_business_impact(df)
    print_limitations()

    print("\n" + "=" * 65)
    print("✅ 05_interpretation.py complete")
    print("   All figures saved in: reports/figures/")
    print("\n   Final figure list:")
    figures_dir = os.path.join(BASE_DIR, 'reports', 'figures')
    for f in sorted(os.listdir(figures_dir)):
        if f.endswith('.png'):
            print(f"   - {f}")
    print("=" * 65)