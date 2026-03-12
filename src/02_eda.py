# =============================================================================
# 02_eda.py
# Exploratory Data Analysis (EDA)
#
# [Analysis Purpose]
# Visualize key patterns related to customer churn and identify
# core variables for clustering and modeling.
#
# [Structure]
#   1. Numeric variable distributions by churn status
#   2. Churn rate by categorical variables
#   3. Correlation heatmap
#   4. Churn pattern by tenure group (key insight)
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# -- 폰트 설정: 한글 대신 영문 사용 (OS 무관하게 안전) --
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
sns.set_style('whitegrid')

# 경로 설정
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'Telco-Customer-Churn.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.append(os.path.join(BASE_DIR, 'src'))
from load_data_utils import load_data


# =============================================================================
# EDA 1. Numeric variable distributions by churn status
# =============================================================================

def plot_numeric_distributions(df: pd.DataFrame):
    """
    Compare KDE distributions of tenure, MonthlyCharges, TotalCharges
    between churned and retained customers.

    Insurance domain interpretation:
    - Short tenure + high churn → need early onboarding service
    - High MonthlyCharges + high churn → need benefits for premium customers
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
            subset.plot.kde(ax=ax, label=legend_labels[churn_val],
                            color=color, linewidth=2)
            ax.axvline(subset.mean(), color=color, linestyle='--',
                       alpha=0.6, linewidth=1)
        ax.set_title(label, fontsize=10, pad=10)
        ax.set_xlabel('')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Numeric Variable Distributions: Churned vs Retained\n'
                 '(Insurance: Contract Duration & Premium)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '02_numeric_distributions.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# EDA 2. Churn rate by categorical variables
# =============================================================================

def plot_categorical_churn_rates(df: pd.DataFrame):
    """
    Bar charts showing churn rate per category for key categorical variables.

    Insurance domain interpretation:
    - Contract type (Month-to-month = monthly premium) → high churn risk
    - PaperlessBilling → digital channel usage pattern
    - TechSupport → customer service usage
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
    avg_churn = df['Churn_binary'].mean() * 100

    for ax, (col, label) in zip(axes, cat_cols):
        churn_rate = (df.groupby(col)['Churn_binary']
                        .mean()
                        .sort_values(ascending=False) * 100)

        bars = ax.bar(range(len(churn_rate)), churn_rate.values,
                      color=['#F44336' if v > 30 else '#2196F3'
                             for v in churn_rate.values],
                      edgecolor='white', linewidth=1.2)

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
    print(f"  Saved: {path}")


# =============================================================================
# EDA 3. Correlation heatmap
# =============================================================================

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Heatmap of correlations among numeric variables.

    Key findings to look for:
    - tenure vs Churn_binary: negative correlation (longer contract = lower churn)
    - MonthlyCharges vs Churn_binary: positive correlation (higher premium = higher churn)
    - tenure vs TotalCharges: high positive (multicollinearity → watch in modeling)
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
    print(f"  Saved: {path}")


# =============================================================================
# EDA 4. Churn pattern by tenure group (key insight)
# =============================================================================

def plot_tenure_churn_pattern(df: pd.DataFrame):
    """
    Churn rate by tenure group — the most critical insight for service planning.

    Insurance domain:
    - If churn is highest in 1~12 months (new customers)
      → Onboarding service & first-year intensive care program needed
      → Directly applicable to Samsung Fire's digital platform planning
    """
    bins   = [0, 12, 24, 36, 48, 60, 72]
    labels = ['1-12mo\n(New)', '13-24mo\n(Early)', '25-36mo\n(Mid)',
              '37-48mo\n(Stable)', '49-60mo\n(Long)', '61-72mo\n(Loyal)']

    df = df.copy()
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    group_stats = df.groupby('tenure_group', observed=True).agg(
        churn_rate=('Churn_binary', lambda x: x.mean() * 100),
        count=('Churn_binary', 'count')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(group_stats['tenure_group'], group_stats['churn_rate'],
                   color=['#F44336' if v > 30 else '#FF9800' if v > 20 else '#2196F3'
                          for v in group_stats['churn_rate']],
                   alpha=0.8, edgecolor='white', linewidth=1.2, zorder=2)

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
    print(f"  Saved: {path}")


# =============================================================================
# EDA summary
# =============================================================================

def print_eda_summary(df: pd.DataFrame):
    """Print key EDA insights mapped to insurance domain."""
    sep = "=" * 60

    print(f"\n{sep}")
    print("[ EDA Key Insights — Insurance Domain Interpretation ]")
    print(sep)

    bins   = [0, 12, 24, 36, 72]
    labels = ['New(1-12mo)', 'Early(13-24mo)', 'Mid(25-36mo)', 'Long(37mo+)']
    df2 = df.copy()
    df2['tenure_group'] = pd.cut(df2['tenure'], bins=bins, labels=labels)
    tenure_churn = df2.groupby('tenure_group', observed=True)['Churn_binary'].mean() * 100
    print("\n[1] Churn Rate by Tenure (Insurance: Contract Duration)")
    for g, r in tenure_churn.items():
        print(f"    {g}: {r:.1f}%")

    contract_churn = df.groupby('Contract')['Churn_binary'].mean() * 100
    print("\n[2] Churn Rate by Contract Type (Insurance: Payment Frequency)")
    for c, r in contract_churn.sort_values(ascending=False).items():
        print(f"    {c}: {r:.1f}%")

    pb_churn = df.groupby('PaperlessBilling')['Churn_binary'].mean() * 100
    print("\n[3] Churn Rate by Paperless Billing (Digital Channel Usage)")
    for c, r in pb_churn.items():
        print(f"    {'Yes (Digital)' if c=='Yes' else 'No (Paper)'}: {r:.1f}%")

    df2['charge_group'] = pd.cut(df['MonthlyCharges'],
                                  bins=[0, 35, 65, 95, 120],
                                  labels=['Low(~35)', 'Mid(35-65)',
                                          'High(65-95)', 'VeryHigh(95+)'])
    charge_churn = df2.groupby('charge_group', observed=True)['Churn_binary'].mean() * 100
    print("\n[4] Churn Rate by Monthly Charge Group (Premium Level)")
    for c, r in charge_churn.items():
        print(f"    {c}: {r:.1f}%")

    print(f"\n{sep}")
    print("[ Service Planning Implications ]")
    print(sep)
    print("  1. New customers (1-12mo) have the highest churn rate (~47%)")
    print("     -> Need onboarding strengthening & first-year care service")
    print("  2. Month-to-month churn (42.7%) >> Two-year churn (2.8%)")
    print("     -> Promote annual payment plan conversion")
    print("  3. High-premium customers show elevated churn risk")
    print("     -> Personalized benefits for high-premium segment")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Insurance Digital Platform - Customer Churn Analysis")
    print("  STEP 2: Exploratory Data Analysis (EDA)")
    print("=" * 60)

    df = load_data(DATA_PATH)

    print("\nGenerating visualizations...")
    plot_numeric_distributions(df)
    plot_categorical_churn_rates(df)
    plot_correlation_heatmap(df)
    plot_tenure_churn_pattern(df)

    print_eda_summary(df)

    print("\n" + "=" * 60)
    print("✅ 02_eda.py complete")
    print("   Figures saved in: reports/figures/")
    print("   -> Next: python src/03_clustering.py")
    print("=" * 60)