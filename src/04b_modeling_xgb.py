# =============================================================================
# 04b_modeling_xgb.py
# Churn Prediction — XGBoost + SHAP Analysis
#
# [Why XGBoost after Logistic Regression?]
# Logistic Regression assumes a LINEAR relationship between features and
# log-odds of churn. XGBoost makes no such assumption — it builds hundreds
# of decision trees that capture non-linear patterns and feature interactions.
#
# Example of what LR cannot capture but XGBoost can:
#   "High MonthlyCharges ONLY increase churn for Month-to-month customers"
#   (interaction between MonthlyCharges × Contract type)
# XGBoost handles this naturally through tree splits.
#
# [Why SHAP?]
# XGBoost is a "black box" — we can't read coefficients like LR.
# SHAP (SHapley Additive exPlanations) solves this by computing each
# feature's marginal contribution to each individual prediction.
# Origin: Shapley values from cooperative game theory.
# Intuition: "How much did feature X contribute to THIS customer's
#             churn probability vs the average prediction?"
#
# [Structure]
#   1. Train XGBoost
#   2. Evaluate & compare with Logistic Regression
#   3. SHAP global importance (bar + beeswarm)
#   4. SHAP waterfall — individual prediction explanation
#   5. Summary & service planning implications
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
# STEP 1. Train XGBoost
# =============================================================================

def train_xgboost(X_train, y_train):
    """
    Train XGBoost with key hyperparameters.

    Hyperparameter rationale:
    - n_estimators=300: number of boosting rounds (trees)
    - max_depth=4: shallow trees reduce overfitting on tabular data
    - learning_rate=0.05: small step size → needs more trees, but generalizes better
    - subsample=0.8: use 80% of rows per tree → acts as regularization
    - colsample_bytree=0.8: use 80% of features per tree → reduces correlation between trees
    - scale_pos_weight: (# negatives) / (# positives) → compensates for class imbalance
      Without this, XGBoost ignores the minority class (churners)
    """
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  scale_pos_weight = {scale_pos:.2f}  (compensates for 73/27 class imbalance)")

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
    print(f"  Trained: {model.n_estimators} trees, max_depth={model.max_depth}")
    return model


# =============================================================================
# STEP 2. Evaluate & Compare with LR
# =============================================================================

def evaluate_xgb(model, X_train, X_test, y_train, y_test):
    """Compute standard classification metrics + 5-fold CV AUC."""
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

    print(f"\n  [XGBoost]")
    for k, v in metrics.items():
        print(f"    {k:<12}: {v:.4f}")
    print(f"    CV AUC (5-fold): {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    return metrics, y_pred, y_prob


def plot_model_comparison(y_test, lr_prob, xgb_prob, lr_metrics, xgb_metrics):
    """
    ROC curve + metrics bar chart comparing LR vs XGBoost.

    How to interpret AUC difference:
    - Δ AUC < 0.02 → linear model is sufficient; prefer LR for interpretability
    - Δ AUC > 0.02 → meaningful non-linear patterns exist; XGBoost adds value
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC curves
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

    # Metrics bar chart
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
    print(f"  Saved: {path}")


# =============================================================================
# STEP 3. SHAP Global Importance
# =============================================================================

def run_shap(model, X_test, feature_names):
    """
    Compute SHAP values using TreeExplainer.

    TreeExplainer is exact (not approximate) for tree-based models,
    unlike permutation importance which is slow and stochastic.

    SHAP value = how much feature X pushes this prediction
                 above or below the baseline (average prediction)
    """
    print("  Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


def plot_shap_bar(shap_values, X_test, feature_names):
    """
    Bar chart of mean |SHAP| per feature — global importance.
    Higher bar = feature has larger average impact on predictions.
    """
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
    print(f"  Saved: {path}")


def plot_shap_beeswarm(shap_values, X_test, feature_names):
    """
    Beeswarm plot — shows direction AND magnitude of each feature's effect.

    How to read:
    - Each dot = one customer
    - X position = SHAP value (impact on churn probability)
    - Color: RED = high feature value, BLUE = low feature value
    - Right side (+): this feature INCREASED churn risk for this customer
    - Left side (−): this feature DECREASED churn risk

    Example reading:
    - 'tenure' row: blue dots (short tenure) on the right
      → short contract duration increases churn probability ✓
    - 'Contract_Two year' row: red dots (has 2yr contract) on the left
      → 2-year contract decreases churn probability ✓
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
    print(f"  Saved: {path}")


# =============================================================================
# STEP 4. SHAP Waterfall — Individual Explanation
# =============================================================================

def plot_shap_waterfall(model, shap_values, X_test, feature_names):
    """
    Waterfall plot for individual customers.
    Shows step-by-step how each feature contributes to the final prediction.

    We show two contrasting cases:
    1. HIGH-RISK customer (churn prob > 0.8) → what's driving their risk?
    2. LOW-RISK customer  (churn prob < 0.2) → what's protecting them?

    Insurance use case:
    Customer service agents can use this to understand:
    "Why did the system flag this customer?" → targeted intervention
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
        print(f"  Saved: {path}")


# =============================================================================
# STEP 5. Summary
# =============================================================================

def print_summary(lr_metrics, xgb_metrics, shap_values, feature_names):
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)

    sep = "=" * 60
    print(f"\n{sep}")
    print("[ Model Comparison Summary ]")
    print(sep)
    print(f"  {'Metric':<12} {'LR':>10} {'XGBoost':>10} {'Δ':>8}")
    print("-" * 45)
    for m in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']:
        delta = xgb_metrics[m] - lr_metrics[m]
        print(f"  {m:<12} {lr_metrics[m]:>10.4f} {xgb_metrics[m]:>10.4f} {delta:>+8.4f}")

    diff = xgb_metrics['AUC-ROC'] - lr_metrics['AUC-ROC']
    print(f"\n  AUC difference: {diff:+.4f}")
    if abs(diff) < 0.02:
        print("  → Δ AUC < 0.02: linear model sufficient for this dataset")
        print("    Logistic Regression preferred (equal performance + interpretability)")
    else:
        print("  → Δ AUC > 0.02: XGBoost captures meaningful non-linear patterns")
        print("    Use XGBoost for prediction, LR coefficients for interpretation")

    print(f"\n{sep}")
    print("[ SHAP Top 10 Features — Insurance Domain Interpretation ]")
    print(sep)
    interp_map = {
        'tenure'              : 'Contract duration → longer = lower churn risk',
        'Contract_Two year'   : '2-year plan → strongest retention factor',
        'Contract_One year'   : '1-year plan → moderate retention effect',
        'MonthlyCharges'      : 'Monthly premium → higher = more at-risk',
        'TotalCharges'        : 'Cumulative premium paid (correlated with tenure)',
        'Internet_Fiber optic': 'Premium digital service → high churn segment',
        'Internet_No'         : 'No digital channel → low churn risk',
        'OnlineSecurity_enc'  : 'Add-on coverage → reduces churn',
        'TechSupport_enc'     : 'Customer service usage → reduces churn',
        'PaperlessBilling_enc': 'Digital billing user → slightly higher churn',
    }
    for feat, val in mean_abs.head(10).items():
        interp = interp_map.get(feat, '')
        print(f"  {feat:<28} |SHAP|={val:.4f}  {interp}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Insurance Digital Platform - Customer Churn Analysis")
    print("  STEP 4b: XGBoost + SHAP Analysis")
    print("=" * 60)

    df = load_data(DATA_PATH)
    X, y = build_features(df)
    feature_names = X.columns.tolist()
    X_arr = X.values

    print("\n[1] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X_arr, y)

    print("\n[2] Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    print("\n[3] Evaluating XGBoost...")
    xgb_metrics, xgb_pred, xgb_prob = evaluate_xgb(
        xgb_model, X_train, X_test, y_train, y_test)

    # Load LR for comparison
    lr_model_path = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
    if os.path.exists(lr_model_path):
        print("\n[4] Comparing with Logistic Regression...")
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
        print("  LR model not found — run 04_modeling.py first")
        lr_metrics = None

    print("\n[5] Running SHAP analysis...")
    explainer, shap_values = run_shap(xgb_model, X_test, feature_names)

    print("\n[6] Plotting SHAP visualizations...")
    plot_shap_bar(shap_values, X_test, feature_names)
    plot_shap_beeswarm(shap_values, X_test, feature_names)
    plot_shap_waterfall(xgb_model, shap_values, X_test, feature_names)

    if lr_metrics:
        print_summary(lr_metrics, xgb_metrics, shap_values, feature_names)

    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    print(f"\n  Model saved: models/xgboost_model.pkl")

    print("\n" + "=" * 60)
    print("✅ 04b_modeling_xgb.py complete")
    print("   -> Next: python src/05_interpretation.py")
    print("=" * 60)