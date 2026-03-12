# =============================================================================
# 03_clustering.py
# Customer Segmentation via K-means Clustering
#
# [Purpose]
# Before predicting churn, we first segment customers into behavioral groups.
# This reflects the actual workflow of Samsung Fire's digital service planning:
#   "Understand WHO our customers are → then decide WHAT service to offer them"
#
# Insurance domain mapping:
#   Cluster 1 (e.g. New + High Premium)  → Early churn risk segment
#   Cluster 2 (e.g. Long + Low Premium)  → Loyal stable segment
#   Cluster 3 (e.g. Mid + Digital user)  → Growth potential segment
#
# [Method]
#   - Features: tenure, MonthlyCharges, TotalCharges + encoded categoricals
#   - Optimal K: Elbow method (inertia) + Silhouette score
#   - Algorithm: K-means (sklearn)
#   - Visualization: 2D scatter (PCA), cluster profile bar charts
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
# STEP 1. Feature Engineering for Clustering
# =============================================================================

def prepare_features(df: pd.DataFrame):
    """
    Select and encode features for K-means clustering.

    Feature selection rationale:
    - tenure, MonthlyCharges, TotalCharges: core behavioral signals
      (insurance: contract duration, premium level, total paid)
    - Contract: payment frequency (monthly vs annual) — strong churn predictor
    - InternetService: digital channel usage proxy
    - PaperlessBilling: digital engagement indicator

    Encoding:
    - Binary Yes/No → 1/0
    - Multi-class (Contract, InternetService) → one-hot encoding
    - All features standardized (mean=0, std=1) for K-means distance calculation
      K-means is distance-based, so scale matters — StandardScaler is essential
    """
    df = df.copy()

    # Binary Yes/No encoding
    binary_cols = ['PaperlessBilling', 'PhoneService', 'Partner', 'Dependents']
    for col in binary_cols:
        df[col + '_enc'] = (df[col] == 'Yes').astype(int)

    # Multi-class one-hot encoding
    contract_dummies = pd.get_dummies(df['Contract'], prefix='Contract')
    internet_dummies = pd.get_dummies(df['InternetService'], prefix='Internet')

    # Final feature set
    feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                    'SeniorCitizen',
                    'PaperlessBilling_enc', 'PhoneService_enc',
                    'Partner_enc', 'Dependents_enc']

    X = pd.concat([df[feature_cols], contract_dummies, internet_dummies], axis=1)

    # Standardize: K-means uses Euclidean distance, so all features must be on same scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, X.columns.tolist(), scaler


# =============================================================================
# STEP 2. Find Optimal K — Elbow + Silhouette
# =============================================================================

def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 9)):
    """
    Determine the optimal number of clusters using two complementary methods:

    1. Elbow Method (Inertia):
       - Inertia = sum of squared distances from each point to its cluster center
       - Plot inertia vs K → look for the 'elbow' where improvement slows
       - Limitation: subjective, always decreases as K increases

    2. Silhouette Score:
       - Measures how similar a point is to its own cluster vs other clusters
       - Range: [-1, 1], higher is better
       - More objective than elbow method
       - Best K = highest silhouette score

    Using both methods together gives a more reliable choice.
    """
    inertias    = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        print(f"  K={k}: Inertia={km.inertia_:.1f}, Silhouette={silhouette_score(X_scaled, labels):.4f}")

    # Plot elbow + silhouette side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Elbow
    ax1.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=7)
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)')
    ax1.set_title('Elbow Method\n(Look for the bend point)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Silhouette
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
    print(f"\n  Saved: {path}")
    print(f"  Best K by Silhouette: {best_k}")

    return best_k, inertias, silhouettes


# =============================================================================
# STEP 3. Fit Final K-means Model
# =============================================================================

def fit_kmeans(X_scaled: np.ndarray, k: int):
    """
    Fit the final K-means model with the selected K.

    n_init=10: run K-means 10 times with different centroid seeds
               and pick the best result — reduces sensitivity to initialization
    random_state=42: reproducibility
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # Save model for reuse in later steps
    model_path = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
    joblib.dump(km, model_path)
    print(f"  Model saved: {model_path}")

    return km, labels


# =============================================================================
# STEP 4. Visualize Clusters (PCA 2D)
# =============================================================================

def plot_clusters_pca(X_scaled: np.ndarray, labels: np.ndarray, k: int):
    """
    Reduce high-dimensional features to 2D using PCA for visualization.

    PCA (Principal Component Analysis):
    - Finds directions of maximum variance in the data
    - Projects all features onto 2 principal components for 2D plotting
    - Does NOT change the clustering — only used for visualization
    - Explained variance ratio tells us how much information is preserved
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    print(f"  PCA explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, "
          f"Total={sum(explained):.1%}")

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
    print(f"  Saved: {path}")


# =============================================================================
# STEP 5. Cluster Profiling
# =============================================================================

def profile_clusters(df: pd.DataFrame, labels: np.ndarray, k: int):
    """
    Analyze the characteristics of each cluster to give them business meaning.

    This is the most important step for service planning:
    - Each cluster should map to a distinct customer persona
    - Churn rate per cluster tells us which segments need intervention
    - Key variables (tenure, charges, contract type) reveal the 'why'

    Insurance domain:
    - Cluster with high churn + short tenure = new customer at-risk segment
    - Cluster with low churn + long tenure = loyal customer segment
    - etc.
    """
    df = df.copy()
    df['Cluster'] = labels + 1  # 1-indexed for readability

    # Numeric profile per cluster
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

    print("\n[ Cluster Numeric Profile ]")
    print(numeric_profile.to_string())

    # Categorical profile: Contract type distribution per cluster
    contract_profile = (df.groupby(['Cluster', 'Contract'])
                          .size()
                          .unstack(fill_value=0))
    contract_pct = contract_profile.div(contract_profile.sum(axis=1), axis=0) * 100
    print("\n[ Contract Type Distribution per Cluster (%) ]")
    print(contract_pct.round(1).to_string())

    # Visualization: cluster profile radar-style bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.Set1(np.linspace(0, 0.8, k))

    # Churn rate per cluster
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

    # Avg tenure per cluster
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

    # Avg monthly charges per cluster
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
    print(f"\n  Saved: {path}")

    return df, numeric_profile


# =============================================================================
# STEP 6. Assign Cluster Labels (Business Personas)
# =============================================================================

def assign_personas(numeric_profile: pd.DataFrame):
    """
    Assign human-readable business personas to each cluster
    based on churn rate and tenure pattern.

    This step translates statistical clusters into actionable segments
    for the service planning team — the core output of this analysis.
    """
    print("\n[ Cluster Personas — Insurance Service Planning ]")
    print("=" * 60)

    for cluster_id, row in numeric_profile.iterrows():
        churn  = row['Churn_rate']
        tenure = row['Tenure_mean']
        charge = row['Monthly_mean']
        count  = int(row['Count'])

        # Assign persona based on behavioral pattern
        if churn > 40:
            persona = "HIGH RISK — At-risk New Customers"
            action  = "Priority: Onboarding service, early engagement push notification"
        elif churn > 20:
            persona = "MEDIUM RISK — Mid-term Wavering Customers"
            action  = "Priority: Retention offer, annual plan conversion promotion"
        else:
            persona = "LOW RISK — Loyal Long-term Customers"
            action  = "Priority: Upsell add-on coverage, VIP loyalty program"

        print(f"\n  Cluster {cluster_id}: {persona}")
        print(f"    Churn Rate : {churn:.1f}%")
        print(f"    Avg Tenure : {tenure:.1f} months")
        print(f"    Avg Premium: ${charge:.1f}/month")
        print(f"    Size       : {count:,} customers")
        print(f"    Action     : {action}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Insurance Digital Platform - Customer Churn Analysis")
    print("  STEP 3: Customer Segmentation (K-means Clustering)")
    print("=" * 60)

    # Load data
    df = load_data(DATA_PATH)

    # 1. Prepare features
    print("\n[1] Preparing features...")
    X_scaled, feature_names, scaler = prepare_features(df)
    print(f"  Feature matrix: {X_scaled.shape[0]} customers × {X_scaled.shape[1]} features")

    # 2. Find optimal K
    print("\n[2] Finding optimal K...")
    best_k, inertias, silhouettes = find_optimal_k(X_scaled)

    # 3. Fit final model
    print(f"\n[3] Fitting K-means with K={best_k}...")
    km, labels = fit_kmeans(X_scaled, best_k)

    # 4. Visualize clusters (PCA)
    print("\n[4] Visualizing clusters via PCA...")
    plot_clusters_pca(X_scaled, labels, best_k)

    # 5. Profile clusters
    print("\n[5] Profiling clusters...")
    df_clustered, numeric_profile = profile_clusters(df, labels, best_k)

    # 6. Assign personas
    assign_personas(numeric_profile)

    # Save clustered data for next step
    out_path = os.path.join(BASE_DIR, 'data', 'telco_clustered.csv')
    df_clustered.to_csv(out_path, index=False)
    print(f"\n  Clustered data saved: {out_path}")

    print("\n" + "=" * 60)
    print("✅ 03_clustering.py complete")
    print("   -> Next: python src/04_modeling.py")
    print("=" * 60)