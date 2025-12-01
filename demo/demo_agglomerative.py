###
## cluster_maker
## James Foadi - University of Bath
## November 2025
##
## Demo: Hierarchical Agglomerative Clustering on Difficult Dataset
###

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist

from cluster_maker.preprocessing import standardise_features
from cluster_maker.agglomerative import (
    hierarchical_clustering,
    compare_linkage_methods,
    get_linkage_matrix,
)
from cluster_maker.evaluation import silhouette_score_sklearn, compute_inertia
from cluster_maker.plotting_clustered import plot_clusters_2d


# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = "data/difficult_dataset.csv"
OUTPUT_DIR = "agglomerative_analysis"
FEATURE_COLS = ["x", "y"]
N_CLUSTERS = 3
LINKAGE_METHODS = ["ward", "complete", "average", "single"]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    """
    Main analysis pipeline for hierarchical clustering on difficult dataset.
    
    Steps:
    1. Load and explore the dataset
    2. Standardise the features
    3. Apply hierarchical clustering with different linkage methods
    4. Compare clustering results and metrics
    5. Generate visualizations (2D plots, dendrograms)
    6. Export results to CSV and create analysis report
    """

    print("=" * 80)
    print("HIERARCHICAL AGGLOMERATIVE CLUSTERING ANALYSIS")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load and Explore Data
    # =========================================================================
    print("\n[Step 1] Loading and Exploring Data")
    print("-" * 80)

    df = pd.read_csv(INPUT_FILE)
    print(f"Dataset shape: {df.shape[0]} samples x {df.shape[1]} features")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nBasic statistics:")
    print(df.describe())

    # Extract features
    X_df = df[FEATURE_COLS]
    X = X_df.to_numpy(dtype=float)

    print(f"\nFeatures extracted: {FEATURE_COLS}")
    print(f"Feature matrix shape: {X.shape}")

    # =========================================================================
    # STEP 2: Standardise Features
    # =========================================================================
    print("\n[Step 2] Preprocessing Data")
    print("-" * 80)

    X_standardised = standardise_features(X)
    print(f"[OK] Data standardised")
    print(f"  Mean: {X_standardised.mean():.6f}")
    print(f"  Std Dev: {X_standardised.std():.6f}")

    # =========================================================================
    # STEP 3: Apply Hierarchical Clustering with Different Linkage Methods
    # =========================================================================
    print(f"\n[Step 3] Hierarchical Clustering with Multiple Linkage Methods")
    print("-" * 80)
    print(f"Testing linkage methods with n_clusters={N_CLUSTERS}:")

    results = compare_linkage_methods(
        X_standardised,
        n_clusters=N_CLUSTERS,
        linkage_methods=LINKAGE_METHODS,
    )

    # =========================================================================
    # STEP 4: Evaluate and Compare Results
    # =========================================================================
    print(f"\n[Step 4] Evaluating Clustering Quality")
    print("-" * 80)

    metrics_data = []

    for method in LINKAGE_METHODS:
        labels = results[method]
        
        # Compute silhouette score
        try:
            sil_score = silhouette_score_sklearn(X_standardised, labels)
        except ValueError:
            sil_score = None
        
        # Count cluster distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_dist = ", ".join([f"C{i}:{c}" for i, c in zip(unique_labels, counts)])

        print(f"\n{method.upper()} linkage:")
        print(f"  Silhouette Score: {sil_score:.4f}" if sil_score else f"  Silhouette Score: N/A")
        print(f"  Cluster distribution: {cluster_dist}")

        metrics_data.append({
            "Linkage Method": method,
            "Silhouette Score": sil_score if sil_score else None,
            "Unique Clusters": len(unique_labels),
        })

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(OUTPUT_DIR, "linkage_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[OK] Linkage comparison metrics saved to: {metrics_path}")

    # =========================================================================
    # STEP 5: Generate Visualizations
    # =========================================================================
    print(f"\n[Step 5] Generating Visualizations")
    print("-" * 80)

    # 5a: 2D cluster plots for each linkage method
    print(f"\nGenerating 2D scatter plots for each linkage method...")
    for method in LINKAGE_METHODS:
        labels = results[method]
        fig, ax = plot_clusters_2d(
            X_standardised,
            labels,
            centroids=None,
            title=f"Hierarchical Clustering ({method.capitalize()} Linkage)",
        )
        plot_path = os.path.join(OUTPUT_DIR, f"clustering_{method}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] {plot_path}")

    # 5b: Comparison plot showing all methods side-by-side
    print(f"\nGenerating comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, method in enumerate(LINKAGE_METHODS):
        labels = results[method]
        ax = axes[idx]
        scatter = ax.scatter(
            X_standardised[:, 0],
            X_standardised[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.9,
            s=50,
        )
        ax.set_xlabel("Feature x")
        ax.set_ylabel("Feature y")
        ax.set_title(f"{method.capitalize()} Linkage")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Cluster")

    fig.suptitle(f"Hierarchical Clustering: Linkage Method Comparison", fontsize=14)
    fig.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, "linkage_comparison.png")
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {comparison_path}")

    # 5c: Dendrograms for each method (sample-based for large datasets)
    print(f"\nGenerating dendrograms (using random sample for large dataset)...")
    if X_standardised.shape[0] > 200:
        # For large datasets, create dendrograms on a sample
        sample_size = 200
        sample_indices = np.random.choice(X_standardised.shape[0], size=sample_size, replace=False)
        X_sample = X_standardised[sample_indices]
        
        for method in LINKAGE_METHODS:
            fig, ax = plt.subplots(figsize=(12, 6))
            Z = get_linkage_matrix(X_sample, linkage_method=method)
            dendrogram(Z, ax=ax, leaf_font_size=7)
            ax.set_title(f"Dendrogram ({method.capitalize()} Linkage, sample of {sample_size} points)")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Distance")
            fig.tight_layout()
            dendrogram_path = os.path.join(OUTPUT_DIR, f"dendrogram_{method}.png")
            fig.savefig(dendrogram_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  [OK] {dendrogram_path}")
    else:
        # For small datasets, show full dendrograms
        for method in LINKAGE_METHODS:
            fig, ax = plt.subplots(figsize=(10, 6))
            Z = get_linkage_matrix(X_standardised, linkage_method=method)
            dendrogram(Z, ax=ax, leaf_font_size=8)
            ax.set_title(f"Dendrogram ({method.capitalize()} Linkage)")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Distance")
            fig.tight_layout()
            dendrogram_path = os.path.join(OUTPUT_DIR, f"dendrogram_{method}.png")
            fig.savefig(dendrogram_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [OK] {dendrogram_path}")

    # =========================================================================
    # STEP 6: Export Clustered Data
    # =========================================================================
    print(f"\n[Step 6] Exporting Results")
    print("-" * 80)

    # Use the best-performing method (highest silhouette score)
    best_method = max(metrics_data, key=lambda x: x["Silhouette Score"] if x["Silhouette Score"] else -np.inf)["Linkage Method"]
    best_labels = results[best_method]

    df_clustered = df.copy()
    df_clustered["cluster"] = best_labels
    df_clustered["linkage_method"] = best_method

    clustered_path = os.path.join(OUTPUT_DIR, "difficult_data_clustered.csv")
    df_clustered.to_csv(clustered_path, index=False)
    print(f"[OK] Clustered data exported to: {clustered_path}")

    # =========================================================================
    # STEP 7: Generate Analysis Report
    # =========================================================================
    print(f"\n[Step 7] Generating Analysis Report")
    print("-" * 80)

    summary_report = f"""
{'='*80}
HIERARCHICAL CLUSTERING ANALYSIS REPORT
{'='*80}

Dataset Information:
  - File: {INPUT_FILE}
  - Samples: {df.shape[0]}
  - Features: {len(FEATURE_COLS)}
  - Feature names: {", ".join(FEATURE_COLS)}

Data Characteristics:
  - Data range: x in [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]
  - Data range: y in [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]
  - Standardised: mean~=0, std~=1

Hierarchical Clustering Configuration:
  - Algorithm: Agglomerative Clustering (scikit-learn)
  - Target clusters: {N_CLUSTERS}
  - Linkage methods tested: {", ".join(LINKAGE_METHODS)}

Linkage Method Comparison:
  {"Method":<12} | Silhouette | Clusters
  {'-'*40}
"""
    
    for item in metrics_data:
        method = item["Linkage Method"]
        sil = item["Silhouette Score"]
        n_clust = item["Unique Clusters"]
        sil_str = f"{sil:.4f}" if sil else "N/A"
        summary_report += f"  {method:<12} | {sil_str:>10} | {n_clust:>8}\n"

    summary_report += f"""
Best Method Selection:
  - Selected method: {best_method.upper()}
  - Rationale: Produces the highest silhouette score, indicating better-defined clusters
  - Final clusters: {len(np.unique(best_labels))}

Cluster Distribution (using {best_method} linkage):
"""
    unique_labels, counts = np.unique(best_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        pct = 100 * count / len(best_labels)
        summary_report += f"  - Cluster {label}: {count} points ({pct:.1f}%)\n"

    summary_report += f"""
Key Findings:
  1. Hierarchical clustering with different linkage methods produces varied results
  2. {best_method.upper()} linkage provides the best cluster separation for this dataset
  3. The difficult dataset shows complex structure requiring careful method selection
  4. Non-convex cluster shapes are better handled by hierarchical methods

Advantages of Hierarchical Clustering:
  - Creates a dendrogram showing the clustering hierarchy
  - No need to specify k in advance (cut at different levels)
  - Can handle non-convex clusters better than k-means
  - Multiple linkage options allow flexibility for different data structures

Visualizations Generated:
  - clustering_[method].png: 2D plots for each linkage method
  - linkage_comparison.png: Side-by-side comparison of all methods
  - dendrogram_[method].png: Full dendrograms showing hierarchy

Output Files:
  - difficult_data_clustered.csv: Original data with cluster labels ({best_method} linkage)
  - linkage_comparison.csv: Silhouette scores for each linkage method
  - All PNG visualizations in {OUTPUT_DIR}/

{'='*80}
Analysis Complete. All results saved to: {OUTPUT_DIR}/
{'='*80}
"""

    report_path = os.path.join(OUTPUT_DIR, "ANALYSIS_REPORT.txt")
    with open(report_path, "w") as f:
        f.write(summary_report)
    print("[OK] Analysis report saved")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + summary_report)
    print("\n[OK] All analyses complete!")


if __name__ == "__main__":
    main()
