# Task 2: Demo Script Bug Fix and Package Overview

## Bug Identification and Fix

### What Was Wrong

The original `cluster_plot.py` script (line 62) contained a critical logic error:

```python
# BUGGY CODE:
k = min(k, 3),
```

This line forced all k values in the loop to be capped at 3, regardless of their intended values:
- Loop iteration k=2: `min(2, 3) = 2` ✓ (correct)
- Loop iteration k=3: `min(3, 3) = 3` ✓ (correct)
- Loop iteration k=4: `min(4, 3) = 3` ✗ (WRONG - should be 4)
- Loop iteration k=5: `min(5, 3) = 3` ✗ (WRONG - should be 5)

**Consequence:** The script only tested k=2 and k=3 independently. When k=4 and k=5 were requested, they were silently converted to k=3, producing identical metrics to the k=3 run. This prevented proper model selection and comparison across the intended range of cluster numbers.

### How It Was Fixed

Changed line 62 to:

```python
# FIXED CODE:
k=k,
```

Now each k value (2, 3, 4, 5) is used as-is, allowing proper independent clustering analysis for each cluster count.

### Verification

**Before fix** (buggy output):
```
k=2: inertia=185.32, silhouette=0.606
k=3: inertia=43.27, silhouette=0.725
k=4: inertia=43.27, silhouette=0.725    <- IDENTICAL to k=3 (BUG!)
k=5: inertia=43.27, silhouette=0.725    <- IDENTICAL to k=3 (BUG!)
```

**After fix** (corrected output):
```
k=2: inertia=185.32, silhouette=0.606
k=3: inertia=43.27, silhouette=0.725    <- Best silhouette score
k=4: inertia=26.09, silhouette=0.672    <- Continues to improve inertia
k=5: inertia=21.30, silhouette=0.611    <- Inertia lowest, silhouette declining
```

---

## What the Corrected Demo Script Does

The `cluster_plot.py` script performs a comprehensive clustering analysis pipeline:

1. **Data Loading & Validation**
   - Accepts a CSV file path as command-line argument
   - Validates file existence and identifies numeric columns
   - Requires minimum 2 numeric features for 2D analysis

2. **Multi-K Analysis**
   - Runs K-means clustering independently for k ∈ {2, 3, 4, 5}
   - Tests different numbers of clusters to find optimal separation
   - Generates separate analysis for each k value

3. **Evaluation Metrics**
   - Computes **inertia** (within-cluster sum of squares)
   - Computes **silhouette score** (cluster cohesion measure)
   - Tracks metrics for comparative analysis

4. **Output Generation**
   - Creates 2D scatter plot for each k showing cluster assignments
   - Generates metrics CSV summarizing performance across k values
   - Produces bar chart comparing silhouette scores
   - Exports clustered data with label assignments

5. **User Feedback**
   - Prints progress for each k iteration
   - Displays metrics for immediate feedback
   - Reports output directory for result location

---

## cluster_maker Package Overview

`cluster_maker` is an educational clustering toolkit providing a complete pipeline from data loading through evaluation. It emphasizes modularity, clear interfaces, and educational value for students learning data science practices.

### Main Components

#### **1. Data Generation** (`dataframe_builder.py`)
- `define_dataframe_structure()` - Creates a seed DataFrame with cluster center specifications
- `simulate_data()` - Generates synthetic clustered data by adding Gaussian noise around specified centers
- **Purpose:** Creates well-structured synthetic datasets for testing and demonstration

#### **2. Data Analysis** (`data_analyser.py`)
- `calculate_descriptive_statistics()` - Computes mean, std, quartiles for each feature
- `calculate_correlation()` - Generates correlation matrices to understand feature relationships
- **Purpose:** Provides exploratory data analysis before clustering

#### **3. Preprocessing** (`preprocessing.py`)
- `select_features()` - Filters to specific numeric columns with validation
- `standardise_features()` - Normalizes features to zero mean and unit variance
- **Purpose:** Ensures data is in proper format and scale for clustering algorithms

#### **4. Clustering Algorithms** (`algorithms.py`)
- `kmeans()` - Custom K-means implementation using Euclidean distance
- `sklearn_kmeans()` - Scikit-learn wrapper for comparison/production use
- `init_centroids()` - Random centroid initialization (sampling without replacement)
- `assign_clusters()` - Assigns points to nearest centroid
- `update_centroids()` - Updates centroid positions and handles empty clusters
- **Purpose:** Provides both educational (manual) and robust (sklearn) clustering implementations

#### **5. Evaluation Metrics** (`evaluation.py`)
- `compute_inertia()` - Within-cluster sum of squares (lower is better)
- `silhouette_score_sklearn()` - Silhouette coefficient (-1 to 1; higher is better)
- `elbow_curve()` - Tests multiple k values to find elbow point
- **Purpose:** Quantifies clustering quality and guides optimal cluster selection

#### **6. Visualization** (`plotting_clustered.py`)
- `plot_clusters_2d()` - 2D scatter plot with cluster colors and centroid markers
- `plot_elbow()` - Line plot of inertia vs k for elbow method visualization
- **Purpose:** Provides interpretable visual feedback on clustering results

#### **7. Data Export** (`data_exporter.py`)
- `export_to_csv()` - Saves DataFrame to CSV with delimiter control
- `export_formatted()` - Exports formatted text tables for reports
- **Purpose:** Enables result persistence and sharing

#### **8. High-Level Interface** (`interface.py`)
- `run_clustering()` - Orchestrates complete pipeline: load → preprocess → cluster → evaluate → plot → export
- **Purpose:** Simplifies workflow for users; reduces boilerplate code

### Design Philosophy

The package embodies professional data science practices:

- **Modularity:** Each function has a single responsibility
- **Type Hints:** Full type annotations for code clarity and IDE support
- **Error Handling:** Validates inputs with clear, informative error messages
- **Documentation:** Comprehensive docstrings following NumPy conventions
- **Separation of Concerns:** Computation, visualization, and I/O separated into distinct modules
- **Reusability:** Functions are composable; users can combine them as needed
- **Educational Value:** Code is readable and comments explain key concepts

### Typical Workflow

```
1. Load data (CSV)
      ↓
2. Select numeric features
      ↓
3. Standardise (zero mean, unit variance)
      ↓
4. Choose clustering algorithm (kmeans or sklearn_kmeans)
      ↓
5. Fit model with chosen k
      ↓
6. Evaluate (inertia, silhouette, elbow curve)
      ↓
7. Visualize 2D clusters
      ↓
8. Export results and plots
```

### Example Usage

```python
from cluster_maker import run_clustering

result = run_clustering(
    input_path="data/demo.csv",
    feature_cols=["x", "y"],
    algorithm="sklearn_kmeans",
    k=3,
    standardise=True,
    compute_elbow=True
)

# Access results
labels = result["labels"]           # Cluster assignments
centroids = result["centroids"]     # Cluster centers
metrics = result["metrics"]         # Inertia and silhouette
fig = result["fig_cluster"]         # 2D plot figure
```

---

## Key Insights from Analysis

Based on the corrected demo output on `demo_data.csv`:

- **k=3 is optimal** - Highest silhouette score (0.725) indicates well-separated, compact clusters
- **Elbow point visible** - Sharp decrease in inertia from k=2→3 (185.32→43.27)
- **Diminishing returns** - Inertia continues to decrease k=3→4→5, but silhouette drops, indicating over-clustering

This analysis demonstrates why the bug was critical: without testing all k values independently, this crucial insight about optimal clustering would be missed.
