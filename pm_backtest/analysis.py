"""
Analysis tools for strategy selection and portfolio construction.

Provides:
- Performance-based clustering
- Parameter robustness analysis  
- Diversified portfolio construction
- Strategy recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


# ============================================================
# PERFORMANCE-BASED CLUSTERING
# ============================================================

def cluster_by_performance(
    df: pd.DataFrame,
    n_clusters: int = 6,
    min_bets: int = 50,
    auto_select_k: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster strategies based on their PERFORMANCE metrics.
    
    This groups strategies by HOW they perform (risk/return profile),
    not by their input parameters. Pick the best from each cluster
    for a diversified set.
    
    Args:
        df: Sweep results DataFrame
        n_clusters: Number of clusters (ignored if auto_select_k=True)
        min_bets: Minimum bets filter
        auto_select_k: If True, automatically select optimal k
        
    Returns:
        Tuple of (clustered DataFrame, cluster statistics DataFrame)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required. Install with: pip install scikit-learn")
    
    df_filtered = df[df["num_bets"] >= min_bets].copy()
    print(f"📊 Clustering {len(df_filtered)} strategies (min {min_bets} bets)")
    
    # Performance features for clustering
    perf_features = ["sharpe_ratio", "calmar_ratio", "win_rate", "avg_roi", "max_drawdown_pct", "total_return_pct"]
    perf_features = [f for f in perf_features if f in df_filtered.columns]
    
    X = df_filtered[perf_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Auto-select k using silhouette score
    if auto_select_k:
        scores = []
        for k in range(2, min(15, len(df_filtered) // 10)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            scores.append((k, score))
        n_clusters = max(scores, key=lambda x: x[1])[0]
        print(f"  Auto-selected k={n_clusters} (silhouette={max(scores, key=lambda x: x[1])[1]:.3f})")
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_filtered["perf_cluster"] = kmeans.fit_predict(X_scaled)
    
    # Calculate cluster statistics
    cluster_stats = df_filtered.groupby("perf_cluster").agg({
        "strategy_name": "count",
        "sharpe_ratio": ["mean", "std", "max"],
        "calmar_ratio": ["mean", "max"],
        "win_rate": ["mean", "max"],
        "total_return_pct": ["mean", "max"],
        "max_drawdown_pct": ["mean", "min"],
        "num_bets": ["sum", "mean"],
    })
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    cluster_stats = cluster_stats.rename(columns={"strategy_name_count": "n_strategies"})
    cluster_stats = cluster_stats.sort_values("sharpe_ratio_mean", ascending=False)
    
    return df_filtered, cluster_stats


def get_best_per_cluster(
    df_clustered: pd.DataFrame,
    top_n: int = 3,
    sort_by: str = "sharpe_ratio"
) -> pd.DataFrame:
    """
    Get top strategies from each performance cluster.
    
    Args:
        df_clustered: DataFrame with 'perf_cluster' column
        top_n: Number of strategies per cluster
        sort_by: Metric to sort by
        
    Returns:
        DataFrame with best strategies per cluster
    """
    best = (
        df_clustered
        .sort_values(sort_by, ascending=False)
        .groupby("perf_cluster")
        .head(top_n)
    )
    
    cols = ["strategy_name", "perf_cluster", "sides", "horizons",
            "price_min", "price_max", "sharpe_ratio", "total_return_pct",
            "win_rate", "num_bets", "max_drawdown_pct"]
    cols = [c for c in cols if c in best.columns]
    
    return best[cols].sort_values(["perf_cluster", sort_by], ascending=[True, False])


def plot_clusters(
    df_clustered: pd.DataFrame,
    cluster_stats: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Visualize performance clusters.
    
    Args:
        df_clustered: DataFrame with cluster assignments
        cluster_stats: Cluster statistics DataFrame
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Sharpe by cluster
    ax = axes[0, 0]
    df_clustered.boxplot(column="sharpe_ratio", by="perf_cluster", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Distribution by Cluster", fontweight='bold')
    plt.suptitle("")
    ax.axhline(y=0, color='gray', linewidth=0.5)
    
    # 2. Risk-Return by cluster
    ax = axes[0, 1]
    for cluster in sorted(df_clustered["perf_cluster"].unique()):
        subset = df_clustered[df_clustered["perf_cluster"] == cluster]
        ax.scatter(
            -subset["max_drawdown_pct"],
            subset["total_return_pct"],
            label=f"Cluster {cluster}",
            alpha=0.6, s=40
        )
    ax.set_xlabel("Risk (|Max DD %|)")
    ax.set_ylabel("Return (%)")
    ax.set_title("Risk-Return by Cluster", fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    
    # 3. Cluster comparison bars
    ax = axes[1, 0]
    x = np.arange(len(cluster_stats))
    width = 0.35
    ax.bar(x - width/2, cluster_stats["sharpe_ratio_mean"], width, label='Mean Sharpe', alpha=0.8)
    ax.bar(x + width/2, cluster_stats["sharpe_ratio_max"], width, label='Max Sharpe', alpha=0.8)
    ax.set_xlabel("Cluster (ranked by mean Sharpe)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Cluster Performance", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_stats.index)
    ax.legend()
    ax.axhline(y=0, color='gray', linewidth=0.5)
    
    # 4. Cluster sizes
    ax = axes[1, 1]
    ax.bar(cluster_stats.index.astype(str), cluster_stats["n_strategies"])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Strategies")
    ax.set_title("Cluster Sizes", fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 CLUSTER SUMMARY:")
    print(cluster_stats.round(2).to_string())


# ============================================================
# PARAMETER ROBUSTNESS ANALYSIS
# ============================================================

def analyze_robustness(
    df: pd.DataFrame,
    param_col: str,
    metric: str = "sharpe_ratio",
    min_bets: int = 50
) -> pd.DataFrame:
    """
    Analyze how robust each parameter value is.
    
    A "robust" parameter has high mean performance AND low variance.
    
    Args:
        df: Sweep results DataFrame
        param_col: Parameter column to analyze
        metric: Performance metric
        min_bets: Minimum bets filter
        
    Returns:
        DataFrame with robustness statistics per parameter value
    """
    df_filtered = df[df["num_bets"] >= min_bets].copy()
    
    robustness = df_filtered.groupby(param_col).agg({
        metric: ["mean", "std", "min", "max", "count"],
        "num_bets": "sum"
    })
    robustness.columns = [f"{metric}_mean", f"{metric}_std", f"{metric}_worst", 
                          f"{metric}_best", "n_strategies", "total_bets"]
    
    # Robustness score = mean / (1 + std)
    robustness["robustness_score"] = (
        robustness[f"{metric}_mean"] / (1 + robustness[f"{metric}_std"].fillna(0))
    )
    
    # Worst case expected = mean - std
    robustness["worst_case_expected"] = (
        robustness[f"{metric}_mean"] - robustness[f"{metric}_std"].fillna(0)
    )
    
    return robustness.sort_values("robustness_score", ascending=False)


def plot_robustness(
    df: pd.DataFrame,
    params: List[str] = None,
    metric: str = "sharpe_ratio",
    min_bets: int = 50,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """
    Show robustness analysis for multiple parameters.
    
    Args:
        df: Sweep results DataFrame
        params: List of parameter columns to analyze
        metric: Performance metric
        min_bets: Minimum bets filter
        figsize: Figure size
    """
    if params is None:
        params = ["sides", "horizons"]
        if "min_volume" in df.columns:
            params.append("min_volume")
    
    n_params = len(params)
    n_cols = min(2, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, param in enumerate(params):
        ax = axes[i]
        
        robustness = analyze_robustness(df, param, metric, min_bets)
        
        x = np.arange(len(robustness))
        ax.bar(x, robustness[f"{metric}_mean"], alpha=0.7, label="Mean")
        ax.errorbar(x, robustness[f"{metric}_mean"], 
                   yerr=robustness[f"{metric}_std"].fillna(0),
                   fmt='none', color='black', capsize=3)
        
        ax.scatter(x, robustness[f"{metric}_worst"], marker='v', color='red', s=30, zorder=5, label="Worst")
        ax.scatter(x, robustness[f"{metric}_best"], marker='^', color='green', s=30, zorder=5, label="Best")
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(v)[:15] for v in robustness.index], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"{param}", fontweight='bold', fontsize=10)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        
        if i == 0:
            ax.legend(fontsize=8)
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(f"Parameter Robustness ({metric})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================

def estimate_similarity(
    df: pd.DataFrame,
    min_bets: int = 50
) -> pd.DataFrame:
    """
    Estimate strategy similarity based on parameter overlap.
    
    Strategies with similar parameters likely have correlated returns.
    
    Args:
        df: Sweep results DataFrame
        min_bets: Minimum bets filter
        
    Returns:
        Similarity matrix (0-1, higher = more similar)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required. Install with: pip install scikit-learn")
    
    from sklearn.metrics.pairwise import euclidean_distances
    
    df_filtered = df[df["num_bets"] >= min_bets].copy()
    
    param_features = ["price_min", "price_max"]
    df_encoded = pd.get_dummies(df_filtered[["sides", "horizons"]], drop_first=False)
    
    X = pd.concat([df_filtered[param_features].reset_index(drop=True), 
                   df_encoded.reset_index(drop=True)], axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dist_matrix = euclidean_distances(X_scaled)
    similarity = np.exp(-dist_matrix / dist_matrix.max())
    
    strategy_names = df_filtered["strategy_name"].values
    return pd.DataFrame(similarity, index=strategy_names, columns=strategy_names)


def build_portfolio(
    df: pd.DataFrame,
    n_strategies: int = 5,
    min_bets: int = 50,
    min_sharpe: float = 1.0,
    max_similarity: float = 0.4
) -> pd.DataFrame:
    """
    Build a diversified portfolio of uncorrelated strategies.
    
    Greedy algorithm:
    1. Start with best Sharpe strategy
    2. Add next-best that has low similarity to all selected
    3. Repeat until n_strategies
    
    Args:
        df: Sweep results DataFrame
        n_strategies: Target portfolio size
        min_bets: Minimum bets filter
        min_sharpe: Minimum Sharpe threshold
        max_similarity: Maximum allowed similarity
        
    Returns:
        DataFrame with selected strategies
    """
    print(f"\n🎯 BUILDING DIVERSIFIED PORTFOLIO")
    print(f"   Target: {n_strategies} strategies")
    print(f"   Min bets: {min_bets}, Min Sharpe: {min_sharpe}")
    print(f"   Max similarity: {max_similarity}")
    
    df_filtered = df[df["num_bets"] >= min_bets].copy()
    similarity = estimate_similarity(df_filtered, min_bets)
    
    # Filter to good strategies
    good = df_filtered[df_filtered["sharpe_ratio"] >= min_sharpe]["strategy_name"].tolist()
    good = [s for s in good if s in similarity.index]
    
    if len(good) == 0:
        print(f"❌ No strategies with Sharpe >= {min_sharpe}")
        return pd.DataFrame()
    
    # Sort by Sharpe
    sharpe_lookup = df_filtered.set_index("strategy_name")["sharpe_ratio"].to_dict()
    good = sorted(good, key=lambda x: sharpe_lookup.get(x, 0), reverse=True)
    
    # Greedy selection
    selected = [good[0]]
    candidates = good[1:]
    
    while len(selected) < n_strategies and len(candidates) > 0:
        candidate_scores = []
        for candidate in candidates:
            max_sim = max(similarity.loc[candidate, s] for s in selected)
            candidate_scores.append((candidate, max_sim, sharpe_lookup.get(candidate, 0)))
        
        valid = [(c, sim, sharpe) for c, sim, sharpe in candidate_scores if sim < max_similarity]
        
        if len(valid) == 0:
            print(f"  No more candidates below similarity {max_similarity}")
            break
        
        best = max(valid, key=lambda x: x[2])
        selected.append(best[0])
        candidates.remove(best[0])
    
    portfolio = df_filtered[df_filtered["strategy_name"].isin(selected)].copy()
    portfolio = portfolio.set_index("strategy_name").loc[selected].reset_index()
    
    print(f"\n✅ Selected {len(portfolio)} strategies:")
    cols = ["strategy_name", "sides", "horizons", "price_min", "price_max",
            "sharpe_ratio", "total_return_pct", "win_rate", "num_bets"]
    print(portfolio[[c for c in cols if c in portfolio.columns]].to_string(index=False))
    
    # Show similarity matrix
    selected_sim = similarity.loc[selected, selected]
    plt.figure(figsize=(8, 6))
    sns.heatmap(selected_sim, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=1, center=0.5)
    plt.title("Portfolio Similarity Matrix\n(lower = more diversified)", fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Portfolio stats
    avg_sharpe = portfolio["sharpe_ratio"].mean()
    avg_return = portfolio["total_return_pct"].mean()
    avg_sim = selected_sim.values[np.triu_indices(len(selected), k=1)].mean() if len(selected) > 1 else 0
    
    print(f"\n📈 PORTFOLIO STATS:")
    print(f"   Avg Sharpe:        {avg_sharpe:.2f}")
    print(f"   Avg Return:        {avg_return:.2f}%")
    print(f"   Avg Similarity:    {avg_sim:.2f}")
    print(f"   Total Bets:        {portfolio['num_bets'].sum():,}")
    
    return portfolio


# ============================================================
# STRATEGY RECOMMENDATIONS
# ============================================================

def get_recommendations(
    df: pd.DataFrame,
    min_bets: int = 50,
    top_n: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Generate strategy recommendations across different criteria.
    
    Args:
        df: Sweep results DataFrame
        min_bets: Minimum bets filter
        top_n: Number of recommendations per category
        
    Returns:
        Dict mapping category name to DataFrame
    """
    df_filtered = df[df["num_bets"] >= min_bets].copy()
    
    recommendations = {}
    
    base_cols = ["strategy_name", "sides", "horizons", "price_min", "price_max"]
    
    # 1. Highest Sharpe
    recommendations["highest_sharpe"] = (
        df_filtered.nlargest(top_n, "sharpe_ratio")
        [base_cols + ["sharpe_ratio", "total_return_pct", "num_bets"]]
    )
    
    # 2. Highest Calmar
    if "calmar_ratio" in df_filtered.columns:
        recommendations["highest_calmar"] = (
            df_filtered.nlargest(top_n, "calmar_ratio")
            [base_cols + ["calmar_ratio", "total_return_pct", "max_drawdown_pct", "num_bets"]]
        )
    
    # 3. Highest Win Rate (with Sharpe >= 1)
    high_sharpe = df_filtered[df_filtered["sharpe_ratio"] >= 1.0]
    if len(high_sharpe) > 0 and "win_rate" in high_sharpe.columns:
        recommendations["highest_win_rate"] = (
            high_sharpe.nlargest(top_n, "win_rate")
            [base_cols + ["win_rate", "sharpe_ratio", "num_bets"]]
        )
    
    # 4. Highest Capacity (most bets with positive Sharpe)
    positive = df_filtered[df_filtered["sharpe_ratio"] > 0]
    if len(positive) > 0:
        recommendations["highest_capacity"] = (
            positive.nlargest(top_n, "num_bets")
            [base_cols + ["num_bets", "sharpe_ratio", "total_return_pct"]]
        )
    
    # 5. Best Composite
    if "composite_score" in df_filtered.columns:
        recommendations["best_composite"] = (
            df_filtered.nlargest(top_n, "composite_score")
            [base_cols + ["composite_score", "sharpe_ratio", "win_rate", "num_bets"]]
        )
    
    return recommendations


def print_recommendations(recommendations: Dict[str, pd.DataFrame]) -> None:
    """Print all recommendations nicely."""
    
    titles = {
        "highest_sharpe": "🏆 HIGHEST SHARPE (Best Risk-Adjusted)",
        "highest_calmar": "📉 HIGHEST CALMAR (Best Return/Drawdown)",
        "highest_win_rate": "🎯 HIGHEST WIN RATE (Sharpe ≥ 1.0)",
        "highest_capacity": "📈 HIGHEST CAPACITY (Most Bets, Sharpe > 0)",
        "best_composite": "⚖️ BEST COMPOSITE (Balanced)"
    }
    
    for key, title in titles.items():
        if key in recommendations and len(recommendations[key]) > 0:
            print(f"\n{title}")
            print("=" * 80)
            print(recommendations[key].to_string(index=False))


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def run_analysis(
    df: pd.DataFrame,
    min_bets: int = 50,
    n_clusters: int = 6
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run complete analysis pipeline.
    
    Args:
        df: Sweep results DataFrame
        min_bets: Minimum bets filter
        n_clusters: Number of performance clusters
        
    Returns:
        Tuple of (clustered df, portfolio df, recommendations dict)
    """
    print("=" * 80)
    print("SWEEP ANALYSIS")
    print("=" * 80)
    print(f"Total strategies: {len(df)}")
    print(f"Strategies with ≥{min_bets} bets: {(df['num_bets'] >= min_bets).sum()}")
    
    # Clustering
    print("\n" + "=" * 80)
    print("PART 1: PERFORMANCE CLUSTERING")
    print("=" * 80)
    df_clustered, cluster_stats = cluster_by_performance(df, n_clusters, min_bets)
    plot_clusters(df_clustered, cluster_stats)
    
    print("\n🎯 BEST PER CLUSTER:")
    best_per = get_best_per_cluster(df_clustered, top_n=2)
    print(best_per.to_string(index=False))
    
    # Robustness
    print("\n" + "=" * 80)
    print("PART 2: PARAMETER ROBUSTNESS")
    print("=" * 80)
    plot_robustness(df, min_bets=min_bets)
    
    # Portfolio
    print("\n" + "=" * 80)
    print("PART 3: PORTFOLIO CONSTRUCTION")
    print("=" * 80)
    portfolio = build_portfolio(df, n_strategies=5, min_bets=min_bets, min_sharpe=1.0)
    
    # Recommendations
    print("\n" + "=" * 80)
    print("PART 4: RECOMMENDATIONS")
    print("=" * 80)
    recommendations = get_recommendations(df, min_bets=min_bets)
    print_recommendations(recommendations)
    
    return df_clustered, portfolio, recommendations
