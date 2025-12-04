"""
Visualization tools for sweep results analysis.

Provides impactful charts for understanding strategy performance
across parameter space.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def prepare_sweep_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare sweep results DataFrame with derived columns.
    
    Args:
        df: Raw sweep results DataFrame
        
    Returns:
        DataFrame with additional derived columns
    """
    df = df.copy()
    
    # Type conversions
    df["horizons"] = df["horizons"].astype(str)
    df["sides"] = df["sides"].astype(str)
    
    # Derived columns
    df["price_range"] = df["price_max"] - df["price_min"]
    df["price_midpoint"] = (df["price_max"] + df["price_min"]) / 2
    df["log_num_bets"] = np.log1p(df["num_bets"])
    
    return df


def plot_parameter_impact(
    df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    min_bets: int = 20,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Show how each parameter affects a performance metric using violin plots.
    
    Args:
        df: Sweep results DataFrame
        metric: Performance metric to analyze
        min_bets: Minimum bets filter
        figsize: Figure size
    """
    df = prepare_sweep_data(df)
    df = df[df["num_bets"] >= min_bets]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"Parameter Impact on {metric} (min {min_bets} bets)", fontsize=14, fontweight='bold')
    
    # 1. By Side
    ax = axes[0, 0]
    order = [s for s in ["NO", "YES", "YES,NO"] if s in df["sides"].unique()]
    if order:
        sns.violinplot(data=df[df["sides"].isin(order)], x="sides", y=metric, order=order, ax=ax, inner="box")
        for i, side in enumerate(order):
            median = df[df["sides"] == side][metric].median()
            ax.annotate(f'{median:.2f}', xy=(i, median), ha='center', va='bottom', fontsize=9)
    ax.set_title("By Side")
    ax.set_xlabel("")
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    
    # 2. By Horizon
    ax = axes[0, 1]
    horizons = sorted(df["horizons"].unique())
    sns.violinplot(data=df, x="horizons", y=metric, order=horizons, ax=ax, inner="box")
    ax.set_title("By Horizon")
    ax.set_xlabel("")
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    
    # 3. By Volume Threshold
    ax = axes[0, 2]
    if "min_volume" in df.columns and df["min_volume"].notna().any():
        df["vol_bucket"] = pd.cut(
            df["min_volume"].fillna(0), 
            bins=[-1, 500, 1000, 5000, np.inf], 
            labels=["<500", "500-1k", "1k-5k", ">5k"]
        )
        sns.violinplot(data=df, x="vol_bucket", y=metric, ax=ax, inner="box")
        ax.set_title("By Volume Threshold")
    else:
        ax.text(0.5, 0.5, "No volume data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("By Volume Threshold")
    ax.set_xlabel("")
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    
    # 4. By Price Range Width
    ax = axes[1, 0]
    df["range_bucket"] = pd.cut(
        df["price_range"], 
        bins=[0, 0.03, 0.05, 0.10, 0.20], 
        labels=["Tight\n(<3%)", "Narrow\n(3-5%)", "Medium\n(5-10%)", "Wide\n(>10%)"]
    )
    sns.violinplot(data=df.dropna(subset=["range_bucket"]), x="range_bucket", y=metric, ax=ax, inner="box")
    ax.set_title("By Price Range Width")
    ax.set_xlabel("")
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    
    # 5. By Price Midpoint
    ax = axes[1, 1]
    df["price_bucket"] = pd.cut(
        df["price_midpoint"], 
        bins=[0.8, 0.85, 0.90, 0.93, 0.96, 1.0],
        labels=["80-85%", "85-90%", "90-93%", "93-96%", "96-100%"]
    )
    sns.violinplot(data=df.dropna(subset=["price_bucket"]), x="price_bucket", y=metric, ax=ax, inner="box")
    ax.set_title("By Price Midpoint")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=15)
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    
    # 6. By Category (if available)
    ax = axes[1, 2]
    if "category_broad" in df.columns or "category_broad_include" in df.columns:
        cat_col = "category_broad" if "category_broad" in df.columns else "category_broad_include"
        # Extract category from strategy name if needed
        if df[cat_col].isna().all():
            df["category"] = df["strategy_name"].apply(
                lambda x: "All" if "catb" not in str(x) else "Filtered"
            )
            cat_col = "category"
        top_cats = df[cat_col].value_counts().head(5).index.tolist()
        sns.violinplot(data=df[df[cat_col].isin(top_cats)], x=cat_col, y=metric, ax=ax, inner="box")
        ax.set_title("By Category")
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, "No category data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("By Category")
    ax.set_xlabel("")
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def plot_heatmap(
    df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    side: Optional[str] = None,
    horizon: Optional[str] = None,
    agg_func: str = "median",
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Heatmap showing performance across price_min x price_max combinations.
    
    Args:
        df: Sweep results DataFrame
        metric: Metric to visualize
        side: Filter to specific side (None = all)
        horizon: Filter to specific horizon (None = all)
        agg_func: Aggregation function ("mean", "median", "max")
        figsize: Figure size
    """
    subset = df.copy()
    
    title_parts = [f"{metric} ({agg_func})"]
    
    if side is not None:
        subset = subset[subset["sides"] == side]
        title_parts.append(f"Side: {side}")
        
    if horizon is not None:
        subset = subset[subset["horizons"] == horizon]
        title_parts.append(f"Horizon: {horizon}")
    
    if len(subset) == 0:
        print(f"No data for the specified filters")
        return
    
    pivot = subset.pivot_table(
        values=metric,
        index="price_min",
        columns="price_max",
        aggfunc=agg_func
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if metric in ["sharpe_ratio", "calmar_ratio", "composite_score"]:
        cmap = "RdYlGn"
        center = 0
    else:
        cmap = "YlGnBu"
        center = None
    
    sns.heatmap(
        pivot, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        center=center,
        ax=ax,
        cbar_kws={"label": metric}
    )
    
    ax.set_title(" | ".join(title_parts), fontweight='bold')
    ax.set_xlabel("Price Max")
    ax.set_ylabel("Price Min")
    
    plt.tight_layout()
    plt.show()


def plot_heatmap_grid(
    df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    agg_func: str = "median",
    figsize: Tuple[int, int] = (18, 12)
) -> None:
    """
    Grid of heatmaps: rows = sides, cols = horizons.
    
    Args:
        df: Sweep results DataFrame
        metric: Metric to visualize
        agg_func: Aggregation function
        figsize: Figure size
    """
    sides = sorted(df["sides"].unique())
    horizons = sorted(df["horizons"].unique())
    
    fig, axes = plt.subplots(len(sides), len(horizons), figsize=figsize)
    fig.suptitle(f"{metric} by Price Range ({agg_func})", fontsize=14, fontweight='bold')
    
    if len(sides) == 1:
        axes = axes.reshape(1, -1)
    if len(horizons) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, side in enumerate(sides):
        for j, horizon in enumerate(horizons):
            ax = axes[i, j]
            
            mask = (df["sides"] == side) & (df["horizons"] == horizon)
            subset = df[mask].copy()
            
            if len(subset) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(f"{side} / {horizon}")
                continue
            
            pivot = subset.pivot_table(
                values=metric,
                index="price_min",
                columns="price_max",
                aggfunc=agg_func
            )
            
            cmap = "RdYlGn" if metric in ["sharpe_ratio", "calmar_ratio", "composite_score"] else "YlGnBu"
            center = 0 if metric in ["sharpe_ratio", "calmar_ratio", "composite_score"] else None
            
            sns.heatmap(
                pivot, 
                annot=True, 
                fmt=".1f", 
                cmap=cmap, 
                center=center,
                ax=ax,
                cbar=False,
                annot_kws={"size": 8}
            )
            
            ax.set_title(f"{side} / {horizon}", fontsize=10)
            if i < len(sides) - 1:
                ax.set_xlabel("")
            if j > 0:
                ax.set_ylabel("")
    
    plt.tight_layout()
    plt.show()


def plot_risk_return(
    df: pd.DataFrame,
    color_by: str = "sides",
    size_by: str = "num_bets",
    min_bets: int = 50,
    figsize: Tuple[int, int] = (12, 8)
) -> pd.DataFrame:
    """
    Scatter plot of risk vs return with Pareto frontier highlighted.
    
    Args:
        df: Sweep results DataFrame
        color_by: Column to color points by
        size_by: Column to size points by
        min_bets: Minimum bets filter
        figsize: Figure size
        
    Returns:
        DataFrame of Pareto-optimal strategies
    """
    from matplotlib.patches import Patch
    
    subset = df[df["num_bets"] >= min_bets].copy()
    
    if len(subset) == 0:
        print(f"No strategies with >= {min_bets} bets")
        return pd.DataFrame()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sizes = (subset[size_by] / subset[size_by].max()) * 200 + 20
    
    if color_by in subset.columns:
        categories = subset[color_by].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        color_map = dict(zip(categories, colors))
        c = [color_map[cat] for cat in subset[color_by]]
        legend_elements = [Patch(facecolor=color_map[cat], label=str(cat)[:20]) for cat in categories]
    else:
        c = subset["sharpe_ratio"]
        legend_elements = None
    
    ax.scatter(
        -subset["max_drawdown_pct"],
        subset["total_return_pct"],
        c=c if legend_elements else subset["sharpe_ratio"],
        s=sizes,
        alpha=0.6,
        cmap="viridis" if not legend_elements else None
    )
    
    # Find Pareto frontier
    pareto_mask = _get_pareto_mask(
        -subset["max_drawdown_pct"].values,
        subset["total_return_pct"].values
    )
    pareto = subset[pareto_mask].sort_values("max_drawdown_pct")
    
    ax.plot(
        -pareto["max_drawdown_pct"],
        pareto["total_return_pct"],
        'r--', linewidth=2, label="Pareto Frontier"
    )
    ax.scatter(
        -pareto["max_drawdown_pct"],
        pareto["total_return_pct"],
        s=100, facecolors='none', edgecolors='red', linewidth=2
    )
    
    ax.set_xlabel("Risk (|Max Drawdown %|) →", fontsize=11)
    ax.set_ylabel("Return (%) →", fontsize=11)
    ax.set_title(f"Risk-Return Frontier (bubble size = {size_by})", fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    if legend_elements:
        legend_elements.append(plt.Line2D([0], [0], color='r', linestyle='--', label='Pareto Frontier'))
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    else:
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 PARETO-OPTIMAL STRATEGIES:")
    cols = ["strategy_name", "total_return_pct", "max_drawdown_pct", "sharpe_ratio", "num_bets"]
    print(pareto[[c for c in cols if c in pareto.columns]].to_string(index=False))
    
    return pareto


def _get_pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find Pareto-optimal points (maximize both x and y)."""
    is_pareto = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                if x[j] >= x[i] and y[j] >= y[i] and (x[j] > x[i] or y[j] > y[i]):
                    is_pareto[i] = False
                    break
    return is_pareto


def plot_sharpe_vs_bets(
    df: pd.DataFrame,
    color_by: str = "sides",
    highlight_sharpe: float = 3.0,
    highlight_bets: int = 100,
    figsize: Tuple[int, int] = (12, 8)
) -> pd.DataFrame:
    """
    Show tradeoff between statistical significance and performance.
    
    Args:
        df: Sweep results DataFrame
        color_by: Column to color by
        highlight_sharpe: Sharpe threshold for highlighting
        highlight_bets: Bet count threshold for highlighting
        figsize: Figure size
        
    Returns:
        DataFrame of high-conviction strategies
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    categories = df[color_by].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))
    
    for cat in categories:
        subset = df[df[color_by] == cat]
        ax.scatter(
            subset["num_bets"],
            subset["sharpe_ratio"],
            c=[color_map[cat]],
            label=str(cat)[:20],
            alpha=0.6,
            s=50
        )
    
    # Highlight high-conviction zone
    good = df[(df["sharpe_ratio"] >= highlight_sharpe) & (df["num_bets"] >= highlight_bets)]
    if len(good) > 0:
        ax.scatter(
            good["num_bets"],
            good["sharpe_ratio"],
            s=150, facecolors='none', edgecolors='green', linewidth=2,
            label=f"High-conviction"
        )
    
    ax.axhline(y=highlight_sharpe, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=highlight_bets, color='gray', linestyle='--', alpha=0.3)
    
    ax.set_xlabel("Number of Bets (log scale)", fontsize=11)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_xscale("log")
    ax.set_title(f"Sharpe Ratio vs Sample Size\n(Green zone: Sharpe≥{highlight_sharpe}, bets≥{highlight_bets})", fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    if len(good) > 0:
        print(f"\n🎯 HIGH-CONVICTION STRATEGIES:")
        cols = ["strategy_name", "sharpe_ratio", "num_bets", "total_return_pct", "win_rate"]
        print(good.sort_values("sharpe_ratio", ascending=False)[[c for c in cols if c in good.columns]].head(10).to_string(index=False))
    
    return good


def plot_dashboard(
    df: pd.DataFrame,
    min_bets: int = 50,
    figsize: Tuple[int, int] = (20, 16)
) -> None:
    """
    Comprehensive single-page dashboard with key insights.
    
    Args:
        df: Sweep results DataFrame
        min_bets: Minimum bets filter
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    subset = df[df["num_bets"] >= min_bets].copy()
    
    if len(subset) == 0:
        print(f"No strategies with >= {min_bets} bets")
        return
    
    # 1. Risk-Return Scatter
    ax1 = fig.add_subplot(gs[0, :2])
    scatter1 = ax1.scatter(
        -subset["max_drawdown_pct"],
        subset["total_return_pct"],
        c=subset["sharpe_ratio"],
        s=50, alpha=0.6, cmap="viridis"
    )
    plt.colorbar(scatter1, ax=ax1, label="Sharpe")
    ax1.set_xlabel("Risk (|Max DD %|)")
    ax1.set_ylabel("Return (%)")
    ax1.set_title("Risk-Return", fontweight='bold')
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    
    # 2. Sharpe by Side
    ax2 = fig.add_subplot(gs[0, 2])
    sides_order = [s for s in ["NO", "YES", "YES,NO"] if s in subset["sides"].unique()]
    if sides_order:
        sns.boxplot(data=subset[subset["sides"].isin(sides_order)], x="sides", y="sharpe_ratio", order=sides_order, ax=ax2)
    ax2.set_title("Sharpe by Side", fontweight='bold')
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    
    # 3. Sharpe by Horizon
    ax3 = fig.add_subplot(gs[1, 0])
    horizons_order = sorted(subset["horizons"].unique())
    sns.boxplot(data=subset, x="horizons", y="sharpe_ratio", order=horizons_order, ax=ax3)
    ax3.set_title("Sharpe by Horizon", fontweight='bold')
    ax3.axhline(y=0, color='gray', linewidth=0.5)
    
    # 4. Sharpe by Price Bucket
    ax4 = fig.add_subplot(gs[1, 1])
    subset["price_bucket"] = pd.cut(
        (subset["price_min"] + subset["price_max"]) / 2,
        bins=[0.8, 0.90, 0.95, 1.0],
        labels=["80-90%", "90-95%", "95-100%"]
    )
    sns.boxplot(data=subset.dropna(subset=["price_bucket"]), x="price_bucket", y="sharpe_ratio", ax=ax4)
    ax4.set_title("Sharpe by Price Bucket", fontweight='bold')
    ax4.axhline(y=0, color='gray', linewidth=0.5)
    
    # 5. Sharpe vs Bets
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(subset["num_bets"], subset["sharpe_ratio"], alpha=0.5, s=30)
    ax5.set_xscale("log")
    ax5.set_xlabel("# Bets")
    ax5.set_ylabel("Sharpe")
    ax5.set_title("Sharpe vs Sample Size", fontweight='bold')
    ax5.axhline(y=0, color='gray', linewidth=0.5)
    
    # 6. Top strategies table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    top_10 = subset.nlargest(10, "sharpe_ratio")[
        ["strategy_name", "sides", "horizons", "price_min", "price_max",
         "sharpe_ratio", "total_return_pct", "win_rate", "num_bets"]
    ].round(3)
    
    table = ax6.table(
        cellText=top_10.values,
        colLabels=top_10.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax6.set_title(f"Top 10 Strategies by Sharpe (min {min_bets} bets)", fontweight='bold', fontsize=12, pad=20)
    
    fig.suptitle(f"Sweep Analysis Dashboard ({len(subset)} strategies)", fontsize=14, fontweight='bold')
    plt.show()
