"""
Parameter sweep and grid search functionality.

Supports:
- Parameter grid sweeps
- Longshot-specific sweeps
- Parallel execution via run_multiple_backtests
- Checkpointing for long sweeps
"""

import pandas as pd
import itertools
from typing import Optional, Any, List
from .strategies import StrategyConfig, create_longshot_strategies
from .backtest import run_multiple_backtests

# Optional tqdm import
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable=None, total=None, **kwargs):
        return iterable


# ----------------------------------------------------------------------
# Helpers for naming
# ----------------------------------------------------------------------

def _slugify(value: str) -> str:
    """Simple slugifier for category names used in strategy_name."""
    if value is None:
        return ""
    v = str(value).strip().lower()
    v = v.replace("&", "and").replace("+", "plus").replace("/", "-").replace(" ", "-")
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    return "".join(ch for ch in v if ch in allowed)


def _format_cat_suffix(tag: str, cats: list[Any], max_items: int = 3) -> str:
    """Build a compact suffix for category filters."""
    if not cats:
        return ""
    slugs = [_slugify(c) for c in cats if c]
    slugs = [s for s in slugs if s]

    if not slugs:
        return ""

    if len(slugs) <= max_items:
        body = "-".join(slugs)
    else:
        head = "-".join(slugs[:max_items])
        body = f"{head}+{len(slugs) - max_items}more"

    return f"_{tag}_{body}"


# ----------------------------------------------------------------------
# Core sweep
# ----------------------------------------------------------------------

def run_parameter_sweep(
    bets_df: pd.DataFrame,
    param_grid: dict[str, list[Any]],
    initial_capital: float = 1000.0,
    stake_mode: str = "fixed",
    base_config: Optional[dict] = None,
    verbose: bool = False,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 50,
    n_jobs: int = -1,
    use_parallel: bool = True,
) -> pd.DataFrame:
    """
    Run a parameter sweep over a grid of strategy configurations.

    Args:
        bets_df: Full bets DataFrame
        param_grid: Dictionary mapping parameter names to lists of values to test
                   Example: {
                       'price_ranges': [[(0.01, 0.05)], [(0.95, 0.99)]],
                       'horizons': [['7d'], ['14d'], ['7d', '14d']],
                       'min_volume': [1000, 3000, 5000],
                   }
        initial_capital: Starting capital for each backtest
        stake_mode: Staking mode
        base_config: Base configuration dict to apply to all strategies
        verbose: If True, show progress bar
        checkpoint_path: Path to save/resume results
        checkpoint_every: Save checkpoint every N strategies
        n_jobs: Number of parallel workers (-1 = all CPUs)
        use_parallel: Whether to use parallel execution

    Returns:
        DataFrame with one row per strategy configuration tested
    """
    if base_config is None:
        base_config = {}

    # Generate all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    print(f"\n{'='*60}")
    print(f"PARAMETER SWEEP")
    print(f"{'='*60}")
    print(f"Parameters: {param_names}")
    print(f"Total combinations: {len(combinations):,}")
    print(f"{'='*60}\n")

    strategies: List[StrategyConfig] = []

    # Build strategies from grid
    for combo in combinations:
        config = dict(zip(param_names, combo))
        config.update(base_config)

        # Handle price_ranges specially
        if "price_ranges" in config:
            price_ranges = config.pop("price_ranges")
            horizons = config.pop("horizons", ["7d"])
            sides = config.pop("sides", ["YES", "NO"])

            cat_inc = config.get("category_include")
            cat_broad_inc = config.get("category_broad_include")
            cat_exc = config.get("category_exclude")
            cat_broad_exc = config.get("category_broad_exclude")

            for price_min, price_max in price_ranges:
                for horizon in horizons:
                    for side in sides:
                        side_str = "both" if side == "both" else side.lower()
                        name = (
                            f"{side_str}_{int(price_min*100)}-{int(price_max*100)}pct_{horizon}"
                        )

                        if "min_volume" in config and config["min_volume"] is not None:
                            name += f"_vol{int(config['min_volume']/1000)}k"

                        if cat_inc:
                            name += _format_cat_suffix("cat", cat_inc)
                        if cat_broad_inc:
                            name += _format_cat_suffix("catb", cat_broad_inc)
                        if cat_exc:
                            name += _format_cat_suffix("catexc", cat_exc)
                        if cat_broad_exc:
                            name += _format_cat_suffix("catbexc", cat_broad_exc)

                        strategy = StrategyConfig(
                            name=name,
                            sides=[side],
                            horizons=[horizon],
                            price_min=price_min,
                            price_max=price_max,
                            **config,
                        )
                        strategies.append(strategy)
        else:
            # Standard parameter combination
            name_parts = []
            if "sides" in config:
                name_parts.append(f"{'_'.join(config['sides']).lower()}")
            if "horizons" in config:
                name_parts.append(f"{'_'.join(config['horizons'])}")
            if "price_min" in config and "price_max" in config:
                name_parts.append(
                    f"{int(config['price_min']*100)}-{int(config['price_max']*100)}pct"
                )

            if "category_include" in config and config["category_include"]:
                name_parts.append(
                    _format_cat_suffix("cat", config["category_include"]).lstrip("_")
                )
            if "category_broad_include" in config and config["category_broad_include"]:
                name_parts.append(
                    _format_cat_suffix("catb", config["category_broad_include"]).lstrip("_")
                )
            if "category_exclude" in config and config["category_exclude"]:
                name_parts.append(
                    _format_cat_suffix("catexc", config["category_exclude"]).lstrip("_")
                )
            if "category_broad_exclude" in config and config["category_broad_exclude"]:
                name_parts.append(
                    _format_cat_suffix("catbexc", config["category_broad_exclude"]).lstrip("_")
                )

            name = "_".join(name_parts) if name_parts else f"strategy_{len(strategies)}"

            strategy = StrategyConfig(
                name=name,
                **config,
            )
            strategies.append(strategy)

    print(f"Generated {len(strategies):,} strategies to test\n")

    # Run backtests with parallel execution
    results_df = run_multiple_backtests(
        bets_df,
        strategies,
        initial_capital=initial_capital,
        stake_mode=stake_mode,
        verbose=verbose,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        n_jobs=n_jobs,
        use_parallel=use_parallel,
    )

    return results_df


def run_longshot_sweep(
    bets_df: pd.DataFrame,
    price_buckets: Optional[list[tuple[float, float]]] = None,
    horizons: Optional[list[str]] = None,
    sides: Optional[list[str]] = None,
    volume_thresholds: Optional[list[float]] = None,
    categories: Optional[list[Optional[list[str]]]] = None,
    categories_broad: Optional[list[Optional[list[str]]]] = None,
    initial_capital: float = 1000.0,
    stake_per_bet: float = 1.0,
    verbose: bool = False,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 50,
    n_jobs: int = -1,
    use_parallel: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to run a sweep over longshot strategy parameters.

    Args:
        bets_df: Full bets DataFrame
        price_buckets: List of (price_min, price_max) tuples to test
        horizons: List of horizons to test
        sides: List of sides to test (can include "both", "YES", "NO")
        volume_thresholds: List of minimum volume thresholds to test
        categories: List of category lists to test
        categories_broad: List of category_broad lists to test
        initial_capital: Starting capital
        stake_per_bet: Fixed stake per bet
        verbose: If True, show progress
        checkpoint_path: Path to save/resume results
        checkpoint_every: Save checkpoint every N strategies
        n_jobs: Number of parallel workers (-1 = all CPUs)
        use_parallel: Whether to use parallel execution

    Returns:
        DataFrame with backtest results for all combinations
    """
    # Default parameters
    if price_buckets is None:
        price_buckets = [
            (0.01, 0.05),
            (0.05, 0.10),
            (0.10, 0.20),
            (0.90, 0.95),
            (0.95, 0.99),
        ]

    if horizons is None:
        horizons = ["7d", "14d", "30d"]

    if sides is None:
        sides = ["YES", "NO"]

    if volume_thresholds is None:
        volume_thresholds = [None, 1000, 3000, 5000]

    if categories is None:
        categories = [None]

    if categories_broad is None:
        categories_broad = [None]

    strategies: List[StrategyConfig] = []

    # Generate all combinations
    for price_min, price_max in price_buckets:
        for horizon in horizons:
            for side in sides:
                for min_vol in volume_thresholds:
                    for cat_list in categories:
                        for cat_broad_list in categories_broad:
                            side_str = "both" if side == "both" else side.lower()
                            name_parts = [
                                side_str,
                                f"{int(price_min*100)}-{int(price_max*100)}pct",
                                horizon,
                            ]

                            config_kwargs: dict[str, Any] = {}

                            if min_vol is not None:
                                name_parts.append(f"vol{int(min_vol/1000)}k")
                                config_kwargs["min_volume"] = min_vol

                            if cat_list is not None:
                                suffix = _format_cat_suffix("cat", cat_list)
                                if suffix:
                                    name_parts.append(suffix.lstrip("_"))
                                config_kwargs["category_include"] = cat_list

                            if cat_broad_list is not None:
                                suffix = _format_cat_suffix("catb", cat_broad_list)
                                if suffix:
                                    name_parts.append(suffix.lstrip("_"))
                                config_kwargs["category_broad_include"] = cat_broad_list

                            name = "_".join(name_parts)

                            strategy = StrategyConfig(
                                name=name,
                                sides=[side],
                                horizons=[horizon],
                                price_min=price_min,
                                price_max=price_max,
                                stake_per_bet=stake_per_bet,
                                **config_kwargs,
                            )
                            strategies.append(strategy)

    print(f"\n{'='*60}")
    print(f"LONGSHOT PARAMETER SWEEP")
    print(f"{'='*60}")
    print(f"Price buckets: {len(price_buckets)}")
    print(f"Horizons: {len(horizons)}")
    print(f"Sides: {len(sides)}")
    print(f"Volume thresholds: {len(volume_thresholds)}")
    print(f"Category filters: {len(categories)}")
    print(f"Category Broad filters: {len(categories_broad)}")
    print(f"Total strategies: {len(strategies):,}")
    print(f"{'='*60}\n")

    # Run backtests with parallel execution
    results_df = run_multiple_backtests(
        bets_df,
        strategies,
        initial_capital=initial_capital,
        stake_mode="fixed",
        verbose=verbose,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        n_jobs=n_jobs,
        use_parallel=use_parallel,
    )

    return results_df


def filter_sweep_results(
    results_df: pd.DataFrame,
    min_bets: int = 10,
    min_sharpe: Optional[float] = None,
    min_return_pct: Optional[float] = None,
    max_drawdown_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter sweep results based on performance criteria.

    Args:
        results_df: DataFrame from run_parameter_sweep or run_longshot_sweep
        min_bets: Minimum number of bets required
        min_sharpe: Minimum Sharpe ratio
        min_return_pct: Minimum total return percentage
        max_drawdown_pct: Maximum drawdown percentage (as negative value)

    Returns:
        Filtered DataFrame
    """
    df = results_df.copy()

    df = df[df["num_bets"] >= min_bets]

    if min_sharpe is not None:
        df = df[df["sharpe_ratio"] >= min_sharpe]

    if min_return_pct is not None:
        df = df[df["total_return_pct"] >= min_return_pct]

    if max_drawdown_pct is not None:
        df = df[df["max_drawdown_pct"] >= max_drawdown_pct]

    return df.reset_index(drop=True)


def analyze_sweep_results(results_df: pd.DataFrame) -> None:
    """
    Print analysis of sweep results.

    Args:
        results_df: DataFrame from run_parameter_sweep or run_longshot_sweep
    """
    if len(results_df) == 0:
        print("No results to analyze")
        return

    print(f"\n{'='*60}")
    print(f"SWEEP RESULTS ANALYSIS")
    print(f"{'='*60}\n")

    print(f"Total strategies tested: {len(results_df):,}")
    print(f"Strategies with positive return: {(results_df['total_return_pct'] > 0).sum():,}")
    print(f"Strategies with Sharpe > 1: {(results_df['sharpe_ratio'] > 1).sum():,}")

    print(f"\n📊 PERFORMANCE DISTRIBUTION")
    print(f"  Return (%):")
    print(f"    Mean:   {results_df['total_return_pct'].mean():>8.2f}")
    print(f"    Median: {results_df['total_return_pct'].median():>8.2f}")
    print(f"    Min:    {results_df['total_return_pct'].min():>8.2f}")
    print(f"    Max:    {results_df['total_return_pct'].max():>8.2f}")

    print(f"\n  Sharpe Ratio:")
    print(f"    Mean:   {results_df['sharpe_ratio'].mean():>8.2f}")
    print(f"    Median: {results_df['sharpe_ratio'].median():>8.2f}")
    print(f"    Min:    {results_df['sharpe_ratio'].min():>8.2f}")
    print(f"    Max:    {results_df['sharpe_ratio'].max():>8.2f}")

    print(f"\n  Drawdown (%):")
    print(f"    Mean:   {results_df['max_drawdown_pct'].mean():>8.2f}")
    print(f"    Median: {results_df['max_drawdown_pct'].median():>8.2f}")
    print(f"    Min:    {results_df['max_drawdown_pct'].min():>8.2f}")
    print(f"    Max:    {results_df['max_drawdown_pct'].max():>8.2f}")

    print(f"\n  Number of Bets:")
    print(f"    Mean:   {results_df['num_bets'].mean():>8,.0f}")
    print(f"    Median: {results_df['num_bets'].median():>8,.0f}")
    print(f"    Min:    {results_df['num_bets'].min():>8,.0f}")
    print(f"    Max:    {results_df['num_bets'].max():>8,.0f}")

    print(f"\n{'='*60}\n")
