"""
Walk-forward testing for prediction market strategies.

Walk-forward analysis tests strategy robustness across time windows
without forward-looking bias.
"""

import warnings
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .strategies import StrategyConfig, select_bets_for_strategy
from .backtest import run_backtest
from .sweep import run_parameter_sweep
from .metrics import calculate_metrics


def _get_bet_key(row: pd.Series) -> tuple:
    """Generate a unique key for a bet to detect duplicates."""
    return (row["condition_id"], row["entry_ts"], row["side"], row["horizon"])


def _filter_duplicate_bets(
    oos_capital_series: pd.DataFrame,
    seen_bets: set,
) -> pd.DataFrame:
    """
    Filter out bets that have already been seen in previous windows.
    
    Args:
        oos_capital_series: Capital series from current OOS window
        seen_bets: Set of bet keys already included
        
    Returns:
        Filtered DataFrame with only new bets
    """
    if len(oos_capital_series) == 0:
        return oos_capital_series
    
    # Identify which bets are new
    new_mask = []
    for _, row in oos_capital_series.iterrows():
        key = _get_bet_key(row)
        new_mask.append(key not in seen_bets)
    
    return oos_capital_series[new_mask].copy()


@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward testing.

    Walk-forward testing divides historical data into rolling windows:
    - In-sample period: optimize parameters
    - Out-of-sample period: test with optimized parameters
    - Step: how far to advance for next window

    Example:
        config = WalkForwardConfig(
            in_sample_days=90,      # Optimize on 90 days
            out_of_sample_days=30,  # Test on next 30 days
            step_days=30,           # Advance 30 days for next window
            optimization_metric="composite_score",
            min_bets=15,
            initial_capital=1000.0,
            stake_per_bet=10.0,
            use_lockup=True,
            start_date="2025-01-01",  # Optional: start from this date
            end_date=None,            # Optional: end at this date (None = use all data)
        )
    """

    in_sample_days: int = 90
    out_of_sample_days: int = 30
    step_days: int = 30
    optimization_metric: str = "composite_score"  # or "calmar_ratio", "sharpe_ratio", "total_return_pct"
    min_bets: int = 15  # Minimum bets required in in-sample for valid optimization
    initial_capital: float = 1000.0  # Starting capital
    stake_per_bet: float = 10.0      # Fixed stake per bet (for sweep)
    use_lockup: bool = True          # Whether to use capital lock-up model
    start_date: Optional[str] = None  # Start date for walk-forward (None = use min date in data)
    end_date: Optional[str] = None    # End date for walk-forward (None = use max date in data)

    def __post_init__(self):
        """Warn if step_days < out_of_sample_days (overlapping OOS windows)."""
        if self.step_days < self.out_of_sample_days:
            warnings.warn(
                f"step_days ({self.step_days}) < out_of_sample_days ({self.out_of_sample_days}): "
                f"OOS windows will overlap. Duplicate bets will be de-duplicated in aggregated metrics.",
                UserWarning,
            )


def run_walk_forward_single(
    bets_df: pd.DataFrame,
    strategy: StrategyConfig,
    config: WalkForwardConfig,
) -> dict:
    """
    Walk-forward test for a SINGLE strategy.

    Tests if the strategy is robust across time windows.
    Capital carries over between windows.

    Args:
        bets_df: Full bets DataFrame
        strategy: Strategy configuration
        config: Walk-forward configuration (includes initial_capital, use_lockup, start_date, end_date)

    Returns:
        Dictionary with:
            - 'window_results': List of per-window metrics
            - 'aggregated_oos_metrics': Combined out-of-sample performance
            - 'aggregated_oos_capital_series': Combined capital series (de-duplicated)
            - 'strategy': The strategy configuration
            - 'num_duplicate_bets_removed': Count of duplicates removed from aggregation
    """
    # Get date range - use config dates if provided, else use data bounds
    data_min_date = bets_df["entry_ts"].min()
    data_max_date = bets_df["entry_ts"].max()
    
    if config.start_date is not None:
        min_date = pd.to_datetime(config.start_date, utc=True)
        if min_date < data_min_date:
            min_date = data_min_date
    else:
        min_date = data_min_date
    
    if config.end_date is not None:
        max_date = pd.to_datetime(config.end_date, utc=True)
        if max_date > data_max_date:
            max_date = data_max_date
    else:
        max_date = data_max_date

    window_results = []
    all_oos_bets = []
    seen_bets = set()  # Track bets already included in aggregation
    num_duplicates_removed = 0
    capital = config.initial_capital

    # Generate windows
    current_start = min_date
    window_num = 0

    while True:
        window_num += 1

        # Define window boundaries
        is_start = current_start
        is_end = is_start + pd.Timedelta(days=config.in_sample_days)
        oos_start = is_end
        oos_end = oos_start + pd.Timedelta(days=config.out_of_sample_days)

        # Check if we have enough data
        if oos_end > max_date:
            break

        # Filter data for this window
        is_data = bets_df[
            (bets_df["entry_ts"] >= is_start) & (bets_df["entry_ts"] < is_end)
        ]
        oos_data = bets_df[
            (bets_df["entry_ts"] >= oos_start) & (bets_df["entry_ts"] < oos_end)
        ]

        # Test strategy on in-sample (for validation)
        is_result = run_backtest(
            is_data,
            strategy,
            initial_capital=capital,
            use_lockup=config.use_lockup,
            verbose=False,
        )

        is_metrics = is_result["metrics"]

        # Check if in-sample has enough bets
        if is_metrics.get("num_bets", 0) < config.min_bets:
            # Not enough data - skip this window
            current_start += pd.Timedelta(days=config.step_days)
            continue

        # Test on out-of-sample
        oos_result = run_backtest(
            oos_data,
            strategy,
            initial_capital=capital,
            use_lockup=config.use_lockup,
            verbose=False,
        )

        oos_metrics = oos_result["metrics"]
        oos_capital_series = oos_result["capital_series"]

        # Update capital for next window
        if len(oos_capital_series) > 0:
            capital = oos_capital_series["capital"].iloc[-1]
            
            # De-duplicate before adding to aggregated results
            original_count = len(oos_capital_series)
            new_bets_only = _filter_duplicate_bets(oos_capital_series, seen_bets)
            num_duplicates_removed += original_count - len(new_bets_only)
            
            # Add new bet keys to seen set
            for _, row in new_bets_only.iterrows():
                seen_bets.add(_get_bet_key(row))
            
            if len(new_bets_only) > 0:
                all_oos_bets.append(new_bets_only)

        # Record window results
        window_results.append({
            "window": window_num,
            "is_start": is_start,
            "is_end": is_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "is_num_bets": is_metrics.get("num_bets", 0),
            "oos_num_bets": oos_metrics.get("num_bets", 0),
            "is_return_pct": is_metrics.get("total_return_pct", 0),
            "oos_return_pct": oos_metrics.get("total_return_pct", 0),
            "is_sharpe": is_metrics.get("sharpe_ratio", 0),
            "oos_sharpe": oos_metrics.get("sharpe_ratio", 0),
            "is_calmar": is_metrics.get("calmar_ratio", 0),
            "oos_calmar": oos_metrics.get("calmar_ratio", 0),
            "is_composite": is_metrics.get("composite_score", 0),
            "oos_composite": oos_metrics.get("composite_score", 0),
        })

        # Advance window
        current_start += pd.Timedelta(days=config.step_days)

    # Aggregate out-of-sample results
    if len(all_oos_bets) > 0:
        aggregated_oos_capital_series = pd.concat(all_oos_bets, ignore_index=True)
        # Re-sort by entry_ts and recalculate capital series
        aggregated_oos_capital_series = _recalculate_capital_series(
            aggregated_oos_capital_series, config.initial_capital
        )
        aggregated_oos_metrics = calculate_metrics(aggregated_oos_capital_series, config.initial_capital)
    else:
        aggregated_oos_capital_series = pd.DataFrame()
        aggregated_oos_metrics = {}

    return {
        "window_results": window_results,
        "aggregated_oos_metrics": aggregated_oos_metrics,
        "aggregated_oos_capital_series": aggregated_oos_capital_series,
        "strategy": strategy,
        "num_duplicate_bets_removed": num_duplicates_removed,
    }


def _recalculate_capital_series(
    capital_series: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    """
    Re-sort and recalculate capital after de-duplication.
    
    Since we removed duplicate bets, we need to:
    1. Sort by entry_ts
    2. Recalculate running capital based on PnL
    """
    if len(capital_series) == 0:
        return capital_series
    
    df = capital_series.sort_values("entry_ts").reset_index(drop=True)
    
    # Recalculate capital from PnL
    df["capital"] = initial_capital + df["pnl"].cumsum()
    
    return df


def run_walk_forward_sweep(
    bets_df: pd.DataFrame,
    param_grid: dict,
    config: WalkForwardConfig,
) -> dict:
    """
    Walk-forward with parameter optimization.

    For each window:
    1. Run sweep on in-sample period
    2. Select best strategy by optimization_metric
    3. Test on out-of-sample period
    4. Roll forward

    Capital carries over between windows.

    Args:
        bets_df: Full bets DataFrame
        param_grid: Parameter grid for sweep
        config: Walk-forward configuration (includes initial_capital, stake_per_bet, use_lockup, start_date, end_date)

    Returns:
        Dictionary with:
            - 'window_results': List with best_params and oos metrics per window
            - 'aggregated_oos_metrics': Realistic combined performance (de-duplicated)
            - 'aggregated_oos_capital_series': Realistic equity curve (de-duplicated)
            - 'param_stability': Dict showing how often each param was selected
            - 'num_duplicate_bets_removed': Count of duplicates removed from aggregation
    """
    # Get date range - use config dates if provided, else use data bounds
    data_min_date = bets_df["entry_ts"].min()
    data_max_date = bets_df["entry_ts"].max()
    
    if config.start_date is not None:
        min_date = pd.to_datetime(config.start_date, utc=True)
        if min_date < data_min_date:
            min_date = data_min_date
    else:
        min_date = data_min_date
    
    if config.end_date is not None:
        max_date = pd.to_datetime(config.end_date, utc=True)
        if max_date > data_max_date:
            max_date = data_max_date
    else:
        max_date = data_max_date

    window_results = []
    all_oos_bets = []
    param_selections = []
    seen_bets = set()  # Track bets already included in aggregation
    num_duplicates_removed = 0
    capital = config.initial_capital

    # Generate windows
    current_start = min_date
    window_num = 0

    while True:
        window_num += 1

        # Define window boundaries
        is_start = current_start
        is_end = is_start + pd.Timedelta(days=config.in_sample_days)
        oos_start = is_end
        oos_end = oos_start + pd.Timedelta(days=config.out_of_sample_days)

        # Check if we have enough data
        if oos_end > max_date:
            break

        # Filter data for this window
        is_data = bets_df[
            (bets_df["entry_ts"] >= is_start) & (bets_df["entry_ts"] < is_end)
        ]
        oos_data = bets_df[
            (bets_df["entry_ts"] >= oos_start) & (bets_df["entry_ts"] < oos_end)
        ]

        # Run parameter sweep on in-sample
        # Modify param_grid to include stake_per_bet
        base_config = {"stake_per_bet": config.stake_per_bet}

        sweep_results = run_parameter_sweep(
            is_data,
            param_grid,
            initial_capital=capital,
            base_config=base_config,
            verbose=False,
        )

        # Filter for minimum bets
        sweep_results = sweep_results[sweep_results["num_bets"] >= config.min_bets]

        if len(sweep_results) == 0:
            # No strategies met criteria - skip this window
            current_start += pd.Timedelta(days=config.step_days)
            continue

        # Select best strategy by optimization metric
        best_idx = sweep_results[config.optimization_metric].idxmax()
        best_row = sweep_results.loc[best_idx]
        best_strategy_name = best_row["strategy_name"]

        # Reconstruct best strategy config
        # Extract parameters from the best strategy
        best_params = {
            "sides": best_row["sides"].split(","),
            "horizons": best_row["horizons"].split(","),
            "price_min": best_row["price_min"],
            "price_max": best_row["price_max"],
            "stake_per_bet": best_row["stake_per_bet"],
        }

        # Record parameter selection
        param_selections.append(best_params)

        # Create strategy config for out-of-sample testing
        best_strategy = StrategyConfig(
            name=f"{best_strategy_name}_window{window_num}",
            sides=best_params["sides"],
            horizons=best_params["horizons"],
            price_min=best_params["price_min"],
            price_max=best_params["price_max"],
            stake_per_bet=best_params["stake_per_bet"],
        )

        # Test on out-of-sample
        oos_result = run_backtest(
            oos_data,
            best_strategy,
            initial_capital=capital,
            use_lockup=config.use_lockup,
            verbose=False,
        )

        oos_metrics = oos_result["metrics"]
        oos_capital_series = oos_result["capital_series"]

        # Update capital for next window
        if len(oos_capital_series) > 0:
            capital = oos_capital_series["capital"].iloc[-1]
            
            # De-duplicate before adding to aggregated results
            original_count = len(oos_capital_series)
            new_bets_only = _filter_duplicate_bets(oos_capital_series, seen_bets)
            num_duplicates_removed += original_count - len(new_bets_only)
            
            # Add new bet keys to seen set
            for _, row in new_bets_only.iterrows():
                seen_bets.add(_get_bet_key(row))
            
            if len(new_bets_only) > 0:
                all_oos_bets.append(new_bets_only)

        # Record window results
        window_results.append({
            "window": window_num,
            "is_start": is_start,
            "is_end": is_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "best_strategy_name": best_strategy_name,
            "best_params": best_params,
            "is_num_strategies": len(sweep_results),
            "is_best_return_pct": best_row["total_return_pct"],
            "is_best_sharpe": best_row["sharpe_ratio"],
            "is_best_metric": best_row[config.optimization_metric],
            "oos_num_bets": oos_metrics.get("num_bets", 0),
            "oos_return_pct": oos_metrics.get("total_return_pct", 0),
            "oos_sharpe": oos_metrics.get("sharpe_ratio", 0),
            "oos_calmar": oos_metrics.get("calmar_ratio", 0),
            "oos_composite": oos_metrics.get("composite_score", 0),
        })

        # Advance window
        current_start += pd.Timedelta(days=config.step_days)

    # Aggregate out-of-sample results
    if len(all_oos_bets) > 0:
        aggregated_oos_capital_series = pd.concat(all_oos_bets, ignore_index=True)
        # Re-sort by entry_ts and recalculate capital series
        aggregated_oos_capital_series = _recalculate_capital_series(
            aggregated_oos_capital_series, config.initial_capital
        )
        aggregated_oos_metrics = calculate_metrics(aggregated_oos_capital_series, config.initial_capital)
    else:
        aggregated_oos_capital_series = pd.DataFrame()
        aggregated_oos_metrics = {}

    # Analyze parameter stability
    param_stability = _analyze_param_stability(param_selections)

    return {
        "window_results": window_results,
        "aggregated_oos_metrics": aggregated_oos_metrics,
        "aggregated_oos_capital_series": aggregated_oos_capital_series,
        "param_stability": param_stability,
        "num_duplicate_bets_removed": num_duplicates_removed,
    }


def _analyze_param_stability(param_selections: list[dict]) -> dict:
    """
    Analyze how often each parameter combination was selected.

    Args:
        param_selections: List of parameter dicts

    Returns:
        Dictionary with parameter frequency analysis
    """
    if len(param_selections) == 0:
        return {}

    # Count selections for each parameter
    side_counts = {}
    horizon_counts = {}
    price_range_counts = {}

    for params in param_selections:
        # Count sides
        sides_key = ",".join(sorted(params["sides"]))
        side_counts[sides_key] = side_counts.get(sides_key, 0) + 1

        # Count horizons
        horizons_key = ",".join(sorted(params["horizons"]))
        horizon_counts[horizons_key] = horizon_counts.get(horizons_key, 0) + 1

        # Count price ranges
        price_key = f"{params['price_min']:.2f}-{params['price_max']:.2f}"
        price_range_counts[price_key] = price_range_counts.get(price_key, 0) + 1

    return {
        "total_windows": len(param_selections),
        "side_counts": side_counts,
        "horizon_counts": horizon_counts,
        "price_range_counts": price_range_counts,
    }


def analyze_walk_forward(results: dict) -> None:
    """
    Print summary analysis of walk-forward results.

    Displays:
    - Number of windows
    - In-sample vs out-of-sample degradation
    - Parameter stability (for sweep results)
    - Aggregated out-of-sample metrics
    - De-duplication stats

    Args:
        results: Dictionary from run_walk_forward_single or run_walk_forward_sweep
    """
    window_results = results.get("window_results", [])
    aggregated_metrics = results.get("aggregated_oos_metrics", {})
    param_stability = results.get("param_stability", {})
    num_duplicates = results.get("num_duplicate_bets_removed", 0)

    if len(window_results) == 0:
        print("No walk-forward windows were completed")
        return

    print(f"\n{'='*80}")
    print("WALK-FORWARD ANALYSIS")
    print(f"{'='*80}\n")

    print(f"Total Windows: {len(window_results)}")
    
    # Show de-duplication info if relevant
    if num_duplicates > 0:
        print(f"⚠️  Duplicate bets removed from aggregation: {num_duplicates}")

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(window_results)

    # In-sample vs out-of-sample comparison
    if "is_return_pct" in df.columns and "oos_return_pct" in df.columns:
        print(f"\n📊 IN-SAMPLE VS OUT-OF-SAMPLE")
        print(f"  In-Sample Return (avg):     {df['is_return_pct'].mean():>8.2f}%")
        print(f"  Out-of-Sample Return (avg): {df['oos_return_pct'].mean():>8.2f}%")
        print(f"  Degradation:                {df['is_return_pct'].mean() - df['oos_return_pct'].mean():>8.2f}%")

        print(f"\n  In-Sample Sharpe (avg):     {df['is_sharpe'].mean():>8.2f}")
        print(f"  Out-of-Sample Sharpe (avg): {df['oos_sharpe'].mean():>8.2f}")

    # Aggregated out-of-sample performance
    if aggregated_metrics:
        print(f"\n🎯 AGGREGATED OUT-OF-SAMPLE PERFORMANCE")
        print(f"  Total Return:      {aggregated_metrics.get('total_return_pct', 0):>8.2f}%")
        print(f"  Annualized Return: {aggregated_metrics.get('annualized_return_pct', 0):>8.2f}%")
        print(f"  Sharpe Ratio:      {aggregated_metrics.get('sharpe_ratio', 0):>8.2f}")
        print(f"  Calmar Ratio:      {aggregated_metrics.get('calmar_ratio', 0):>8.2f}")
        print(f"  Composite Score:   {aggregated_metrics.get('composite_score', 0):>8.2f}")
        print(f"  Max Drawdown:      {aggregated_metrics.get('max_drawdown_pct', 0):>8.2f}%")
        print(f"  Total Bets:        {aggregated_metrics.get('num_bets', 0):>8,}")
        print(f"  Win Rate:          {aggregated_metrics.get('win_rate', 0)*100:>8.2f}%")

    # Parameter stability (for sweep results)
    if param_stability:
        print(f"\n🔄 PARAMETER STABILITY")
        print(f"  Total Windows: {param_stability.get('total_windows', 0)}")

        print(f"\n  Side Selections:")
        for side, count in sorted(param_stability.get("side_counts", {}).items(), key=lambda x: -x[1]):
            pct = count / param_stability["total_windows"] * 100
            print(f"    {side:<20} {count:>3} ({pct:>5.1f}%)")

        print(f"\n  Horizon Selections:")
        for horizon, count in sorted(param_stability.get("horizon_counts", {}).items(), key=lambda x: -x[1]):
            pct = count / param_stability["total_windows"] * 100
            print(f"    {horizon:<20} {count:>3} ({pct:>5.1f}%)")

        print(f"\n  Price Range Selections:")
        for price_range, count in sorted(param_stability.get("price_range_counts", {}).items(), key=lambda x: -x[1]):
            pct = count / param_stability["total_windows"] * 100
            print(f"    {price_range:<20} {count:>3} ({pct:>5.1f}%)")

    print(f"\n{'='*80}\n")
