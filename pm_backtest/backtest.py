"""
Backtesting engine for prediction market strategies.
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal
from .strategies import StrategyConfig, select_bets_for_strategy
from .metrics import calculate_metrics, format_metrics


def run_backtest(
    bets_df: pd.DataFrame,
    strategy: StrategyConfig,
    initial_capital: float = 1000.0,
    stake_mode: Literal["fixed"] = "fixed",
    verbose: bool = True,
) -> dict:
    """
    Run a backtest for a given strategy.

    The backtest simulates placing bets in chronological order based on the
    strategy configuration. Each bet is held until resolution, and capital
    is updated accordingly.

    Args:
        bets_df: Full bets DataFrame
        strategy: Strategy configuration
        initial_capital: Starting capital
        stake_mode: Staking mode ("fixed" for fixed stake per bet)
        verbose: If True, print progress and results

    Returns:
        Dictionary containing:
            - 'capital_series': DataFrame with time series of capital, bets, etc.
            - 'metrics': Dictionary of performance metrics
            - 'strategy': The strategy configuration used
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running backtest: {strategy.name}")
        print(f"{'='*60}")

    # Select bets matching the strategy
    selected_bets = select_bets_for_strategy(bets_df, strategy)

    if len(selected_bets) == 0:
        if verbose:
            print("⚠️  No bets match this strategy")
        return {
            "capital_series": pd.DataFrame(),
            "metrics": {},
            "strategy": strategy,
        }

    if verbose:
        print(f"✓ Selected {len(selected_bets):,} bets matching strategy")
        print(f"  Date range: {selected_bets['entry_ts'].min()} to {selected_bets['entry_ts'].max()}")

    # Initialize backtest state
    capital = initial_capital
    results = []

    # Iterate through bets in chronological order
    for idx, bet in selected_bets.iterrows():
        # Determine stake
        if stake_mode == "fixed":
            stake = strategy.stake_per_bet
        else:
            raise ValueError(f"Unsupported stake_mode: {stake_mode}")

        # Check if we have enough capital
        if capital < stake:
            # Insufficient capital - skip this bet
            continue

        # Place the bet
        # PnL calculation: stake * ROI
        pnl = stake * bet["roi_per_stake_net"]

        # Update capital (bet is resolved immediately in our simulation)
        # In reality, capital is tied up until resolution, but for simplicity
        # we update capital at entry time with the known outcome
        capital += pnl

        # Record this bet
        results.append({
            "entry_ts": bet["entry_ts"],
            "resolution_ts": bet["resolution_ts"],
            "condition_id": bet["condition_id"],
            "side": bet["side"],
            "horizon": bet["horizon"],
            "entry_price": bet["entry_price"],
            "stake": stake,
            "realized": bet["realized"],
            "roi_per_stake_net": bet["roi_per_stake_net"],
            "pnl": pnl,
            "capital": capital,
            "strategy_name": strategy.name,
        })

    # Create capital series DataFrame
    if len(results) == 0:
        if verbose:
            print("⚠️  No bets were placed (insufficient capital or all filtered out)")
        return {
            "capital_series": pd.DataFrame(),
            "metrics": {},
            "strategy": strategy,
        }

    capital_series = pd.DataFrame(results)

    # Calculate metrics
    metrics = calculate_metrics(capital_series, initial_capital)

    if verbose:
        print(f"\n{format_metrics(metrics, verbose=True)}")

    return {
        "capital_series": capital_series,
        "metrics": metrics,
        "strategy": strategy,
    }


def run_multiple_backtests(
    bets_df: pd.DataFrame,
    strategies: list[StrategyConfig],
    initial_capital: float = 1000.0,
    stake_mode: Literal["fixed"] = "fixed",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run backtests for multiple strategies and return a summary DataFrame.

    Args:
        bets_df: Full bets DataFrame
        strategies: List of strategy configurations
        initial_capital: Starting capital for each backtest
        stake_mode: Staking mode
        verbose: If True, print progress for each strategy

    Returns:
        DataFrame with one row per strategy, containing metrics
    """
    results = []

    for i, strategy in enumerate(strategies):
        if not verbose:
            print(f"Running backtest {i+1}/{len(strategies)}: {strategy.name}", end="\r")

        backtest_result = run_backtest(
            bets_df,
            strategy,
            initial_capital=initial_capital,
            stake_mode=stake_mode,
            verbose=verbose,
        )

        metrics = backtest_result["metrics"]

        if metrics:
            # Add strategy parameters to metrics
            result_row = {
                "strategy_name": strategy.name,
                "sides": ",".join(strategy.sides),
                "horizons": ",".join(strategy.horizons),
                "price_min": strategy.price_min,
                "price_max": strategy.price_max,
                "stake_per_bet": strategy.stake_per_bet,
                **metrics,
            }
            results.append(result_row)

    if not verbose:
        print()  # New line after progress

    if len(results) == 0:
        print("⚠️  No strategies produced results")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by total return descending
    df = df.sort_values("total_return_pct", ascending=False).reset_index(drop=True)

    return df


def compare_strategies(
    results_df: pd.DataFrame,
    top_n: int = 10,
    sort_by: str = "sharpe_ratio",
) -> None:
    """
    Print a comparison of strategy results.

    Args:
        results_df: DataFrame from run_multiple_backtests
        top_n: Number of top strategies to display
        sort_by: Metric to sort by
    """
    if len(results_df) == 0:
        print("No results to compare")
        return

    if sort_by not in results_df.columns:
        print(f"Warning: '{sort_by}' not found in results, using 'total_return_pct'")
        sort_by = "total_return_pct"

    # Sort by the specified metric
    df = results_df.sort_values(sort_by, ascending=False).head(top_n)

    print(f"\n{'='*80}")
    print(f"TOP {top_n} STRATEGIES (sorted by {sort_by})")
    print(f"{'='*80}\n")

    for idx, row in df.iterrows():
        print(f"{row['strategy_name']}")
        print(f"  Return: {row['total_return_pct']:>8.2f}%  |  "
              f"Sharpe: {row['sharpe_ratio']:>6.2f}  |  "
              f"Max DD: {row['max_drawdown_pct']:>7.2f}%  |  "
              f"Bets: {row['num_bets']:>6,.0f}  |  "
              f"Win Rate: {row['win_rate']*100:>5.1f}%")

    print(f"{'='*80}\n")


def export_backtest_results(
    capital_series: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Export backtest capital series to CSV.

    Args:
        capital_series: DataFrame from run_backtest()['capital_series']
        output_path: Path to save CSV file
    """
    capital_series.to_csv(output_path, index=False)
    print(f"✓ Exported backtest results to {output_path}")
