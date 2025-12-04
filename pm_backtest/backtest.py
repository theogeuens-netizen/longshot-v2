"""
Backtesting engine for prediction market strategies.

Supports:
- Single strategy backtesting
- Multiple strategy backtesting with parallel execution
- Capital lock-up model
- Checkpointing for long sweeps
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal, List
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Optional tqdm import
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable=None, total=None, **kwargs):
        return iterable

from .strategies import StrategyConfig, select_bets_for_strategy
from .metrics import calculate_metrics, format_metrics


def run_backtest(
    bets_df: pd.DataFrame,
    strategy: StrategyConfig,
    initial_capital: float = 1000.0,
    stake_mode: Literal["fixed"] = "fixed",
    use_lockup: bool = False,
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
        use_lockup: If True, use capital lock-up model (tracks pending bets)
        verbose: If True, print progress and results

    Returns:
        Dictionary containing:
            - 'capital_series': DataFrame with time series of capital, bets, etc.
            - 'metrics': Dictionary of performance metrics
            - 'strategy': The strategy configuration used
    """
    # Use lockup model if requested
    if use_lockup:
        return run_backtest_with_lockup(
            bets_df=bets_df,
            strategy=strategy,
            initial_capital=initial_capital,
            stake_mode=stake_mode,
            verbose=verbose,
        )

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


def run_backtest_with_lockup(
    bets_df: pd.DataFrame,
    strategy: StrategyConfig,
    initial_capital: float = 1000.0,
    stake_mode: Literal["fixed"] = "fixed",
    verbose: bool = True,
) -> dict:
    """
    Run a backtest with capital lock-up model.

    This version tracks pending bets (placed but not resolved) and locks capital
    until resolution. Only allows new bets if available capital is sufficient.

    Args:
        bets_df: Full bets DataFrame
        strategy: Strategy configuration
        initial_capital: Starting capital
        stake_mode: Staking mode ("fixed" for fixed stake per bet)
        verbose: If True, print progress and results

    Returns:
        Dictionary containing:
            - 'capital_series': DataFrame with capital, available_capital, locked_capital, etc.
            - 'metrics': Dictionary of performance metrics
            - 'strategy': The strategy configuration used
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running backtest with lockup: {strategy.name}")
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
    pending_bets = []  # List of dicts: {stake, pnl, resolution_ts, ...}
    results = []

    # Iterate through bets in chronological order
    for idx, bet in selected_bets.iterrows():
        current_time = bet["entry_ts"]

        # Settle any pending bets that have resolved by now
        still_pending = []
        for pending in pending_bets:
            if pending["resolution_ts"] <= current_time:
                # This bet has resolved - update capital
                capital += pending["pnl"]
            else:
                # Still pending
                still_pending.append(pending)
        pending_bets = still_pending

        # Calculate locked and available capital
        locked_capital = sum(p["stake"] for p in pending_bets)
        available_capital = capital - locked_capital

        # Determine stake
        if stake_mode == "fixed":
            stake = strategy.stake_per_bet
        else:
            raise ValueError(f"Unsupported stake_mode: {stake_mode}")

        # Check if we have enough available capital
        if available_capital < stake:
            # Insufficient available capital - skip this bet
            continue

        # Place the bet
        pnl = stake * bet["roi_per_stake_net"]

        # Add to pending bets
        pending_bets.append({
            "stake": stake,
            "pnl": pnl,
            "resolution_ts": bet["resolution_ts"],
        })

        # Record this bet at entry time
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
            "available_capital": available_capital,
            "locked_capital": locked_capital + stake,  # Include this bet's stake
            "num_pending_bets": len(pending_bets),
            "strategy_name": strategy.name,
        })

    # Settle any remaining pending bets at the end
    for pending in pending_bets:
        capital += pending["pnl"]

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
        print(f"\n💼 CAPITAL LOCKUP STATS")
        print(f"  Max Locked:         ${capital_series['locked_capital'].max():>10,.2f}")
        print(f"  Avg Locked:         ${capital_series['locked_capital'].mean():>10,.2f}")
        print(f"  Max Pending Bets:   {capital_series['num_pending_bets'].max():>10,}")
        print(f"  Avg Pending Bets:   {capital_series['num_pending_bets'].mean():>10.1f}")

    return {
        "capital_series": capital_series,
        "metrics": metrics,
        "strategy": strategy,
    }


# =============================================================================
# Worker Pool with Shared DataFrame (avoids pickling DataFrame per task)
# =============================================================================

# Global variable for worker processes - initialized once per worker
_worker_bets_df = None
_worker_initial_capital = None
_worker_stake_mode = None


def _init_worker(bets_df: pd.DataFrame, initial_capital: float, stake_mode: str):
    """
    Initializer for worker processes.
    
    Called ONCE per worker when the pool starts. The DataFrame is pickled
    only once per worker, not once per task. This is the key optimization.
    """
    global _worker_bets_df, _worker_initial_capital, _worker_stake_mode
    _worker_bets_df = bets_df
    _worker_initial_capital = initial_capital
    _worker_stake_mode = stake_mode


def _strategy_to_dict(strategy: StrategyConfig) -> dict:
    """Convert StrategyConfig to a plain dict for pickling."""
    return {
        "name": strategy.name,
        "sides": strategy.sides,
        "horizons": strategy.horizons,
        "price_min": strategy.price_min,
        "price_max": strategy.price_max,
        "stake_per_bet": strategy.stake_per_bet,
        "min_volume": getattr(strategy, 'min_volume', None),
        "max_volume": getattr(strategy, 'max_volume', None),
        "min_liquidity": getattr(strategy, 'min_liquidity', None),
        "max_liquidity": getattr(strategy, 'max_liquidity', None),
        "category_include": getattr(strategy, 'category_include', None),
        "category_exclude": getattr(strategy, 'category_exclude', None),
        "category_broad_include": getattr(strategy, 'category_broad_include', None),
        "category_broad_exclude": getattr(strategy, 'category_broad_exclude', None),
        "category_field": getattr(strategy, 'category_field', 'category_1'),
        "volume_field": getattr(strategy, 'volume_field', 'volumeNum'),
        "liquidity_field": getattr(strategy, 'liquidity_field', 'liquidityNum'),
        "max_bets_per_day": getattr(strategy, 'max_bets_per_day', None),
        "start_date": getattr(strategy, 'start_date', None),
        "end_date": getattr(strategy, 'end_date', None),
    }


def _dict_to_strategy(d: dict) -> StrategyConfig:
    """Convert a plain dict back to StrategyConfig."""
    return StrategyConfig(
        name=d["name"],
        sides=d["sides"],
        horizons=d["horizons"],
        price_min=d["price_min"],
        price_max=d["price_max"],
        stake_per_bet=d["stake_per_bet"],
        min_volume=d.get("min_volume"),
        max_volume=d.get("max_volume"),
        min_liquidity=d.get("min_liquidity"),
        max_liquidity=d.get("max_liquidity"),
        category_include=d.get("category_include"),
        category_exclude=d.get("category_exclude"),
        category_broad_include=d.get("category_broad_include"),
        category_broad_exclude=d.get("category_broad_exclude"),
        category_field=d.get("category_field", "category_1"),
        volume_field=d.get("volume_field", "volumeNum"),
        liquidity_field=d.get("liquidity_field", "liquidityNum"),
        max_bets_per_day=d.get("max_bets_per_day"),
        start_date=d.get("start_date"),
        end_date=d.get("end_date"),
    )


def _run_single_backtest_worker(strategy_dict: dict) -> Optional[dict]:
    """
    Worker function for parallel backtest execution.
    
    Uses global _worker_bets_df initialized by _init_worker().
    Only the strategy_dict is passed per task (small, fast to pickle).
    
    Args:
        strategy_dict: Plain dict representation of StrategyConfig
        
    Returns:
        Dictionary with strategy results, or None if no results
    """
    global _worker_bets_df, _worker_initial_capital, _worker_stake_mode
    
    try:
        # Reconstruct StrategyConfig from dict
        strategy = _dict_to_strategy(strategy_dict)
        
        backtest_result = run_backtest(
            _worker_bets_df,
            strategy,
            initial_capital=_worker_initial_capital,
            stake_mode=_worker_stake_mode,
            verbose=False,
        )
        
        metrics = backtest_result["metrics"]
        
        if metrics:
            return {
                "strategy_name": strategy.name,
                "sides": ",".join(strategy.sides),
                "horizons": ",".join(strategy.horizons),
                "price_min": strategy.price_min,
                "price_max": strategy.price_max,
                "stake_per_bet": strategy.stake_per_bet,
                "min_volume": strategy.min_volume,
                "category_include": strategy.category_include,
                "category_broad_include": strategy.category_broad_include,
                "category_exclude": strategy.category_exclude,
                "category_broad_exclude": strategy.category_broad_exclude,
                **metrics,
            }
    except Exception as e:
        print(f"⚠️  Error in strategy {strategy_dict.get('name', '?')}: {e}")
    
    return None


def run_multiple_backtests(
    bets_df: pd.DataFrame,
    strategies: List[StrategyConfig],
    initial_capital: float = 1000.0,
    stake_mode: Literal["fixed"] = "fixed",
    verbose: bool = False,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 50,
    n_jobs: int = -1,
    use_parallel: bool = True,
) -> pd.DataFrame:
    """
    Run backtests for multiple strategies with parallel execution and checkpointing.

    Args:
        bets_df: Full bets DataFrame
        strategies: List of strategy configurations
        initial_capital: Starting capital for each backtest
        stake_mode: Staking mode
        verbose: If True, show progress bar (individual backtest prints are suppressed)
        checkpoint_path: Optional path to a parquet file for saving/resuming results
        checkpoint_every: Save checkpoint every N strategies (default: 50)
        n_jobs: Number of parallel workers. -1 = use all CPUs, 1 = sequential
        use_parallel: Whether to use parallel execution (default: True)

    Returns:
        DataFrame with one row per strategy, containing metrics
    """
    results: List[dict] = []
    start_index = 0

    # Determine number of workers
    # Determine number of workers
    # For ProcessPoolExecutor with initializer, use CPU count for optimal parallelism
    if n_jobs == -1:
        n_workers = mp.cpu_count()
    elif n_jobs <= 0:
        n_workers = max(1, mp.cpu_count() + n_jobs)
    else:
        n_workers = n_jobs
    
    # Disable parallel for very small sweeps (overhead not worth it)
    if len(strategies) < 20 or not use_parallel or n_workers == 1:
        use_parallel = False
        n_workers = 1

    # --------------------------------------------------
    # Resume from checkpoint if available
    # --------------------------------------------------
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            checkpoint_df = pd.read_parquet(checkpoint_path)
            results = checkpoint_df.to_dict("records")
            start_index = len(results)
            print(f"✓ Resuming from checkpoint: {start_index}/{len(strategies)} strategies completed")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint from {checkpoint_path}: {e}")
            results = []
            start_index = 0

    remaining_strategies = strategies[start_index:]
    
    if len(remaining_strategies) == 0:
        print("✓ All strategies already completed!")
        return pd.DataFrame(results)

    # --------------------------------------------------
    # Run backtests
    # --------------------------------------------------
    print(f"\n{'='*60}")
    print(f"RUNNING BACKTESTS")
    print(f"{'='*60}")
    print(f"  Total strategies: {len(strategies):,}")
    print(f"  Remaining: {len(remaining_strategies):,}")
    print(f"  Mode: {'Parallel' if use_parallel else 'Sequential'} ({n_workers} workers)")
    print(f"  Checkpoint: {'Enabled' if checkpoint_path else 'Disabled'} (every {checkpoint_every})")
    print(f"{'='*60}\n")

    if use_parallel:
        results = _run_parallel_backtests(
            bets_df=bets_df,
            strategies=remaining_strategies,
            initial_capital=initial_capital,
            stake_mode=stake_mode,
            existing_results=results,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            n_workers=n_workers,
            start_index=start_index,
            total_strategies=len(strategies),
        )
    else:
        results = _run_sequential_backtests(
            bets_df=bets_df,
            strategies=remaining_strategies,
            initial_capital=initial_capital,
            stake_mode=stake_mode,
            existing_results=results,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            start_index=start_index,
            total_strategies=len(strategies),
        )

    if len(results) == 0:
        print("⚠️  No strategies produced results")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by total return descending
    df = df.sort_values("total_return_pct", ascending=False).reset_index(drop=True)

    # Final checkpoint save
    if checkpoint_path is not None:
        df.to_parquet(checkpoint_path, index=False)
        print(f"\n✓ Final results saved to {checkpoint_path}")

    return df


def _run_sequential_backtests(
    bets_df: pd.DataFrame,
    strategies: List[StrategyConfig],
    initial_capital: float,
    stake_mode: str,
    existing_results: List[dict],
    checkpoint_path: Optional[str],
    checkpoint_every: int,
    start_index: int,
    total_strategies: int,
) -> List[dict]:
    """Run backtests sequentially with progress bar."""
    results = existing_results.copy()
    
    # Progress bar
    pbar = tqdm(
        strategies,
        total=len(strategies),
        desc="Backtesting",
        unit="strat",
        initial=0,
        ncols=100,
    ) if TQDM_AVAILABLE else strategies

    for i, strategy in enumerate(pbar):
        actual_index = start_index + i
        
        backtest_result = run_backtest(
            bets_df,
            strategy,
            initial_capital=initial_capital,
            stake_mode=stake_mode,
            verbose=False,
        )

        metrics = backtest_result["metrics"]

        if metrics:
            result_row = {
                "strategy_name": strategy.name,
                "sides": ",".join(strategy.sides),
                "horizons": ",".join(strategy.horizons),
                "price_min": strategy.price_min,
                "price_max": strategy.price_max,
                "stake_per_bet": strategy.stake_per_bet,
                "min_volume": getattr(strategy, 'min_volume', None),
                "category_include": getattr(strategy, 'category_include', None),
                "category_broad_include": getattr(strategy, 'category_broad_include', None),
                "category_exclude": getattr(strategy, 'category_exclude', None),
                "category_broad_exclude": getattr(strategy, 'category_broad_exclude', None),
                **metrics,
            }
            results.append(result_row)

        # Update progress bar description
        if TQDM_AVAILABLE and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'done': f"{actual_index + 1}/{total_strategies}",
                'results': len(results),
            })

        # Checkpoint
        if checkpoint_path is not None and (actual_index + 1) % checkpoint_every == 0:
            pd.DataFrame(results).to_parquet(checkpoint_path, index=False)

    return results


def _run_parallel_backtests(
    bets_df: pd.DataFrame,
    strategies: List[StrategyConfig],
    initial_capital: float,
    stake_mode: str,
    existing_results: List[dict],
    checkpoint_path: Optional[str],
    checkpoint_every: int,
    n_workers: int,
    start_index: int,
    total_strategies: int,
) -> List[dict]:
    """
    Run backtests in parallel using ThreadPoolExecutor.
    
    We use ThreadPoolExecutor instead of ProcessPoolExecutor because:
    - ProcessPoolExecutor hangs in Jupyter/Colab environments
    - ThreadPoolExecutor shares memory (no pickle overhead)
    - Pandas releases the GIL for most operations, so we still get parallelism
    """
    from concurrent.futures import ThreadPoolExecutor
    
    results = existing_results.copy()

    # Progress bar
    pbar = tqdm(
        total=len(strategies),
        desc=f"Backtesting ({n_workers} threads)",
        unit="strat",
        ncols=100,
    ) if TQDM_AVAILABLE else None

    completed = 0
    
    def run_single(strategy: StrategyConfig) -> Optional[dict]:
        """Run a single backtest."""
        try:
            result = run_backtest(
                bets_df,
                strategy,
                initial_capital=initial_capital,
                stake_mode=stake_mode,
                verbose=False,
            )
            
            metrics = result["metrics"]
            if metrics:
                return {
                    "strategy_name": strategy.name,
                    "sides": ",".join(strategy.sides),
                    "horizons": ",".join(strategy.horizons),
                    "price_min": strategy.price_min,
                    "price_max": strategy.price_max,
                    "stake_per_bet": strategy.stake_per_bet,
                    "min_volume": strategy.min_volume,
                    "category_include": strategy.category_include,
                    "category_broad_include": strategy.category_broad_include,
                    "category_exclude": strategy.category_exclude,
                    "category_broad_exclude": strategy.category_broad_exclude,
                    **metrics,
                }
        except Exception as e:
            print(f"⚠️  Error in {strategy.name}: {e}")
        return None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(run_single, s): i for i, s in enumerate(strategies)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            actual_index = start_index + completed
            
            try:
                result = future.result(timeout=60)  # 60s timeout per strategy
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"⚠️  Strategy {idx} failed: {e}")
            
            completed += 1
            
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    'done': f"{actual_index + 1}/{total_strategies}",
                    'results': len(results),
                })
            
            # Checkpoint
            if checkpoint_path is not None and completed % checkpoint_every == 0:
                pd.DataFrame(results).to_parquet(checkpoint_path, index=False)

    if pbar is not None:
        pbar.close()

    return results


def run_multiple_backtests_chunked(
    bets_df: pd.DataFrame,
    strategies: List[StrategyConfig],
    initial_capital: float = 1000.0,
    stake_mode: Literal["fixed"] = "fixed",
    chunk_size: int = 100,
    checkpoint_path: Optional[str] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Run backtests in chunks with automatic checkpointing.
    
    Useful for very large sweeps where you want finer-grained progress.
    
    Args:
        bets_df: Full bets DataFrame
        strategies: List of strategy configurations
        initial_capital: Starting capital
        stake_mode: Staking mode
        chunk_size: Number of strategies per chunk
        checkpoint_path: Path to save results
        n_jobs: Number of parallel workers
        
    Returns:
        DataFrame with results
    """
    all_results = []
    
    # Load existing checkpoint if available
    start_chunk = 0
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            existing = pd.read_parquet(checkpoint_path)
            all_results = existing.to_dict("records")
            start_chunk = len(all_results) // chunk_size
            print(f"✓ Resuming from chunk {start_chunk}")
        except Exception:
            pass
    
    # Process in chunks
    n_chunks = (len(strategies) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(start_chunk, n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(strategies))
        chunk_strategies = strategies[chunk_start:chunk_end]
        
        print(f"\n📦 Processing chunk {chunk_idx + 1}/{n_chunks} ({chunk_start} - {chunk_end})")
        
        chunk_results = run_multiple_backtests(
            bets_df=bets_df,
            strategies=chunk_strategies,
            initial_capital=initial_capital,
            stake_mode=stake_mode,
            verbose=True,
            checkpoint_path=None,  # We handle checkpointing here
            n_jobs=n_jobs,
        )
        
        all_results.extend(chunk_results.to_dict("records"))
        
        # Save after each chunk
        if checkpoint_path is not None:
            pd.DataFrame(all_results).to_parquet(checkpoint_path, index=False)
            print(f"  ✓ Checkpoint saved ({len(all_results)} total)")
    
    return pd.DataFrame(all_results)


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
