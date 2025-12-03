"""
Walk-forward testing examples for the Polymarket backtesting framework.

This script demonstrates:
1. Testing a single strategy with walk-forward analysis
2. Parameter optimization with walk-forward sweep
3. Capital lock-up model
"""

# For Colab: Mount Google Drive
# Uncomment these lines when running in Google Colab:
# from google.colab import drive
# drive.mount("/content/drive")

import pandas as pd
from pm_backtest import (
    load_bets,
    StrategyConfig,
    run_backtest,
    WalkForwardConfig,
    run_walk_forward_single,
    run_walk_forward_sweep,
    analyze_walk_forward,
)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Path to your bets data on Google Drive
# Adjust this path to match your actual file location
DATA_PATH = "/content/drive/MyDrive/longshot_backups/2025-12-02/db/bets_table_long.parquet"

# For local testing without Google Drive, you can use a different path:
# DATA_PATH = "/path/to/local/bets_table_long.parquet"

try:
    bets_df = load_bets(DATA_PATH)
except FileNotFoundError:
    print(f"\n⚠️  File not found at: {DATA_PATH}")
    print("Please adjust DATA_PATH to point to your bets_table_long.parquet file")
    exit(1)

# ============================================================================
# 2. SIMPLE BACKTEST WITH CAPITAL LOCK-UP
# ============================================================================

print("\n" + "="*80)
print("STEP 2: SIMPLE BACKTEST WITH CAPITAL LOCK-UP")
print("="*80)

# Define a simple strategy
strategy = StrategyConfig(
    name="longshot_no_95-99_7d",
    sides=["NO"],
    horizons=["7d"],
    price_min=0.95,
    price_max=0.99,
    min_volume=3000,
    stake_per_bet=10.0,
)

print(strategy.describe())

# Run backtest WITHOUT lock-up
print("\n--- Without Capital Lock-up ---")
result_no_lockup = run_backtest(
    bets_df=bets_df,
    strategy=strategy,
    initial_capital=1000.0,
    use_lockup=False,
    verbose=True,
)

# Run backtest WITH lock-up
print("\n--- With Capital Lock-up ---")
result_with_lockup = run_backtest(
    bets_df=bets_df,
    strategy=strategy,
    initial_capital=1000.0,
    use_lockup=True,
    verbose=True,
)

# ============================================================================
# 3. WALK-FORWARD TEST FOR SINGLE STRATEGY
# ============================================================================

print("\n" + "="*80)
print("STEP 3: WALK-FORWARD TEST FOR SINGLE STRATEGY")
print("="*80)

# Configure walk-forward parameters
wf_config = WalkForwardConfig(
    in_sample_days=90,      # Optimize on 90 days
    out_of_sample_days=30,  # Test on next 30 days
    step_days=30,           # Advance 30 days for next window
    optimization_metric="composite_score",
    min_bets=15,
    initial_capital=1000.0,
    use_lockup=True,
)

print(f"\nWalk-Forward Configuration:")
print(f"  In-sample period:     {wf_config.in_sample_days} days")
print(f"  Out-of-sample period: {wf_config.out_of_sample_days} days")
print(f"  Step size:            {wf_config.step_days} days")
print(f"  Optimization metric:  {wf_config.optimization_metric}")
print(f"  Min bets required:    {wf_config.min_bets}")

# Run walk-forward test
wf_result = run_walk_forward_single(
    bets_df=bets_df,
    strategy=strategy,
    config=wf_config,
)

# Analyze results
analyze_walk_forward(wf_result)

# ============================================================================
# 4. WALK-FORWARD SWEEP WITH PARAMETER OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("STEP 4: WALK-FORWARD SWEEP WITH PARAMETER OPTIMIZATION")
print("="*80)

# Define parameter grid to search
param_grid = {
    'price_ranges': [
        [(0.01, 0.05)],  # Extreme longshots 1-5%
        [(0.05, 0.10)],  # Longshots 5-10%
        [(0.95, 0.99)],  # Extreme reverse 95-99%
    ],
    'horizons': [['7d'], ['14d'], ['7d', '14d']],
    'sides': [['YES'], ['NO'], ['both']],
    'min_volume': [None, 3000],
}

print("\nParameter Grid:")
print(f"  Price ranges: {len(param_grid['price_ranges'])} options")
print(f"  Horizons:     {len(param_grid['horizons'])} options")
print(f"  Sides:        {len(param_grid['sides'])} options")
print(f"  Min volume:   {len(param_grid['min_volume'])} options")
print(f"  Total combos: {len(param_grid['price_ranges']) * len(param_grid['horizons']) * len(param_grid['sides']) * len(param_grid['min_volume'])}")

# Run walk-forward sweep
wf_sweep_result = run_walk_forward_sweep(
    bets_df=bets_df,
    param_grid=param_grid,
    config=wf_config,
)

# Analyze sweep results
analyze_walk_forward(wf_sweep_result)

# ============================================================================
# 5. EXPORT RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: EXPORTING RESULTS")
print("="*80)

# Export walk-forward sweep results
if len(wf_sweep_result["window_results"]) > 0:
    window_df = pd.DataFrame(wf_sweep_result["window_results"])
    window_df.to_csv("walkforward_windows.csv", index=False)
    print(f"✓ Saved walk-forward windows to walkforward_windows.csv")

# Export aggregated out-of-sample capital series
if len(wf_sweep_result["aggregated_oos_capital_series"]) > 0:
    wf_sweep_result["aggregated_oos_capital_series"].to_csv("walkforward_oos_capital.csv", index=False)
    print(f"✓ Saved OOS capital series to walkforward_oos_capital.csv")

# ============================================================================
# 6. COMPARING DIFFERENT OPTIMIZATION METRICS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: COMPARING OPTIMIZATION METRICS")
print("="*80)

# Test different optimization metrics
optimization_metrics = ["composite_score", "sharpe_ratio", "calmar_ratio", "total_return_pct"]

metric_results = {}

for metric in optimization_metrics:
    print(f"\nTesting with optimization metric: {metric}")

    config = WalkForwardConfig(
        in_sample_days=90,
        out_of_sample_days=30,
        step_days=30,
        optimization_metric=metric,
        min_bets=15,
        initial_capital=1000.0,
        stake_per_bet=10.0,
        use_lockup=True,
    )

    result = run_walk_forward_sweep(
        bets_df=bets_df,
        param_grid=param_grid,
        config=config,
    )

    metric_results[metric] = result["aggregated_oos_metrics"]

# Compare results
print("\n" + "="*80)
print("OPTIMIZATION METRIC COMPARISON")
print("="*80)

comparison_df = pd.DataFrame(metric_results).T
print("\nAggregated Out-of-Sample Performance:")
print(comparison_df[['total_return_pct', 'sharpe_ratio', 'calmar_ratio', 'composite_score', 'max_drawdown_pct']].to_string())

print("\n" + "="*80)
print("WALK-FORWARD TESTING COMPLETE")
print("="*80)
