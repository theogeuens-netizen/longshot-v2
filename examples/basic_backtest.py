"""
Basic example of using the Polymarket backtesting framework.

This script demonstrates:
1. Loading bets data from Google Drive (Colab environment)
2. Defining a simple longshot strategy
3. Running a backtest
4. Running a parameter sweep
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
    compare_strategies,
)
from pm_backtest.sweep import run_longshot_sweep, filter_sweep_results, analyze_sweep_results

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
    print("\nFor Colab users:")
    print("1. Uncomment the Google Drive mount lines at the top of this script")
    print("2. Run the cells to mount your drive")
    print("3. Update DATA_PATH to match your file location")
    exit(1)

# ============================================================================
# 2. DEFINE A SIMPLE STRATEGY
# ============================================================================

print("\n" + "="*80)
print("STEP 2: DEFINING A SIMPLE LONGSHOT STRATEGY")
print("="*80)

# Example: Bet NO when NO price is between 95% and 99%
# at 7 days before resolution, for markets with volume >= 3000

strategy = StrategyConfig(
    name="longshot_no_95-99_7d",
    sides=["NO"],
    horizons=["7d"],
    price_min=0.95,
    price_max=0.99,
    min_volume=3000,
    stake_per_bet=10.0,  # Bet $10 per opportunity
)

print(strategy.describe())

# ============================================================================
# 3. RUN A BACKTEST
# ============================================================================

print("\n" + "="*80)
print("STEP 3: RUNNING BACKTEST")
print("="*80)

result = run_backtest(
    bets_df=bets_df,
    strategy=strategy,
    initial_capital=1000.0,
    stake_mode="fixed",
    verbose=True,
)

# Access the results
capital_series = result["capital_series"]
metrics = result["metrics"]

# ============================================================================
# 4. ADDITIONAL STRATEGY EXAMPLES
# ============================================================================

print("\n" + "="*80)
print("STEP 4: TESTING MULTIPLE STRATEGIES")
print("="*80)

# Define several strategies to compare
strategies = [
    # Extreme longshots (1-5%)
    StrategyConfig(
        name="yes_1-5_7d",
        sides=["YES"],
        horizons=["7d"],
        price_min=0.01,
        price_max=0.05,
        min_volume=1000,
        stake_per_bet=10.0,
    ),
    # Reverse longshots (95-99%)
    StrategyConfig(
        name="no_95-99_7d",
        sides=["NO"],
        horizons=["7d"],
        price_min=0.95,
        price_max=0.99,
        min_volume=1000,
        stake_per_bet=10.0,
    ),
    # Mid-range (10-20%)
    StrategyConfig(
        name="yes_10-20_14d",
        sides=["YES"],
        horizons=["14d"],
        price_min=0.10,
        price_max=0.20,
        min_volume=1000,
        stake_per_bet=10.0,
    ),
    # With category filter (example: Sports)
    StrategyConfig(
        name="yes_5-10_7d_sports",
        sides=["YES"],
        horizons=["7d"],
        price_min=0.05,
        price_max=0.10,
        category_include=["Sports"],
        min_volume=1000,
        stake_per_bet=10.0,
    ),
]

from pm_backtest.backtest import run_multiple_backtests

results_df = run_multiple_backtests(
    bets_df=bets_df,
    strategies=strategies,
    initial_capital=1000.0,
    verbose=False,
)

print("\nComparison of strategies:")
compare_strategies(results_df, top_n=10, sort_by="sharpe_ratio")

# ============================================================================
# 5. PARAMETER SWEEP
# ============================================================================

print("\n" + "="*80)
print("STEP 5: RUNNING PARAMETER SWEEP")
print("="*80)

# Run a comprehensive sweep over multiple parameters
sweep_results = run_longshot_sweep(
    bets_df=bets_df,
    price_buckets=[
        (0.01, 0.05),  # 1-5% extreme longshots
        (0.05, 0.10),  # 5-10%
        (0.90, 0.95),  # 90-95% reverse
        (0.95, 0.99),  # 95-99% extreme reverse
    ],
    horizons=["7d", "14d", "30d"],
    sides=["YES", "NO"],
    volume_thresholds=[None, 1000, 3000],
    categories=[None],  # No category filter for simplicity
    initial_capital=1000.0,
    stake_per_bet=10.0,
    verbose=False,
)

# Analyze overall results
analyze_sweep_results(sweep_results)

# Filter to find best strategies
print("\n" + "="*80)
print("FILTERING FOR BEST STRATEGIES")
print("="*80)

best_strategies = filter_sweep_results(
    sweep_results,
    min_bets=20,  # At least 20 bets
    min_sharpe=0.5,  # Sharpe ratio > 0.5
    min_return_pct=5.0,  # At least 5% return
)

print(f"\nFound {len(best_strategies)} strategies meeting criteria:")
compare_strategies(best_strategies, top_n=10, sort_by="sharpe_ratio")

# ============================================================================
# 6. EXPORT RESULTS (OPTIONAL)
# ============================================================================

print("\n" + "="*80)
print("STEP 6: EXPORTING RESULTS")
print("="*80)

# Save sweep results to CSV
output_path = "sweep_results.csv"
sweep_results.to_csv(output_path, index=False)
print(f"✓ Saved sweep results to {output_path}")

# Save capital series for best strategy
if len(best_strategies) > 0:
    best_strategy_name = best_strategies.iloc[0]["strategy_name"]
    print(f"\nBest strategy: {best_strategy_name}")

    # Re-run the best strategy to get capital series
    best_config = None
    for s in strategies:
        if s.name == best_strategy_name:
            best_config = s
            break

    if best_config:
        best_result = run_backtest(
            bets_df=bets_df,
            strategy=best_config,
            initial_capital=1000.0,
            verbose=False,
        )

        capital_path = "best_strategy_capital.csv"
        best_result["capital_series"].to_csv(capital_path, index=False)
        print(f"✓ Saved best strategy capital series to {capital_path}")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)
