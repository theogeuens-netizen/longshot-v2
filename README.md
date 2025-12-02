# Polymarket Backtesting Framework

A modular Python framework for backtesting prediction market (Polymarket) strategies with a focus on longshot bias opportunities.

## Overview

This framework allows you to:
- Define parametric betting strategies (price ranges, time horizons, filters)
- Backtest strategies on historical Polymarket data
- Calculate performance metrics (Sharpe ratio, drawdown, returns, etc.)
- Run parameter sweeps to find optimal configurations
- Easily extend with new features (ML models, advanced position sizing, etc.)

## Project Structure

```
pm_backtest/
├── __init__.py          # Package initialization
├── data.py              # Data loading and validation
├── strategies.py        # Strategy configuration and selection
├── metrics.py           # Performance metrics calculation
├── backtest.py          # Backtesting engine
└── sweep.py             # Parameter sweep utilities

examples/
└── basic_backtest.py    # Example usage script

README.md                # This file
```

## Installation

This framework is designed to work in Google Colab or any Python 3.x environment.

**Required dependencies:**
```bash
pip install pandas numpy
```

**Optional (for visualization):**
```bash
pip install matplotlib seaborn
```

## Quick Start

### 1. Load Your Data

```python
from google.colab import drive
from pm_backtest import load_bets

# Mount Google Drive (Colab only)
drive.mount("/content/drive")

# Load bets data
DATA_PATH = "/content/drive/MyDrive/longshot_backups/2025-12-02/db/bets_table_long.parquet"
bets_df = load_bets(DATA_PATH)
```

### 2. Define a Strategy

```python
from pm_backtest import StrategyConfig

# Example: Bet NO on extreme longshots (95-99% price) 7 days before resolution
strategy = StrategyConfig(
    name="longshot_no_95-99_7d",
    sides=["NO"],
    horizons=["7d"],
    price_min=0.95,
    price_max=0.99,
    min_volume=3000,
    stake_per_bet=10.0,
)
```

### 3. Run a Backtest

```python
from pm_backtest import run_backtest

result = run_backtest(
    bets_df=bets_df,
    strategy=strategy,
    initial_capital=1000.0,
    verbose=True,
)

# Access results
capital_series = result["capital_series"]
metrics = result["metrics"]
```

### 4. Run a Parameter Sweep

```python
from pm_backtest.sweep import run_longshot_sweep

sweep_results = run_longshot_sweep(
    bets_df=bets_df,
    price_buckets=[(0.01, 0.05), (0.05, 0.10), (0.90, 0.95), (0.95, 0.99)],
    horizons=["7d", "14d", "30d"],
    sides=["YES", "NO"],
    volume_thresholds=[None, 1000, 3000, 5000],
    initial_capital=1000.0,
    stake_per_bet=10.0,
)

# Analyze results
from pm_backtest.sweep import analyze_sweep_results
analyze_sweep_results(sweep_results)
```

## Data Format

The framework expects a **long-format bets table** where each row represents one hypothetical bet opportunity:

**Required columns:**
- `condition_id`: Market identifier
- `side`: "YES" or "NO"
- `horizon`: Time horizon (e.g., "1h", "1d", "7d", "14d", "30d")
- `entry_price`: Price at entry time (0-1 range)
- `winner_side`: Actual outcome ("YES" or "NO")
- `realized`: 1 if bet won, 0 if lost
- `resolution_ts`: Resolution timestamp (datetime)
- `entry_ts`: Entry timestamp (datetime)
- `roi_per_stake_gross`: Gross return on investment per unit stake
- `roi_per_stake_net`: Net return on investment per unit stake

**Optional columns (for filtering):**
- `category_1`, `category_2`, etc.: Market categories
- `volumeNum`, `liquidityNum`: Market volume and liquidity
- `event_category`, `series_title`, etc.: Additional metadata

## Strategy Configuration

`StrategyConfig` allows you to define complex filtering rules:

```python
strategy = StrategyConfig(
    name="my_strategy",

    # Which side(s) to bet
    sides=["YES", "NO"],

    # Which time horizons
    horizons=["7d", "14d"],

    # Price range filter
    price_min=0.05,
    price_max=0.10,

    # Category filters
    category_include=["Sports", "Politics"],  # Only these categories
    category_exclude=["Crypto"],               # Exclude these categories
    category_field="category_1",               # Which column to use

    # Volume/liquidity filters
    min_volume=3000,
    max_volume=100000,
    volume_field="volumeNum",
    min_liquidity=1000,

    # Throttling
    max_bets_per_day=5,  # Limit number of bets per day

    # Position sizing
    stake_per_bet=10.0,  # Fixed $10 per bet
)
```

## Performance Metrics

The framework calculates comprehensive metrics:

**Return Metrics:**
- Total return (%)
- Annualized return (%)

**Risk Metrics:**
- Maximum drawdown (%)
- Sharpe ratio
- Calmar ratio (return / max drawdown)

**Bet Statistics:**
- Number of bets
- Win rate
- Average PnL per bet
- Average ROI per bet
- Total capital deployed

**Time-Based:**
- Backtest duration
- Start and end dates

## Parameter Sweep

The sweep module provides tools for systematic strategy exploration:

```python
from pm_backtest.sweep import run_parameter_sweep, filter_sweep_results

# Define parameter grid
param_grid = {
    "price_ranges": [
        [(0.01, 0.05)],
        [(0.05, 0.10)],
        [(0.95, 0.99)],
    ],
    "horizons": [["7d"], ["14d"], ["7d", "14d"]],
    "min_volume": [1000, 3000, 5000],
}

# Run sweep
results = run_parameter_sweep(bets_df, param_grid, initial_capital=1000.0)

# Filter for good strategies
best = filter_sweep_results(
    results,
    min_bets=20,
    min_sharpe=0.5,
    min_return_pct=5.0,
)
```

## Example Strategies

### 1. Extreme Longshots (YES)
Bet YES on markets priced 1-5%, looking for underpriced unlikely outcomes:
```python
StrategyConfig(
    name="extreme_yes_longshots",
    sides=["YES"],
    horizons=["7d"],
    price_min=0.01,
    price_max=0.05,
    min_volume=1000,
    stake_per_bet=10.0,
)
```

### 2. Reverse Longshots (NO)
Bet NO on markets priced 95-99%, fading overconfident favorites:
```python
StrategyConfig(
    name="reverse_longshots",
    sides=["NO"],
    horizons=["7d"],
    price_min=0.95,
    price_max=0.99,
    min_volume=3000,
    stake_per_bet=10.0,
)
```

### 3. Category-Specific Strategy
Focus on a specific market category:
```python
StrategyConfig(
    name="sports_longshots",
    sides=["YES"],
    horizons=["7d", "14d"],
    price_min=0.05,
    price_max=0.15,
    category_include=["Sports"],
    min_volume=5000,
    stake_per_bet=10.0,
)
```

### 4. Multi-Horizon Strategy
Test different entry times:
```python
StrategyConfig(
    name="multi_horizon",
    sides=["NO"],
    horizons=["1d", "7d", "14d", "30d"],
    price_min=0.90,
    price_max=0.99,
    min_volume=2000,
    stake_per_bet=10.0,
)
```

## Extending the Framework

The framework is designed to be easily extensible. Here are some ideas:

### Adding ML Models
```python
# In strategies.py, add an ML-based selection function:
def select_bets_with_ml_model(bets_df, strategy, model):
    # 1. Get base selection
    selected = select_bets_for_strategy(bets_df, strategy)

    # 2. Generate features
    features = generate_features(selected)

    # 3. Predict
    predictions = model.predict(features)

    # 4. Filter by prediction threshold
    selected = selected[predictions > 0.5]

    return selected
```

### Advanced Position Sizing
```python
# In backtest.py, add new stake modes:
if stake_mode == "kelly":
    # Kelly criterion based on edge and odds
    stake = capital * kelly_fraction(edge, odds)
elif stake_mode == "percentage":
    # Fixed percentage of capital
    stake = capital * 0.02
```

### Custom Metrics
```python
# In metrics.py, add new metrics:
def calculate_sortino_ratio(returns, target=0):
    downside_returns = returns[returns < target]
    downside_std = downside_returns.std()
    return (returns.mean() - target) / downside_std if downside_std > 0 else 0
```

## Tips for Use

1. **Start simple**: Test basic strategies first before adding complex filters
2. **Validate assumptions**: Check that your price ranges and filters are selecting reasonable numbers of bets
3. **Avoid overfitting**: Don't over-optimize on historical data; save some data for out-of-sample testing
4. **Use volume filters**: Low-volume markets may have stale prices that don't reflect true probabilities
5. **Consider transaction costs**: The framework uses `roi_per_stake_net` which should include fees
6. **Monitor drawdowns**: High returns with high drawdowns may not be practical to trade

## Common Issues

**Issue: "No bets match this strategy"**
- Check that your price range is reasonable (e.g., not too narrow)
- Verify that category/volume filters aren't too restrictive
- Ensure the data actually contains bets in your selected horizons

**Issue: "FileNotFoundError"**
- Make sure Google Drive is mounted (in Colab)
- Check that the path to your parquet file is correct
- Verify the file exists at the specified location

**Issue: Low Sharpe ratio despite positive returns**
- This indicates high volatility; consider tighter filters or more diversification
- Try combining multiple uncorrelated strategies
- Consider reducing position size

## Future Enhancements

Potential additions to the framework:

- [ ] ML model integration (scikit-learn, XGBoost)
- [ ] Advanced position sizing (Kelly criterion, volatility targeting)
- [ ] Portfolio-level backtesting (multiple strategies simultaneously)
- [ ] Walk-forward optimization
- [ ] Visualization tools (capital curves, drawdown charts)
- [ ] Transaction cost modeling (more sophisticated than fixed fees)
- [ ] Market impact modeling
- [ ] Real-time strategy monitoring

## License

MIT License - feel free to use and modify as needed.