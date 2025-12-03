"""
Performance metrics calculation for backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_metrics(
    capital_series: pd.DataFrame,
    initial_capital: float,
    freq: str = "D",
) -> dict:
    """
    Calculate comprehensive performance metrics from a backtest capital series.

    Args:
        capital_series: DataFrame with at least 'timestamp' and 'capital' columns
        initial_capital: Starting capital
        freq: Frequency for return calculations ('D' for daily, 'W' for weekly)

    Returns:
        Dictionary of performance metrics
    """
    if len(capital_series) == 0:
        return _empty_metrics()

    df = capital_series.copy()

    # Ensure we have a timestamp column
    if "entry_ts" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = df["entry_ts"]

    # Basic return metrics
    final_capital = df["capital"].iloc[-1]
    total_return = (final_capital - initial_capital) / initial_capital
    total_return_pct = total_return * 100

    # Time-based metrics
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    total_days = (end_date - start_date).total_seconds() / 86400
    total_years = total_days / 365.25

    # Annualized return
    if total_years > 0:
        annualized_return = (final_capital / initial_capital) ** (1 / total_years) - 1
        annualized_return_pct = annualized_return * 100
    else:
        annualized_return = 0
        annualized_return_pct = 0

    # Drawdown metrics
    df["peak"] = df["capital"].cummax()
    df["drawdown"] = (df["capital"] - df["peak"]) / df["peak"]
    max_drawdown = df["drawdown"].min()
    max_drawdown_pct = max_drawdown * 100

    # Resample to calculate periodic returns for Sharpe ratio
    df_resampled = df.set_index("timestamp").resample(freq).last().ffill()
    df_resampled["returns"] = df_resampled["capital"].pct_change()

    # Sharpe ratio (assuming 0 risk-free rate for simplicity)
    returns = df_resampled["returns"].dropna()
    if len(returns) > 1 and returns.std() > 0:
        # Annualize based on frequency
        periods_per_year = {"D": 252, "W": 52, "M": 12}.get(freq, 252)
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(periods_per_year)
        )
    else:
        sharpe_ratio = 0

    # Bet-level statistics
    num_bets = len(df)

    # Calculate PnL per bet if available
    if "pnl" in df.columns:
        total_pnl = df["pnl"].sum()
        avg_pnl_per_bet = df["pnl"].mean()
    else:
        total_pnl = final_capital - initial_capital
        avg_pnl_per_bet = total_pnl / num_bets if num_bets > 0 else 0

    # Win rate and other bet statistics
    if "realized" in df.columns:
        win_rate = df["realized"].mean()
        num_wins = df["realized"].sum()
        num_losses = num_bets - num_wins
    else:
        win_rate = None
        num_wins = None
        num_losses = None

    if "roi_per_stake_net" in df.columns:
        avg_roi = df["roi_per_stake_net"].mean()
        median_roi = df["roi_per_stake_net"].median()
    else:
        avg_roi = None
        median_roi = None

    if "stake" in df.columns:
        avg_stake = df["stake"].mean()
        total_staked = df["stake"].sum()
    else:
        avg_stake = None
        total_staked = None

    # Compile metrics
    metrics = {
        # Return metrics
        "total_return": total_return,
        "total_return_pct": total_return_pct,
        "annualized_return": annualized_return,
        "annualized_return_pct": annualized_return_pct,
        # Risk metrics
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        # Capital metrics
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_pnl": total_pnl,
        # Bet metrics
        "num_bets": num_bets,
        "win_rate": win_rate,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "avg_pnl_per_bet": avg_pnl_per_bet,
        "avg_roi": avg_roi,
        "median_roi": median_roi,
        "avg_stake": avg_stake,
        "total_staked": total_staked,
        # Time metrics
        "start_date": start_date,
        "end_date": end_date,
        "total_days": total_days,
        "total_years": total_years,
    }

    return metrics


def _empty_metrics() -> dict:
    """Return empty metrics dict for cases with no data."""
    return {
        "total_return": 0,
        "total_return_pct": 0,
        "annualized_return": 0,
        "annualized_return_pct": 0,
        "max_drawdown": 0,
        "max_drawdown_pct": 0,
        "sharpe_ratio": 0,
        "initial_capital": 0,
        "final_capital": 0,
        "total_pnl": 0,
        "num_bets": 0,
        "win_rate": 0,
        "num_wins": 0,
        "num_losses": 0,
        "avg_pnl_per_bet": 0,
        "avg_roi": 0,
        "median_roi": 0,
        "avg_stake": 0,
        "total_staked": 0,
        "start_date": None,
        "end_date": None,
        "total_days": 0,
        "total_years": 0,
    }


def format_metrics(metrics: dict, verbose: bool = True) -> str:
    """
    Format metrics dictionary as a readable string.

    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        verbose: If True, include all metrics; if False, only key metrics

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST PERFORMANCE SUMMARY")
    lines.append("=" * 60)

    # Return metrics
    lines.append("\n📈 RETURNS")
    lines.append(f"  Total Return:       {metrics['total_return_pct']:>10.2f}%")
    lines.append(f"  Annualized Return:  {metrics['annualized_return_pct']:>10.2f}%")

    # Risk metrics
    lines.append("\n📉 RISK")
    lines.append(f"  Max Drawdown:       {metrics['max_drawdown_pct']:>10.2f}%")
    lines.append(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}")

    # Capital
    lines.append("\n💰 CAPITAL")
    lines.append(f"  Initial:            ${metrics['initial_capital']:>10,.2f}")
    lines.append(f"  Final:              ${metrics['final_capital']:>10,.2f}")
    lines.append(f"  Total PnL:          ${metrics['total_pnl']:>10,.2f}")

    # Bet statistics
    lines.append("\n🎯 BET STATISTICS")
    lines.append(f"  Total Bets:         {metrics['num_bets']:>10,}")

    if metrics['win_rate'] is not None:
        lines.append(f"  Win Rate:           {metrics['win_rate']*100:>10.2f}%")
        lines.append(f"  Wins / Losses:      {metrics['num_wins']:>5,} / {metrics['num_losses']:<5,}")

    if metrics['avg_pnl_per_bet'] is not None:
        lines.append(f"  Avg PnL per Bet:    ${metrics['avg_pnl_per_bet']:>10.2f}")

    if metrics['avg_roi'] is not None:
        lines.append(f"  Avg ROI:            {metrics['avg_roi']*100:>10.2f}%")

    if verbose:
        lines.append("\n📅 TIME PERIOD")
        lines.append(f"  Start:              {metrics['start_date']}")
        lines.append(f"  End:                {metrics['end_date']}")
        lines.append(f"  Duration:           {metrics['total_days']:.1f} days ({metrics['total_years']:.2f} years)")

        if metrics['avg_stake'] is not None:
            lines.append("\n💵 STAKING")
            lines.append(f"  Avg Stake:          ${metrics['avg_stake']:>10,.2f}")
            lines.append(f"  Total Staked:       ${metrics['total_staked']:>10,.2f}")

    lines.append("=" * 60)

    return "\n".join(lines)


def calculate_rolling_sharpe(
    capital_series: pd.DataFrame,
    window: int = 30,
    freq: str = "D",
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.

    Args:
        capital_series: DataFrame with 'timestamp' and 'capital' columns
        window: Rolling window size in days
        freq: Frequency for return calculations

    Returns:
        Series of rolling Sharpe ratios
    """
    df = capital_series.copy()

    if "entry_ts" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = df["entry_ts"]

    df = df.set_index("timestamp").resample(freq).last().ffill()
    df["returns"] = df["capital"].pct_change()

    periods_per_year = {"D": 252, "W": 52, "M": 12}.get(freq, 252)

    rolling_sharpe = (
        df["returns"].rolling(window=window).mean()
        / df["returns"].rolling(window=window).std()
        * np.sqrt(periods_per_year)
    )

    return rolling_sharpe


def calculate_calmar_ratio(
    total_return: float,
    max_drawdown: float,
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        total_return: Total return (e.g., 0.5 for 50%)
        max_drawdown: Maximum drawdown (e.g., -0.2 for -20%)

    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return 0

    return total_return / abs(max_drawdown)
