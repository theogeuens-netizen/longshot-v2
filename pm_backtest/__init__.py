"""
Polymarket Backtesting Framework

A modular Python framework for backtesting prediction market (Polymarket) strategies.
"""

from .data import load_bets
from .strategies import StrategyConfig, select_bets_for_strategy
from .backtest import run_backtest, run_backtest_with_lockup
from .metrics import calculate_metrics, calculate_sharpe_per_bet
from .sweep import run_parameter_sweep
from .walkforward import (
    WalkForwardConfig,
    run_walk_forward_single,
    run_walk_forward_sweep,
    analyze_walk_forward,
)

__version__ = "0.2.0"

__all__ = [
    "load_bets",
    "StrategyConfig",
    "select_bets_for_strategy",
    "run_backtest",
    "run_backtest_with_lockup",
    "calculate_metrics",
    "calculate_sharpe_per_bet",
    "run_parameter_sweep",
    "WalkForwardConfig",
    "run_walk_forward_single",
    "run_walk_forward_sweep",
    "analyze_walk_forward",
]
