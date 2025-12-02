"""
Polymarket Backtesting Framework

A modular Python framework for backtesting prediction market (Polymarket) strategies.
"""

from .data import load_bets
from .strategies import StrategyConfig, select_bets_for_strategy
from .backtest import run_backtest
from .metrics import calculate_metrics
from .sweep import run_parameter_sweep

__version__ = "0.1.0"

__all__ = [
    "load_bets",
    "StrategyConfig",
    "select_bets_for_strategy",
    "run_backtest",
    "calculate_metrics",
    "run_parameter_sweep",
]
