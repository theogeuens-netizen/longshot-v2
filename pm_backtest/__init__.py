"""
Polymarket Backtesting Framework

A modular Python framework for backtesting prediction market (Polymarket) strategies.
"""

from .data import load_bets
from .strategies import StrategyConfig, select_bets_for_strategy
from .backtest import (
    run_backtest,
    run_backtest_with_lockup,
    run_multiple_backtests,
    run_multiple_backtests_chunked,
    compare_strategies,
)
from .metrics import calculate_metrics, calculate_sharpe_per_bet
from .sweep import (
    run_parameter_sweep,
    run_longshot_sweep,
    filter_sweep_results,
    analyze_sweep_results,
)
from .walkforward import (
    WalkForwardConfig,
    run_walk_forward_single,
    run_walk_forward_sweep,
    analyze_walk_forward,
)

# Optional imports - only if dependencies available
try:
    from .visualization import (
        plot_parameter_impact,
        plot_heatmap,
        plot_heatmap_grid,
        plot_risk_return,
        plot_sharpe_vs_bets,
        plot_dashboard,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from .analysis import (
        cluster_by_performance,
        get_best_per_cluster,
        plot_clusters,
        analyze_robustness,
        plot_robustness,
        build_portfolio,
        get_recommendations,
        print_recommendations,
        run_analysis,
    )
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

__version__ = "0.4.0"

__all__ = [
    # Core
    "load_bets",
    "StrategyConfig",
    "select_bets_for_strategy",
    "run_backtest",
    "run_backtest_with_lockup",
    "run_multiple_backtests",
    "run_multiple_backtests_chunked",
    "compare_strategies",
    "calculate_metrics",
    "calculate_sharpe_per_bet",
    # Sweep
    "run_parameter_sweep",
    "run_longshot_sweep",
    "filter_sweep_results",
    "analyze_sweep_results",
    # Walk-forward
    "WalkForwardConfig",
    "run_walk_forward_single",
    "run_walk_forward_sweep",
    "analyze_walk_forward",
]
