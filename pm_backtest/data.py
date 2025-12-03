"""
Data loading and validation utilities for the backtesting framework.
"""

from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np


# Required columns that must be present in the bets DataFrame
REQUIRED_COLUMNS = [
    "condition_id",
    "side",
    "horizon",
    "entry_price",
    "winner_side",
    "realized",
    "resolution_ts",
    "entry_ts",
    "roi_per_stake_gross",
    "roi_per_stake_net",
]


def load_bets(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the long-format bets table from a parquet file.

    Expected to be used with Google Drive mounted in Colab:
        from google.colab import drive
        drive.mount("/content/drive")
        path = "/content/drive/MyDrive/longshot_backups/2025-12-02/db/bets_table_long.parquet"
        bets_df = load_bets(path)

    Args:
        path: Path to the parquet file containing bets data

    Returns:
        DataFrame with validated and cleaned bets data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing or data validation fails
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Bets file not found at: {path}")

    # Load the parquet file
    df = pd.read_parquet(path)

    # Validate required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure timestamps are datetime objects
    for ts_col in ["resolution_ts", "entry_ts"]:
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

    # Basic data validation
    if df.empty:
        raise ValueError("Loaded DataFrame is empty")

    # Validate side values
    valid_sides = {"YES", "NO"}
    invalid_sides = set(df["side"].unique()) - valid_sides
    if invalid_sides:
        raise ValueError(f"Invalid side values found: {invalid_sides}. Must be 'YES' or 'NO'")

    # Validate realized values (should be 0 or 1)
    if not df["realized"].isin([0, 1]).all():
        raise ValueError("Column 'realized' must contain only 0 or 1")

    # Validate entry_price is in valid range (0, 1)
    if (df["entry_price"] <= 0).any() or (df["entry_price"] >= 1).any():
        raise ValueError("Column 'entry_price' must be in range (0, 1)")

    # Sort by entry timestamp for efficient backtesting
    df = df.sort_values("entry_ts").reset_index(drop=True)

    print(f"✓ Loaded {len(df):,} bets from {path.name}")
    print(f"  Date range: {df['entry_ts'].min()} to {df['entry_ts'].max()}")
    print(f"  Unique markets: {df['condition_id'].nunique():,}")
    print(f"  Horizons: {sorted(df['horizon'].unique())}")

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the bets DataFrame.

    Args:
        df: Bets DataFrame

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_bets": len(df),
        "unique_markets": df["condition_id"].nunique(),
        "date_range": (df["entry_ts"].min(), df["entry_ts"].max()),
        "horizons": sorted(df["horizon"].unique()),
        "sides": sorted(df["side"].unique()),
        "win_rate": df["realized"].mean(),
        "avg_entry_price": df["entry_price"].mean(),
        "avg_roi_gross": df["roi_per_stake_gross"].mean(),
        "avg_roi_net": df["roi_per_stake_net"].mean(),
    }

    # Category breakdown if available
    if "category_1" in df.columns:
        summary["top_categories"] = df["category_1"].value_counts().head(10).to_dict()

    return summary


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Union[str, pd.Timestamp, None] = None,
    end_date: Union[str, pd.Timestamp, None] = None,
) -> pd.DataFrame:
    """
    Filter bets by entry date range.

    Args:
        df: Bets DataFrame
        start_date: Start date (inclusive), or None for no lower bound
        end_date: End date (inclusive), or None for no upper bound

    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()

    if start_date is not None:
        start_date = pd.to_datetime(start_date, utc=True)
        filtered = filtered[filtered["entry_ts"] >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date, utc=True)
        filtered = filtered[filtered["entry_ts"] <= end_date]

    return filtered
