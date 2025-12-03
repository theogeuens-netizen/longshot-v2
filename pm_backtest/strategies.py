"""
Strategy definition and bet selection logic.
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class StrategyConfig:
    """
    Configuration for a prediction market betting strategy.

    A strategy defines which bets to take based on:
    - Side (YES/NO/both)
    - Time horizon before resolution
    - Price range (entry_price)
    - Market metadata filters (category, volume, liquidity, etc.)
    - Position sizing (stake per bet)

    Example:
        # Bet on longshots on both sides
        strategy = StrategyConfig(
            name="longshot_both_7d",
            sides=["both"],  # Accepts YES or NO bets in price range
            horizons=["7d"],
            price_min=0.01,
            price_max=0.05,
            min_volume=3000,
            stake_per_bet=10.0,
        )

        # Or bet on specific side with category filters
        strategy = StrategyConfig(
            name="longshot_no_7d",
            sides=["NO"],
            horizons=["7d"],
            price_min=0.95,
            price_max=0.99,
            category_include=["Grand Prix"],
            category_broad_include=["Sports"],
            min_volume=3000,
            stake_per_bet=10.0,
        )
    """

    name: str
    sides: list[str] = field(default_factory=lambda: ["YES", "NO"])
    horizons: list[str] = field(default_factory=lambda: ["7d"])
    price_min: float = 0.0
    price_max: float = 1.0
    category_include: Optional[list[str]] = None
    category_exclude: Optional[list[str]] = None
    category_broad_include: Optional[list[str]] = None
    category_broad_exclude: Optional[list[str]] = None
    min_volume: Optional[float] = None
    max_volume: Optional[float] = None
    min_liquidity: Optional[float] = None
    max_liquidity: Optional[float] = None
    max_bets_per_day: Optional[int] = None
    stake_per_bet: float = 1.0

    # Additional filter fields
    volume_field: str = "volumeNum"  # Which volume column to use
    liquidity_field: str = "liquidityNum"  # Which liquidity column to use
    category_field: str = "category_1"  # Which category column to use

    # Date filtering
    start_date: Optional[str] = None  # e.g., "2024-01-01" - filter by entry_ts >= start_date
    end_date: Optional[str] = None    # e.g., "2024-12-31" - filter by entry_ts <= end_date

    def __post_init__(self):
        """Validate strategy configuration."""
        if not self.name:
            raise ValueError("Strategy name cannot be empty")

        if not self.sides:
            raise ValueError("Must specify at least one side")

        # Handle "both" as a special case - expand to ["YES", "NO"]
        if "both" in self.sides:
            if len(self.sides) > 1:
                raise ValueError("Cannot mix 'both' with other side specifications")
            self.sides = ["YES", "NO"]

        valid_sides = {"YES", "NO"}
        for side in self.sides:
            if side not in valid_sides:
                raise ValueError(f"Invalid side: {side}. Must be 'YES', 'NO', or 'both'")

        if not self.horizons:
            raise ValueError("Must specify at least one horizon")

        if not (0 <= self.price_min < self.price_max <= 1):
            raise ValueError(
                f"Invalid price range: [{self.price_min}, {self.price_max}]. "
                "Must satisfy 0 <= price_min < price_max <= 1"
            )

        if self.stake_per_bet <= 0:
            raise ValueError("stake_per_bet must be positive")

    def describe(self) -> str:
        """Return a human-readable description of the strategy."""
        desc = [f"Strategy: {self.name}"]
        desc.append(f"  Sides: {', '.join(self.sides)}")
        desc.append(f"  Horizons: {', '.join(self.horizons)}")
        desc.append(f"  Price range: [{self.price_min:.3f}, {self.price_max:.3f}]")

        if self.category_include:
            desc.append(f"  Categories (include): {', '.join(self.category_include)}")
        if self.category_exclude:
            desc.append(f"  Categories (exclude): {', '.join(self.category_exclude)}")

        if self.category_broad_include:
            desc.append(f"  Category Broad (include): {', '.join(self.category_broad_include)}")
        if self.category_broad_exclude:
            desc.append(f"  Category Broad (exclude): {', '.join(self.category_broad_exclude)}")

        if self.min_volume is not None:
            desc.append(f"  Min volume ({self.volume_field}): {self.min_volume:,.0f}")
        if self.max_volume is not None:
            desc.append(f"  Max volume ({self.volume_field}): {self.max_volume:,.0f}")

        if self.min_liquidity is not None:
            desc.append(f"  Min liquidity ({self.liquidity_field}): {self.min_liquidity:,.0f}")
        if self.max_liquidity is not None:
            desc.append(f"  Max liquidity ({self.liquidity_field}): {self.max_liquidity:,.0f}")

        if self.max_bets_per_day is not None:
            desc.append(f"  Max bets per day: {self.max_bets_per_day}")

        if self.start_date is not None:
            desc.append(f"  Start date: {self.start_date}")
        if self.end_date is not None:
            desc.append(f"  End date: {self.end_date}")

        desc.append(f"  Stake per bet: ${self.stake_per_bet:.2f}")

        return "\n".join(desc)


def select_bets_for_strategy(
    bets_df: pd.DataFrame,
    strategy: StrategyConfig,
) -> pd.DataFrame:
    """
    Filter bets based on strategy configuration.

    Returns a DataFrame of candidate bets that match the strategy criteria,
    sorted by entry_ts (chronological order).

    Args:
        bets_df: Full bets DataFrame
        strategy: Strategy configuration

    Returns:
        Filtered DataFrame of bets matching the strategy
    """
    df = bets_df.copy()

    # Filter by side
    df = df[df["side"].isin(strategy.sides)]

    # Filter by horizon
    df = df[df["horizon"].isin(strategy.horizons)]

    # Filter by price range
    df = df[(df["entry_price"] >= strategy.price_min) & (df["entry_price"] <= strategy.price_max)]

    # Date filters
    if strategy.start_date is not None:
        start_ts = pd.to_datetime(strategy.start_date, utc=True)
        df = df[df["entry_ts"] >= start_ts]

    if strategy.end_date is not None:
        end_ts = pd.to_datetime(strategy.end_date, utc=True)
        df = df[df["entry_ts"] <= end_ts]

    # Category filters
    if strategy.category_include is not None:
        if strategy.category_field in df.columns:
            df = df[df[strategy.category_field].isin(strategy.category_include)]
        else:
            print(f"Warning: category field '{strategy.category_field}' not found in DataFrame")

    if strategy.category_exclude is not None:
        if strategy.category_field in df.columns:
            df = df[~df[strategy.category_field].isin(strategy.category_exclude)]
        else:
            print(f"Warning: category field '{strategy.category_field}' not found in DataFrame")

    # Category broad filters
    if strategy.category_broad_include is not None:
        if "category_broad" in df.columns:
            df = df[df["category_broad"].isin(strategy.category_broad_include)]
        else:
            print(f"Warning: column 'category_broad' not found in DataFrame")

    if strategy.category_broad_exclude is not None:
        if "category_broad" in df.columns:
            df = df[~df["category_broad"].isin(strategy.category_broad_exclude)]
        else:
            print(f"Warning: column 'category_broad' not found in DataFrame")

    # Volume filters
    if strategy.min_volume is not None:
        if strategy.volume_field in df.columns:
            df = df[df[strategy.volume_field] >= strategy.min_volume]
        else:
            print(f"Warning: volume field '{strategy.volume_field}' not found in DataFrame")

    if strategy.max_volume is not None:
        if strategy.volume_field in df.columns:
            df = df[df[strategy.volume_field] <= strategy.max_volume]
        else:
            print(f"Warning: volume field '{strategy.volume_field}' not found in DataFrame")

    # Liquidity filters
    if strategy.min_liquidity is not None:
        if strategy.liquidity_field in df.columns:
            df = df[df[strategy.liquidity_field] >= strategy.min_liquidity]
        else:
            print(f"Warning: liquidity field '{strategy.liquidity_field}' not found in DataFrame")

    if strategy.max_liquidity is not None:
        if strategy.liquidity_field in df.columns:
            df = df[df[strategy.liquidity_field] <= strategy.max_liquidity]
        else:
            print(f"Warning: liquidity field '{strategy.liquidity_field}' not found in DataFrame")

    # Max bets per day throttle (optional)
    if strategy.max_bets_per_day is not None:
        df = df.copy()
        df["entry_date"] = df["entry_ts"].dt.date
        # Sample up to max_bets_per_day from each day
        df = df.groupby("entry_date", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), strategy.max_bets_per_day), random_state=42)
        )
        df = df.drop(columns=["entry_date"])

    # Sort by entry timestamp (chronological order)
    df = df.sort_values("entry_ts").reset_index(drop=True)

    return df


def create_longshot_strategies(
    price_buckets: list[tuple[float, float]],
    horizons: list[str],
    sides: list[str] = None,
    stake_per_bet: float = 1.0,
    **kwargs,
) -> list[StrategyConfig]:
    """
    Factory function to create multiple longshot strategies based on price buckets.

    Args:
        price_buckets: List of (price_min, price_max) tuples
        horizons: List of horizons to test
        sides: List of sides to test (default: ["YES", "NO"]) - can include "both"
        stake_per_bet: Fixed stake per bet
        **kwargs: Additional strategy parameters (category_include, min_volume, etc.)

    Returns:
        List of StrategyConfig objects

    Example:
        # Create strategies for both sides
        strategies = create_longshot_strategies(
            price_buckets=[(0.01, 0.05), (0.05, 0.10)],
            horizons=["7d", "14d"],
            sides=["both"],
            min_volume=3000,
        )

        # Or create for specific sides
        strategies = create_longshot_strategies(
            price_buckets=[(0.90, 0.95), (0.95, 0.99)],
            horizons=["7d", "14d"],
            sides=["NO"],
            min_volume=3000,
            category_include=["Grand Prix"],
        )
    """
    if sides is None:
        sides = ["YES", "NO"]

    strategies = []

    for price_min, price_max in price_buckets:
        for horizon in horizons:
            for side in sides:
                # Handle "both" in naming
                side_str = "both" if side == "both" else side.lower()
                name = f"{side_str}_{int(price_min*100)}-{int(price_max*100)}pct_{horizon}"

                strategy = StrategyConfig(
                    name=name,
                    sides=[side],
                    horizons=[horizon],
                    price_min=price_min,
                    price_max=price_max,
                    stake_per_bet=stake_per_bet,
                    **kwargs,
                )
                strategies.append(strategy)

    return strategies
