"""Feature processing methods."""

import datetime
from typing import Union

import numpy as np
import pandas as pd
import polars as pl


def get_features(
    ts: pd.Timestamp, win: int, order_book: pl.DataFrame, time_step: float
) -> list:
    """
    Get the features from the data.

    Args:
        ts: The timestamp.
        win: The window size.
        order_book: The order book data.
        time_step: The normalized time step.

    Returns:
        The features.
    """
    data = order_book.filter(
        pl.col("received_time").is_between(ts - pd.Timedelta(seconds=win), ts)
    ).collect()
    data = compute_features(data)
    data = rolling_normalization(data, win)
    row = data.row(-1)
    features = [
        time_step,  # Normalize time step
        row[1],  # Bid 0 price
        row[2],  # Bid 0 size
        row[3],  # Bid 1 price
        row[4],  # Bid 1 size
        row[5],  # Bid 2 price
        row[6],  # Bid 2 size
        row[41],  # Ask 0 price
        row[42],  # Ask 0 size
        row[43],  # Ask 1 price
        row[44],  # Ask 1 size
        row[45],  # Ask 2 price
        row[46],  # Ask 2 size
        row[84],  # Mid price
        row[85],  # Mid price change
        row[86],  # Spread
    ]
    features = np.clip(np.array(features, dtype=np.float32), a_min=-2, a_max=2)
    return features


def verify_nans(data: pd.DataFrame):
    """
    Verifies that there are no NaN values in the dataset.

    Args:
        data: The dataset to verify.

    Raises:
        ValueError: If there are NaN values in the dataset.

    Returns:
        True if there are no NaN values in the dataset, False otherwise.
    """
    test = data.isnull().values.any()
    if test:
        raise ValueError("There are NaN values in the dataset.")
    return test


def filter_data(
    data: Union[pd.DataFrame, pl.DataFrame],
    ts_start: datetime.datetime,
    ts_end: datetime.datetime,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Filters the data to the specified time range.

    Args:
        data: The data to filter.
        ts_start: The start timestamp.
        ts_end: The end timestamp.

    Returns:
        The filtered data.
    """
    return data[(data.index >= ts_start) & (data.index <= ts_end)]


def compute_features(order_book: pl.DataFrame) -> pl.DataFrame:
    """
    Computes the features for the orderbook data. If both ts_start and ts_end
    are specified, the data are filtered out to the specified time range after
    the computation of features.

    Args:
        order_book: The orderbook data.
        ts_start: The start timestamp.
        ts_end: The end timestamp.

    Returns:
        The orderbook data with the computed features.
    """
    order_book = order_book.with_columns(
        ((order_book["bid_0_price"] + order_book["ask_0_price"]) / 2).alias(
            "mid_price"
        )
    )
    order_book = order_book.with_columns(
        (
            (order_book["mid_price"] - order_book["mid_price"].shift(1))
            / order_book["mid_price"].shift(1)
        ).alias("mid_price_change")
    )
    order_book = order_book.with_columns(
        (order_book["ask_0_price"] - order_book["bid_0_price"]).alias("spread")
    )

    # Transform bid prices
    for i in range(20):
        order_book = order_book.with_columns(
            (
                (order_book[f"bid_{i}_price"] - order_book["mid_price"])
                / order_book["mid_price"]
            ).alias(f"bid_{i}_price")
        )

    # Transform ask prices
    for i in range(20):
        order_book = order_book.with_columns(
            (
                (order_book[f"ask_{i}_price"] - order_book["mid_price"])
                / order_book["mid_price"]
            ).alias(f"ask_{i}_price")
        )

    return order_book


def rolling_normalization(
    order_book: pl.DataFrame, win_size: int
) -> pl.DataFrame:
    """
    Normalize the dataset with rolling window mean and std.

    Args:
        order_book: Order book data to normalize.
        win_size: Size of the rolling window.

    Returns:
        Normalized order book data.
    """
    # Columns to normalize
    columns = (
        [f"bid_{i}_price" for i in range(20)]
        + [f"bid_{i}_size" for i in range(20)]
        + [f"ask_{i}_price" for i in range(20)]
        + [f"ask_{i}_size" for i in range(20)]
        + ["mid_price", "mid_price_change", "spread"]
    )

    # Compute rolling mean and standard deviation
    for i in columns:
        mean = order_book[i].rolling_mean(
            window_size=win_size, min_periods=win_size
        )
        std = order_book[i].rolling_std(
            window_size=win_size, min_periods=win_size
        )
        order_book = order_book.with_columns(
            ((order_book[i] - mean) / std).alias(i)
        )

    return order_book
