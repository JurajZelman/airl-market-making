"""Functions for data downloading."""

import datetime
import os
from typing import Literal

import lakeapi


def download_data(
    date: datetime.datetime,
    symbol: str,
    exchange: str,
    data_type: Literal["book", "trades"],
    path_save: str,
) -> None:
    """
    Download the limit order book / trades data for a given date.

    Args:
        date: The date to download the data for.
        symbol: The symbol to download the data for.
        exchange: The exchange to download the data for.
        data_type: The type of data to download. Either "book" or "trades".
        path_save: The path to save the data to.
    """
    # Check the type
    if data_type not in ["book", "trades"]:
        raise ValueError("Invalid type. Must be either 'book' or 'trade'.")

    # Get the parquet arguments
    parquet_args = get_parquet_args()

    # Load the trades data
    data = lakeapi.load_data(
        table=data_type,
        start=date,
        end=date + datetime.timedelta(days=1),
        symbols=[symbol],
        exchanges=[exchange],
    ).sort_values(by="received_time")

    # Save the data
    os.makedirs(path_save, exist_ok=True)
    if data_type == "book":
        prefix, postfix = f"{exchange}_{symbol}_order_book", "_original"
    elif data_type == "trades":
        prefix, postfix = f"{exchange}_{symbol}_trades", ""
    file_name = f"{prefix}_{date.strftime('%Y_%m_%d')}{postfix}.parquet"
    data.to_parquet(os.path.join(path_save, file_name), **parquet_args)


def get_parquet_args():
    """
    Returns the parquet arguments for saving the data and avoiding the timestamp
    conversion issues.
    """
    return {
        "coerce_timestamps": "us",  # Coerce timestamps to microseconds
        "allow_truncated_timestamps": True,  # Allow truncated timestamps
    }
