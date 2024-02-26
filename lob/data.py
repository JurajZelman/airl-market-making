"""Methods for data handling and processing."""

import os

import pandas as pd
import polars as pl
from lob.time import TimeManager


def scan_parquet(
    name: str,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
    win: int,
    path: str,
    time_manager: TimeManager,
) -> pl.DataFrame:
    """
    Scan the parquet datasets into one dataframe.

    Args:
        name: Name of the dataset. It is assumed that the datafiles are
            using this name convention with the date appended to the end.
        ts_start: Start timestamp.
        ts_end: End timestamp.
        win: Window size to pre-load.
        path: Path to the direcory containing datasets.
        time_manager: Time manager for timestamp operations.
    """
    # Detect the days between the start and end
    first_day, last_day = ts_start.date(), ts_end.date()
    n_days = (last_day - first_day).days + 1
    days = [first_day + pd.Timedelta(days=i) for i in range(n_days)]

    # Merge the dataframes
    df = []
    for day in days:
        file_name = f"{name}_{day.strftime('%Y_%m_%d')}.parquet"
        df.append(pl.scan_parquet(os.path.join(path, file_name)))
    df = pl.concat(df, how="vertical")

    # Filter the data
    ts_start = time_manager.get_ts_larger_equal_than(ts_start)
    ts_end = time_manager.get_ts_smaller_equal_than(ts_end)
    ts_start_win = time_manager.get_ts_n_steps_from(ts_start, -win)
    df = df.filter(pl.col("received_time").is_between(ts_start_win, ts_end))

    # Read the data for the previous day if needed to account for the window
    date_start_win = ts_start_win.date()
    if date_start_win < first_day:
        file_name = f"{name}_{date_start_win.strftime('%Y_%m_%d')}.parquet"
        df_prev = pl.scan_parquet(os.path.join(path, file_name))
        df_prev = df_prev.filter(pl.col("received_time").ge(ts_start_win))
        df = pl.concat([df_prev, df], how="vertical")

    return df
