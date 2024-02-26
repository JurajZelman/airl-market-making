"""Methods for handling of time and timestamps."""

import datetime

import polars as pl


class TimeManager:
    """Timeline class for timestamps management."""

    def __init__(
        self,
        exchange: str,
        symbol: str,
        ts_start: datetime.datetime,
        ts_end: datetime.datetime,
        path: str,
        win_size: int = None,
        max_steps: int = None,
    ) -> None:
        """
        Initialize the timeline.

        Args:
            exchange: Exchange to load the timeline for.
            symbol: Symbol to load the timeline for.
            ts_start: Start timestamp.
            ts_end: End timestamp.
            path: Path to the directory containing datasets.
            win_size: Window size for the number of timestamps to preload before
                the start timestamp. If None, preload no timestamps.
            max_steps: Maximum number of steps to load. If None, load all
                timestamps in the given range.
        """
        self.exchange = exchange
        self.symbol = symbol
        self.ts_start = ts_start
        self.ts_end = ts_end
        self.path = path
        self.win_size = win_size
        self.max_steps = max_steps
        self.timeline = self.load_timeline()

    def get_current_time_step(self) -> int:
        """Get the current time step."""
        if self.win_size is None:
            return self.iter
        return self.iter - self.win_size

    def get_current_time_ratio(self) -> float:
        """Get the current time ratio."""
        return self.get_current_time_step() / self.last_time_step

    def get_last_time_step(self) -> int:
        """Get the last time step."""
        return self.last_time_step

    def get_current_index(self) -> int:
        """Get the current index."""
        return self.iter

    def get_current_ts(self) -> datetime.datetime:
        """Get the current timestamp."""
        if self.iter >= len(self.timeline):
            return None
        return self.timeline[self.iter]

    def step_forward(self) -> datetime.datetime:
        """Step forward and return the next timestamp."""
        self.iter += 1
        if self.iter < len(self.timeline):
            return self.timeline[self.iter]
        return None

    def get_next_ts(self) -> datetime.datetime:
        """
        Get the next timestamp. If the current timestamp is the last one,
        return None.
        """
        next_idx = self.iter + 1
        if next_idx < len(self.timeline):
            return self.timeline[next_idx]
        return None

    def get_previous_ts(self) -> datetime.datetime:
        """
        Get the previous timestamp. If the current timestamp is the first one,
        return None.
        """
        prev_idx = self.iter - 1
        if prev_idx >= 0:
            return self.timeline[prev_idx]
        return None

    def load_timeline(self) -> list[datetime.datetime]:
        """Load the timeline for the given exchange and symbol."""
        timeline = []

        # 1. Load the timeline for all days in the given range.
        dates = []
        date = self.ts_start.date()
        while date <= self.ts_end.date():
            dates.append(date)
            date += datetime.timedelta(days=1)

        for date in dates:
            dt = date.strftime("%Y_%m_%d")
            name = f"{self.path}/{self.exchange}_{self.symbol}_order_book_{dt}"
            df = pl.scan_parquet(f"{name}.parquet")
            timestamps = df.select(pl.col("received_time")).collect()
            timestamps = timestamps.get_columns()[0].to_list()
            timeline += timestamps

        # 2. Check whether the timeline contains a sufficient window and if not,
        #    load the previous day.
        if self.win_size is not None:
            idx_start = next(
                i for i, x in enumerate(timeline) if x > self.ts_start
            )
            idx_win_start = idx_start - self.win_size

            # Preload one more file for sufficient window
            if idx_win_start < 0:
                dt = (dates[0] - datetime.timedelta(days=1)).strftime(
                    "%Y_%m_%d"
                )
                name = (
                    f"{self.path}/{self.exchange}_{self.symbol}_order_book_{dt}"
                )
                df = pl.scan_parquet(f"{name}.parquet")
                timestamps = df.select(pl.col("received_time")).collect()
                timestamps = timestamps.get_columns()[0].to_list()
                timeline = timestamps + timeline

        # 3. Filter the timeline to the given range.
        idx_start = next(i for i, x in enumerate(timeline) if x > self.ts_start)
        idx_win_start = idx_start - self.win_size
        try:
            idx_end = next(i for i, x in enumerate(timeline) if x > self.ts_end)
        except StopIteration:
            idx_end = None
        if not idx_end:
            timeline = timeline[idx_win_start:]
        else:
            timeline = timeline[idx_win_start:idx_end]
        if self.max_steps is not None:
            timeline = timeline[
                : min(self.max_steps + self.win_size, len(timeline))
            ]

        # 4. Set the current index to the start index.
        self.iter = self.win_size if self.win_size is not None else 0
        self.last_time_step = len(timeline) - self.iter - 1
        return timeline

    def get_timeline(self, with_win: bool = False) -> list[datetime.datetime]:
        """
        Get the timeline.

        Args:
            with_win: Whether to return the timeline with the window or not.

        Returns:
            The timeline.
        """
        if with_win and self.win_size is not None:
            return self.timeline
        if with_win and self.win_size is None:
            raise ValueError("Window size is None.")
        return self.timeline[self.win_size :]

    def get_ts_n_steps_from(
        self, ts: datetime.datetime, n: int
    ) -> datetime.datetime:
        """
        Get the timestamp n steps from the given timestamp. Positive n
        represents n steps forward, negative n represents n steps backward.

        Args:
            ts: Timestamp to get the n steps from.
            n: Number of steps to get.
        """
        index = self.get_index_for_ts(ts)
        return self.get_ts_for_index(index + n)

    def get_index_for_ts(self, ts: datetime.datetime) -> int:
        """
        Get the index for the given timestamp.

        Args:
            ts: Timestamp to get the index for.
        """
        return self.timeline.index(ts)

    def get_ts_for_index(self, index: int) -> datetime.datetime:
        """
        Get the timestamp for the given index.

        Args:
            index: Index of the timestamp to get.
        """
        if index < 0 or index >= len(self.timeline):
            raise IndexError("Index out of bounds.")
        return self.timeline[index]

    def get_ts_larger_equal_than(self, ts) -> datetime.datetime:
        """
        Get the first timestamp larger than the given timestamp.

        Args:
            ts: Timestamp to get the next timestamp for.
        """
        return next(x for x in self.timeline if x >= ts)

    def get_ts_smaller_equal_than(self, ts) -> datetime.datetime:
        """
        Get the first timestamp smaller than the given timestamp.

        Args:
            ts: Timestamp to get the previous timestamp for.

        Returns:
            The first smaller timestamp.
        """
        return next(x for x in reversed(self.timeline) if x <= ts)
