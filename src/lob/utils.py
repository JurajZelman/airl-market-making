"""Various helper functions for the lob package."""

import os
import random
import string

import numpy as np
import pandas as pd


def generate_second_timestamps(ts_start: pd.Timestamp, ts_end: pd.Timestamp):
    """
    Generate a list of timestamps for each second between the start and end.

    Args:
        ts_start: Start timestamp.
        ts_end: End timestamp.

    Returns:
        List of timestamps.
    """
    return pd.date_range(ts_start, ts_end, freq="S").tolist()


def round_to_tick(price: float, tick_size: float):
    """
    Round a price to the nearest multiple of the tick size.

    Args:
        price: The original price to be rounded.
        tick_size: The minimum tick size for rounding. Smallest allowed value
            is 0.00001 (due to a rounding to 5 decimal places to avoid floating
            point errors).

    Returns:
        The rounded price.
    """
    return round(round(price / tick_size) * tick_size, 5)


def round_to_lot(volume: float, lot_size: float):
    """
    Round a volume to the nearest multiple of the lot size.

    Args:
        volume: The original volume to be rounded.
        lot_size: The minimum lot size for rounding.

    Returns:
        The rounded volume.
    """
    return round(round(volume / lot_size) * lot_size, 7)


def get_lot_size(exchange: "str") -> float:
    """
    Returns the lot size for the given exchange.

    Args:
        exchange: The exchange to get the lot size for.

    Returns:
        The lot size for the given exchange.
    """
    match exchange:
        case "BINANCE":
            return 0.01
        case "OKX":
            return 0.000001
        case "GATEIO":
            return 0.000001
        case "BIT.COM":
            return 0.01
        case _:
            raise ValueError(f"Lot size for exchange {exchange} not set.")


def get_tick_size(exchange: "str") -> float:
    """
    Returns the tick size for the given exchange.

    Args:
        exchange: The exchange to get the tick size for.

    Returns:
        The tick size for the given exchange.
    """
    match exchange:
        case "BINANCE":
            return 0.01
        case "OKX":
            return 0.001
        case "GATEIO":
            return 0.001
        case "BIT.COM":
            return 0.01
        case _:
            raise ValueError(f"Tick size for exchange {exchange} not set.")


def get_rnd_str(length: int = 3) -> str:
    """Get a random string of given length."""
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


def get_rnd_id(length: int = 6) -> int:
    """Get a random int of given length."""
    return random.randint(10 ** (length - 1), 10**length - 1)


def get_rnd_side() -> bool:
    """Get a random boolean."""
    return random.choice([True, False])


def get_rnd_price_around_mean(mean: float, spread: float, tick: float) -> float:
    """Get a random price around the mean value."""
    prices = list(np.arange(mean - spread, mean + spread, tick))
    return round(random.choice(prices), 2)


def get_rnd_volume() -> int:
    """Get a random volume between 1 and 200."""
    return random.randint(1, 200)


def ensure_dir_exists(path: str) -> None:
    """Check if a directory exists. If not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)
