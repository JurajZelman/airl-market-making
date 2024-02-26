"""Methods for dowloading and processing the data."""

import datetime
import os

import lakeapi
import pandas as pd

from data.utils import (
    get_list_of_second_timestamps,
    get_parquet_args,
    get_rnd_id,
)


def download_lob_data(
    date: datetime.datetime,
    symbol: str,
    exchange: str,
    path: str,
    second: bool = True,
    raw: bool = False,
) -> None:
    """
    Download limit order book snapshots for a given date.

    Args:
        date: The date to download the data for. Data are downloaded for the
            whole day.
        symbol: The symbol to download the data for.
        exchange: The exchange to download the data for.
        path: The path to save the data to.
        second (optional): Whether to download second data or tick data.
            Defaults to `True`.
        raw (optional): Whether to download the raw unprocessed data. Defaults
            to `False`.
    """
    # Get the parquet arguments
    parquet_args = get_parquet_args()

    # Download the data
    book_data = lakeapi.load_data(
        table="book",
        start=date,
        end=date + datetime.timedelta(days=1),
        symbols=[symbol],
        exchanges=[exchange],
    ).sort_values(by="received_time")
    book_data.set_index("received_time", inplace=True)

    if raw:
        # Save the data
        if not os.path.exists(path):
            os.makedirs(path)
        prefix = "order_book"
        file_name = (
            f"{exchange}_{symbol}_{prefix}_{date.strftime('%Y_%m_%d')}.parquet"
        )
        book_data.to_parquet(os.path.join(path, file_name), **parquet_args)
        return

    # Process the dataset to second data
    if second:
        book_data = book_data.resample("1S").first().ffill()
        book_data.drop(columns=["origin_time"], inplace=True)

    # Filter data
    book_data = book_data[book_data.index >= date]
    book_data = book_data[book_data.index < date + datetime.timedelta(days=1)]
    book_data.sort_index(inplace=True)

    # # Sanity checks
    # for i in range(20):
    #     # Check not none
    #     assert (book_data[f"bid_{i}_price"].notnull()).all()
    #     assert (book_data[f"bid_{i}_size"].notnull()).all()
    #     assert (book_data[f"ask_{i}_price"].notnull()).all()
    #     assert (book_data[f"ask_{i}_size"].notnull()).all()

    #     # Check positive / non-negative
    #     assert (book_data[f"bid_{i}_price"] >= 0).all()
    #     assert (book_data[f"bid_{i}_size"] > 0).all()
    #     assert (book_data[f"ask_{i}_price"] >= 0).all()
    #     assert (book_data[f"ask_{i}_size"] > 0).all()

    # # Check indices are unique, sorted and in the correct range
    # assert len(book_data.index.unique()) == len(book_data.index)
    # assert (book_data.index == book_data.index.sort_values()).all()
    # if second:
    #     seconds = get_list_of_second_timestamps(date)
    #     assert set(book_data.index) == set(seconds)

    # Save the data
    if not os.path.exists(path):
        os.makedirs(path)
    prefix = "order_book_second" if second else "order_book"
    file_name = (
        f"{exchange}_{symbol}_{prefix}_{date.strftime('%Y_%m_%d')}.parquet"
    )
    book_data.to_parquet(os.path.join(path, file_name), **parquet_args)


def download_trade_data(
    date: datetime.datetime,
    symbol: str,
    exchange: str,
    path: str,
    tick_size: float = 0.01,
    raw: bool = False,
) -> None:
    """
    Download trade data for a given date. The data are downloaded for the whole
    day, the 'fake' trades are detected and removed, and the data are aggregated
    to second data.

    Args:
        date: The date to download the data for.
        symbol: The symbol to download the data for.
        exchange: The exchange to download the data for.
        path: The path to save the data to.
        tick_size (optional): The tick size of the symbol. Defaults to 0.01.
        raw (optional): Whether to download the raw unprocessed data. Defaults
            to `False`.
    """
    # Get the parquet arguments
    parquet_args = get_parquet_args()

    # Load the trades data
    trades = lakeapi.load_data(
        table="trades",
        start=date,
        end=date + datetime.timedelta(days=1),
        symbols=[symbol],
        exchanges=[exchange],
    ).sort_values(by="received_time")

    if raw:
        # Save the data
        if not os.path.exists(path):
            os.makedirs(path)
        prefix = "trades"
        file_name = (
            f"{exchange}_{symbol}_{prefix}_{date.strftime('%Y_%m_%d')}.parquet"
        )
        trades.to_parquet(os.path.join(path, file_name), **parquet_args)
        return

    # Load the top level order book data
    book_data = lakeapi.load_data(
        table="book",
        start=date,
        end=date + datetime.timedelta(days=1),
        symbols=[symbol],
        exchanges=[exchange],
    ).sort_values(by="received_time")
    cols = ["received_time", "symbol", "exchange", "bid_0_price", "ask_0_price"]
    book_data = book_data[cols]

    # Merge trades and book data
    book_data["future_bid"] = book_data.bid_0_price.shift(-1)
    book_data["future_ask"] = book_data.ask_0_price.shift(-1)
    df = pd.merge_asof(
        left=trades.rename(columns={"received_time": "trade_received_time"}),
        right=book_data.rename(
            columns={"received_time": "depth_received_time"}
        ),
        left_on="trade_received_time",
        right_on="depth_received_time",
        tolerance=pd.Timedelta(minutes=60),
    )
    df = df.dropna().reset_index(drop=True)

    # Detection of fake trades
    epsilon = 3 * tick_size
    df["fake"] = (
        # We consider trade to be fake when it is inside the spread (+- epsilon)
        (df["price"] > df["bid_0_price"] + epsilon)
        & (df["price"] < df["ask_0_price"] - epsilon)
        &
        # To prevent false positives, we also test for the future spread
        (df["price"] > df["future_bid"] + epsilon)
        & (df["price"] < df["future_ask"] - epsilon)
    )

    # Drop the fake trades
    df = df[df.fake == False].reset_index(drop=True)  # noqa: E712
    trades = df.drop(
        columns=[
            "fake",
            "future_bid",
            "future_ask",
            "bid_0_price",
            "ask_0_price",
            "depth_received_time",
        ]
    )
    trades = trades.rename(columns={"trade_received_time": "received_time"})
    trades = trades.set_index("received_time")

    # Aggregate the data to second data
    buy_trades = trades[trades["side"] == "buy"]
    sell_trades = trades[trades["side"] == "sell"]
    buy_trade_data_second = buy_trades.resample("S").agg(
        {"price": "ohlc", "quantity": "sum"}
    )
    sell_trade_data_second = sell_trades.resample("S").agg(
        {"price": "ohlc", "quantity": "sum"}
    )
    buy_trade_data_second["side"] = "buy"
    sell_trade_data_second["side"] = "sell"
    trades_second = pd.concat(
        [buy_trade_data_second, sell_trade_data_second], axis=0
    )
    trades_second.sort_index(inplace=True)
    trades_second.columns = [
        "_".join(col).strip() for col in trades_second.columns.values
    ]
    trades_second.rename(
        columns={"side_": "side", "quantity_quantity": "quantity"}, inplace=True
    )

    # Generate random ids for the trades
    trades_second["id"] = trades_second.apply(
        lambda x: str(get_rnd_id()), axis=1
    )

    # Impute missing seconds
    buy_trades = trades_second[trades_second["side"] == "buy"]
    sell_trades = trades_second[trades_second["side"] == "sell"]
    seconds = get_list_of_second_timestamps(date)

    # Missing second timestamps
    missing_df_buy = get_missing_trade_dataframe(buy_trades, "buy", seconds)
    missing_df_sell = get_missing_trade_dataframe(sell_trades, "sell", seconds)
    trades_second = pd.concat(
        [trades_second, missing_df_buy, missing_df_sell], axis=0
    )
    trades_second.sort_index(inplace=True)

    # Fill the null prices (just technicality, since volume is zero)
    trades_second.fillna(method="ffill", inplace=True)
    trades_second.fillna(method="bfill", inplace=True)

    # Sanity checks
    buy_trades = trades_second[trades_second["side"] == "buy"]
    sell_trades = trades_second[trades_second["side"] == "sell"]
    assert (trades_second["price_open"].notnull()).all()
    assert (trades_second["price_high"].notnull()).all()
    assert (trades_second["price_low"].notnull()).all()
    assert (trades_second["price_close"].notnull()).all()
    assert (trades_second["quantity"].notnull()).all()
    assert (trades_second["price_open"] >= 0).all()
    assert (trades_second["price_high"] >= 0).all()
    assert (trades_second["price_low"] >= 0).all()
    assert (trades_second["price_close"] >= 0).all()
    assert (trades_second["quantity"] >= 0).all()
    assert set(buy_trades.index) == set(seconds)
    assert set(sell_trades.index) == set(seconds)
    assert len(buy_trades.index.unique()) == len(buy_trades.index)
    assert len(sell_trades.index.unique()) == len(sell_trades.index)

    # Save the data
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = (
        f"{exchange}_{symbol}_trades_second_{date.strftime('%Y_%m_%d')}.parquet"
    )
    trades_second.to_parquet(os.path.join(path, file_name), **parquet_args)


def get_missing_trade_dataframe(
    data: pd.DataFrame, side: str, seconds: list
) -> pd.DataFrame:
    """
    Get a dataframe with missing seconds for a given side.

    Args:
        data: The dataframe with the data.
        side: The side to get the missing data for.
        seconds: The list of seconds to check for.
    """
    missing_seconds = set(seconds) - set(data.index.to_list())
    rnd_ids = [str(get_rnd_id()) for _ in range(len(missing_seconds))]
    empty_df = pd.DataFrame(
        {
            "price_open": [None] * len(missing_seconds),
            "price_high": [None] * len(missing_seconds),
            "price_low": [None] * len(missing_seconds),
            "price_close": [None] * len(missing_seconds),
            "quantity": [0] * len(missing_seconds),
            "side": [side] * len(missing_seconds),
            "id": rnd_ids,
            "received_time": list(missing_seconds),
        }
    )
    empty_df = empty_df.set_index("received_time")
    return empty_df
