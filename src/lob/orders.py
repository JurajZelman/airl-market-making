"""Classes representing orders available at the market."""

import sys
from abc import ABC, abstractmethod
from datetime import datetime


class Order(ABC):
    """Abstract base class for orders."""

    @abstractmethod
    def __init__(
        self,
        ticker: str,
        id: str,
        trader_id: int,
        side: bool,
        volume: float,
        entry_time: datetime,
    ) -> None:
        """
        Initialize an order.

        Args:
            ticker: Ticker of the traded security.
            id: Unique identifier of the order.
            trader_id: Unique identifier of the trader who posted the order.
            side: Side of the order (True for buy, False for sell).
            volume: Volume of the security to buy or sell.
            entry_time: Datetime when the order was posted.
        """
        self.ticker = ticker
        self.id = id
        self.trader_id = trader_id
        self.side = side
        self.volume = volume
        self.entry_time = entry_time


class LimitOrder(Order):
    """Limit order class."""

    def __init__(
        self,
        ticker: str,
        id: str,
        trader_id: int,
        side: bool,
        volume: float,
        entry_time: datetime,
        price: float,
    ) -> None:
        """
        Initialize a limit order.

        Args:
            ticker: Ticker of the traded security.
            id: Unique identifier of the order.
            trader_id: Unique identifier of the trader who posted the order.
            side: Side of the order (True for buy, False for sell).
            volume: Volume of the security to buy or sell.
            entry_time: Datetime when the order was posted.
            price: Limit price of the order.
        """
        super().__init__(ticker, id, trader_id, side, volume, entry_time)
        self.price = price

    def __repr__(self) -> str:
        """
        Return a string representation of the limit order.

        Returns:
            A string representation of the limit order.
        """
        return (
            f"LimitOrder(ticker={self.ticker}, id={self.id}, "
            f"trader_id={self.trader_id}, side={self.side}, "
            f"volume={self.volume}, entry_time={self.entry_time}, "
            f"price={self.price})"
        )


class MarketOrder(LimitOrder):
    """Market order class."""

    def __init__(
        self,
        ticker: str,
        id: str,
        trader_id: int,
        side: bool,
        volume: float,
        entry_time: datetime,
    ) -> None:
        """
        Initialize a market order.

        Args:
            ticker: Ticker of the traded security.
            id: Unique identifier of the order.
            trader_id: Unique identifier of the trader who posted the order.
            side: Side of the order (True for buy, False for sell).
            volume: Volume of the security to buy or sell.
            entry_time: Datetime when the order was posted.
        """
        price = sys.maxsize if side else -sys.maxsize
        super().__init__(ticker, id, trader_id, side, volume, entry_time, price)

    def __repr__(self) -> str:
        """
        Return a string representation of the market order.

        Returns:
            A string representation of the market order.
        """
        return (
            f"MarketOrder(ticker={self.ticker}, id={self.id}, "
            f"trader_id={self.trader_id}, side={self.side}, "
            f"volume={self.volume}, entry_time={self.entry_time})"
        )
