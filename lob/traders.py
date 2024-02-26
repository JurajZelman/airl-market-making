"""Implementations of market participants."""

import datetime
import math
import pickle
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np
import polars as pl

from lob.commissions import CommissionModel
from lob.limit_order_book import LimitOrderBook
from lob.orders import LimitOrder, MarketOrder, Order
from lob.utils import get_rnd_str, round_to_lot, round_to_tick

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")
COIN_SYMBOL = "SOL-USDT"


class Trader(ABC):
    """Abstract class representing a trader at the exchange."""

    @abstractmethod
    def __init__(self, id: str):
        """
        Initialize a trader.

        Args:
            id: Unique identifier of the trader.
        """
        self.id = id  # ID of the trader
        self.lob = None  # Reference to the limit order book
        self.active_orders = []  # List of trader's active orders

    def set_lob(self, lob: LimitOrderBook) -> None:
        """
        Set the limit order book for the trader.

        Args:
            lob: Limit order book.
        """
        self.lob = lob

    @abstractmethod
    def place_orders(
        self, time_step: int, *args, **kwargs
    ) -> tuple[list, list]:
        """
        Create lists of orders to be canceled and added to the lob.

        Args:
            time_step: Current time step.

        Returns:
            Tuple of lists of orders to cancel and add.
        """

    @abstractmethod
    def cancel_orders(self) -> list:
        """Cancel all active orders."""

    @abstractmethod
    def add_order(self, order: Order) -> None:
        """
        Add an order to the trader's active orders.

        Args:
            order: Order to add.
        """
        self.active_orders.append(order)

    @abstractmethod
    def remove_order(self, order: Order) -> None:
        """
        Remove an order from the trader's active orders.

        Args:
            order: Order to remove.
        """

    @abstractmethod
    def process_trade(
        self,
        time_step: int,
        price: float,
        volume: float,
        order: Order,
        is_make: bool,
    ) -> None:
        """
        Process a trade.

        Args:
            time_step: Current time step.
            price: Price of the trade.
            volume: Volume of the trade.
            order: Order that was matched.
            is_make: True if the order was a maker, False if taker.
        """

    @abstractmethod
    def save_stats(self, path: int, date: str) -> None:
        """
        Save the trader's statistics.

        Args:
            path: Path to the directory where the statistics should be saved.
            date: Date of the simulation start.
        """


class ExchangeTrader(Trader):
    """
    Trader that is used to replicate the state of the real limit order book from
    input dataset. Currently, the trader supports datasets from Crypto lake.
    """

    def __init__(self, id: str, depth: int = 10) -> None:
        """
        Initialize an exchange trader.

        Args:
            id: Unique identifier of the trader.
            depth: Depth of the limit order book to replicate.
        """
        super().__init__(id)
        self.depth = depth  # Depth of the limit order book to replicate
        self.timestamps = None

    def place_orders(
        self, time_step: int, book_data: pl.DataFrame
    ) -> tuple[list, list]:
        """
        Create lists of orders to be canceled and added to the lob.

        Args:
            time_step: Current time step.
            book_data: Dataframe with limit order book data for the current
                time step.

        Returns:
            Tuple of lists of orders to cancel and add.
        """
        entry_time = book_data["received_time"][0]
        order_id = str(int(book_data["sequence_number"][0]))

        # Delete previous bid / ask orders
        cancel_orders = self.active_orders

        # Process all bid / ask orders for a chosen depth
        new_orders = []
        order_ids = [
            (
                order_id + f"B{chr(65+j)}"
                if j < self.depth
                else order_id + f"A{chr(65+j-self.depth)}"
            )
            for j in range(self.depth * 2)
        ]
        for j in range(self.depth * 2):
            name = f"bid_{j}" if j < self.depth else f"ask_{j - self.depth}"
            if book_data[name + "_size"][0] and book_data[name + "_price"][0]:
                order = LimitOrder(
                    book_data["symbol"][0],
                    order_ids[j],
                    self.id,
                    True if j < self.depth else False,
                    book_data[name + "_size"][0],
                    entry_time,
                    book_data[name + "_price"][0],
                )
                new_orders.append(order)

        return cancel_orders, new_orders

    def cancel_orders(self) -> list:
        """Cancel all active orders."""
        return self.active_orders

    def add_order(self, order: Order) -> None:
        """
        Add an order to the trader's active orders.

        Args:
            order: Order to add.
        """
        self.active_orders.append(order)

    def remove_order(self, order: Order) -> None:
        """
        Remove an order from the trader's active orders.

        Args:
            order: Order to remove.
        """
        if order in self.active_orders:
            self.active_orders.remove(order)

    def process_trade(
        self,
        time_step: int,
        price: float,
        volume: float,
        order: Order,
        is_make: bool,
    ) -> None:
        """
        Process a trade.

        Args:
            time_step: Current time step.
            price: Price of the trade.
            volume: Volume of the trade.
            order: Order that was matched.
            is_make: True if the order was a maker, False if taker.
        """
        if order in self.active_orders and math.isclose(0, order.volume):
            self.active_orders.remove(order)

    def process_historical_trades(
        self, data: pl.DataFrame, ts, side: bool
    ) -> list[Order]:
        """Load historical trades data."""
        side_str = "buy" if side else "sell"
        data = data.filter(pl.col("side") == side_str)
        orders = []
        for row in data.rows(named=True):
            order = MarketOrder(
                COIN_SYMBOL,
                id=get_rnd_str(4),  # TODO: Preprocess the ids
                trader_id=self.id,
                side=side,
                volume=row["quantity"],
                entry_time=row["received_time"],
            )
            orders.append(order)

        return orders

    def save_stats(self, path: int, date: str) -> None:
        """
        Save the trader's statistics.

        Args:
            path: Path to the directory where the statistics should be saved.
            date: Date of the simulation start.
        """
        pass


class PureMarketMaker(Trader):
    """
    Pure market making strategy that places orders at the specified level of the
    limit order book.
    """

    def __init__(
        self,
        id: str,
        com_model: CommissionModel,
        volume: float = 1,
        priority: int = 0,
        inventory_manage: bool = False,
    ) -> None:
        """
        Initialize a pure market maker.

        Args:
            id: Unique identifier of the trader.
            volume: Volume of the orders to be placed.
            com_model: Commission model.
            priority: Priority of the orders to be placed. 0 means that the
                orders will be placed at the best bid and ask prices or one tick
                better if the spread is at least 3 ticks wide. 1 means that the
                orders will be placed at the first price level, 2 at the second
                price level, etc.
            inventory_manage: True if the trader should manage its inventory
                to minimize inventory risk.
        """
        super().__init__(id)
        self.volume = volume  # Volume of the orders to be placed
        self.priority = priority  # Priority of the orders to be placed
        self.com_model = com_model  # Commission model
        self.inventory_manage = inventory_manage  # Inventory management
        self.reset()

    def reset(self) -> None:
        """Reset the trader's statistics."""
        self.lob = None
        self.active_orders = []
        self.realized_pnl = 0  # Realized pnl
        self.adj_pnl = 0  # Adjusted pnl
        self.inventory = 0  # Long (positive) or short (negative) inventory
        self.cum_costs = 0  # Cumulative transaction costs
        self.total_volume = 0  # Total volume traded
        self.trade_count = 0  # Number of trades in one time step

        self.stats = {
            "realized_pnl": [],
            "adj_pnl": [],
            "inventory": [],
            "cum_costs": [],
            "total_volume": [],
            "trade_count": [],
            "quoted_bid_price": [],
            "quoted_ask_price": [],
            "quoted_bid_volume": [],
            "quoted_ask_volume": [],
        }

    def place_orders(self, time_step: int, timestamp) -> tuple[list, list]:
        """
        Create lists of orders to be canceled and added to the lob.

        Args:
            time_step: Current time step.

        Returns:
            Tuple of lists of orders to cancel and add.
        """
        # Cancel old orders
        cancel_orders = self.active_orders

        # Set the order prices
        bids, asks = self.lob.get_bids(), self.lob.get_asks()

        if not bids or not asks:
            self.stats["quoted_bid_price"].append(np.nan)
            self.stats["quoted_ask_price"].append(np.nan)
            self.stats["quoted_bid_volume"].append(0)
            self.stats["quoted_ask_volume"].append(0)
            return cancel_orders, []

        if self.priority == 0:
            diff = round_to_lot(asks[0] - bids[0], self.lob.lot_size)
            if diff >= 3 * self.lob.tick_size:
                bid_price, ask_price = (
                    bids[0] + self.lob.tick_size,
                    asks[0] - self.lob.tick_size,
                )
            else:
                bid_price = (
                    bids[1] + self.lob.tick_size if 2 <= len(bids) else bids[-1]
                )
                ask_price = (
                    asks[1] - self.lob.tick_size if 2 <= len(asks) else asks[-1]
                )

        else:
            bid_price = (
                bids[self.priority] + self.lob.tick_size
                if self.priority + 1 <= len(bids)
                else bids[-1]
            )
            ask_price = (
                asks[self.priority] - self.lob.tick_size
                if self.priority + 1 <= len(asks)
                else asks[-1]
            )

        new_orders = []
        if self.inventory_manage and self.inventory < 0:
            bid_volume = min(self.volume, abs(self.inventory))
        elif self.inventory_manage and self.inventory > 0:
            bid_volume = 0
        else:
            bid_volume = self.volume

        if bid_volume != 0:
            new_orders.append(
                LimitOrder(
                    COIN_SYMBOL,
                    get_rnd_str(6),
                    self.id,
                    True,
                    bid_volume,
                    timestamp,
                    bid_price,
                )
            )

        if self.inventory_manage and self.inventory > 0:
            ask_volume = min(self.volume, self.inventory)
        elif self.inventory_manage and self.inventory < 0:
            ask_volume = 0
        else:
            ask_volume = self.volume

        if ask_volume != 0:
            new_orders.append(
                LimitOrder(
                    COIN_SYMBOL,
                    get_rnd_str(6),
                    self.id,
                    False,
                    ask_volume,
                    timestamp,
                    ask_price,
                )
            )

        self.stats["quoted_bid_price"].append(bid_price)
        self.stats["quoted_ask_price"].append(ask_price)
        self.stats["quoted_bid_volume"].append(bid_volume)
        self.stats["quoted_ask_volume"].append(ask_volume)

        return cancel_orders, new_orders

    def cancel_orders(self) -> list:
        """Cancel all active orders."""
        return self.active_orders

    def add_order(self, order: Order) -> None:
        """
        Add an order to the trader's active orders.

        Args:
            order: Order to add.
        """
        self.active_orders.append(order)

    def remove_order(self, order: Order) -> None:
        """
        Remove an order from the trader's active orders.

        Args:
            order: Order to remove.
        """
        if order in self.active_orders:
            self.active_orders.remove(order)

    def process_trade(
        self,
        time_step: int,
        price: float,
        volume: float,
        order: Order,
        is_make: bool,
    ) -> None:
        """
        Process a trade.

        Args:
            time_step: Current time step.
            price: Price of the trade.
            volume: Volume of the trade.
            order: Order that was matched.
            is_make: True if the order was a maker, False if taker.
        """
        if order in self.active_orders and math.isclose(0, order.volume):
            self.active_orders.remove(order)

        if order.side:
            self.inventory += volume
            self.realized_pnl -= price * volume
        else:
            self.inventory -= volume
            self.realized_pnl += price * volume

        mid_price_median = self.lob.get_mid_price_median()
        if is_make:
            costs = self.com_model.maker_fee(volume, price)
        else:
            costs = self.com_model.taker_fee(volume, price)
        self.realized_pnl -= costs
        self.adj_pnl = self.realized_pnl + self.inventory * mid_price_median
        self.cum_costs += costs
        self.total_volume += volume * price
        self.trade_count += 1

    def update_stats(self, time_step: int) -> None:
        """
        Update the trader's statistics.

        Args:
            time_step: Current time step.
        """
        self.inventory = round_to_lot(self.inventory, self.lob.lot_size)
        self.stats["realized_pnl"].append(self.realized_pnl)
        self.stats["adj_pnl"].append(self.adj_pnl)
        self.stats["inventory"].append(self.inventory)
        self.stats["cum_costs"].append(self.cum_costs)
        self.stats["total_volume"].append(self.total_volume)
        self.stats["trade_count"].append(self.trade_count)
        self.trade_count = 0

    def save_stats(self, path: int, date: str) -> None:
        """
        Save the trader's statistics.

        Args:
            path: Path to the directory where the statistics should be saved.
            date: Date of the simulation start.
        """
        file_name = f"{path}/trader_{self.id}_{date}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(self.stats, f)


class AvellanedaStoikov(Trader):
    """Avellaneda-Stoikov market making strategy."""

    def __init__(
        self,
        id: str,
        com_model: CommissionModel,
        gamma: float = 0.001,  # Risk aversion parameter
        sigma: float = 0.0426,  # Volatility of the asset
        kappa: float = 200,  # Liquidity parameter
        volume: float = 1,
        inventory_manage: bool = False,
    ) -> None:
        """
        Initialize an Avellaneda-Stoikov market maker.

        Args:
            id: Unique identifier of the trader.
            com_model: Commission model.
            gamma: Risk aversion parameter.
            sigma: Volatility of the asset.
            kappa: Liquidity parameter.
            volume: Volume of the orders to be placed.
            inventory_manage: True if the trader should manage its inventory
                to minimize inventory risk.
        """
        super().__init__(id)
        self.volume = volume  # Volume of the orders to be placed
        self.com_model = com_model  # Commission model
        self.inventory_manage = inventory_manage  # Inventory management
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.reset()

    def reset(self) -> None:
        """Reset the trader's statistics."""
        self.lob = None
        self.active_orders = []
        self.realized_pnl = 0  # Realized pnl
        self.adj_pnl = 0  # Adjusted pnl
        self.inventory = 0  # Long (positive) or short (negative) inventory
        self.cum_costs = 0  # Cumulative transaction costs
        self.total_volume = 0  # Total volume traded
        self.trade_count = 0  # Number of trades in one time step

        self.stats = {
            "realized_pnl": [],
            "adj_pnl": [],
            "inventory": [],
            "cum_costs": [],
            "total_volume": [],
            "trade_count": [],
            "quoted_bid_price": [],
            "quoted_ask_price": [],
            "quoted_bid_volume": [],
            "quoted_ask_volume": [],
        }

    def place_orders(
        self, time_step: int, timestamp: datetime.datetime, last_time_step: int
    ) -> tuple[list, list]:
        """
        Create lists of orders to be canceled and added to the lob.

        Args:
            time_step: Current time step.
            timestamp: Current timestamp.
            last_time_step: Last time step.

        Returns:
            Tuple of lists of orders to cancel and add.
        """
        # Cancel old orders
        cancel_orders = self.active_orders

        mid_price = self.lob.get_mid_price()
        r = mid_price - self.inventory * self.gamma * self.sigma**2 * (
            last_time_step - time_step
        )
        spread = self.gamma * self.sigma**2 * (
            last_time_step - time_step
        ) + 2 / self.gamma * math.log(1 + self.gamma / self.kappa)

        bid_price = round_to_tick(r - spread / 2, self.lob.tick_size)
        ask_price = round_to_tick(r + spread / 2, self.lob.tick_size)

        if math.isclose(bid_price, ask_price):
            self.stats["quoted_bid_price"].append(np.nan)
            self.stats["quoted_ask_price"].append(np.nan)
            self.stats["quoted_bid_volume"].append(0)
            self.stats["quoted_ask_volume"].append(0)
            return cancel_orders, []

        new_orders = []
        if self.inventory_manage and self.inventory < 0:
            bid_volume = min(self.volume, abs(self.inventory))
        elif self.inventory_manage and self.inventory > 0:
            bid_volume = 0
        else:
            bid_volume = self.volume

        if bid_volume != 0:
            new_orders.append(
                LimitOrder(
                    COIN_SYMBOL,
                    get_rnd_str(6),
                    self.id,
                    True,
                    bid_volume,
                    timestamp,
                    bid_price,
                )
            )

        if self.inventory_manage and self.inventory > 0:
            ask_volume = min(self.volume, self.inventory)
        elif self.inventory_manage and self.inventory < 0:
            ask_volume = 0
        else:
            ask_volume = self.volume

        if ask_volume != 0:
            new_orders.append(
                LimitOrder(
                    COIN_SYMBOL,
                    get_rnd_str(6),
                    self.id,
                    False,
                    ask_volume,
                    timestamp,
                    ask_price,
                )
            )

        self.stats["quoted_bid_price"].append(bid_price)
        self.stats["quoted_ask_price"].append(ask_price)
        self.stats["quoted_bid_volume"].append(bid_volume)
        self.stats["quoted_ask_volume"].append(ask_volume)

        return cancel_orders, new_orders

    def cancel_orders(self) -> list:
        """Cancel all active orders."""
        return self.active_orders

    def add_order(self, order: Order) -> None:
        """
        Add an order to the trader's active orders.

        Args:
            order: Order to add.
        """
        self.active_orders.append(order)

    def remove_order(self, order: Order) -> None:
        """
        Remove an order from the trader's active orders.

        Args:
            order: Order to remove.
        """
        if order in self.active_orders:
            self.active_orders.remove(order)

    def process_trade(
        self,
        time_step: int,
        price: float,
        volume: float,
        order: Order,
        is_make: bool,
    ) -> None:
        """
        Process a trade.

        Args:
            time_step: Current time step.
            price: Price of the trade.
            volume: Volume of the trade.
            order: Order that was matched.
            is_make: True if the order was a maker, False if taker.
        """
        if order in self.active_orders and math.isclose(0, order.volume):
            self.active_orders.remove(order)

        if order.side:
            self.inventory += volume
            self.realized_pnl -= price * volume
        else:
            self.inventory -= volume
            self.realized_pnl += price * volume

        mid_price_median = self.lob.get_mid_price_median()
        if is_make:
            costs = self.com_model.maker_fee(volume, price)
        else:
            costs = self.com_model.taker_fee(volume, price)
        self.realized_pnl -= costs
        self.adj_pnl = self.realized_pnl + self.inventory * mid_price_median
        self.cum_costs += costs
        self.total_volume += volume * price
        self.trade_count += 1

    def update_stats(self, time_step: int) -> None:
        """
        Update the trader's statistics.

        Args:
            time_step: Current time step.
        """
        self.inventory = round_to_lot(self.inventory, self.lob.lot_size)
        self.stats["realized_pnl"].append(self.realized_pnl)
        self.stats["adj_pnl"].append(self.adj_pnl)
        self.stats["inventory"].append(self.inventory)
        self.stats["cum_costs"].append(self.cum_costs)
        self.stats["total_volume"].append(self.total_volume)
        self.stats["trade_count"].append(self.trade_count)
        self.trade_count = 0

    def save_stats(self, path: int, date: str) -> None:
        """
        Save the trader's statistics.

        Args:
            path: Path to the directory where the statistics should be saved.
            date: Date of the simulation start.
        """
        file_name = f"{path}/trader_{self.id}_{date}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(self.stats, f)


class RLMarketMaker(Trader):
    """Reinforcement learning market making strategy."""

    def __init__(
        self,
        id: str,
        com_model: CommissionModel,
        volume: float = 1,
        policy: Any = None,
    ) -> None:
        """
        Initialize a reinforcement learning market maker.

        Args:
            id: Unique identifier of the trader.
            volume: Volume of the orders to be placed.
            com_model: Commission model.
            inventory_manage: True if the trader should manage its inventory
                to minimize inventory risk.
            policy: Policy to use for the RL agent.
        """
        super().__init__(id)
        self.volume = volume  # Volume of the orders to be placed
        self.com_model = com_model  # Commission model
        self.policy = policy  # Policy
        self.reset()

    def reset(self):
        """Reset the agent's statistics."""
        self.lob = None
        self.active_orders = []
        self.realized_pnl = 0  # Realized pnl
        self.adj_pnl = 0  # Adjusted pnl
        self.inventory = 0  # Long (positive) or short (negative) inventory
        self.cum_costs = 0  # Cumulative transaction costs
        self.total_volume = 0  # Total volume traded
        self.trade_count = 0  # Number of trades in one time step
        self.reward = 0  # Reward

        self.stats = {
            "realized_pnl": [],
            "adj_pnl": [],
            "inventory": [],
            "cum_costs": [],
            "total_volume": [],
            "trade_count": [],
            "quoted_bid_price": [],
            "quoted_ask_price": [],
            "quoted_bid_volume": [],
            "quoted_ask_volume": [],
            "reward": [],
        }

    def place_orders(
        self, time_step: int, timestamp, action: ActType, obs: ObsType
    ) -> tuple[list, list]:
        """
        Create lists of orders to be canceled and added to the lob.

        Args:
            time_step: Current time step.
            timestamp: Current timestamp.
            action: Action to take.
            obs: Observation of the environment.

        Returns:
            Tuple of lists of orders to cancel and add.
        """
        if action is None and self.policy is not None:
            action = int(self.policy.predict(obs, deterministic=True)[0])
        elif action is None and self.policy is None:
            raise ValueError("Policy is not defined.")

        # ----------------------------------------------------------------------
        # First handle trivial action (20)
        # ----------------------------------------------------------------------
        # Action 20: Wait and no new quote
        if action == 20:
            cancel_orders = []
            new_orders = []
            self.stats["quoted_bid_price"].append(np.nan)
            self.stats["quoted_ask_price"].append(np.nan)
            self.stats["quoted_bid_volume"].append(0)
            self.stats["quoted_ask_volume"].append(0)
            return cancel_orders, new_orders

        # ----------------------------------------------------------------------
        # Next handle other actions (0-19) consisting of bid-ask combinations
        # ----------------------------------------------------------------------
        # Cancel old orders and get the current bid and ask prices
        cancel_orders = self.active_orders
        bids, asks = self.lob.get_bids(), self.lob.get_asks()
        if not bids or not asks:
            self.stats["quoted_bid_price"].append(np.nan)
            self.stats["quoted_ask_price"].append(np.nan)
            self.stats["quoted_bid_volume"].append(0)
            self.stats["quoted_ask_volume"].append(0)
            return cancel_orders, []

        # Set the correct bid price
        # ----------------------------------------------------------------------
        # Bid - priority 0
        if action in [0, 4, 8, 12, 16]:
            diff = round_to_lot(asks[0] - bids[0], self.lob.lot_size)
            if diff >= 3 * self.lob.tick_size:
                bid_price = bids[0] + self.lob.tick_size
            else:
                bid_price = (
                    bids[1] + self.lob.tick_size if len(bids) >= 2 else bids[-1]
                )
        # Bid - priority 1
        elif action in [1, 5, 9, 13, 17]:
            bid_price = (
                bids[1] + self.lob.tick_size if len(bids) >= 2 else bids[-1]
            )
        # Bid - priority 2
        elif action in [2, 6, 10, 14, 18]:
            bid_price = (
                bids[2] + self.lob.tick_size if len(bids) >= 3 else bids[-1]
            )
        # Bid - priority 3
        elif action in [3, 7, 11, 15, 19]:
            bid_price = (
                bids[3] + self.lob.tick_size if len(bids) >= 4 else bids[-1]
            )

        # Set the correct ask price
        # ----------------------------------------------------------------------
        # Ask - priority 0
        if action in [0, 1, 2, 3, 16]:
            diff = round_to_lot(asks[0] - bids[0], self.lob.lot_size)
            if diff >= 3 * self.lob.tick_size:
                ask_price = asks[0] - self.lob.tick_size
            else:
                ask_price = (
                    asks[1] - self.lob.tick_size if len(asks) >= 2 else asks[-1]
                )
        # Ask - priority 1
        elif action in [4, 5, 6, 7, 17]:
            ask_price = (
                asks[1] - self.lob.tick_size if len(asks) >= 2 else asks[-1]
            )
        # Ask - priority 2
        elif action in [8, 9, 10, 11, 18]:
            ask_price = (
                asks[2] - self.lob.tick_size if len(asks) >= 3 else asks[-1]
            )
        # Ask - priority 3
        elif action in [12, 13, 14, 15, 19]:
            ask_price = (
                asks[3] - self.lob.tick_size if len(asks) >= 4 else asks[-1]
            )

        # Set quoted volumes
        # ----------------------------------------------------------------------
        if action >= 0 and action <= 15:
            bid_volume, ask_volume = self.volume, self.volume
        elif action >= 16 and action <= 19:
            if self.inventory > 0:
                bid_volume, ask_volume = 0, abs(self.inventory)
            elif self.inventory < 0:
                bid_volume, ask_volume = abs(self.inventory), 0
            else:
                bid_volume, ask_volume = 0, 0
        else:
            raise ValueError(f"Invalid action. {action}")

        # ----------------------------------------------------------------------
        # Create orders
        # ----------------------------------------------------------------------
        new_orders = []
        if bid_volume != 0:
            new_orders.append(
                LimitOrder(
                    COIN_SYMBOL,
                    get_rnd_str(6),
                    self.id,
                    True,
                    bid_volume,
                    timestamp,
                    bid_price,
                )
            )

        if ask_volume != 0:
            new_orders.append(
                LimitOrder(
                    COIN_SYMBOL,
                    get_rnd_str(6),
                    self.id,
                    False,
                    ask_volume,
                    timestamp,
                    ask_price,
                )
            )

        # ----------------------------------------------------------------------
        # Save stats and set reward
        # ----------------------------------------------------------------------
        self.stats["quoted_bid_price"].append(bid_price)
        self.stats["quoted_ask_price"].append(ask_price)
        self.stats["quoted_bid_volume"].append(bid_volume)
        self.stats["quoted_ask_volume"].append(ask_volume)

        if abs(self.inventory) > 0 and action == 17:
            self.reward = 1
        elif abs(self.inventory) == 0 and action == 5:
            self.reward = 1

        return cancel_orders, new_orders

    def cancel_orders(self) -> list:
        """Cancel all active orders."""
        return self.active_orders

    def add_order(self, order: Order) -> None:
        """
        Add an order to the trader's active orders.

        Args:
            order: Order to add.
        """
        self.active_orders.append(order)

    def remove_order(self, order: Order) -> None:
        """
        Remove an order from the trader's active orders.

        Args:
            order: Order to remove.
        """
        if order in self.active_orders:
            self.active_orders.remove(order)

    def process_trade(
        self,
        time_step: int,
        price: float,
        volume: float,
        order: Order,
        is_make: bool,
    ) -> None:
        """
        Process a trade.

        Args:
            time_step: Current time step.
            price: Price of the trade.
            volume: Volume of the trade.
            order: Order that was matched.
            is_make: True if the order was a maker, False if taker.
        """
        if order in self.active_orders and math.isclose(0, order.volume):
            self.active_orders.remove(order)

        if order.side:
            self.inventory += volume
            self.realized_pnl -= price * volume
        else:
            self.inventory -= volume
            self.realized_pnl += price * volume

        mid_price_median = self.lob.get_mid_price_median()
        if is_make:
            costs = self.com_model.maker_fee(volume, price)
        else:
            costs = self.com_model.taker_fee(volume, price)
        self.realized_pnl -= costs
        self.adj_pnl = self.realized_pnl + self.inventory * mid_price_median
        self.cum_costs += costs
        self.total_volume += volume * price
        self.trade_count += 1

    def update_stats(self, time_step: int) -> None:
        """
        Update the trader's statistics.

        Args:
            time_step: Current time step.
        """
        self.inventory = round_to_lot(self.inventory, self.lob.lot_size)
        self.stats["realized_pnl"].append(self.realized_pnl)
        if len(self.stats["adj_pnl"]) == 0:
            prev_pnl = 0
        else:
            prev_pnl = self.stats["adj_pnl"][-1]
        self.stats["adj_pnl"].append(self.adj_pnl)
        self.stats["inventory"].append(self.inventory)
        self.stats["cum_costs"].append(self.cum_costs)
        self.stats["total_volume"].append(self.total_volume)
        self.stats["trade_count"].append(self.trade_count)
        self.trade_count = 0

        prev_pnl = 1e-8 if prev_pnl == 0 else prev_pnl
        self.pnl_change = (self.adj_pnl - prev_pnl) / abs(prev_pnl)
        self.pnl_change = np.clip(self.pnl_change, a_min=-2, a_max=2)

        self.stats["reward"].append(self.reward)
        self.reward = 0

    def save_stats(self, path: int, date: str) -> None:
        """
        Save the trader's statistics.

        Args:
            path: Path to the directory where the statistics should be saved.
            date: Date of the simulation start.
        """
        file_name = f"{path}/trader_{self.id}_{date}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(self.stats, f)
