"""Limit order book."""

import math
import os
from datetime import datetime
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyllist import dllistnode
from sortedcontainers import SortedDict

from src.lob.order_queue import OrderQueue
from src.lob.orders import Order
from src.lob.plots import set_plot_style
from src.lob.utils import ensure_dir_exists, round_to_lot, round_to_tick


class LimitOrderBook:
    """Limit order book class."""

    def __init__(
        self,
        tick_size: float,
        lot_size: float,
        logging: bool = False,
        ts_save: datetime = None,
    ) -> None:
        """
        Initialize a limit order book.

        Args:
            tick_size: Minimum tick size for rounding prices.
            lot_size: Minimum lot size for rounding volumes.
            logging: Indicates whether to log the events.
            ts_save: Start time of the simulation. Used for logging.
        """
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.logging = logging
        self.ts_save = ts_save

        self.orders = {}
        self.bid_side = SortedDict()
        self.ask_side = SortedDict()
        self.num_bid_orders = 0
        self.num_ask_orders = 0
        self.bid_volume = 0
        self.ask_volume = 0
        self.best_bid_price = None
        self.last_best_bid_price = None  # Last non None bid price
        self.best_ask_price = None
        self.last_best_ask_price = None  # Last non None ask price
        self.parquet_writer = None
        self.log_file_name = None
        self.log = None
        self.ts = None
        self.mid_price_history = []
        ensure_dir_exists(os.path.join(os.getcwd(), "data", "results_backtest"))
        self.initialize_log()
        set_plot_style()

    def add_order(self, order: Order) -> tuple[list, Order]:
        """
        Add an order to the limit order book.

        Args:
            order: Order to add.

        Returns:
            A tuple of the list of trades and the remaining order.
        """
        # Round the order price to the tick size
        order.price = round_to_tick(order.price, self.tick_size)
        order.volume = round_to_lot(order.volume, self.lot_size)

        # Try to match the order with existing orders
        trades, order = self.match_order(order)
        if order is None:
            return trades, None

        # Add the order to the order book and update the order queue
        order_tree = self.bid_side if order.side else self.ask_side
        if not order_tree.__contains__(order.price):
            order_tree[order.price] = OrderQueue(self.lot_size)
        node = order_tree[order.price].add_order(order)
        self.orders[order.id] = node

        if order.side:
            self.num_bid_orders += 1
            self.bid_volume += order.volume
            self.update_bid_price(order.price)
        else:
            self.num_ask_orders += 1
            self.ask_volume += order.volume
            self.update_ask_price(order.price)

        if self.logging:
            self.update_log(
                max(order.entry_time, self.ts),
                "Insert",
                order.ticker,
                order.id,
                order.trader_id,
                order.side,
                order.price,
                order.volume,
            )
        return trades, order

    def remove_order_by_id(self, order_id: str, log: bool = True) -> Order:
        """
        Remove an order from the limit order book.

        Args:
            order_id: ID of the order to remove.
            log: Indicated whether to log the order removal.

        Returns:
            The removed order.
        """
        node = self.get_order_by_id(order_id, node=True)
        if node is not None:
            order = node.value
        else:
            return None
        order_tree = self.bid_side if order.side else self.ask_side
        if order.side:
            self.num_bid_orders -= 1
            self.bid_volume -= order.volume
        else:
            self.num_ask_orders -= 1
            self.ask_volume -= order.volume
        del self.orders[order_id]
        order_tree[order.price].remove_order(order, node)
        if order_tree[order.price].num_orders == 0:
            del order_tree[order.price]
            self.update_bid_price() if order.side else self.update_ask_price()

        # Log the order removal
        if log and self.logging:
            self.update_log(
                max(order.entry_time, self.ts),
                "Cancel",
                order.ticker,
                order.id,
                order.trader_id,
                order.side,
                order.price,
                order.volume,
            )
        return order

    def update_order_volume(
        self, order_id: str, volume: int, log: bool = True
    ) -> None:
        """
        Update the volume of an order. This method is only used for order
        matching by the exchange. It should not be used by agents, since they
        should not be able to change the volume of an order without posting
        a new order with a new timestamp. This is the case for the Binance
        exchange, where each change of the order volume updates the timestamp
        of the order as well.

        Args:
            order_id: ID of the order to update.
            volume: New volume.
            log: Whether to log the order update.
        """
        order = self.get_order_by_id(order_id)
        if order is None:
            return
        if order.side:
            self.bid_volume = self.bid_volume - order.volume + volume
        else:
            self.ask_volume = self.ask_volume - order.volume + volume
        order_tree = self.bid_side if order.side else self.ask_side
        order_tree[order.price].update_order_volume(order, volume)
        if log and self.logging:
            self.update_log(
                max(order.entry_time, self.ts),
                "Update",
                order.ticker,
                order.id,
                order.trader_id,
                order.side,
                order.price,
                volume,
            )

    def match_order(self, order: Order) -> tuple[list, Order]:
        """
        If possible, partially or fully match an order. If the order is fully
        matched, it won't be added to the order book and None will be returned.
        If the order is partially matched, the remaining volume will be
        returned as a new order, so that it can be added to the order book.

        Args:
            order: Order to be matched.

        Returns:
            A tuple of the list of trades and the remaining order.
        """
        bid, ask = self.best_bid_price, self.best_ask_price

        # Return the order if it cannot be matched
        if order.side and (ask is None or order.price < ask):
            return [], order
        if not order.side and (bid is None or order.price > bid):
            return [], order

        # Match the order
        orders = self.ask_side if order.side else self.bid_side
        match_price = ask if order.side else bid
        match_order = orders[match_price].first_order
        trade_price = ask if order.side else bid
        trade_volume = min(order.volume, match_order.volume)

        # Log the trade
        if self.logging:
            self.update_log(
                max(order.entry_time, match_order.entry_time, self.ts),
                "Trade",
                order.ticker,
                order.id,
                order.trader_id,
                order.side,
                trade_price,
                trade_volume,
                match_order.id,
                match_order.trader_id,
                match_order.side,
            )

        # Update the order quantities
        if math.isclose(order.volume, match_order.volume):
            order.volume = 0
            match_order.volume = 0
            self.remove_order_by_id(match_order.id, log=False)
            return [
                {
                    "price": trade_price,
                    "volume": trade_volume,
                    "order_take": order,
                    "order_make": match_order,
                }
            ], None

        elif order.volume < match_order.volume:
            diff = round_to_lot(
                match_order.volume - order.volume, self.lot_size
            )
            self.update_order_volume(match_order.id, diff, log=False)
            order.volume = 0
            return [
                {
                    "price": trade_price,
                    "volume": trade_volume,
                    "order_take": order,
                    "order_make": match_order,
                }
            ], None

        else:
            order.volume -= match_order.volume
            order.volume = round_to_lot(order.volume, self.lot_size)
            match_order.volume = 0
            self.remove_order_by_id(match_order.id, log=False)

            trades, remaining_order = self.match_order(order)
            trades.insert(
                0,
                {
                    "price": trade_price,
                    "volume": trade_volume,
                    "order_take": order,
                    "order_make": match_order,
                },
            )
            return trades, remaining_order

    def update_bid_price(self, price: float = None) -> None:
        """
        Update the best bid price.

        Args:
            price: New best bid price, if known.
        """
        if self.num_bid_orders == 0:
            self.best_bid_price = None
        elif price is not None and (
            self.best_bid_price is None or price > self.best_bid_price
        ):
            self.best_bid_price = price
            self.last_best_bid_price = price
        else:
            self.best_bid_price = self.bid_side.peekitem(index=-1)[0]
            self.last_best_bid_price = self.best_bid_price

    def update_ask_price(self, price: float = None) -> None:
        """
        Update the best ask price.

        Args:
            price: New best ask price, if known.
        """
        if self.num_ask_orders == 0:
            self.best_ask_price = None
        elif price is not None and (
            self.best_ask_price is None or price < self.best_ask_price
        ):
            self.best_ask_price = price
            self.last_best_ask_price = price
        else:
            self.best_ask_price = self.ask_side.peekitem(index=0)[0]
            self.last_best_ask_price = self.best_ask_price

    def get_order_by_id(
        self, order_id: str, node: bool = False
    ) -> Union[Order, dllistnode]:
        """
        Get an order object by its ID. If node is True, returns the node
        containing the order in the double-linked list.
        """
        try:
            if node:
                return self.orders[order_id]
            else:
                return self.orders[order_id].value
        except KeyError:
            return None
        except Exception as e:
            raise e

    def get_best_bid(self) -> float:
        """Returns the best bid price."""
        return self.best_bid_price

    def get_best_bid_volume(self) -> float:
        """Returns the volume at the best bid."""
        return self.get_volume_at_price(self.best_bid_price)

    def get_best_ask(self) -> float:
        """Returns the best ask price."""
        return self.best_ask_price

    def get_best_ask_volume(self) -> float:
        """Returns the volume at the best ask."""
        return self.get_volume_at_price(self.best_ask_price)

    def get_bid_ask_spread(self) -> float:
        """Returns the bid-ask spread."""
        if self.best_bid_price is None or self.best_ask_price is None:
            return None
        return self.best_ask_price - self.best_bid_price

    def get_mid_price(self) -> float:
        """Returns the mid price."""
        bid = (
            self.best_bid_price
            if self.best_bid_price is not None
            else self.last_best_bid_price
        )
        ask = (
            self.best_ask_price
            if self.best_ask_price is not None
            else self.last_best_ask_price
        )
        return (bid + ask) / 2

    def get_bids(self) -> list[float]:
        """Returns a list of all bid prices, decreasing from the best bid."""
        return list(self.bid_side.keys())[::-1]

    def get_asks(self) -> list[float]:
        """Returns a list of all ask prices, increasing from the best ask."""
        return list(self.ask_side.keys())

    def get_book_info(
        self, max_depth: int = 5
    ) -> dict[str, list[tuple[float, float]]]:
        """
        Returns a dictionary with information about the order book. The
        dictionary contains price-depth pairs for the best bid and ask prices,
        up to a given maximum depth.

        Args:
            max_depth: Maximum depth to return.
        """
        bids, asks = self.get_bids(), self.get_asks()
        bids = bids[: min(max_depth, len(bids))]
        asks = asks[: min(max_depth, len(asks))]
        return {
            "bid_side": [(p, self.get_volume_at_price(p)) for p in bids],
            "ask_side": [(p, self.get_volume_at_price(p)) for p in asks],
        }

    def get_volume_at_price(self, price: float) -> float:
        """
        Get the volume of shares for a given price. Assumes that the price
        cannot be matched and is one of the prices waiting in the lob.

        Args:
            price: Price to get the volume for.

        Returns:
            The volume of shares at the given price.
        """
        best_bid = self.get_best_bid()
        if best_bid is None or price > best_bid:
            order_tree = self.ask_side
        else:
            order_tree = self.bid_side
        # order_tree = self.bid_side if price <= best_bid else self.ask_side
        if order_tree.__contains__(price):
            return order_tree[price].volume
        else:
            return 0

    def get_bid_volume(self) -> float:
        """Returns the total volume of orders on the bid side."""
        return self.bid_volume

    def get_ask_volume(self) -> float:
        """Returns the total volume of orders on the ask side."""
        return self.ask_volume

    def get_total_volume(self) -> float:
        """Returns the total volume of orders on both sides."""
        return self.bid_volume + self.ask_volume

    def get_order_position(self, order_id: str) -> Union[int, None]:
        """
        Returns the position of an order in the order book. If the output is
        1, the order is the first order on its side (bid/ask) of the order book.

        Args:
            order_id: ID of the order to get the position for.

        Returns:
            The position of the order in the order book.
        """
        order = self.get_order_by_id(order_id)
        if order is None:
            return None
        order_side = self.bid_side if order.side else self.ask_side
        prices = self.get_bids() if order.side else self.get_asks()
        prices = prices[: prices.index(order.price)]
        num_orders = sum(order_side[p].num_orders for p in prices)
        temp = order.prev
        while temp is not None:
            num_orders += 1
            temp = temp.prev
        return num_orders + 1

    def get_num_orders_per_price(self, price: float) -> int:
        """
        Get the number of orders for a given price. Assumes that the price
        cannot be matched and is one of the prices waiting in the lob.

        Args:
            price: Price to get the number of orders for.

        Returns:
            The number of orders at the given price.
        """
        best_bid = self.get_best_bid()
        order_tree = self.bid_side if price <= best_bid else self.ask_side
        if order_tree.__contains__(price):
            return order_tree[price].num_orders
        else:
            return 0

    def update_mid_price_history(self) -> None:
        """
        Update the mid price history list. This list is used for finding
        the median of recent mid prices since in illiquid markets the mid
        price can sometime jump by a large amount for a short period.
        """
        if len(self.mid_price_history) < 11:
            self.mid_price_history.append(self.get_mid_price())
        else:
            self.mid_price_history.pop(0)
            self.mid_price_history.append(self.get_mid_price())

    def get_mid_price_median(self) -> float:
        """Returns the median of the mid price history."""
        return sorted(self.mid_price_history)[len(self.mid_price_history) // 2]

    def get_vamp(self, q: float, max_depth: int = 10) -> float:
        """
        Get volume adjusted mid price, suggested by Stoikov and Covario. They
        suggest the difference between the VAMP and the mid price as a good
        predictor for the direction of the stock price in various timescales
        between 1-60 seconds.

        Args:
            q: Parameter for the VAMP, should be the desired trading volume.
            max_depth: Maximum depth of the order book to consider.
        """
        if self.best_bid_price is None or self.best_ask_price is None:
            return None
        return (self.get_vabp(q, max_depth) + self.get_vaap(q, max_depth)) / 2

    def get_vabp(self, q: float, max_depth: int = 10) -> float:
        """
        Get volume adjusted bid price (Stoikov, Covario). The VABP is defined as
        the weighted average of the bid prices, where the weights are the
        volumes at each price level (up to a maximum depth) divided by a
        parameter trading volume parameter q specified by a trader.

        Args:
            q: Parameter for the VABP, should be the desired trading volume.
            max_depth: Maximum depth of the order book to consider.
        """
        if self.best_bid_price is None:
            return None
        bids = self.get_bids()
        bids = bids[: min(max_depth, len(bids))]
        volumes = [self.get_volume_at_price(bid) for bid in bids]
        return sum(bids[i] * volumes[i] for i in range(len(bids))) / q

    def get_vaap(self, q: float, max_depth: int = 10) -> float:
        """
        Get volume adjusted ask price (Stoikov, Covario). The VAAP is defined as
        the weighted average of the ask prices, where the weights are the
        volumes at each price level (up to a maximum depth) divided by a
        parameter trading volume parameter q specified by a trader.

        Args:
            q: Parameter for the VAAP, should be the desired trading volume.
            max_depth: Maximum depth of the order book to consider.
        """
        if self.best_ask_price is None:
            return None
        asks = self.get_asks()
        asks = asks[: min(max_depth, len(asks))]
        volumes = [self.get_volume_at_price(ask) for ask in asks]
        return sum(asks[i] * volumes[i] for i in range(len(asks))) / q

    def visualize(self, depth: int = 6) -> None:
        """Creates a plot of the limit order book."""
        bids, asks = self.get_bids(), self.get_asks()
        bid_volumes = [self.get_volume_at_price(p) for p in bids]
        ask_volumes = [self.get_volume_at_price(p) for p in asks]
        if len(bids) > depth:
            bids = bids[:depth]
            bid_volumes = bid_volumes[:depth]
        if len(asks) > depth:
            asks = asks[:depth]
            ask_volumes = ask_volumes[:depth]
        spread_space = 1  # Number of ticks to leave in the middle
        x_axis = np.arange(0, len(bids) + len(asks) + spread_space, 1)

        plt.figure(figsize=(12, 5))
        plt.bar(
            x_axis[: len(bids)],
            bid_volumes[::-1],
            label="Bid",
            color="#9ED166",
            width=1,
            edgecolor="black",
            linewidth=1.3,
        )
        plt.bar(
            x_axis[len(bids) + spread_space :],
            ask_volumes,
            label="Ask",
            color="#EB735F",
            width=1,
            edgecolor="black",
            linewidth=1.3,
        )
        x_ticks = np.append(bids[::-1], asks)
        x_ticks = [str(x) for x in x_ticks]
        x_ticks = np.insert(x_ticks, len(bids), "")
        plt.xticks(x_axis, x_ticks, rotation=45, size=12)
        plt.xlabel("Price")
        plt.ylabel("Volume")
        plt.show()
        # Save figure as pdf image
        # plt.savefig("lob.pdf", format="pdf")

    def initialize_log(self):
        """Initialize the log with the column names."""
        self.log = {
            "ts": [],
            "type": [],
            "ticker": [],
            "id": [],
            "trader_id": [],
            "side": [],
            "price": [],
            "volume": [],
            "id2": [],
            "trader_id2": [],
            "side2": [],
        }

    def update_log(
        self,
        ts,
        order_type,
        ticker,
        order_id,
        trader_id,
        side,
        price,
        volume,
        id2=None,
        trader_id2=None,
        side2=None,
    ) -> None:
        """
        Update the log with the order entry.

        Args:
            ts: _description_
            order_type: _description_
            ticker: _description_
            order_id: _description_
            trader_id: _description_
            side: _description_
            price: _description_
            volume: _description_
            id2: _description_. Defaults to None.
            trader_id2: _description_. Defaults to None.
            side2: _description_. Defaults to None.
        """
        self.log["ts"].append(ts)
        self.log["type"].append(order_type)
        self.log["ticker"].append(ticker)
        self.log["id"].append(order_id)
        self.log["trader_id"].append(trader_id)
        self.log["side"].append(side)
        self.log["price"].append(price)
        self.log["volume"].append(volume)
        self.log["id2"].append(id2)
        self.log["trader_id2"].append(trader_id2)
        self.log["side2"].append(side2)

        if len(self.log["ts"]) > 30000:
            self.write_log_to_parquet()

    def write_log_to_parquet(self, date: Optional[datetime] = None) -> None:
        """
        Save the log to a parquet file. The file will be saved in the logs
        folder with the name log_{current_date}.parquet.

        Args:
            date: Date to use for the log file name. If None, the current
                date will be used.
        """
        if self.logging:
            if self.log_file_name is None:
                self.log_file_name = os.path.join(
                    os.getcwd(),
                    "data",
                    "results_backtest",
                    f"log_{self.ts_save}.parquet",
                )
            df = pa.Table.from_arrays(
                [
                    pa.array(self.log["ts"]),
                    pa.array(self.log["type"]),
                    pa.array(self.log["ticker"]),
                    pa.array(self.log["id"]),
                    pa.array(self.log["trader_id"]),
                    pa.array(self.log["side"]),
                    pa.array(self.log["price"]),
                    pa.array(self.log["volume"]),
                    pa.array(self.log["id2"]),
                    pa.array(self.log["trader_id2"]),
                    pa.array(self.log["side2"]),
                ],
                names=[
                    "ts",
                    "type",
                    "ticker",
                    "id",
                    "trader_id",
                    "side",
                    "price",
                    "volume",
                    "id2",
                    "trader_id2",
                    "side2",
                ],
            )
            if self.parquet_writer is None:
                self.parquet_writer = pq.ParquetWriter(
                    self.log_file_name, df.schema
                )
            self.parquet_writer.write_table(df)
            self.initialize_log()

    def close_parquet_writer(self):
        """Close and reset the parquet writer."""
        if self.logging:
            self.write_log_to_parquet()
            self.parquet_writer.close()
            self.parquet_writer = None
