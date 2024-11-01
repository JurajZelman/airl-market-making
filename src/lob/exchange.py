"""Exchange simulator."""

import copy
import os
import pickle
from datetime import datetime
from typing import TypeVar

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from src.lob.data import scan_parquet
from src.lob.distributions import EmpiricalOrderVolumeDistribution
from src.lob.limit_order_book import LimitOrderBook
from src.lob.orders import Order
from src.lob.time import TimeManager
from src.lob.traders import ExchangeTrader, Trader
from src.lob.utils import get_rnd_str

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

pl.enable_string_cache()  # Fix polars issues

DEFAULT_LATENCY_COMP_PARAMS = {}
DEFAULT_RNG = np.random.default_rng(seed=42)


class Exchange:
    """
    Class representing an exchange simulator which handles the interactions
    between traders and the limit order book.
    """

    def __init__(
        self,
        exchange_name: str,
        symbol_name: str,
        tick_size: float,
        lot_size: float,
        depth: int,
        traders: dict[str:Trader],
        max_steps: int,
        ts_start: pd.Timestamp,
        ts_end: pd.Timestamp,
        win: int,
        path: str,
        path_vol_distr: str,
        rl_trader_id: str,
        initialize: bool = True,
        logging: bool = False,
        latency_comp_params: dict = DEFAULT_LATENCY_COMP_PARAMS,
        ts_save: datetime = None,
        description: str = "",
        rng: np.random.Generator = DEFAULT_RNG,
    ) -> None:
        """
        Initialize an exchange simulator.

        Args:
            exchange_name: Name of the exchange.
            symbol_name: Name of the symbol.
            tick_size: Tick size of the symbol.
            lot_size: Lot size of the symbol.
            depth: Max depth of the limit order book to load.
            traders: List of traders participating in the exchange.
            max_steps: Maximum number of steps to run in the simulation.
            ts_start: Start timestamp.
            ts_end: End timestamp.
            win: Window size for features.
            path: Path to the directory containing the datasets.
            path_vol_distr: Path to the directory containing the distributions
                for the EmpiricalOrderVolumeDistribution.
            rl_trader_id: ID of the RL trader.
            initialize: Indicates whether to process the first step in the
                simulation. Since RL algorithms need the first observation we
                allow to reset manually to get this observation.
            logging: Indicates whether to log the limit order book.
            latency_comp_params: Parameters for the latency compensation model.
                Each number represents the level of the order book to from which
                a volume is sampled for a front running order that is placed
                before the actual order with a specified probability.
            ts_save: Timestamp to include in the file names.
            description: Description of the simulation.
            rng: Random number generator.
        """
        self.exchange_name = exchange_name
        self.symbol_name = symbol_name
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.depth = depth
        self.logging = logging
        self.ts_save = ts_save
        self.lob = LimitOrderBook(
            tick_size=self.tick_size,
            lot_size=self.lot_size,
            logging=self.logging,
            ts_save=self.ts_save,
        )
        self.exchange_trader = ExchangeTrader(id="Exchange", depth=self.depth)
        self.traders = traders
        self.traders["Exchange"] = self.exchange_trader
        self.stats = {
            "ts": [],
            "bids": [],
            "asks": [],
            "bid_volumes": [],
            "ask_volumes": [],
            "trader_ids": [],
        }
        self.ts_start = ts_start
        self.ts_end = ts_end
        self.win = win
        self.path = path
        self.latency_comp_params = latency_comp_params
        self.rl_trader_id = rl_trader_id
        self.description = description
        self.rng = rng
        self.sampler = EmpiricalOrderVolumeDistribution(
            path=path_vol_distr, rng=self.rng
        )

        if ts_save is None:
            ts_save = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ts_save = ts_save

        # Set the limit order book reference for each trader
        self.exchange_trader.set_lob(self.lob)
        for i in self.traders.keys():
            traders[i].set_lob(self.lob)
            if i != "Exchange":
                self.stats["trader_ids"].append(i)

        # Initialize the timestamp iterator
        self.time_manager = TimeManager(
            exchange=self.exchange_name,
            symbol=self.symbol_name,
            ts_start=self.ts_start,
            ts_end=self.ts_end,
            path=self.path,
            win_size=self.win,
            max_steps=max_steps,
        )
        self.max_steps = self.time_manager.get_last_time_step() + 1
        self.stats["ts"] = self.time_manager.get_timeline()

        book_name = f"{self.exchange_name}_{self.symbol_name}_order_book"
        self.book_data = scan_parquet(
            name=book_name,
            path=self.path,
            ts_start=self.ts_start,
            ts_end=self.ts_end,
            win=self.win,
            time_manager=self.time_manager,
        )

        trades_name = f"{self.exchange_name}_{self.symbol_name}_trades"
        self.trades_data = scan_parquet(
            name=trades_name,
            path=self.path,
            ts_start=self.ts_start,
            ts_end=self.ts_end,
            win=self.win,
            time_manager=self.time_manager,
        )

        if initialize:
            self.initialize_first_observation()

    def initialize_first_observation(self) -> ObsType:
        """
        Process the initial state of the limit order book. This is done by
        processing the first limit orders in the dataframe. The initial state
        is returned as an observation for RL algorithms.

        Returns:
            Initial observation of the limit order book.
        """
        # ----------------------------------------------------------------------
        # Load market data
        # ----------------------------------------------------------------------
        current_ts = self.time_manager.get_current_ts()
        time_step = self.time_manager.get_current_time_step()
        self.lob.ts = current_ts
        book_data_now = self.book_data.filter(
            pl.col("received_time") == current_ts
        ).collect()

        # ----------------------------------------------------------------------
        # Process exchange trader orders
        # ----------------------------------------------------------------------
        cancel_orders, new_orders = self.exchange_trader.place_orders(
            time_step, book_data_now
        )
        cancel_orders.sort(key=lambda x: x.entry_time)
        new_orders.sort(key=lambda x: x.entry_time)

        while len(cancel_orders) > 0:
            order = cancel_orders.pop(0)
            self.remove_order_from_lob(order)
        while len(new_orders) > 0:
            order = new_orders.pop(0)
            self.add_order_to_lob(order)

        self.lob.update_mid_price_history()

        # ----------------------------------------------------------------------
        # Compute features
        # ----------------------------------------------------------------------
        # Compute scaled price distances to mid price
        mid_price = self.lob.get_mid_price()
        snapshot = self.lob.get_book_info(max_depth=3)
        bid_dist = [(item[0] / mid_price - 1) for item in snapshot["bid_side"]]
        ask_dist = [(item[0] / mid_price - 1) for item in snapshot["ask_side"]]
        bid_dist[0] = bid_dist[0] / 0.006255
        bid_dist[1] = bid_dist[1] / 0.011612
        bid_dist[2] = bid_dist[2] / 0.014852
        bid_dist = bid_dist[:3]
        ask_dist[0] = ask_dist[0] / 0.006255
        ask_dist[1] = ask_dist[1] / 0.008974
        ask_dist[2] = ask_dist[2] / 0.011694
        ask_dist = ask_dist[:3]

        # Compute the mid-price change
        mid_price_change = 0

        # Compute the spread change
        spread = self.lob.best_ask_price - self.lob.best_bid_price
        spread_change = 0
        self.prev_spread = spread

        # Compute the order book imbalances
        bid_vols = [item[1] for item in snapshot["bid_side"]]
        ask_vols = [item[1] for item in snapshot["ask_side"]]
        lob_imbalances = [
            (ask_vols[0] - bid_vols[0]) / (ask_vols[0] + bid_vols[0]),
            (ask_vols[1] - bid_vols[1]) / (ask_vols[1] + bid_vols[1]),
            (ask_vols[2] - bid_vols[2]) / (ask_vols[2] + bid_vols[2]),
        ]

        # Inventory ratio
        if self.rl_trader_id:
            agent = self.traders[self.rl_trader_id]
            inventory = agent.inventory / agent.volume
        else:
            inventory = 0

        feat = (
            [inventory]
            + bid_dist
            + ask_dist
            + [mid_price_change, spread_change]
            + lob_imbalances
        )
        features = np.array(feat, dtype=np.float32)
        features = np.clip(features, -1, 1)

        self.last_obs = features

        return features

    def process_timestep(
        self, action: ActType = None
    ) -> tuple[ObsType, float, bool, dict]:
        """
        Process orders from all traders for one timestep. This includes the
        following steps:
            1. Load market data.
            2. Process the actions of traders.
            3. Update the LOB stats (best bid/ask, volumes, etc.)
            4. Process the incoming market orders.
            5. Compute rewards and check termination.
            6. Update trader statistics.
            7. Process limit orders from the exchange at time t+1.
            8. Compute features for t+1 and return them as observation.

        Args:
            action: Action from the RL agent.

        Returns:
            obs: Observation of the environment.
            reward: Reward for the current timestep.
            terminated: Whether the simulation is terminated.
            info: Additional information about the environment.
        """
        # ----------------------------------------------------------------------
        # 1. Load market data
        # ----------------------------------------------------------------------
        current_ts = self.time_manager.get_current_ts()
        time_step = self.time_manager.get_current_time_step()
        self.lob.ts = current_ts
        next_ts = self.time_manager.get_next_ts()
        trades_data_now = self.trades_data.filter(
            pl.col("received_time").is_between(current_ts, next_ts)
        ).collect()

        # ----------------------------------------------------------------------
        # 2. Process orders from all agents (traders)
        # ----------------------------------------------------------------------
        cancel_orders, new_orders = [], []

        for key in self.traders.keys():
            # Ignore exchange trader
            if key == self.exchange_trader.id:
                continue

            # Process action from agents (traders)
            if key == self.rl_trader_id:
                c, n = self.traders[key].place_orders(
                    time_step,
                    current_ts,
                    action,
                    self.last_obs,
                )
            elif key == "Avellaneda-Stoikov":
                c, n = self.traders[key].place_orders(
                    time_step,
                    current_ts,
                    self.time_manager.get_last_time_step(),
                )
            else:
                c, n = self.traders[key].place_orders(
                    time_step,
                    current_ts,
                )
            cancel_orders.extend(c)
            new_orders.extend(n)

        # Sort both lists based on arrival time
        cancel_orders.sort(key=lambda x: x.entry_time)
        new_orders.sort(key=lambda x: x.entry_time)

        # Process the cancel orders
        while len(cancel_orders) > 0:
            order = cancel_orders.pop(0)
            self.remove_order_from_lob(order)

        # Process the new orders (including front running)
        while len(new_orders) > 0:
            order = new_orders.pop(0)

            if self.latency_comp_params != {}:
                # # Detect the level for bid price
                # price = order.price
                # if order.side:
                #     bids = self.lob.get_bids()
                #     lev = 0
                #     while lev < len(bids) and price <= bids[lev] and lev < 3:
                #         lev += 1
                # # Detect the level for bid price
                # else:
                #     asks = self.lob.get_asks()
                #     lev = 0
                #     while lev < len(asks) and price >= asks[lev] and lev < 3:
                #         lev += 1

                # Always fix the level to 2 for enough volume
                lev = 2

                # Front run the order with a specified probability
                rnd_unif = self.rng.uniform()
                if rnd_unif < self.latency_comp_params[lev]["prob"]:
                    order_copy = copy.deepcopy(order)
                    order_copy.volume = (
                        self.sampler.sample(level=max(lev - 1, 0))
                        / self.latency_comp_params[lev]["divisor"]
                    )
                    order_copy.id = "FrontRun" + get_rnd_str(4)
                    order_copy.trader_id = self.exchange_trader.id
                    self.add_order_to_lob(order_copy)

            self.add_order_to_lob(order)

        # ----------------------------------------------------------------------
        # 3. Update the LOB statistics
        # ----------------------------------------------------------------------
        self.stats["bids"].append(self.lob.best_bid_price)
        self.stats["asks"].append(self.lob.best_ask_price)
        self.stats["bid_volumes"].append(self.lob.get_best_bid_volume())
        self.stats["ask_volumes"].append(self.lob.get_best_ask_volume())

        # ----------------------------------------------------------------------
        # 4. Process the incoming market orders at [t, t+1)
        # ----------------------------------------------------------------------
        buy_orders = self.exchange_trader.process_historical_trades(
            trades_data_now, ts=current_ts, side=True
        )
        sell_orders = self.exchange_trader.process_historical_trades(
            trades_data_now, ts=current_ts, side=False
        )

        # Limit volume of the market orders to lob volume and ignore the rest
        #  to avoid order book depletion
        while len(buy_orders) > 0:
            order = buy_orders.pop(0)
            max_volume = self.lob.get_ask_volume()
            order.volume = min(max_volume, order.volume)
            if order.volume > 0:
                self.add_order_to_lob(order)
        while len(sell_orders) > 0:
            order = sell_orders.pop(0)
            max_volume = self.lob.get_bid_volume()
            order.volume = min(max_volume, order.volume)
            if order.volume > 0:
                self.add_order_to_lob(order)

        # ----------------------------------------------------------------------
        # 5. Compute rewards and check termination
        # ----------------------------------------------------------------------
        reward = 0
        if self.rl_trader_id in self.traders.keys():
            reward = self.traders[self.rl_trader_id].reward

        # Update the time step and check termination
        next_ts = self.time_manager.step_forward()
        terminated = False if next_ts else True

        # ----------------------------------------------------------------------
        # 6.Update the trader statistics
        # ----------------------------------------------------------------------
        for key in self.traders.keys():
            if key == self.exchange_trader.id:
                continue
            self.traders[key].update_stats(time_step)

        # ----------------------------------------------------------------------
        # 7. Process limit orders from the exchange at time t+1
        # ----------------------------------------------------------------------
        # Cancel agent orders
        cancel_orders = []

        for key in self.traders.keys():
            # Ignore exchange trader
            if key == self.exchange_trader.id:
                continue
            c = self.traders[key].cancel_orders()
            cancel_orders.extend(c)

        # Sort both lists based on arrival time
        cancel_orders.sort(key=lambda x: x.entry_time)

        # Process the orders
        while len(cancel_orders) > 0:
            order = cancel_orders.pop(0)
            self.remove_order_from_lob(order)

        prev_mid_price = self.lob.mid_price_history[-1]
        if not terminated:
            book_data_next = self.book_data.filter(
                pl.col("received_time") == next_ts
            ).collect()
            cancel_orders, new_orders = self.exchange_trader.place_orders(
                time_step, book_data_next
            )

            # cancel_orders.sort(key=lambda x: x.entry_time)
            new_orders.sort(key=lambda x: x.entry_time)

            while len(cancel_orders) > 0:
                order = cancel_orders.pop(0)
                self.remove_order_from_lob(order)
            while len(new_orders) > 0:
                order = new_orders.pop(0)
                self.add_order_to_lob(order)

            self.lob.update_mid_price_history()

        # ----------------------------------------------------------------------
        # 8. Compute features for t+1 and return them as observation
        # ----------------------------------------------------------------------
        # Compute scaled price distances to mid price
        mid_price = self.lob.get_mid_price()
        snapshot = self.lob.get_book_info(max_depth=3)
        bid_dist = [(item[0] / mid_price - 1) for item in snapshot["bid_side"]]
        ask_dist = [(item[0] / mid_price - 1) for item in snapshot["ask_side"]]
        bid_dist[0] = bid_dist[0] / 0.006255
        bid_dist[1] = bid_dist[1] / 0.011612
        bid_dist[2] = bid_dist[2] / 0.014852
        bid_dist = bid_dist[:3]
        ask_dist[0] = ask_dist[0] / 0.006255
        ask_dist[1] = ask_dist[1] / 0.008974
        ask_dist[2] = ask_dist[2] / 0.011694
        ask_dist = ask_dist[:3]

        # Compute the scaled mid-price change
        mid_price_change = mid_price / prev_mid_price - 1
        mid_price_change = mid_price_change / 0.17

        # Compute the spread change
        spread = self.lob.best_ask_price - self.lob.best_bid_price
        spread_change = spread / self.prev_spread - 1
        spread_change = spread_change / 0.14
        self.prev_spread = spread

        # Compute the order book imbalances
        bid_vols = [item[1] for item in snapshot["bid_side"]]
        ask_vols = [item[1] for item in snapshot["ask_side"]]
        lob_imbalances = [
            (ask_vols[0] - bid_vols[0]) / (ask_vols[0] + bid_vols[0]),
            (ask_vols[1] - bid_vols[1]) / (ask_vols[1] + bid_vols[1]),
            (ask_vols[2] - bid_vols[2]) / (ask_vols[2] + bid_vols[2]),
        ]

        # Inventory ratio
        if self.rl_trader_id:
            agent = self.traders[self.rl_trader_id]
            inventory = agent.inventory / agent.volume
            inventory = np.clip(inventory, -1, 1)
        else:
            inventory = 0

        feat = (
            [inventory]
            + bid_dist
            + ask_dist
            + [mid_price_change, spread_change]
            + lob_imbalances
        )
        features = np.array(feat, dtype=np.float32)
        features = np.clip(features, -1, 1)
        truncated = False

        # Save the observation
        self.last_obs = features

        return (
            features,
            float(reward),
            terminated,
            truncated,
            {},
        )

    def run(self, visualize_step: int = None) -> None:
        """
        Run the exchange simulation.

        Args:
            max_steps: Maximum number of steps to run in the simulation.
            visualize_step: Visualize the limit order book every n steps.
        """
        # Run the simulation until the end of the data or until the max steps
        iterable = range(self.max_steps)
        for i in tqdm(iterable, desc="Running the exchange simulation"):
            self.process_timestep()

            if visualize_step and i % visualize_step == 0:
                self.lob.visualize()

        # Finish the simulation
        self.lob.close_parquet_writer()
        self.save_exchange_stats()

    def add_order_to_lob(self, order: Order) -> None:
        """Add an order to the limit order book."""
        trades, new_order = self.lob.add_order(order)
        time_step = self.time_manager.get_current_time_step()

        # Process the trades
        for trade in trades:
            trade_price, trade_volume = trade["price"], trade["volume"]
            order_make, order_take = trade["order_make"], trade["order_take"]

            self.traders[order_make.trader_id].process_trade(
                time_step, trade_price, trade_volume, order_make, True
            )
            self.traders[order_take.trader_id].process_trade(
                time_step, trade_price, trade_volume, order_take, False
            )

        # Add the new order to the trader's active orders
        if new_order:
            self.traders[new_order.trader_id].add_order(new_order)

    def remove_order_from_lob(self, order: Order) -> None:
        """Remove an order from the limit order book."""
        order = self.lob.remove_order_by_id(order.id)
        if order:
            self.traders[order.trader_id].remove_order(order)

    def save_exchange_stats(self) -> None:
        """Save the exchange statistics to a pickle file."""
        if self.logging:
            file_name = f"exchange_stats_{self.ts_save}.pkl"
            with open(os.path.join("results_backtest", file_name), "wb") as f:
                pickle.dump(self.stats, f)

            # Save the trader stats
            for key in self.traders.keys():
                if key == self.exchange_trader.id:
                    continue
                path = "results_backtest"
                self.traders[key].save_stats(path, self.ts_save)

            # Save the description
            file_name = f"description_{self.ts_save}.txt"
            with open(os.path.join("results_backtest", file_name), "w") as f:
                f.write(self.description)
