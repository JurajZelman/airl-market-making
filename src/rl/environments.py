"""Reinforcement learning environments."""

from datetime import datetime, timedelta
from typing import TypeVar

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from src.lob.exchange import Exchange
from src.rl.utils import random_timestamp

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class LimitOrderBookGym(gym.Env):
    """Gym environment with limit order book simulator."""

    def __init__(
        self,
        exchange_name: str,
        symbol_name: str,
        tick_size: float,
        lot_size: float,
        depth: int,
        traders: list,
        max_steps: int,
        ts_start: datetime,
        ts_end: datetime,
        deterministic: bool,
        win: int,
        path: str,
        rl_trader_id: str,
        latency_comp_params: dict,
        logging: bool,
        ts_save: datetime,
        description: str,
        rng: np.random.Generator,
    ) -> None:
        """
        Initialize the LimitOrderBookGym environment.

        Args:
            exchange_name: Name of the exchange.
            symbol_name: Name of the symbol.
            tick_size: Tick size of the symbol.
            lot_size: Lot size of the symbol.
            depth: Depth of the limit order book.
            traders: List of traders participating in the exchange.
            max_steps: Maximum number of steps in the environment.
            ts_start: Start timestamp.
            ts_end: End timestamp.
            deterministic: Whether to use a deterministic environment. If False,
                the environment will randomly sample trajectories between the
                start and end timestamps of the desired length.
            win: Window size for features.
            path: Path to the directory containing the datasets.
            rl_trader_id: ID of the RL trader.
            latency_comp_params: Parameters for the latency compensation model.
                Each number represents the level of the order book to from which
                a volume is sampled for a front running order that is placed
                before the actual order with a specified probability.
            logging: Whether to log the environment.
            ts_save: Timestamp to include in the file names.
            description: Description of the environment.
            rng: Random number generator.
        """
        # Load the parameters (for reset method)
        self.exchange_name = exchange_name
        self.symbol_name = symbol_name
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.depth = depth
        self.traders = traders
        self.max_steps = max_steps
        self.ts_start = ts_start
        self.ts_end = ts_end
        self.deterministic = deterministic
        self.win = win
        self.path = path
        self.rl_trader_id = rl_trader_id
        self.latency_comp_params = latency_comp_params
        self.logging = logging
        self.ts_save = ts_save
        self.description = description
        self.rng = rng

        self.action_space = Discrete(21)
        self.observation_space = Box(low=-1, high=1, shape=(12,))

        # Initialize the environment
        self.reset()

    def reset(self, seed: int = None, options: dict = None) -> ObsType:
        """
        Reinitialize the LimitOrderBookGym environment.

        Args:
            seed: Seed for the environment.
            options: Options for the environment.

        Returns:
            obs: Observation of the environment.
        """
        # Reset the agents
        for trader in self.traders:
            if trader != "Exchange":
                self.traders[trader].reset()

        # Set the start timestamp (randomly if not deterministic)
        if self.deterministic is False:
            ts_end_lag = self.ts_end - timedelta(hours=1)
            ts = random_timestamp(self.ts_start, ts_end_lag)
        else:
            ts = self.ts_start

        # Initialize the exchange
        self.exchange = Exchange(
            exchange_name=self.exchange_name,
            symbol_name=self.symbol_name,
            tick_size=self.tick_size,
            lot_size=self.lot_size,
            depth=self.depth,
            traders=self.traders,
            max_steps=self.max_steps,
            ts_start=ts,
            ts_end=self.ts_end,
            win=self.win,
            path=self.path,
            rl_trader_id=self.rl_trader_id,
            latency_comp_params=self.latency_comp_params,
            logging=self.logging,
            ts_save=self.ts_save,
            description=self.description,
            initialize=False,
            rng=self.rng,
        )
        obs = self.exchange.initialize_first_observation()
        info = {}

        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Action to take in the environment.

        Returns:
            obs: Observation of the environment.
            reward: Reward from the environment.
            term: Whether the episode terminated.
            trunc: Whether the episode was truncated.
            info: Additional information about the environment.

        """
        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = self.exchange.process_timestep(action=action)

        # Close environment when the episode terminates
        if terminated:
            self.close()

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment."""
        pass

    def close(self) -> None:
        """Close the environment."""
        self.exchange.lob.close_parquet_writer()
