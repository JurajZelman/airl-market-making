"""Plotting functionalities."""

import datetime
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lob.backtest_metrics import drawdowns

COLOR_GREEN = "#13961a"
COLOR_RED = "#eb5c14"


def set_plot_style() -> None:
    """Set the plotting style."""
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update(
        {"axes.prop_cycle": plt.cycler("color", plt.cm.tab10.colors)}
    )
    # Change to computer modern font and increase font size
    plt.rcParams.update({"font.family": "cmr10", "font.size": 12})
    plt.rcParams.update({"axes.formatter.use_mathtext": True})

    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def visualize_backtest(
    ts: list[datetime.datetime], trader_stats: dict, initial_cost: float
) -> None:
    """
    Visualize the backtest results.

    Args:
        ts: Timestamps.
        trader_stats: Trader statistics.
        initial_cost: Initial cost of the trader used for the computation of the
            equity curve.
    """
    # PLOT - Adjusted PnL
    make_plot(
        x=ts,
        y=trader_stats["adj_pnl"],
        title="P&L",
        xlabel="Timestamp",
        ylabel="P&L (USDT)",
    )

    # PLOT - Returns
    equity = pd.Series(np.array(trader_stats["adj_pnl"]) + initial_cost)
    make_plot(
        x=ts,
        y=equity.pct_change() * 100,
        title="Returns",
        xlabel="Timestamp",
        ylabel="Returns (%)",
    )

    # PLOT - Drawdowns
    dd = drawdowns(equity)
    make_drawdown_plot(
        x=ts,
        y=dd,
        title="Drawdowns",
        xlabel="Timestamp",
        ylabel="Drawdown (%)",
    )

    # PLOT - Inventory
    make_plot(
        x=ts,
        y=trader_stats["inventory"],
        title="Inventory",
        xlabel="Timestamp",
        ylabel="Inventory (SOL)",
        color="darkorange",
    )

    # PLOT - Total traded volume
    make_plot(
        x=ts,
        y=trader_stats["total_volume"],
        title="Total traded volume",
        xlabel="Timestamp",
        ylabel="Total traded volume (USDT)",
        color="darkorange",
    )

    # PLOT - Transaction costs
    make_plot(
        x=ts,
        y=trader_stats["cum_costs"],
        title="Cumulative transaction fees",
        xlabel="Timestamp",
        ylabel="Transaction fees (USDT)",
    )

    # PLOT - Number of trades
    make_plot(
        x=ts,
        y=trader_stats["trade_count"],
        title="Number of trades",
        xlabel="Timestamp",
        ylabel="Number of trades",
    )

    # PLOT - Quoted spreads
    asks = np.array(trader_stats["quoted_ask_price"])
    bids = np.array(trader_stats["quoted_bid_price"])
    spreads = np.where(
        np.isnan(asks) | np.isnan(bids), np.nan, np.subtract(asks, bids)
    )
    make_plot(
        x=ts,
        y=spreads,
        title="Quoted spread",
        xlabel="Timestamp",
        ylabel="Quoted spread",
        color="black",
    )

    # PLOT - Quoted bid volume
    make_plot(
        x=ts,
        y=trader_stats["quoted_bid_volume"],
        title="Quoted bid volume",
        xlabel="Timestamp",
        ylabel="Quoted bid volume (SOL)",
        color=COLOR_GREEN,
    )

    # PLOT - Quoted ask volume
    make_plot(
        x=ts,
        y=trader_stats["quoted_ask_volume"],
        title="Quoted ask volume",
        xlabel="Timestamp",
        ylabel="Quoted ask volume (SOL)",
        color=COLOR_RED,
    )


def make_plot(
    x: Union[list, np.ndarray, pd.Series],
    y: Union[list, np.ndarray, pd.Series],
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    ylim: tuple = None,
    legend: bool = False,
    figsize: tuple = (12, 4.5),
    color: str = None,
    save_path: str = None,
) -> None:
    """
    Make a plot.

    Args:
        x: X-axis data.
        y: Y-axis data.
        title: Title of the plot.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        legend: Whether to show the legend.
        figsize: Size of the figure.
        color: Color of the plot.
    """
    default_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    color = color if color else default_color
    plt.figure(figsize=figsize)
    plt.plot(x, y, color=color)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if legend:
        plt.legend()
    plt.tight_layout()
    if ylim:
        plt.ylim(ylim)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def make_drawdown_plot(
    x: Union[list, np.ndarray, pd.Series],
    y: Union[list, np.ndarray, pd.Series],
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    legend: bool = False,
    figsize: tuple = (12, 5),
    save_path: str = None,
) -> None:
    """
    Make a drawdown plot.

    Args:
        x: X-axis data.
        y: Y-axis data.
        title: Title of the plot.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        legend: Whether to show the legend.
        figsize: Size of the figure.
    """
    plt.figure(figsize=figsize)
    plt.fill_between(x, y, 0, color="red", alpha=0.3)
    plt.plot(x, y, color="red", alpha=0.5)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if legend:
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
