"""Plotting functionalities."""

import datetime
from typing import Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from src.lob.backtest_metrics import drawdowns

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

    # small_size = 16
    # medium_size = 18
    # bigger_size = 20
    small_size = 20
    medium_size = 22
    bigger_size = 24

    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small_size)  # legend fontsize
    plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title


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
    loc_interval: int = None,
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
        ylim: Limits of the y-axis.
        legend: Whether to show the legend.
        figsize: Size of the figure.
        loc_interval: Interval of the x-axis labels.
        color: Color of the plot.
        save_path: Path to save the plot.
    """
    default_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    color = color if color else default_color
    plt.figure(figsize=figsize)
    plt.plot(x, y, color=color)
    # Show every 5th label
    plt.gca().xaxis.set_major_locator(MultipleLocator(3))
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
    if loc_interval:
        locator = mdates.DayLocator(interval=loc_interval)
        plt.gca().xaxis.set_major_locator(locator)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def make_drawdown_plot(
    x: Union[list, np.ndarray, pd.Series],
    y: Union[list, np.ndarray, pd.Series],
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    ylim: tuple = None,
    legend: bool = False,
    figsize: tuple = (12, 4.5),
    loc_interval: int = None,
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
        ylim: Limits of the y-axis.
        legend: Whether to show the legend.
        figsize: Size of the figure.
        loc_interval: Interval of the x-axis labels.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=figsize)
    plt.fill_between(x, y, 0, color="red", alpha=0.3)
    plt.plot(x, y, color="red", alpha=0.5)
    plt.gca().xaxis.set_major_locator(MultipleLocator(3))
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if ylim:
        plt.ylim(ylim)
    if loc_interval:
        locator = mdates.DayLocator(interval=loc_interval)
        plt.gca().xaxis.set_major_locator(locator)
    if title:
        plt.title(title)
    if legend:
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_multi_results(
    results: dict,
    metric: str,
    ylabel: str,
    ylim: tuple = None,
    legend: bool = True,
    title: str = None,
    print_stats: bool = True,
    figsize: tuple = (12, 4.5),
    locator_interval: int = None,
    save_path: str = None,
) -> None:
    """
    Generalized function to plot results for a given metric (P&L, volume, etc.)
    over multiple strategies.

    Args:
        results: Dictionary containing strategy results with timestamps and
            trader stats.
        metric: Metric to plot (e.g., 'adj_pnl', 'total_volume', or 'spread').
        ylabel: Label for the y-axis.
        ylim: Limits for the y-axis. Default is None.
        legend: Whether to show the legend. Default is True.
        title: Title of the plot. Default is None.
        print_stats: Whether to print statistics for the given metric. Default
            is True.
        figsize: Size of the figure. Default is (12, 4.5).
        locator_interval: Interval for the x-axis locator. Default is None.
        save_path: Path to save the plot. Default is None.
    """
    plt.figure(figsize=figsize)

    for i, value in enumerate(results.values()):
        x = value["timestamps"]
        y = np.array(value["trader_stats"][metric])
        label = f"PMM (priority {i})"
        plt.plot(x, y, label=label)

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(ylim)
    if legend:
        plt.legend()
    if title:
        plt.title(title)
    if locator_interval:
        locator = mdates.DayLocator(interval=locator_interval)
        plt.gca().xaxis.set_major_locator(locator)
    plt.tight_layout()

    # Save the figure if required
    if save_path:
        plt.savefig(save_path)

    plt.show()

    if print_stats:
        # Print the statistics for the given metric
        print(f"Statistics for {metric}:")
        for key, value in results.items():
            y = np.array(value["trader_stats"][metric])
            print(f"  {key} - Final {metric.replace('_', ' ')}: {y[-1]:.2f}")
        print()


def plot_comparison(
    x_data_list: list[list],
    y_data_list: list[list],
    labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str = None,
    figsize: tuple = (12, 4.5),
    save_path: str = None,
) -> None:
    """
    Plot a comparison between multiple time series.

    Args:
        x_data_list: List of lists containing x-axis values.
        y_data_list: List of lists containing y-axis values.
        labels: List of labels for the legend.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Title of the plot. Default is None.
        figsize: Size of the figure. Default is (12, 4.5).
        save_path: Path to save the plot. Default is None.
    """
    plt.figure(figsize=figsize)

    for x, y, label in zip(x_data_list, y_data_list, labels):
        plt.plot(x, y, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.legend(loc="upper left")
    if title:
        plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
