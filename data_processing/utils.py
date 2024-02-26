"""Helper functions for the data analysis."""

import datetime
import os
import random

import matplotlib.pyplot as plt


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


def get_parquet_args():
    """
    Returns the parquet arguments for saving the data and avoiding the timestamp
    conversion issues.
    """
    return {
        "coerce_timestamps": "us",  # Coerce timestamps to microseconds
        "allow_truncated_timestamps": True,  # Allow truncated timestamps
    }


def get_rnd_id(length: int = 6) -> int:
    """Get a random int of given length."""
    return random.randint(10 ** (length - 1), 10**length - 1)


def get_list_of_second_timestamps(date: datetime.datetime) -> list:
    """Generate a list of second timestamps for a given date."""
    seconds = [date + datetime.timedelta(seconds=x) for x in range(86400)]
    return seconds


def get_list_of_dates_between(
    start_date: datetime.datetime, end_date: datetime.datetime
) -> list:
    """Generate a list of dates between two dates."""
    days = [
        start_date + datetime.timedelta(days=x)
        for x in range((end_date - start_date).days + 1)
    ]
    return days


def ensure_dir_exists(path: str) -> None:
    """Ensure that the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
