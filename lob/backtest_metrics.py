"""Metrics for evaluating the performance of a trading strategy."""

import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    """
    Compute the total return of a strategy in percent.

    Args:
        equity: Equity curve of the strategy.

    Returns:
        The total return in percent.
    """
    equity = equity.to_numpy()
    return (equity[-1] / equity[0] - 1) * 100


def win_rate(equity: pd.Series) -> float:
    """
    Compute the win rate of a strategy in percent.

    Args:
        equity: Equity curve of the strategy.

    Returns:
        The win rate in percent.
    """
    returns = get_returns_from_equity(equity)
    return (returns > 0).mean() * 100


def get_returns_from_equity(equity: pd.Series) -> pd.Series:
    """
    Compute the returns of a strategy.

    Args:
        equity: Equity curve of the strategy.

    Returns:
        The returns of the strategy.
    """
    return pd.Series(np.diff(equity), index=equity.index[1:])


def best_return(equity: pd.Series) -> pd.Series:
    """
    Compute the best return of a strategy in percent.

    Args:
        equity: Equity curve of the strategy.

    Returns:
        The best return in percent.
    """
    returns = get_returns_from_equity(equity)
    return returns.max() * 100


def worst_return(equity: pd.Series) -> pd.Series:
    """
    Compute the worst return of a strategy in percent.

    Args:
        equity: Equity curve of the strategy.

    Returns:
        The worst return in percent.
    """
    returns = get_returns_from_equity(equity)
    return returns.min() * 100


def average_return(equity: pd.Series) -> pd.Series:
    """
    Compute the average return of a strategy in percent.

    Args:
        equity: Equity curve of the strategy.

    Returns:
        The average return in percent.
    """
    returns = get_returns_from_equity(equity)
    return returns.mean() * 100


def skewness(returns: pd.Series) -> float:
    """
    Compute the skewness of the returns.

    Args:
        returns: Returns of the strategy.

    Returns:
        The skewness.
    """
    return returns.skew(axis=0, skipna=True)


def kurtosis(returns: pd.Series) -> float:
    """
    Compute the kurtosis of the returns.

    Args:
        returns: Returns of the strategy.

    Returns:
        The kurtosis.
    """
    return returns.kurt(axis=0, skipna=True)


def volatility(returns: pd.Series) -> float:
    """
    Compute the volatility of the returns.

    Args:
        returns: Returns of the strategy.

    Returns:
        The volatility.
    """
    return returns.std(axis=0, skipna=True)


def downside_volatility(returns: pd.Series, threshold=0) -> float:
    """
    Compute downside volatility of returns below a specified threshold.

    Args:
        returns: Returns of the strategy.
        threshold: Minimum acceptable return (default is 0).

    Returns:
        The downside volatility.
    """
    excess_returns = np.minimum(returns - threshold, 0)
    downside_volatility = np.std(excess_returns, ddof=1)
    return downside_volatility


def drawdowns(equity: pd.Series) -> pd.Series:
    """
    Compute the drawdowns of a strategy. Values are expressed in percentage.

    Args:
        pnl: Profit and loss of the strategy.

    Returns:
        A DataFrame containing the wealth index and the previous peaks.
    """
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks
    return drawdowns * 100


def max_drawdown(equity: pd.Series) -> float:
    """
    Compute the maximum drawdown of a strategy. Value is expressed in
    percentage.

    Args:
        pnl: Profit and loss of the strategy.

    Returns:
        The maximum drawdown.
    """
    dd = drawdowns(equity)
    max_drawdown = np.min(dd)
    return max_drawdown


def max_drawdown_duration(equity: pd.Series) -> float:
    """
    Compute the maximum drawdown duration of a strategy. Value is expressed in
    number of time steps.

    Args:
        pnl: Profit and loss of the strategy.

    Returns:
        The maximum drawdown duration in the number of time steps.
    """
    dd = drawdowns(equity)
    counter, max_length = 0, 0
    for value in dd:
        if value == 0:
            counter = 0
        else:
            counter += 1
            max_length = max(max_length, counter)

    return max_length
