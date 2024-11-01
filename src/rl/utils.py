"""Various helper functions for RL algorithms."""

import os
import pickle
from datetime import datetime, timedelta
from random import uniform

import torch as th
from imitation.rewards import reward_nets
from stable_baselines3.common import base_class
from stable_baselines3.ppo import PPO


def send_notification(message: str, time: int = 10000) -> None:
    """
    Send a notification to the user.

    Args:
        message: The message to send.
        time: The time for which the notification should be displayed.
    """
    os.system(
        f'notify-send -t {time} "VSCode notification manager" "{message}"'
    )  # nosec: B605


def save_model(
    learner: base_class.BaseAlgorithm,
    reward_net: reward_nets,
    stats: dict,
    path: str,
    ts: datetime,
) -> None:
    """
    Saves the model to the specified path.

    Args:
        learner: Learner policy.
        reward_net: Reward network.
        stats: Training statistics.
        path: Path to save the model to.
        ts: Timestamp to include in the file names.
    """
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # Save the learner
    learner.save(f"{path}/{ts}_learner")

    # Save the reward net
    th.save(reward_net, f"{path}/{ts}_reward_nn")

    # Save the training statistics
    with open(f"{path}/{ts}_stats.pkl", "wb") as f:
        pickle.dump(stats, f)


def load_model(path: str, ts: datetime) -> tuple:
    """
    Loads the model from the specified path.

    Args:
        path: Path to load the model from.
        ts: Timestamp to include in the file names.

    Returns:
        The learner, reward net, and training statistics.
    """
    # Load the learner
    learner = PPO.load(f"{path}/{ts}_learner", verbose=1)

    # Load the reward net
    reward_net = th.load(f"{path}/{ts}_reward_nn")

    # Load the training statistics
    with open(f"{path}/{ts}_stats.pkl", "rb") as f:
        stats = pickle.load(f)

    return learner, reward_net, stats


def random_timestamp(
    start_timestamp: datetime, end_timestamp: datetime
) -> datetime:
    """
    Return a random timestamp between the start and end timestamps.

    Args:
        start_timestamp: Start timestamp.
        end_timestamp: End timestamp.

    Returns:
        Random timestamp between the start and end timestamps.
    """
    # Calculate the time difference
    diff = end_timestamp - start_timestamp

    # Generate a random timedelta within the time difference
    random_timedelta = timedelta(seconds=uniform(0, diff.total_seconds()))

    # Add the random timedelta to the start datetime
    random_timestamp = start_timestamp + random_timedelta

    return random_timestamp
