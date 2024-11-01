"""Methods related to actions of reinforcement learning agents."""

from typing import TypeVar

ActType = TypeVar("ActType")


# TODO: Update to the latest version of the environment
def decode_action_v1(action: ActType):
    """
    Decode an action.

    Args:
        action: Action to decode.
    """
    # If action 0, do not place any orders
    if action == 0:
        return [], []

    # If action 1, place orders on both sides
    elif action == 1:
        return 1
