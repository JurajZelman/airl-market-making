"""Expert policies for imitation learning."""

import math

import gymnasium as gym
import numpy as np


class RandomPolicy:
    """Random policy."""

    def __init__(self, action_space: gym.spaces.Space) -> None:
        """
        Initialize the random policy.

        Args:
            action_space: Action space of the environment.
        """
        self.action_space = action_space

    def predict(self, obs, state, *args, **kwargs) -> tuple:
        """
        Get the random actions for the given observations, states and dones.

        Args:
            obs: Observations of the environment.
            state: States.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Actions and states.
        """
        return np.array([self.action_space.sample()]), state


class ExpertPolicy:
    """Expert policy."""

    def __init__(self) -> None:
        """Initialize the expert policy."""

    def predict(self, obs, state, *args, **kwargs) -> tuple:
        """
        Get the expert actions for the given observations, states and dones.

        Args:
            obs: Observations of the environment.
            state: States.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Actions and states.
        """
        if math.isclose(obs[0][0], 0):
            return np.array([5]), None
        else:
            return np.array([17]), None
