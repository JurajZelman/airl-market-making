"""Module for sampling from the empirical distributions."""

import os

import numpy as np
import pandas as pd


class EmpiricalOrderVolumeDistribution:
    """
    Class for sampling order volumes from the empirical distribution estimated
    on the insample order book data.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        """
        Initialize the class by loading the volume distributions from the pickle
        files.

        Args:
            rng: Numpy random generator.
        """
        base_path = os.path.join(os.getcwd(), "distributions")
        self.vols_level_0 = pd.read_pickle(
            os.path.join(base_path, "volumes_level_0.pkl")
        ).to_numpy()
        self.vols_level_1 = pd.read_pickle(
            os.path.join(base_path, "volumes_level_1.pkl")
        ).to_numpy()
        self.vols_level_2 = pd.read_pickle(
            os.path.join(base_path, "volumes_level_2.pkl")
        ).to_numpy()
        self.rng = rng

    def sample(self, level: int) -> float:
        """
        Sample a volume from the empirical distribution.

        Args:
            level: The level of the order book to sample from.

        Returns:
            The sampled volume.
        """
        if level == 0:
            return self.rng.choice(self.vols_level_0)
        elif level == 1:
            return self.rng.choice(self.vols_level_1)
        elif level == 2:
            return self.rng.choice(self.vols_level_2)
        else:
            raise ValueError("Level must be between 0 and 2.")
