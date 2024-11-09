![Adversarial Inverse Reinforcement Learning for Market Making](.images/title_image.png)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?style=flat&logo=python)](https://www.python.org/downloads/release/python-3119/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Stable-Baselines3 Badge](https://img.shields.io/badge/stable--baselines3-v2.2.1-green)](https://stable-baselines3.readthedocs.io/en/master/)
[![Imitation Badge](https://img.shields.io/badge/imitation-v1.0.0-green)](https://imitation.readthedocs.io/en/latest/)
[![GPL-3.0 License](https://img.shields.io/badge/License-GPL%203.0-green)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains the code for the paper *Adversarial Inverse Reinforcement Learning for Market Making* (2024) [[arxiv](), [ACM]() - both will be added soon] by Juraj Zelman (Richfox Capital &  ETH Zürich), Martin Stefanik (Richfox Capital &  ETH Zürich), Moritz Weiß (ETH Zürich) and Prof. Dr. Josef Teichmann (ETH Zürich). The paper was published and presented at the [5th ACM International Conference on AI in Finance (ICAIF ’24)](https://ai-finance.org/). The links of both affiliations: [Richfox Capital](https://www.richfox.com/) and [ETH Zürich (Dept. of Mathematics)](https://math.ethz.ch/).

The full training pipeline can be found in [`main.ipynb`](main.ipynb). Beforehand, see the [Installation](#installation) section below.

We hope you will learn something new here and that this project sparks your curiosity to explore reinforcement learning methods, as well as other fields!

## Abstract

In this paper, we propose a novel application of the Adversarial Inverse Reinforcement Learning (AIRL) algorithm combining the framework of generative adversarial networks and inverse reinforcement learning for automated reward acquisition in the market making problem. We demonstrate that this algorithm can be used to learn a pure market making strategy. An advantage of such an approach is the possibility for further training in the reinforcement learning framework while potentially preserving its explainability and already acquired domain knowledge incorporated in an expert's decision process.

## Installation

In order to run the code, we recommend you to install [`pyenv`](https://github.com/pyenv/pyenv) (for managing Python versions) and [`poetry`](https://python-poetry.org/) (for managing dependencies) or alternative management tools. The project can be installed as follows:

1. Clone and cd into the repository:

    ```bash
    git clone git@github.com:JurajZelman/airl-market-making.git
    cd airl-market-making
    ```

2. Set the Python version to `3.11.9`:

    ```bash
    pyenv install 3.11.9
    pyenv local 3.11.9
    ```

3. Install the dependencies:

    ```bash
    poetry install
    ```

    We recommend you to install the dependencies this way, as the repository uses two modified packages located in the [`.packages`](.packages) directory for better monitoring of the training process and a minor bug fix.
4. Run the [`main.ipynb`](main.ipynb) notebook. Note that in order to query the pricing and trades data, you will need an active data provider subscription as described in Section 1 of the notebook.

## Package modifications ([SB3](https://stable-baselines3.readthedocs.io/en/master/) & [imitation](https://imitation.readthedocs.io/en/latest/index.html))

The repository uses two modified packages located in the [`.packages`](.packages) directory (all modifications are marked with `# MODIFIED` comments). The modifications are as follows:

- [`common.py`](.packages/imitation-1.0.0/src/imitation/algorithms/adversarial/common.py): In this file we implement multiple enhancements that seemed to empirically improve the training process. Firstly, we update perform the discriminator optimization step for every batch of data instead of as an aggregate. Further, we scale the rewards for the PPO (generator) training and implement balancing of expert samples in the training process to improve the stability of the training process. Lastly, we enhance the monitoring of the training statistics.
- [`utils.py`](.packages/stable_baselines3-2.2.1/stable_baselines3/common/utils.py): Here we add a minor fix to avoid overflow warnings that might sometimes appear during the training process of the PPO algorithm (generator).
- [`logger.py`](.packages/stable_baselines3-2.2.1/stable_baselines3/common/logger.py): In this file we suppress the logging printouts during the training process.
- [`bc.py`](.packages/imitation-1.0.0/src/imitation/algorithms/bc.py): Lastly, this enhancement of training process monitoring for the behavioral cloning algorithm is an artifact of the initial research and is not used in the final implementation.

## Contributing

In case you find any bugs, have any suggestions for improvements or have any questions, feel free to open an issue or a pull request. I am happy to help and discuss any potential improvements.

## Disclaimer

This repository was prepared for informational purposes, in part, by members, in their personal capacity, of the Research Department of Richfox Capital Investment Management AG hereinafter "Richfox" and is not a product of Richfox. The information is not intended to be relied upon as investment advise or as a recommendation to buy, sell, or hold any securities or financial instruments. The contents of this repository should not be construed as an offer, solicitation, or recommendation to engage in any investment activity. The members have taken reasonable care in preparing this repository, but do not warrant or guarantee the accuracy, completeness, or timeliness of the information contained herein. The repository may include information obtained from third-party sources, which we believe to be reliable. However, Richfox does not endorse or make any representation or warranty regarding the accuracy or completeness of such third-party information. Any reliance on the information contained is at the reader’s own risk. By accessing or using this repository, you acknowledge and agree to the terms of this disclaimer. If you do not agree with these terms, please do not rely on the information contained herein.
