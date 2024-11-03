![Adversarial Inverse Reinforcement Learning for Market Making](.images/title_image.png)

This repository contains the code for the paper *Adversarial Inverse Reinforcement Learning* published in the proceedings of the [ICAIF'24](https://ai-finance.org/) conference. The paper is a joint work with Martin Stefanik (Richfox Capital &  ETH Zürich), Moritz Weiß (ETH Zürich) and Prof. Dr. Josef Teichmann (ETH Zürich).

The full training pipeline can be found in [`main.ipynb`](main.ipynb).

## TODO list

- [x] New poetry setup.
- [x] Ensure the reproducibility of the environment with modified packages.
- [x] Organize all modules under the `src` directory.
- [x] Update the title image.
- [x] Data download pipeline.
- [x] Data cleaning pipeline.
- [x] Generation of volume distributions.
- [x] Generation of rollouts.
- [ ] Backtest results pipeline.
- [ ] Remove unnecessary codes and files.
- [x] Add PRINT constant.
- [ ] Add Appendix section to the notebook with data analysis.
- [ ] Update the README with the new structure.
- [ ] Add appendix with additional results?
- [ ] Add the link to the paper.
- [ ] Upload the saved models.
- [ ] Update the documentation (installation, repository description, modified packages, training details, strategy limitations, etc.).

## Abstract

In this paper, we propose a novel application of the Adversarial Inverse Reinforcement Learning (AIRL) algorithm combining the framework of generative adversarial networks and inverse reinforcement learning for automated reward acquisition in the market making problem. We demonstrate that this algorithm can be used to learn a pure market making strategy. An advantage of such an approach is the possibility for further training in the reinforcement learning framework while potentially preserving its explainability and already acquired domain knowledge incorporated in an expert's decision process.

## Disclaimer

This repository was prepared for informational purposes, in part, by members, in their personal capacity, of the Research Department of Richfox Capital Investment Management AG hereinafter "Richfox" and is not a product of Richfox. The information is not intended to be relied upon as investment advise or as a recommendation to buy, sell, or hold any securities or financial instruments. The contents of this repository should not be construed as an offer, solicitation, or recommendation to engage in any investment activity. The members have taken reasonable care in preparing this repository, but do not warrant or guarantee the accuracy, completeness, or timeliness of the information contained herein. The repository may include information obtained from third-party sources, which we believe to be reliable. However, Richfox does not endorse or make any representation or warranty regarding the accuracy or completeness of such third-party information. Any reliance on the information contained is at the reader’s own risk. By accessing or using this repository, you acknowledge and agree to the terms of this disclaimer. If you do not agree with these terms, please do not rely on the information contained herein.
