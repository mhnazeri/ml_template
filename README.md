[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

Inspired by [Stanford CS230 blog post](https://cs230.stanford.edu/blog/pytorch/) I created this repository to use as a template for my ML projects. `util.py` will be updated with helper functions through time. The `logging.py` is a wrapper around three main libraries [Comet.ml](https://www.comet.ml), [Wandb](https://www.wandb.ai) and [Tensorboard](https://github.com/lanpa/tensorboardX). You can choose between any of them by passing the name of the library as an argument to the logger class (note that the functionality of logger is not tested yet).

Main libraries:
* [PyTorch](pytorch.org/): as the main ML framework
* [Comet.ml](https://www.comet.ml): tracking code, logging experiments
* [Wandb](https://www.wandb.ai): tracking code, logging experiments
* [OmegaConf](https://omegaconf.readthedocs.io/en/latest/): for managing configuration files

