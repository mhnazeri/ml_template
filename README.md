[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Inspired by [Stanford CS230 blog post](https://cs230.stanford.edu/blog/pytorch/), I created this repository to use as a template for my ML projects. `util.py` will be updated with helper functions through time.

Main libraries:
* [PyTorch](https://www.pytorch.org/): as the main ML framework
* [Comet.ml](https://www.comet.ml): tracking code, logging experiments
* [OmegaConf](https://omegaconf.readthedocs.io/en/latest/): for managing configuration files

## Installation
First create a virtual env for the project. 
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the latest version of PyTorch from the [official site](htpps://www.pytorch.org/). Finally, run the following:
```bash
pip install -r requirements
```
To set up Comet.Ml follow the [official documentations](https://www.comet.ml/docs/).
## What's new
* (July 3, 2021) Added `run.sh` bash file to easily run different parts of the code.
* (July 3, 2021) Added UUID generation for sample names.
* (July 3, 2021) Added MNIST as an example (not SOTA).

## Run
To run different parts, just edit [run.sh](./run.sh) file and run it using `.` calling.

