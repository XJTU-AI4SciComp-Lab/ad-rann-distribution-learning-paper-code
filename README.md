# AD-RaNN

Official MATLAB implementation of **Adaptive-Distribution Randomized Neural Networks (AD-RaNN)** for PDE solving and operator learning.

## Paper Information

- **Title:** Adaptive-Distribution Randomized Neural Networks for PDEs: A Low-Dimensional Distribution-Learning Framework
- **Authors:** You Yang and Fei Wang
- **Status:** arXiv preprint
- **arXiv:** [arXiv:2604.23999](https://arxiv.org/abs/2604.23999)
- **Corresponding author:** Fei Wang (`feiwang.xjtu@xjtu.edu.cn`)
- **Institution:** XJTU AI for Scientific Computing Lab, Xi'an Jiaotong University

## Method Summary

AD-RaNN learns a low-dimensional parameterization of the hidden-feature sampling distribution while retaining the efficient linear least-squares structure of randomized neural networks. This repository contains PDE-Driven Adaptive Distribution (PDAD), Data-Driven Adaptive Distribution (DDAD), and numerical examples for PDEs and operator-learning problems.

## Installation

1. Clone or download this repository.
2. Open MATLAB and set the repository root as the current folder.
3. Run scripts from the repository root so that relative paths are resolved correctly.

MATLAB R2022b or later is recommended. No Python environment is required for the MATLAB examples. The code has been tested with MATLAB R2024b. No additional MATLAB toolboxes are required unless otherwise specified in an experiment README.

## Quick Start

Run the minimal 2-D Poisson example:

```matlab
run("examples/poisson_2d/run.m")
```

Experiment parameters are defined in:

```text
examples/poisson_2d/config.m
```

## Reproduce Results

Run the Poisson random-seed study:

```matlab
run("experiments/poisson_2d/run_seed_study.m")
```

All full numerical studies are organized under `experiments/`, including Allen-Cahn, Burgers, Helmholtz, Black-Scholes, L-shaped Poisson problems, activation comparisons, parameter studies, and operator-learning experiments. Use the corresponding `run_*.m` script as the entry point and edit the nearby `config*.m` file when necessary.

> **Reproducibility note:** The paper reports results from multiple experiments with different problem settings and random trials. Therefore, fully reproducing a specific table or figure may require adjusting parameters in the corresponding `config.m` or `config_*.m` file, including the random seed, number of features, learning rate, regularization parameter, iteration count, data path, or activation function.

Some full-scale experiments may require substantial runtime and memory. Start with the Poisson quick-start example before running larger studies.

## Repository Structure

- `src/`: reusable AD-RaNN algorithms and utilities
- `examples/`: the minimal single-seed 2-D Poisson example
- `experiments/`: full numerical studies and result-reproduction scripts
- `data/`: datasets or data-generation resources
- `results/`: generated numerical results
- `tests/`: test scripts
- `docs/`: additional documentation

## Tests

```matlab
run("tests/run_all_tests.m")
```

## Citation

If you use this code, please cite:

```bibtex
@misc{yang2026adrann,
  title         = {Adaptive-Distribution Randomized Neural Networks for PDEs: A Low-Dimensional Distribution-Learning Framework},
  author        = {Yang, You and Wang, Fei},
  year          = {2026},
  eprint        = {2604.23999},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2604.23999}
}
```

## License

This project is released under the BSD 3-Clause License. See `LICENSE` for details.
