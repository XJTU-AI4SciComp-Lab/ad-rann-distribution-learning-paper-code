# AD-RaNN-Distribution-Learning-Paper-Code

[![DOI](https://zenodo.org/badge/1319973836.svg)](https://doi.org/10.5281/zenodo.21759015)
[![Paper](https://img.shields.io/badge/arXiv-2604.23999-b31b1b.svg)](https://arxiv.org/abs/2604.23999)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

Official MATLAB implementation of **Adaptive-Distribution Randomized Neural Networks (AD-RaNN)** for PDE solving and operator learning.

This repository contains the MATLAB implementation and numerical experiments for the paper:

**Adaptive-Distribution Randomized Neural Networks for PDEs: A Low-Dimensional Distribution-Learning Framework**

by You Yang and Fei Wang.

## Paper Information

- **Title:** Adaptive-Distribution Randomized Neural Networks for PDEs: A Low-Dimensional Distribution-Learning Framework
- **Authors:** You Yang and Fei Wang
- **Status:** arXiv preprint
- **Paper:** [arXiv:2604.23999](https://arxiv.org/abs/2604.23999)
- **Corresponding author:** Fei Wang (`feiwang.xjtu@xjtu.edu.cn`)
- **Code maintainer:** You Yang
- **Repository:** [XJTU-AI4SciComp-Lab/ad-rann-distribution-learning-paper-code](https://github.com/XJTU-AI4SciComp-Lab/ad-rann-distribution-learning-paper-code)

## Method Summary

AD-RaNN adapts a low-dimensional parameterization of the hidden-feature sampling distribution while retaining the efficient linear least-squares structure of randomized neural networks. The repository includes PDE-Driven Adaptive Distribution (PDAD), Data-Driven Adaptive Distribution (DDAD), discrete-time solvers, layer-growth experiments, and operator-learning models.

## Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- MANIFEST.md
|-- requirements.txt
|-- src/                  Reusable MATLAB source code
|-- examples/             Minimal single-seed Poisson example
|-- experiments/          Full numerical and reproduction experiments
|-- data/                 Data documentation and local reference data
|-- results/              Generated numerical outputs
|-- docs/                 Additional documentation
`-- tests/                Repository tests
```

## Requirements
MATLAB requirements

Required:
- MATLAB R2024b or later

Optional:
- Parallel Computing Toolbox for GPU-enabled experiments

## Quick Start

Open MATLAB in the repository root and run the lightweight single-seed Poisson example:

```matlab
run("examples/poisson_2d/run.m")
```

The corresponding configuration is:

```text
examples/poisson_2d/config.m
```

Generated numerical outputs are written under `results/`.

## Reproducing Paper Results

Full studies are organized under `experiments/`. Representative entry points include:

```matlab
% Poisson seed study
run("experiments/poisson_2d/run_seed_study.m")

% Activation comparison
run("experiments/activation_experiment/run_activation_experiment.m")

% DDAD with three initializations
run("experiments/DDAD_poisson_three_initializations/run_DDAD_three_initializations.m")

% Nonlinear sharp-layer experiments
run("experiments/nonlinear_sharp_layer/PDAD/run_PDAD_sharp_layer.m")
run("experiments/nonlinear_sharp_layer/DDAD/run_DDAD_sharp_layer.m")

% Allen-Cahn and Burgers discrete-time experiments
run("experiments/allen_cahn_1d_dt/run_PDAD_DT_AC.m")
run("experiments/burgers_1d_dt/run_PDAD_DT_burgers.m")

% Operator-learning experiments
run("experiments/diffusion_reaction_deeponet/run_DDAD_RaNN_DeepONet.m")
run("experiments/burgers_deeponet/run_RaNN_DeepONet.m")
```

Additional Helmholtz, Black-Scholes, L-shaped Poisson, two-dimensional time-dependent PDE, learning-rate, feature-number, and differential-evolution studies are also provided under `experiments/`.

Large multi-seed, high-dimensional, time-dependent, and operator-learning experiments may require substantial runtime, memory, or external datasets.

## Reproducibility

Random seeds and experiment parameters are specified in the corresponding MATLAB configuration or experiment script. Run scripts from the repository root and keep the same MATLAB version, toolbox set, input data, and random seed when comparing results.

> **Important:** The paper reports multiple experiments with different settings and random trials. Fully reproducing a specific table or figure may require adjusting the corresponding `config.m` or `config_*.m` file. Relevant settings can include the random seed, number of features, learning rate, regularization parameter, iteration count, data path, activation function, and GPU configuration.

Large datasets are intentionally excluded from Git. Their expected filenames, variables, generation procedures, and source references are documented in [`data/README.md`](data/README.md).

## Tests

Run the basic test suite:

```matlab
run("tests/run_all_tests.m")
```

Run the extended activation and dimension tests:

```matlab
run("tests/run_extended_tests.m")
```

## Citation

If you use this code, please cite the paper:

```bibtex
@marticle{yang2026adrann,
  title         = {Adaptive-Distribution Randomized Neural Networks for PDEs: A Low-Dimensional Distribution-Learning Framework},
  author        = {Yang, You and Wang, Fei},
  year          = {2026},
  eprint        = {2604.23999},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  url           = {https://arxiv.org/abs/2604.23999}
}
```

The machine-readable citation metadata are available in [`CITATION.cff`](CITATION.cff).

## License

This repository is released under the BSD 3-Clause License. See [`LICENSE`](LICENSE).
## Release

Version 1.0.3 contains minor documentation updates.
