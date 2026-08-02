# Project manifest

## Minimal example

The `examples/` directory intentionally contains only the minimal single-seed
2-D Poisson example:

- `examples/poisson_2d/config.m`
- `examples/poisson_2d/run.m`

## Numerical experiments

All full studies, parameter sweeps, ablation studies, multi-seed runs, and
operator-learning benchmarks are organized under `experiments/`.

Important experiment groups include:

- `experiments/poisson_2d/`
- `experiments/activation_experiment/`
- `experiments/DDAD_poisson_three_initializations/`
- `experiments/nonlinear_sharp_layer/`
- `experiments/allen_cahn_1d_dt/` and `allen_cahn_2d_dt/`
- `experiments/burgers_1d_dt/`, `burgers_2d_dt/`, and `burgers_deeponet/`
- `experiments/helmholtz_2d/`
- `experiments/poisson_lshape_2d/`
- `experiments/black_scholes_highdim/`
- `experiments/diffusion_reaction_deeponet/`

## Reusable implementation

Reusable algorithms, activation functions, least-squares routines, metrics,
sampling utilities, and problem-dependent implementations are stored under
`src/`. Generated numerical outputs belong under `results/`, and generated
figures should be stored in the corresponding result directory or `figures/`.
