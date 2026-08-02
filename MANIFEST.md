# Project manifest

## Preserved Gaussian implementation

The following current Gaussian files are copied unchanged:

- `examples/poisson_2d/evaluate_poisson_reduced_fast.m`
- `examples/poisson_2d/prepare_poisson_cache.m`
- `examples/poisson_2d/build_system.m`
- `examples/poisson_2d/frequency_initialization.m`
- `examples/poisson_2d/seed_example.m`
- `examples/poisson_2d/lambda_study/lambda_example.m`

The existing Gaussian feature, least-squares, metric, and sampling files are
retained as well.

## New reusable source modules

- `src/pde_driven/evaluate_pde_reduced_generic.m`
- `src/data_driven/*`
- `src/features/build_random_weights_nd.m`
- `src/features/build_preactivation.m`
- `src/features/evaluate_activation.m`
- `src/features/activation_derivatives.m`
- `src/features/activation_features.m`
- `src/features/feature_derivatives_2d.m`
- `src/features/tanh_features.m`
- `src/features/tanh_derivatives.m`
- `src/features/sin_features.m`
- `src/features/sin_derivatives.m`

## New Poisson activation support

- `examples/poisson_2d/build_system_activation.m`
- `examples/poisson_2d/evaluate_poisson_reduced_fast_gaussian.m`
- `examples/poisson_2d/evaluate_poisson_reduced_fast_tanh.m`
- `examples/poisson_2d/evaluate_poisson_reduced_fast_sin.m`
- `examples/poisson_2d/frequency_initialization_activation.m`

`config.m` adds `cfg.activation`, defaulting to `gaussian`.
`run.m` routes to the legacy Gaussian functions when that default is selected.
