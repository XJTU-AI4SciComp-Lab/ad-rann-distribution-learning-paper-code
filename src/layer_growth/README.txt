Generic layer-growth module for AD-RaNN
=======================================

Intended project location
-------------------------
    src/layer_growth/

Purpose
-------
This folder contains only PDE-independent layer-growth logic.

The base method can be PDAD or DDAD.  The layer-growth scale rho is trained
by DDAD in both cases.

Growth feature
--------------
For residual center c_j, random direction beta_j, and a frozen pre-growth
base solution u_0,

    psi_j(x;rho) = exp(-rho^2 S_j(x))

with

    S_j(x)
      = ||beta_j||^2 [u_0(x)-u_0(c_j)]^2
        + 1/2 sum_k beta_{k,j}^2 (x_k-c_{j,k})^2.

This is the compact form of the legacy two-stage Gaussian construction.
It avoids repeatedly rebuilding w2/b2 and repeatedly calling nested
gauss_dif2/G_dif routines.

Files
-----
build_growth_directions.m
    Reproducible U(-1,1) random directions.

select_growth_centers.m
    PDE-independent top-residual center selection.

prepare_growth_model.m
    Stores centers, random directions, and frozen center values.

prepare_growth_ddad_cache.m
    Precomputes S on solution-data points.

evaluate_growth_ddad_reduced_fast.m
    Ridge-reduced DDAD objective and analytic gradient with respect to rho.

optimize_growth_scale_ddad.m
    Reuses the common optimize_distribution_adam optimizer.

evaluate_growth_features.m
    Analytic feature values, first derivatives, and diagonal second
    derivatives for arbitrary input dimension.

fit_growth_block_ddad.m
    Small high-level wrapper used by examples.

Important design rule
---------------------
No PDE operator is hard-coded here.

A PDE-specific example should:
  1. compute a residual/indicator and choose centers;
  2. train rho with fit_growth_block_ddad;
  3. call evaluate_growth_features;
  4. combine phi/d1/d2 according to its own PDE operator.

This is what makes the same module reusable later for discrete-time
IMEX-BDF2 problems.
