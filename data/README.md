# Data

Large datasets are **not included in this GitHub repository**. Place the required files in the locations below before running the corresponding examples.

## Diffusion-reaction operator-learning data

Used by:

```matlab
run("examples/diffusion_reaction_deeponet/run_DDAD_RaNN_DeepONet.m")
run("examples/diffusion_reaction_deeponet/run_DDAD_PI_RaNN_DeepONet.m")
```

Required files:

```text
data/Diffusion10000.mat
data/Diffusion_test1000.mat
```

Each MAT file must contain:

- `f`: source functions sampled at 100 branch sensors;
- `y`: query coordinates with columns `[x,t]`;
- `u`: the corresponding PDE solution values.

The dataset corresponds to the nonlinear diffusion-reaction equation

```text
u_t - 0.01 u_xx - 0.01 u^2 = f(x),  (x,t) in (0,1) x (0,1],
```

with homogeneous initial and boundary conditions. Source functions are sampled from a zero-mean Gaussian random field with squared-exponential covariance and length scale `l = 0.2`. The training set contains 10,000 source-function realizations; the test set contains another 1,000 realizations evaluated on a `100 x 100` grid.

This dataset follows the data-generation setup of the PI-DeepONet paper. The original implementation is available at:

https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets

Generate or obtain the data using that implementation, convert/save the arrays as MATLAB v7.3 MAT files with variables `f`, `y`, and `u`, and use the filenames shown above. MATLAB v7.3 is recommended because the local loader uses partial `matfile` indexing for these large files.

## Burgers operator-learning data

Used by:

```matlab
run("examples/burgers_deeponet/run_RaNN_DeepONet.m")
```

Required layout:

```text
data/burgers_deeponet/
  ic.mat
  bc.mat
  ir.mat
  Burger_t.mat
```

Expected variables:

- `ic.mat`: `f`, `y`, and `u` for initial-condition points;
- `bc.mat`: `y` and `u` for boundary points;
- `ir.mat`: `y` and `u` for interior points;
- `Burger_t.mat`: `output`, containing the solution fields used for testing.

The data correspond to the one-dimensional viscous Burgers equation on `t in (0,1)` with viscosity `nu = 0.01` and periodic boundary conditions. Initial conditions are sampled from the periodic Gaussian random field

```text
GRF(0, 625(-Delta + 25 I)^(-4)).
```

Following the RaNN-DeepONet paper, generate 1,000 training realizations and 100 additional test realizations on a `101 x 101` uniform space-time grid. Each initial condition is represented at 101 branch sensors. The paper uses the Burgers data-generation procedure provided by Lu et al. for DeepONet/PI-DeepONet experiments.

After generation, split and save the initial, boundary, interior, and full solution arrays using the filenames and variable names listed above.

## Other PDE reference data

The smaller files `AC_new.mat`, `AC_2D_Dirichlet_spectral_10snap.mat`, and `burgers.mat` are reference solutions for the discrete-time PDE examples. If these files are also excluded from the repository, document their numerical solver, grid, time step, and expected variables before public release.


Only this README and any small data-generation scripts should be tracked by Git.
