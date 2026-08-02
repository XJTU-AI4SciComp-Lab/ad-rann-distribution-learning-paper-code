# Figures

This directory stores figures generated from numerical data in `results/`.

The recommended workflow is:

1. Run an experiment under `experiments/`.
2. Load its saved `.mat` or `.csv` file from `results/`.
3. Run the corresponding plotting script.
4. Save the generated figure under `figures/generated/<experiment>/`.

For example, the nonlinear sharp-layer plotting workflow uses the result files under:

```text
results/nonlinear_sharp_layer/PDAD/
results/nonlinear_sharp_layer/DDAD/
```

and the plotting routines associated with:

```text
experiments/nonlinear_sharp_layer/
```

Generated figures are not committed by default. Selected final paper figures may be added manually when needed.
