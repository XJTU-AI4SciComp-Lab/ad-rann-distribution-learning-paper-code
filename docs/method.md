# AD-RaNN Method

AD-RaNN learns a low-dimensional sampling distribution for randomized hidden features while retaining a linear least-squares solve for the output coefficients.

## Random Weights and Distribution Parameterization

The hidden weights are generated from fixed reference random variables and scaled by a low-dimensional parameter vector $\boldsymbol p$, allowing the randomized feature distribution to adapt to different spatial or temporal directions.

## Two-Stage Training Strategy

AD-RaNN first optimizes $\boldsymbol p$ through a ridge-regularized reduced least-squares problem for numerical stability and then recomputes the final output coefficients by unregularized least squares.

## PDAD and DDAD

PDAD learns the feature distribution directly from the discretized PDE residual system, whereas DDAD learns it from available numerical solution data before performing the final PDE solve.

## Reduced Gradient

The gradient of the reduced objective is computed without differentiating through the least-squares solution, using the residual, the derivative of the system matrix, and the current output coefficients.

## Space-Time and Discrete-Time Frameworks

AD-RaNN can be used either in a unified space-time formulation or in a discrete-time scheme where the distribution parameters are updated adaptively during time stepping.

## Reduced Training and Full Refit

Distribution adaptation may use fewer randomized features and training points to reduce cost, after which the learned distribution is transferred to the full model for the final unregularized refit.

## Localized Layer Growth

When the remaining error is strongly localized, residual-driven layer growth adds adaptive local basis functions around large-residual points to improve the resolution of sharp structures.

## AD-RaNN-DeepONet

AD-RaNN-DeepONet applies the same distribution-learning principle to randomized branch and trunk features for efficient operator learning.

## Theory, Advantages, and Limitations

The reduced ridge problem is well posed and consistent with the unregularized problem under suitable assumptions, while the method preserves low-cost least-squares training but still involves a nonconvex outer optimization and does not by itself provide a complete PDE-level convergence theory.