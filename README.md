# B0Map

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cganter.github.io/B0Map.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cganter.github.io/B0Map.jl/dev/)
[![Build Status](https://github.com/cganter/B0Map.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cganter/B0Map.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/cganter/B0Map.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cganter/B0Map.jl)

## Scope

Fitting data $\mathbf{y}$ from quantitative magnetic resonance imaging (qMRI) 
locally to some model $\mathbf{A}$ typically leads to a least squares (LS) problem of the form
```math
\chi^2\left(\mathbf{x}, \mathbf{c}\right) \;=\;
\left\|\,\mathbf{y} - \mathbf{A}(\mathbf{x}) \cdot \mathbf{c}\,\right\|^2_2
```
to be minimized with respect to $\mathbf{x}$ and $\mathbf{c}$
```math
\hat{\mathbf{x}}, \hat{\mathbf{c}} \;=\;
\underset{\mathbf{x}, \mathbf{c}}{\mathop{\text{argmin}}}\;
\chi^2\left(\mathbf{x}, \mathbf{c}\right)
```
The matrix $\mathbf{A}\left(\mathbf{x}\right)$ depends on the details of the MR scan and 
the underlying tissue model. Besides other factors, the linear coefficients $\mathbf{c}$ are
proportional to the local receive coil sensitivities.
[VARPRO](https://doi.org/10.1137/0710036) allows to reduce the dimensionality of this optimization problem,
by eliminating the dependence on the linear coefficients $\mathbf{c}$.

Based upon [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/), the 
[B0Map](https://github.com/cganter/B0Map.jl) package provides a collection
of specific model implementations with relevance to quantitative MRI.

## Installation

The package is not registered yet. Until then, just clone it and add it to your load path.

## Contributing

Feedback and bug reports, which adhere to the 
[Julia Community Standards](https://julialang.org/community/standards/), are welcome.
README.md (END)

