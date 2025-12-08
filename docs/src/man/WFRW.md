```@meta
CurrentModule = B0Map
```

# Real-valued water-fat mixture

## Description

We consider a set of `Ny` complex local signals ``y_j`` acquired at 
different echo times ``t_j`` in an RF-spoiled multi-echo GRE sequence. A water-fat signal model is assumed in the usual manner
```math
  \hat{y}_{j}
  \; := \;
  e^{i\,\theta}\,\bigl(r_{w} + r_{f}\,\beta_j\bigr)\,e_j
  \qquad\qquad
  e_{j}
  \; := \;
  e^{i\,\omega\,t_{j}}\,e^{-R_2^{\ast}\,t_{j}}
```
with the multi-peak fat model given by
```math
  \beta_{j} \; := \;\sum_{k=1}^{m}\,\alpha_{k}\,e^{i\,\omega_{k}t_{j}}
  \qquad\qquad
  \sum_{k=1}^m\,\alpha_k \;= \;1
```
The real-valued amplitudes ``\alpha_k>0`` and chemical shifts 
``\omega_k`` are fixed and assumed to be known. 

The variable parameters of this
model are therefore given by ``r_w,r_f \in \mathbb{R}`` and 
``\omega, R_2^\ast, \theta\in \mathbb{R}``.

Writing the signal model in vector notation ``\bm{y}=\bm{A}\bm{r}``,
leads to the following definition of the 
`Ny` ``\times`` `Nc` matrix ``\bm{A}``
```math
A_{j1} \;=\; e^{i\,\theta}\,e_j \qquad\text{and}\qquad A_{j2} \;:=\; e^{i\,\theta}\,\beta_j\,e_j
```
together with the real-valued linear coefficient vector
``\bm{r} = \left[\,r_w\,r_f\,\right]^T``.

!!! note
    Note that the incorporating ``\theta`` into ``\bm{A}`` increases the complexity of the VARPRO 
    cost function ``\chi^2``. As shown by [Bydder et al.](https://doi.org/10.1016/j.mri.2010.08.011), 
    it is possible to determine ``\theta=\theta\left(\omega,R_2^\ast\right)`` efficiently and thereby remove its
    explicit appearance in ``\chi^2``. The model [GREMultiEchoWFRW](@ref GREMultiEchoWFRW) already takes
    care of this and no user action is needed.
    
In terms of variable projection, specifically the notation
``\bm{A}\left(\bm{x}\right)`` introduced in
[VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/), 
the elements in the variable vector ``\bm{x}`` are therefore given by a 
(sub)set ``\subseteq \left\{\omega, R_2^\ast\right\}``.

!!! note
    Different from [Unconstrained water-fat mixture](@ref), this model does not allow water and fat
    to point in arbitrary directions at ``t=0``, but aligns them. This significantly improves stability 
    in case of noisy data and/or a small number of echo times. While forced to be parallel, water and fat
    still may point in opposed directions though, which can still cause stability issues for challenging data.

    An alternative fully constrained model has been developed in [link to follow] and is described in
    [Constrained water-fat mixture](@ref).

## Type

```@docs
GREMultiEchoWFRW
```

## Constructor parameters

The parameters for the constructor are defined in
```@docs
ModParWFRW
```
It can be set up with [VP4Optim.modpar](https://cganter.github.io/VP4Optim.jl/stable/man/guide/#VP4Optim.modpar).

## Constructor

```@docs
GREMultiEchoWFRW(::ModParWFRW)
```

## Support for [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)

- All methods are fully implemented with one exception:
- No partial derivatives are implemented.

## Specific API

!!! note
    The fat fraction is defined as
    ```math
    \frac{\left|r_{f}\right|}{\left|r_w\right| + \left|r_f\right|}
    ```
    and returned by [`fat_fraction`](@ref fat_fraction).
    
```@docs
coil_phase
```