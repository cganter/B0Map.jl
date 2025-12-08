```@meta
CurrentModule = B0Map
```

# Unconstrained water-fat mixture

## Description

We consider a set of `Ny` complex local signals ``y_j`` acquired at 
different echo times ``t_j`` in an RF-spoiled multi-echo GRE sequence. A water-fat signal model is assumed in the usual manner
```math
  \hat{y}_{j}
  \; := \;
  \bigl(c_{w} + c_{f}\,\beta_j\bigr)\,e_j
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
model are therefore given by ``c_w,c_f \in \mathbb{C}`` and 
``\omega, R_2^\ast\in \mathbb{R}``.

Writing the signal model in vector notation ``\bm{y}=\bm{A}\bm{c}``,
leads to the following definition of the 
`Ny` ``\times`` `Nc` matrix ``\bm{A}``
```math
A_{j1} \;=\; e_j \qquad\text{and}\qquad A_{j2} \;:=\; \beta_j\, e_j
```
together with the linear coefficient vector
``\bm{c} = \left[\,c_w\,c_f\,\right]^T``.

In terms of variable projection, specifically the notation
``\bm{A}\left(\bm{x}\right)`` introduced in
[VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/), 
the elements in the variable vector ``\bm{x}`` are given by a 
(sub)set ``\subseteq \left\{\omega, R_2^\ast\right\}``.

!!! note
    In VARPRO, the coefficients ``c_w`` and ``c_f`` are calculated in closed form (cf. the 
    intro section in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)). This means that
    there is not way to enforce them to have equal phase. But since all ``\alpha_k`` are assumed to be
    real and positive, this would be a consistency requirement.
    For alternative models, see
    - [Real-valued water-fat mixture](@ref): Aligns water and fat, but allows for opposite direction
    - [Constrained water-fat mixture](@ref): Full implementation of the constraint

## Type

```@docs
GREMultiEchoWFFW
```

## Constructor parameters

The parameters for the constructor are defined in
```@docs
ModParWFFW
```
It can be set up with [VP4Optim.modpar](https://cganter.github.io/VP4Optim.jl/stable/man/guide/#VP4Optim.modpar).

## Constructor

```@docs
GREMultiEchoWFFW(::ModParWFFW)
```

## Support for [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)

- All methods are fully implemented.
- Partial derivatives up to second order are provided.

## Specific API

!!! note
    The fat fraction is defined as
    ```math
    \frac{\left|c_{f}\right|}{\left|c_w\right| + \left|c_f\right|}
    ```
    and returned by [`fat_fraction`](@ref fat_fraction).