```@meta
CurrentModule = B0Map
```

# Constrained water-fat mixture

## Description

We consider a set of `Ny = Nt * Nc` complex local RF-spoiled multi-echo GRE signals ``y_{j\gamma}`` 
acquired at `Nt` different echo times ``t_j`` in `Nc` (``\ge 1``) coil elements (indexed by ``\gamma``). 
Assuming a water-fat mixture, the signal model is set up as follows
```math
  \hat{y}_{j\gamma}
  \; := \;
  c_\gamma 
  \left[\,
  1 + f\,\left(\beta_j - 1\right)\,\right]\,e_j
  \qquad\qquad
  e_{j}
  \; := \;
  e^{i\,\omega\,t_{j}}\,e^{-R_2^{\ast}\,t_{j}}
```
with ``0\le f \le 1`` and the multi-peak fat model given by
```math
  \beta_{j} \; := \;\sum_{k=1}^{m}\,\alpha_{k}\,e^{i\,\omega_{k}t_{j}}
  \qquad\qquad
  \sum_{k=1}^m\,\alpha_k \;= \;1
```
The real-valued amplitudes ``\alpha_k>0`` and chemical shifts 
``\omega_k`` are fixed and assumed to be known. 

!!! note
    This model guarantees the water peak and all fat peaks to have equal phase at ``t=0``.
    One can recover the [Unconstrained water-fat mixture](@ref) model via
    ```math
    c_{w\gamma} = c_\gamma\left(1-f\right)
    \qquad\qquad
    c_{f\gamma} = c_\gamma\,f
    ```
    After regrouping these equations
    ```math
    c_{\gamma} = c_{w\gamma} + c_{f\gamma}
    \qquad\qquad
    f = \frac{\left|c_{f\gamma}\right|}{\left|c_{f\gamma}\right| + \left|c_{w\gamma}\right|}
    ```
    it becomes evident that ``f`` is just the relative fat fraction and ``c_\gamma`` an in-phase 
    signal, weighted by the coil sensitivity.
    
Due to finite coil overlap, data noise ``\eta_{j\gamma} = y_{j\gamma} - \hat{y}_{j\gamma}`` can be correlated, 
which is then expressed by a Hermitian coil noise covariance matrix ``\bm{\Psi}``
```math
    \left\langle \eta_{j,\gamma}\, \eta^\ast_{j^\prime\gamma^\prime}\right\rangle =:
    \Psi_{\gamma\gamma^\prime}\cdot \delta_{jj^\prime}
    = \sum_k\,\sigma^2_k\cdot U_{\gamma k}U^*_{\gamma^\prime k}
```
Based upon the eigendecomposition in the last equation, the transformations
``c_\gamma \to c_k`` and ``y_{j\gamma} \to y_{jk}``
```math
    c_k := \sum_\gamma\,\frac{1}{\sigma_k}\,U^*_{\gamma k}\,c_\gamma
    \qquad\qquad
    y_{jk} := \sum_\gamma\,\frac{1}{\sigma_k}\,U^*_{\gamma k}\,y_{j\gamma}
```
allow to write the multi-coil ML cost function ``\chi^2``
in the conventional form as expected by VARPRO. (See [link to follow] for more details.)

In summary, the variable parameters of this
model are therefore given by ``c_k \in \mathbb{C}`` and 
``\omega, R_2^\ast, f\in \mathbb{R}``.

Writing the signal model in vector notation ``\bm{y}=\bm{A}\bm{c}``,
leads to the following definition of the 
`Ny` ``\times`` `Nc` matrix ``\bm{A}``
```math
  A_{jk} \equiv \left[\,
  1 + f\,\left(\beta_j - 1\right)\,\right]\,e_j
  \qquad\qquad
  \forall \,k
```
and the linear coefficient vector ``[\bm{c}]_k= c_k``.

!!! note
    Note that the incorporating ``f`` into ``\bm{A}`` increases the complexity of the VARPRO 
    cost function ``\chi^2``. As shown in [link to follow], it is possible to determine 
    ``f=f\left(\omega,R_2^\ast\right)`` efficiently and thereby remove its explicit appearance
    in ``\chi^2``. This correponds to the default setting `mode = :autofat` in the constructor
    below.

In terms of variable projection, specifically the notation
``\bm{A}\left(\bm{x}\right)`` introduced in
[VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/), 
the elements in the variable vector ``\bm{x}`` are therefore given by a 
(sub)set ``\subseteq \left\{\omega, R_2^\ast\right\}``.

## Type

```@docs
GREMultiEchoWF
```
## Constructor parameters

The parameters for the constructor are defined in
```@docs
ModParWF
```
It can be set up with [VP4Optim.modpar](https://cganter.github.io/VP4Optim.jl/stable/man/guide/#VP4Optim.modpar).

## Constructor

```@docs
GREMultiEchoWF(::ModParWF)
```

## Support for [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)

- All methods are fully implemented.
- Partial derivatives up to second order are provided.

## Specific API

!!! note
    - For ``\Psi \ne \mathrm{I}``, data must be (re)set with `VP.set_data!(gre, data)` instead of `VP.y!(gre, data)`. 
    - The `data` argument must be ordered such that the expression `SMatrix{Nt,Nc}(data)` produces the correct data matrix ``y_{j\gamma}``.

!!! note
    The fat fraction is simply given by ``f(\omega, R_2^\ast)``.

```@docs
coil_sensitivities
```