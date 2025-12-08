```@meta
CurrentModule = B0Map
```

# Local fit

Local ``B_0`` fitting can be done in two ways:

- Splitting the interval ``[-\pi, \pi]`` into `n_ϕ` subintervals of equal size and fitting them separately. The final maximum likelihood (ML) estimator is defined by the minimal ``\chi^2``. Use [`set_num_phase_intervals`](@ref set_num_phase_intervals) to define the `n_ϕ` intervals in `fitopt`.
- Starting from some initial guess `ϕ`, search for the closest local minimum of ``\chi^2`` with a nonlinear optimizer.

## Method

Local data fitting is accomplished by the routine

```@docs
local_fit!
```

## Golden section search (GSS)

An auxiliary golden section search (GSS) implementation is provided in  

```@docs
GSS
```

