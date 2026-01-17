```@meta
CurrentModule = B0Map
```
# Get results

## Fit estimators

!!! note
    Estimators, calculated by [`local_fit!`](@ref local_fit!) 
    are stored in the supplied structure of type [`FitPar`](@ref FitPar).

## Derived quantities

Based upon these estimators, further model-dependent information can be calculated with the routine

```@docs
calc_par
```

## Predefined maps
For convenience (and partly efficiency) some more direct calls are provided as well:

```@docs
fat_fraction_map
freq_map
```