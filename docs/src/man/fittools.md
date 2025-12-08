```@meta
CurrentModule = B0Map
```
# Set up parameters

## Sequence and tissue

Any available information about the acquisition and tissue, including the data and allocated 
arrays for the fit results is stored in a single data structure

```@docs
FitPar
```

To create an instance, we just call a constructor

```@docs
fitPar
```

## Fit options

All options, affecting the fit routines provided in `B0Map`, are collected in the structure

```@docs
FitOpt
```

To set them up, we generate an instance with default values, using the constructor

```@docs
fitOpt
```

!!! note
    To change the default phase search intervals, do not directly change the field `fitopt.n_ϕ`, but call
    the following routine:

```@docs
set_num_phase_intervals
```