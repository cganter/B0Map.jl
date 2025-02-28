```@meta
CurrentModule = B0Map
```
# General steps

Fitting a set of acquired complex multi-echo GRE images, typically involves the following steps:

- Storing the data in a suitable array.
- Defining a spatial ROI mask.
- Selecting an `AbstractGREMultiEcho` model and initialize it, based upon the measurement contitions (sequence parameters, B0) and the chosen tissue model.
- Define options, how the data should be fitted.
- Call the fitting routine.
- Extract results.

## Define model, data and fit parameters

Any available information about the acquisition and tissue, including the data and allocated 
arrays for the fit results is stored in a single data structure

```@docs
FitPar
```

To create an instance, we just call a constructor

```@docs
fitPar
```

just as in the following example

```@julia
import VP4Optim as VP
import B0Map as BM

# Define acquisition parameters
nTE = 6                             # number of echoes
t0 = 0.5                            # first echo time [ms]
ΔTE = 1.0                           # echo spacing [ms]
TEs = collect(range(t0, t0 + (nTE-1) * ΔTE, nTE))  # resulting echo times
B0 = 3.0                            # scanner field strength [T]
precession = :counterclockwise      # orientation of precession (depends on the scanner)
n_coils = 6                         # number of coils (if reconstructed separately)
cov_mat = get_cov_mat(...)          # coil covariance matrix (if available)

# Define water-fat tissue model (we only need to specify the fat component)
ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

# set up model constructor parameters
pars = VP.modpar(BM.ModParWF;
    ts = TEs,
    B0 = B0,
    ppm_fat = ppm_fat,
    ampl_fat = ampl_fat,
    precession = precession,
    n_coils = n_coils,
    cov_mat = cov_mat)
    
# Create an instance of AbstractGREMultiEcho (here we use the recommended constrained model)
gre = BM.greMultiEchoWF(pars)

# Prepare data and ROI
# (For the required dimensions, see the docs of the constructor fitPar().)
data = get_from_somewhere(...)
S = define_ROI_somehow(data, ...)

# create an instance of FitPar
fitpar = BM.fitPar(gre, data, S) 
```

## Specify fit options

All options, affecting the fit routines provided in `B0Map`, are collected in the structure

```@docs
FitOpt
```

To set them up, we first generate a default instance with the constructor

```@docs
fitOpt
```

and (optionally) modify the default settings like this

```@julia
# all steps from the example above
...

# generate FitOpt instance with default options
fitopt = BM.fitOpt(gre)

# modify whatever we like to modify
BM.set_num_phase_intervals(fitpar, fitopt, 4)
fitopt.ϕ_acc = 1.e-6
...
```

!!! note
    To change the default phase search intervals, we should use 
    [`set_num_phase_intervals`](@ref set_num_phase_intervals), like in the example.

```@docs
set_num_phase_intervals
phase_search_intervals
```

## Fit the data

See [Local Fitting](@ref) and [PHASER](@ref) for how this can be done.

## Extract information

!!! note
    After fitting the data, the estimators returned in the supplied structure `fitpar::FitPar`
    - `fitpar.ϕ`: phase ``\phi := \omega \cdot \Delta t``
    - `fitpar.R2s`: relaxation rate ``R_2^\ast``
    - `fitpar.c`: linear VARPRO coefficients ``\bm{c}`` (note: `eltype(c) <: AbstractVector`)
    - `fitpar.χ2`: least-squares residual ``\chi^2``

Based upon these estimators, further model-dependent information can be calculated with the routine

```@docs
calc_par
```

The following example shows how this looks like for the fat fraction

```@julia
# fitting has been done and the results are stored in fitpar
...

# The fat fraction can now be extracted as follows
ff = zeros(size(fitpar.S))                          # allocate space for the results
BM.calc_par(fitpar, fitopt, BM.fat_fraction, ff)    # do the job
```

For convenience (and partly efficiency) some more direct calls are provided as well:

```@docs
fat_fraction_map
freq_map
```