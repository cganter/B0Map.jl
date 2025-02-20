```@meta
CurrentModule = B0Map
```

# B0Map

*Tools for rapid B0-mapping in quantitative magnetic resonance imaging (qMRI).*

## Purpose

Mapping the main magnetic field strength B0 is an important and recurrent
task in quantitative magnetic resonance imaging (qMRI). 
This package provides some tools to analyze related multi-echo data from 
(RF-)spoiled gradient echo (GRE) sequences.

## Features

- Multi-echo GRE signal models for water-fat mixtures.
- Optional support for multiple coils and coil noise covariance.
- Local and regularized data fitting (B0-map, fat fraction, coil sensitivities)
- Based upon variable projection ([VARPRO](https://doi.org/10.1137/0710036)) (package: [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/))
- Support for multi-threaded execution.

## Installation

The package is not registered yet. Until then, just clone it and add it to your load path.

## Multi-echo GRE signal models

```@contents
    Pages=[
        "man/GREgeneral.md",
        "man/WFFW.md",
        "man/WFRW.md",
        "man/WF.md",
        ]
    Depth=1
```

!!! note
    For actual applications, the [Constrained water-fat mixture](@ref) model should be used.

## Data fitting

```@contents
    Pages=[
        "man/fittools.md",
        "man/localfit.md",
        "man/phaser.md",
        ]
    Depth=1
```

## API

```@contents
    Pages=[
        "man/api.md",
        ]
    Depth=1
```
