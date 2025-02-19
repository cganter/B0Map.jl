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

## Global conventions

Some informations about the acquired MRI data are not restricted to specific models:

- `B0::Real`: Magnetic field strength of the scanner. \[T\]
- `precession::Symbol`: Specifies, which convention about the orientation of precession is assumed in the data. Assuming positive frequencies, such that ``\omega_{water} > \omega_{fat}``
    * `precession == :counterclockwise` means a phase evolution ``\propto e^{i\omega t}``.
    * `precession == :clockwise` means a phase evolution ``\propto e^{-\,i\omega t}``.
- Chemical shift [ppm] is defined with the following sign convention: With the water peak at 0 ppm, the chemical shift of the main fat peak shall be negative and approximately located at - 3.4 ppm.

## Multi-echo GRE signal models

!!! note
    In case of equidistant echos, separated by a constant ``\Delta\text{TE}``, the local off-resonance 
    frequency ``\omega`` cannot be uniquely determined from the data, since adding any integer multiple
    of ``2\pi/\Delta\text{TE}`` to ``\omega`` does not affect the fit (apart from an unimportant 
    global phase).

    The actual implementations of the signal models below actually do no rely on ``\omega``, but rather
    work with the phase ``\varphi = \omega\cdot\Delta\text{TE}``. This makes no difference, apart from that
    this now restricts the unambiguous phase information to some ``2\pi`` interval, say, ``(-\pi, \pi]``.
    
    For non-equidistant echo times, the periodicity of frequency/phase is no longer valid.
    In this case, the phase definition reads ``\varphi = \omega\cdot\Delta t``, where by default
    ``\Delta t`` is taken as the average echo spacing. For cases, when this choice is not satisfactory,
    all models allow to overrule this setting by explicitly providing ``\Delta t`` in the constructor.

```@contents
    Pages=[
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
