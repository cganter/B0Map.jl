```@meta
CurrentModule = B0Map
```

# B0Map

*Tools for rapid ``B_0`` mapping in quantitative magnetic resonance imaging (qMRI).*

## Scope and Features

- Sequence: Multi-echo GRE
- Tissue: Water-fat mixture with spectral fat model
- Local and regularized field maps
- Support for multiple coil elements (with optional noise covariance)
- Maps: ``B_0``, ``R_2^\ast``, proton density fat fraction (PDFF), coil sensitivities
- Based upon the variable projection ([VARPRO](https://doi.org/10.1137/0710036)) package [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/).
- Acceleration: Multi-threading

## Installation

The package is not registered yet. Until then, just clone it and add it to your load path.

## Quick Start

```@contents
    Pages=[
        "man/fittools.md",
        "man/localfit.md",
        "man/phaser.md",
        "man/results.md",
        ]
    Depth=1
```

## Signal models

```@contents
    Pages=[
        "man/GREgeneral.md",
        "man/WFFW.md",
        "man/WFRW.md",
        "man/WF.md",
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
