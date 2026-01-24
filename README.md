# B0Map

*A Julia package for rapid B0-mapping in MRI.*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cganter.github.io/B0Map.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cganter.github.io/B0Map.jl/dev/)
[![Build Status](https://github.com/cganter/B0Map.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cganter/B0Map.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/cganter/B0Map.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cganter/B0Map.jl)

## Purpose

Mapping the main magnetic field strength B0 is an important and recurrent
task in quantitative magnetic resonance imaging (qMRI). 

This package provides some tools to analyze related multi-echo data from 
(RF-)spoiled gradient echo (GRE) sequences.

See the [documentation](https://cganter.github.io/B0Map.jl/stable/) for more details.

## Installation

The package can be installed with the Julia package manager.

In the Julia REPL, either issue the commands

```julia
julia> import Pkg; Pkg.add("B0Map")
```
or type `]` to enter the Pkg REPL mode and run

```
pkg> add B0Map
```

## Contributing

Feedback and bug reports, which adhere to the 
[Julia Community Standards](https://julialang.org/community/standards/), are welcome.

