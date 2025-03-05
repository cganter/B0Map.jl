```@meta
CurrentModule = B0Map
```

# PHASER

## Smooth basis

```@docs
BSmooth{N}
BFourierLin
fourier_lin
Nρ
Nρ_orig
Nκ
phase_map(bs::BSmooth, b::Float64, c::AbstractVector)
phase_map(::BSmooth, ::AbstractVector)
```

## Regularized phase map

```@docs
calc!
phaser
smooth_projection!
```