```@meta
CurrentModule = B0Map
```

# General

## Global conventions

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

!!! note
    Some informations about the acquired MRI data are not restricted to specific models:

    - `B0::Real`: Magnetic field strength of the scanner. \[T\]
    - `precession::Symbol`: Specifies, which convention about the orientation of precession is assumed in the data. Assuming positive frequencies, such that ``\omega_{water} > \omega_{fat}`` (main fat peak)
        * `precession == :counterclockwise` means a phase evolution ``\propto e^{i\omega t}``.
        * `precession == :clockwise` means a phase evolution ``\propto e^{-\,i\omega t}``.
    - Chemical shift [ppm] is defined with the following sign convention: With the water peak at 0 ppm, the chemical shift of the main fat peak shall be negative and approximately located at - 3.4 ppm.

## Abstract multi-echo GRE type

All (concrete) multi-echo GRE types must be subtypes of

```@docs
AbstractGREMultiEcho
make(::Type{<: GREMultiEchoWF}, ::Any) 
```

## Generic Routines

```@docs
Δt
fat_fraction(::AbstractGREMultiEcho)
```
