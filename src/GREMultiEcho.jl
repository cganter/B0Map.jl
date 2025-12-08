using Compat
import VP4Optim as VP
@compat public AbstractGREMultiEcho, nTE, Δt, fat_fraction, max_derivative

"""
    AbstractGREMultiEcho{Ny,Nx,Nc,T} <: VP4Optim.Model{Ny,Nx,Nc,T}

Abstract supertype of multi-echo GRE sequences

# Type parameters
- `Ny::Int`: Number of acquired data == number(echoes) * number(coils)
- `Nx::Int`: Number of *variable* parameters `x ⊆ {ϕ, R2s, ...}` (relevant for optimization)
- `Nc::Int`: Number of linear coefficients (e.g. number of coil elements)
- `T::Union{Float64, ComplexF64}`: acquired data type
"""
abstract type AbstractGREMultiEcho{Ny,Nx,Nc,T} <: VP.Model{Ny,Nx,Nc,T} end

"""
    nTE(::AbstractGREMultiEcho)

Return number of recorded echoes
"""
function nTE(::AbstractGREMultiEcho) end

"""
    Δt(gre::T) where T <: AbstractGREMultiEcho

Return the echo spacing `Δt`.

## Remark

Defaults to the mean echo spacing for non-equidistant sampling.
"""
function Δt(gre::T) where T <: AbstractGREMultiEcho
    @assert hasfield(T, :Δt)
    gre.Δt
end

"""
    fat_fraction(::AbstractGREMultiEcho)

Calculate and return the fat fraction. (if applicable)

## Remark
- The precise implementation depends on the concrete subtype of `AbstractGREMultiEcho`.
- Generates an error if no such implementation exists.
"""
function fat_fraction(gre::AbstractGREMultiEcho)
    error("No implementation of fat_fraction() available for " * string(typeof(gre)))
end

"""
    max_derivative(::AbstractGREMultiEcho)

Return max. implemented derivative order.
"""
function max_derivative(::AbstractGREMultiEcho) end