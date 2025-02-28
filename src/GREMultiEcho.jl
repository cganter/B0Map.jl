import VP4Optim as VP

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
    make(GRE::Type{<: AbstractGREMultiEcho}, args; x_sym=nothing, Δt=nothing)

Generate a `GRE` instance.

## Arguments
- `args::Tuple`: Arguments of `GRE` constuctor without keyword arguments
"""
function make(GRE::Type{<: AbstractGREMultiEcho}, args; x_sym=nothing, Δt=nothing)
    make(GRE, args..., x_sym=x_sym, Δt=Δt)
end

"""
    Δt(gre::T) where T <: AbstractGREMultiEcho

Return the effective echo spacing `Δt`.
"""
function Δt(gre::T) where T <: AbstractGREMultiEcho
    @assert hasfield(T, :Δt)
    gre.Δt
end

"""
    fat_fraction(::AbstractGREMultiEcho)

Calculate and return the fat fraction.

## Remark
- The precise implementation depends on the concrete subtype of `AbstractGREMultiEcho`.
- Generates an error if no such implementation exists.
"""
function fat_fraction(gre::AbstractGREMultiEcho)
    error("No implementation of fat_fraction() available for " * string(typeof(gre)))
end