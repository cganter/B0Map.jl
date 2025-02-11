import VP4Optim as VP

"""
    AbstractGREMultiEcho{Ny,Nx,Nc,T}

Abstract supertype of multi-echo GRE sequences

# Type parameters
- `Ny::Int`: Number of acquired data == number(echoes) * number(coils)
- `Nx::Int`: Number of *variable* parameters `x ⊆ {ϕ, R2s, ...}` (relevant for optimization)
- `Nc::Int`: Number of linear coefficients (e.g. number of coil elements)
- `T::Union{Float64, ComplexF64}`: acquired data type
"""
abstract type AbstractGREMultiEcho{Ny,Nx,Nc,T} <: VP.Model{Ny,Nx,Nc,T} end
