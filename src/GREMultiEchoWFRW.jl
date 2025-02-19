#=
==================================================================  
Real-valued VARPRO for water-fat mixtures
==================================================================  
=#

using LinearAlgebra, StaticArrays, Statistics, Compat
import VP4Optim as VP
@compat public GREMultiEchoWFRW, greMultiEchoWFRW, fat_fraction, coil_phase

"""
[VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/) model

## Scope
- RF spoiled multi-echo GRE sequence
- Water-fat tissue model
## Specifics
- Water and fat components are described by real weights, multiplied with a common phase factor.
- Single coil only.
- No partial derivatives implemented
"""
mutable struct GREMultiEchoWFRW{Ny,Nx,Nc,Nt} <: AbstractGREMultiEcho{Ny,Nx,2,Float64}
    # common parameters of any model
    sym::Vector{Symbol}
    x_sym::Vector{Symbol}
    par_sym::Vector{Symbol}
    val::Vector{Float64}
    x_ind::SVector{Nx,Int}
    par_ind::Vector{Int}
    y::SVector{Ny,Float64}
    y2::Float64
    cy::SVector{Nt,ComplexF64}

    # model specific information
    c_phase::Float64
    # measurement conditions
    ts::SVector{Nt,Float64}     # echo times [ms]
    B0::Float64             # field strength [T]
    precession::Symbol      # orientation of precession
    # fat model specification
    ppm_fat::Vector{Float64}    # ppm of fat peaks
    # convention: ppm(water) == 0 and ppm(main fat peak) < 0 (!)
    ampl_fat::Vector{Float64}   # normalized fat peak amplitudes
    # general auxiliary elements
    ΔTE::Float64
    Δt::Float64
    ϕ_scale::Float64
    iΔt::ComplexF64
    w::SVector{Nt,ComplexF64}
    A::SMatrix{Ny,Nc,Float64}
end

"""
    fat_fraction(gre::GREMultiEchoWFRW)

Calculate and return fat fraction ``|r_f| / (|r_w| + |r_f|)``.
"""
function fat_fraction(gre::GREMultiEchoWFRW)
    (ar_w, ar_f) = abs.(VP.c(gre))
    ar_f / (ar_w + ar_f)
end

"""
    coil_phase(gre::GREMultiEchoWFRW)

Return (coil) phase ``\\theta``.
"""
function coil_phase(gre::GREMultiEchoWFRW)
    gre.c_phase
end

"""
    greMultiEchoWFRW(ts, B0, ppm_fat, ampl_fat, precession; x_sym=[:ϕ, :R2s], Δt=nothing)

Constructor 

# Arguments 
- `ts::Vector{Float64}`: Echo times [ms]
- `B0::Float64`: Main field strength [T]
- `ppm_fat::Vector{Float64}`: Chemical shift of fat peaks
- `ampl_fat::Vector{Float64}`: relative fat peak amplitudes (all positive with `sum(ampl_fat) ≈ 1`)
- `precession::Symbol`: Orientation of precession `∈ {:clockwise, :counterclockwise}`
- `mode::Symbol`: By default (`mode == :auto_fat`) the fat fraction `f` is calculated automatically.
    Alternatively (`mode == :manual_fat`), it can also be set manually.
- `x_sym::Vector{Symbol}`: Vector of variable parameters `∈ {:ϕ, :R2s}`, default: `[:ϕ, :R2s]`
- `Δt::Union{Float64, Nothing}`: Allows to adjust the frequency bandwidth ``2π/Δt`` in case of 
    non-equidistant echoes. Default: `Δt` equals the average echo spacing.
"""
function greMultiEchoWFRW(ts, B0, ppm_fat, ampl_fat, precession; x_sym=[:ϕ, :R2s], Δt=nothing) 
    Nt = length(ts)
    Ny, Nx, Nc = 2Nt, length(x_sym), 2

    GREMultiEchoWFRW(Val(Ny), Val(Nx), Val(Nc), Val(Nt), ts, B0, ppm_fat, ampl_fat, precession, x_sym, Δt)
end

"""
    GREMultiEchoWFRW(::Val{Ny}, ::Val{Nx}, ::Val{Nc}, ::Val{Nt}, ts, B0, ppm_fat, ampl_fat,
    precession, x_sym, Δt) where {Ny,Nx,Nc,Nt}

Auxiliary routine
"""
function GREMultiEchoWFRW(::Val{Ny}, ::Val{Nx}, ::Val{Nc}, ::Val{Nt}, ts, B0, ppm_fat, ampl_fat,
    precession, x_sym, Δt) where {Ny,Nx,Nc,Nt}
    @assert precession ∈ (:counterclockwise, :clockwise)
    # automatic or manual definition of fat fraction
    sym = [:ϕ, :R2s]
    n_sym = length(sym)
    # variable parameters; equal to par_sym, if not specified
    x_sym === nothing && (x_sym = deepcopy(sym))
    # confirm that all *submitted* variables are known
    @assert all(sy -> sy ∈ sym, x_sym)
    # initialize storage and indexing of parameters
    x_ind = SVector{Nx,Int}(findfirst(x -> x == x_sym[i], sym) for i in 1:Nx) # indices of variable parameters
    par_ind = filter(x -> x ∉ x_ind, 1:n_sym)
    par_sym = sym[par_ind]
    # real vector of all parameters (variable and constant)
    val = zeros(2)
    # data
    y = SVector{Ny,Float64}(zeros(Float64, Ny))
    y2 = 0.0
    cy = SVector{Nt,ComplexF64}(zeros(ComplexF64, Nt))
    # initialize mandatory fields
    ts = SVector{Nt,Float64}(collect(ts))
    ΔTE = mean(ts[2:end] - ts[1:end-1])
    if Δt === nothing
        Δt = ΔTE
        ϕ_scale = 1.0
    else
        @assert Δt isa Real
        ϕ_scale = Δt / ΔTE
    end
    iΔt = 1im / Δt
    fac = im * 2π * 0.042577 * B0
    precession == :clockwise && (iΔt = -iΔt; fac = -fac)
    ppm_fat = collect(ppm_fat)
    ampl_fat = collect(ampl_fat)
    w = SVector{Nt,ComplexF64}(sum(ampl_fat' .* exp.(fac * ppm_fat' .* ts), dims=2))
    A = SMatrix{Ny,Nc,Float64}(zeros(Float64, Ny, Nc))
    
    GREMultiEchoWFRW{Ny,Nx,Nc,Nt}(sym, x_sym, par_sym, val, x_ind, par_ind, y, y2, cy,
        0.0, ts, B0, precession, ppm_fat, ampl_fat, ΔTE, Δt, ϕ_scale, iΔt, w, A)
end

#=
==================================================================  
VP4Optim routines and specializations
==================================================================  
=#

"""
    VP.set_data!(gre::GREMultiEchoWFRW{Ny,Nx,Nc,Nt}, data::AbstractArray) where {Ny,Nx,Nc,Nt}

Stores data

# Arguments
- `data::AbstractArray`: Complex data as a vector of length `Ny/2`.
    
## Remarks
- Input is transformed into a real vector and handed over to `VP4Optim.y!`.
"""
function VP.set_data!(gre::GREMultiEchoWFRW{Ny,Nx,Nc,Nt}, data::AbstractArray) where {Ny,Nx,Nc,Nt}
    gre.cy = SVector{Nt,ComplexF64}(data)
    VP.y!(gre, reinterpret(Float64, collect(data)))
end

"""
    VP.x_changed!(gre::GREMultiEchoWFRW)

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.x_changed!(gre::GREMultiEchoWFRW)
    update!(gre)
end

"""
    VP.par_changed!(gre::GREMultiEchoWFRW)

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
TBW
"""
function VP.par_changed!(gre::GREMultiEchoWFRW)
    update!(gre)
end

"""
    VP.y!(gre::GREMultiEchoWFRW{Ny,Nx,Nc,Nt}, new_y::AbstractArray) where {Ny,Nx,Nc,Nt}

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.y!(gre::GREMultiEchoWFRW{Ny,Nx,Nc,Nt}, new_y::AbstractArray) where {Ny,Nx,Nc,Nt}
    gre.y = SVector{Ny,Float64}(new_y)
    gre.y2 = gre.y' * gre.y
    update!(gre)
end

#=
==================================================================  
Auxiliary routines
==================================================================  
=#

"""
    update!(gre::GREMultiEchoWFRW{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}

Auxiliary routine
"""
function update!(gre::GREMultiEchoWFRW{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}
    e = exp.((gre.iΔt * gre.val[1] - gre.val[2]) .* gre.ts)
    ew = e .* gre.w
    A0 = StaticArrays.sacollect(SMatrix{Nt,2,ComplexF64}, j == 1 ? e[i] : ew[i] for i in 1:Nt, j in 1:2)
    b0 = A0' * gre.cy
    gre.c_phase = 0.5angle(transpose(b0) * (real.(A0' * A0) \ b0))
    gre.A = SMatrix{Ny,Nc}(reinterpret(Float64, exp(im * gre.c_phase) * A0))
end
