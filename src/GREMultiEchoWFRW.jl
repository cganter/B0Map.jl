#=
==================================================================  
Real-valued VARPRO for water-fat mixtures
==================================================================  
=#

using LinearAlgebra, StaticArrays, Statistics, Compat
import VP4Optim as VP
@compat public ModParWFRW, check, GREMultiEchoWFRW, fat_fraction

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
    nTE(::GREMultiEchoWFRW{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}

Return number of recorded echoes
"""
function nTE(::GREMultiEchoWFRW{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}
    Nt
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
    max_derivative(::GREMultiEchoWFRW)

Return max. implemented derivative order.
"""
function max_derivative(::GREMultiEchoWFRW)
    0
end

"""
    coil_phase(gre::GREMultiEchoWFRW)

Return (coil) phase ``\\theta``.
"""
function coil_phase(gre::GREMultiEchoWFRW)
    gre.c_phase
end

"""
    ModParWFRW <: VP4Optim.ModPar{GREMultiEchoWFRW}

Parameters to setup an instance of `GREMultiEchoWF`

## Fields
- `ts::Vector{Float64}`: Echo times [ms]
- `B0::Float64`: Field strength [T]
- `ppm_fat::Vector{Float64}`: Chemical shift of fat peaks [ppm]
- `ampl_fat::Vector{Float64}`: Relative amplitudes of fat peaks (`≥ 0`, add up to one)
- `precession::Symbol`: Direction of `precession ∈ (:clockwise, :counterclockwise)`
- `x_sym::Vector{Symbol}`: Variable parameters
- `Δt::Float64`: Effective echo spacing (see docs), `Δt == 0` means `Δt = mean(ΔTE)`
"""
struct ModParWFRW <: VP.ModPar{GREMultiEchoWFRW}
    ts::Vector{Float64}
    B0::Float64
    ppm_fat::Vector{Float64}
    ampl_fat::Vector{Float64}
    precession::Symbol
    x_sym::Vector{Symbol}
    Δt::Float64
end

"""
    ModParWFRW()

Return default instance of `ModParWFRW`
"""
function VP.ModPar(::Type{GREMultiEchoWFRW})
    ts = Float64[]
    B0 = 0.0
    ppm_fat = Float64[]
    ampl_fat = Float64[]
    precession = :unknown
    x_sym = [:ϕ, :R2s]
    Δt = 0.0
    
    ModParWFRW(ts, B0, ppm_fat, ampl_fat, precession, x_sym, Δt)
end

"""
    VP.check(pars::ModParWFRW)

Throws an exception, if the fields in `pars` are defined inconsistently.
"""
function VP.check(pars::ModParWFRW)
    @assert length(pars.ts) > 1
    @assert pars.B0 > 0
    @assert length(pars.ppm_fat) == length(pars.ampl_fat) > 0
    @assert pars.precession ∈ [:clockwise, :counterclockwise]
    @assert all(sy -> sy ∈ [:ϕ, :R2s], pars.x_sym)
    @assert pars.Δt ≥ 0.0
end

"""
    GREMultiEchoWFRW(pars::ModParWFRW)

Constructor 

# Arguments 
- `pars::ModParWFRW`: Model parameters to instantiate the model. See [ModParWFRW](@ref ModParWFRW).
"""
function GREMultiEchoWFRW(pars::ModParWFRW)
    # before doing anything else: check parameters
    VP.check(pars)

    # set parametric type parameters
    Nt = length(pars.ts)
    Ny, Nx, Nc = 2Nt, length(pars.x_sym), 2

    GREMultiEchoWFRW(Val(Ny), Val(Nx), Val(Nc), Val(Nt), pars)
end

"""
    GREMultiEchoWFRW(::Val{Ny}, ::Val{Nx}, ::Val{Nc}, ::Val{Nt}, ts, B0, ppm_fat, ampl_fat,
    precession, x_sym, Δt) where {Ny,Nx,Nc,Nt}

Auxiliary routine
"""
function GREMultiEchoWFRW(::Val{Ny}, ::Val{Nx}, ::Val{Nc}, ::Val{Nt}, pars::ModParWFRW) where {Ny,Nx,Nc,Nt}
    # all parameters in val (variable or not)
    sym = [:ϕ, :R2s]
    n_sym = length(sym)
    x_sym = deepcopy(pars.x_sym)
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
    ts = SVector{Nt,Float64}(pars.ts)
    ΔTE = mean(ts[2:end] - ts[1:end-1])
    if pars.Δt == 0.0
        Δt = ΔTE
        ϕ_scale = 1.0
    else
        Δt = pars.Δt
        ϕ_scale = Δt / ΔTE
    end
    iΔt = 1im / Δt
    fac = im * 2π * 0.042577 * pars.B0
    pars.precession == :clockwise && (iΔt = -iΔt; fac = -fac)
    ppm_fat = deepcopy(pars.ppm_fat)
    ampl_fat = deepcopy(pars.ampl_fat)
    w = SVector{Nt,ComplexF64}(sum(ampl_fat' .* exp.(fac * ppm_fat' .* ts), dims=2))
    A = SMatrix{Ny,Nc,Float64}(zeros(Float64, Ny, Nc))
    
    GREMultiEchoWFRW{Ny,Nx,Nc,Nt}(sym, x_sym, par_sym, val, x_ind, par_ind, y, y2, cy,
        0.0, ts, pars.B0, pars.precession, ppm_fat, ampl_fat, ΔTE, Δt, ϕ_scale, iΔt, w, A)
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
