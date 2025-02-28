#= 
==================================================================  
Unconstrained VARPRO for water-fat mixtures
==================================================================  
=#

using LinearAlgebra, StaticArrays, Statistics, Compat
import VP4Optim as VP
@compat public ModParWFFW, check, GREMultiEchoWFFW, fat_fraction

"""
    ModParWFFW <: VP4Optim.ModPar

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
struct ModParWFFW <: VP.ModPar
    ts::Vector{Float64}
    B0::Float64
    ppm_fat::Vector{Float64}
    ampl_fat::Vector{Float64}
    precession::Symbol
    x_sym::Vector{Symbol}
    Δt::Float64
end

"""
    ModParWFFW()

Return default instance of `ModParWFFW`
"""
function ModParWFFW()
    ts = Float64[]
    B0 = 0.0
    ppm_fat = Float64[]
    ampl_fat = Float64[]
    precession = :unknown
    x_sym = [:ϕ, :R2s]
    Δt = 0.0
    
    ModParWFFW(ts, B0, ppm_fat, ampl_fat, precession, x_sym, Δt)
end

"""
    VP.check(pars::ModParWFFW)

Throws an exception, if the fields in `pars` are defined inconsistently.
"""
function VP.check(pars::ModParWFFW)
    @assert length(pars.ts) > 1
    @assert pars.B0 > 0
    @assert length(pars.ppm_fat) == length(pars.ampl_fat) > 0
    @assert pars.precession ∈ [:clockwise, :counterclockwise]
    @assert all(sy -> sy ∈ [:ϕ, :R2s], pars.x_sym)
    @assert pars.Δt ≥ 0.0
end

"""
[VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/) model

## Scope
- RF spoiled multi-echo GRE sequence
- Water-fat tissue model
## Specifics
- Water and fat are *not constrained* to have equal phase at zero echo time.
- Single coil only
"""
mutable struct GREMultiEchoWFFW{Ny,Nx} <: AbstractGREMultiEcho{Ny,Nx,2,ComplexF64}
    # common parameters of any model
    sym::Vector{Symbol}
    x_sym::Vector{Symbol}
    par_sym::Vector{Symbol}
    val::Vector{Float64}
    x_ind::SVector{Nx,Int}
    par_ind::Vector{Int}
    y::SVector{Ny,ComplexF64}
    y2::Float64
    
    # model specific information
    # measurement conditions
    ts::SVector{Ny,Float64}     # echo times [ms]
    B0::Float64             # field strength [T]
    precession::Symbol      # orientation of precession
    # fat model specification
    ppm_fat::Vector{Float64}    # ppm of fat peaks
    # convention: ppm(water) == 0 and ppm(main fat peak) < 0 (!)
    ampl_fat::Vector{Float64}   # normalized fat peak amplitudes
    # auxiliary elements
    ΔTE::Float64
    Δt::Float64
    ϕ_scale::Float64
    Δt2::Float64
    iΔt::ComplexF64
    ty::SVector{Ny,ComplexF64}
    w::SVector{Ny,ComplexF64}
end

"""
    GREMultiEchoWFFW(pars::ModParWFFW)

Constructor 

# Arguments 
- `pars::ModParWFFW`: Model parameters to instantiate the model. See [ModParWFFW](@ref ModParWFFW).
"""
function GREMultiEchoWFFW(pars::ModParWFFW)
    # before doing anything else: check parameters
    VP.check(pars)

    # set parametric type parameters
    Ny, Nx = length(pars.ts), length(pars.x_sym)

    GREMultiEchoWFFW(Val(Ny), Val(Nx), pars)
end

"""
    GREMultiEchoWFFW(::Val{Ny}, ::Val{Nx}, pars::ModParWFFW)

Auxiliary function
"""
function GREMultiEchoWFFW(::Val{Ny}, ::Val{Nx}, pars::ModParWFFW) where {Ny, Nx}
    # all parameters in val (variable or not)
    sym = [:ϕ, :R2s]
    n_sym = length(sym)
    x_sym = deepcopy(pars.x_sym)
    # initialize storage and indexing of parameters
    x_ind = SVector{Nx,Int}(findfirst(x -> x == x_sym[i], sym) for i in 1:Nx) # indices of variable parameters
    par_ind = filter(x -> x ∉ x_ind, 1:n_sym)
    par_sym = sym[par_ind]
    # real vector of all parameters (variable and constant)
    val = zeros(n_sym) 
    # data
    y = SVector{Ny,ComplexF64}(zeros(ComplexF64, Ny))
    y2 = 0.0
    # initialize mandatory fields
    ts = SVector{Ny,Float64}(pars.ts)
    ΔTE = mean(ts[2:end] - ts[1:end-1])
    if pars.Δt == 0.0
        Δt = ΔTE
        ϕ_scale = 1.0
    else
        Δt = pars.Δt
        ϕ_scale = Δt / ΔTE
    end
    Δt2 = 1.0 / Δt^2
    iΔt = 1im / Δt
    fac = im * 2π * 0.042577 * pars.B0
    pars.precession == :clockwise && (iΔt = - iΔt; fac = - fac)
    ppm_fat = deepcopy(pars.ppm_fat)
    ampl_fat = deepcopy(pars.ampl_fat)
    ty = SVector{Ny,ComplexF64}(zeros(ComplexF64, Ny))
    w = SVector{Ny,ComplexF64}(sum(ampl_fat' .* exp.(fac * ppm_fat' .* ts), dims=2))
    
    GREMultiEchoWFFW{Ny,Nx}(sym, x_sym, par_sym, val, x_ind, par_ind, y, y2, ts, pars.B0, 
        pars.precession, ppm_fat, ampl_fat, ΔTE, Δt, ϕ_scale, Δt2, iΔt, ty, w)
end

"""
    fat_fraction(gre::GREMultiEchoWFFW)

Calculates and returns fat fraction.
"""
function fat_fraction(gre::GREMultiEchoWFFW)
    (w, f) = abs.(VP.c(gre))

    return f / (w + f)
end

"""
    VP.y!(gre::GREMultiEchoWFFW{Ny,Nx}, new_y::AbstractArray) where {Ny,Nx}

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.y!(gre::GREMultiEchoWFFW{Ny,Nx}, new_y::AbstractArray) where {Ny,Nx}
    gre.y = SVector{Ny,ComplexF64}(new_y)
    gre.y2 = real(gre.y' * gre.y)
    gre.ty = gre.ts .* gre.y
end

"""
    VP.A(gre::GREMultiEchoWFFW)

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.A(gre::GREMultiEchoWFFW)
    A = calc_A(gre)

    return A
end

"""
    calc_A(gre::GREMultiEchoWFFW{Ny,Nx}) where {Ny,Nx}

Auxiliary method
"""
function calc_A(gre::GREMultiEchoWFFW{Ny,Nx}) where {Ny,Nx}
    e = exp.((gre.iΔt * gre.val[1] - gre.val[2]) .* gre.ts)
    ew = e .* gre.w
    
    return StaticArrays.sacollect(SMatrix{Ny,2,ComplexF64}, j == 1 ? e[i] : ew[i] for i in 1:Ny, j in 1:2)
end

"""
    calc_Bb(gre::GREMultiEchoWFFW, A)

Auxiliary method
"""
function calc_Bb(gre::GREMultiEchoWFFW, A)
    B = A' * A
    b = A' * gre.y

    return (B, b)
end

"""
    calc_∂Bb(gre::GREMultiEchoWFFW{Ny,Nx}, A) where {Ny,Nx}

Auxiliary method
"""
function calc_∂Bb(gre::GREMultiEchoWFFW{Ny,Nx}, A) where {Ny,Nx}
    tA = gre.ts .* A

    ∂B = SVector{Nx,SMatrix{2,2,ComplexF64}}(begin
        if gre.x_sym[i] == :ϕ
            @SMatrix zeros(ComplexF64, 2, 2)
        else # :R2s
            -2(A' * tA)
        end
    end for i in 1:Nx)

    Aty = tA' * gre.y

    ∂b = SVector{Nx,SVector{2,ComplexF64}}(begin
        if gre.x_sym[i] == :ϕ
            -gre.iΔt * Aty
        else # :R2s
            -Aty
        end
    end for i in 1:Nx)

    (∂B, ∂b, tA)
end

"""
    calc_∂∂Bb(gre::GREMultiEchoWFFW{Ny,Nx}, tA) where {Ny,Nx}

Auxiliary method
"""
function calc_∂∂Bb(gre::GREMultiEchoWFFW{Ny,Nx}, tA) where {Ny,Nx}
    ∂∂B = SMatrix{Nx,Nx,SMatrix{2,2,ComplexF64}}(begin
        if gre.x_sym[i] == :ϕ || gre.x_sym[j] == :ϕ
            @SMatrix zeros(ComplexF64, 2, 2)
        else # (:R2s, :R2s)
            4(tA' * tA)
        end
    end for i in 1:Nx, j in 1:Nx)

    At2y = tA' * gre.ty

    ∂∂b = SMatrix{Nx,Nx,SVector{2,ComplexF64}}(begin
        if gre.x_sym[i] == :ϕ && gre.x_sym[j] == :ϕ
            -gre.Δt2 * At2y
        elseif (gre.x_sym[i] == :ϕ && gre.x_sym[j] == :R2s) || (gre.x_sym[i] == :R2s && gre.x_sym[j] == :ϕ)
            gre.iΔt * At2y
        else # (:R2s, R2s)
            At2y
        end
    end for i in 1:Nx, j in 1:Nx)

    return (∂∂B, ∂∂b)
end

"""
    VP.Bb!(gre::GREMultiEchoWFFW)

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.Bb!(gre::GREMultiEchoWFFW)
    A = calc_A(gre)
    (B, b) = calc_Bb(gre, A)

    return (B, b)
end

"""
    VP.∂Bb!(gre::GREMultiEchoWFFW)

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.∂Bb!(gre::GREMultiEchoWFFW)
    A = calc_A(gre)
    (B, b) = calc_Bb(gre, A)
    (∂B, ∂b, _) = calc_∂Bb(gre, A)

    return (B, b, ∂B, ∂b)
end

"""
    VP.∂∂Bb!(gre::GREMultiEchoWFFW{Ny,Nx}) where {Ny,Nx}

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.∂∂Bb!(gre::GREMultiEchoWFFW{Ny,Nx}) where {Ny,Nx}
    A = calc_A(gre)
    (B, b) = calc_Bb(gre, A)
    (∂B, ∂b, tA) = calc_∂Bb(gre, A)
    (∂∂B, ∂∂b) = calc_∂∂Bb(gre, tA)

    return (B, b, ∂B, ∂b, ∂∂B, ∂∂b)
end