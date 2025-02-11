#= 
==================================================================  
Unconstrained VARPRO for water-fat mixtures
==================================================================  
=#

using LinearAlgebra, StaticArrays, Statistics
import VP4Optim as VP

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
    greMultiEchoWFFW(ts, B0, ppm_fat, ampl_fat, precession; x_sym=[:ϕ, :R2s], Δt=nothing)

Constructor 

# Arguments 
- `ts::Vector{Float64}`: Echo times [ms]
- `B0::Float64`: Main field strength [T]
- `ppm_fat::Vector{Float64}`: Chemical shift of fat peaks
- `ampl_fat::Vector{Float64}`: relative fat peak amplitudes (all positive with `sum(ampl_fat) ≈ 1`)
- `precession::Symbol`: Orientation of precession `∈ {:clockwise, :counterclockwise}`
- `x_sym::Vector{Symbol}`: Vector of variable parameters `∈ {:ϕ, :R2s}`, default: `[:ϕ, :R2s]`
- `Δt::Union{Float64, Nothing}`: Allows to adjust the frequency bandwidth ``2π/Δt`` in case of 
    non-equidistant echoes. Default: `Δt` equals the average echo spacing.
"""
function greMultiEchoWFFW(ts, B0, ppm_fat, ampl_fat, precession; x_sym=[:ϕ, :R2s], Δt=nothing)
    GREMultiEchoWFFW(Val(length(ts)), Val(length(x_sym)), ts, B0, ppm_fat, ampl_fat, precession, x_sym, Δt)
end

"""
    GREMultiEchoWFFW(::Val{Ny}, ::Val{Nx}, ts, B0, ppm_fat, ampl_fat, 
        precession, x_sym, Δt) where {Ny,Nx}

Auxiliary function
"""
function GREMultiEchoWFFW(::Val{Ny}, ::Val{Nx}, ts, B0, ppm_fat, ampl_fat, 
        precession, x_sym, Δt) where {Ny, Nx}
    @assert precession ∈ (:counterclockwise, :clockwise)
    # all parameters in val (variable or not)
    sym = [:ϕ, :R2s]
    n_sym = length(sym)
    # confirm that all *submitted* variables are known
    @assert all(sy -> sy ∈ sym, x_sym)
    # initialize storage and indexing of parameters
    x_ind = SVector{Nx,Int}(findfirst(x -> x == x_sym[i], sym) for i in 1:Nx) # indices of variable parameters
    par_ind = filter(x -> x ∉ x_ind, 1:n_sym)
    par_sym = sym[par_ind]
    val = zeros(n_sym) # real vector of all parameters (variable and constant)
    # data
    y = SVector{Ny,ComplexF64}(zeros(ComplexF64, Ny))
    y2 = 0.0
    # initialize mandatory fields
    ts = SVector{Ny,Float64}(collect(ts))
    ΔTE = mean(ts[2:end] - ts[1:end-1])
    if Δt === nothing
        Δt = ΔTE
        ϕ_scale = 1.0
    else
        @assert Δt isa Real
        ϕ_scale = Δt / ΔTE
    end
    Δt2 = 1.0 / Δt^2
    iΔt = 1im / Δt
    fac = im * 2π * 0.042577 * B0
    precession == :clockwise && (iΔt = - iΔt; fac = - fac)
    ppm_fat = collect(ppm_fat)
    ampl_fat = collect(ampl_fat)
    ty = SVector{Ny,ComplexF64}(zeros(ComplexF64, Ny))
    w = SVector{Ny,ComplexF64}(sum(ampl_fat' .* exp.(fac * ppm_fat' .* ts), dims=2))
    
    GREMultiEchoWFFW{Ny,Nx}(sym, x_sym, par_sym, val, x_ind, par_ind, y, y2, ts, B0, precession, ppm_fat, ampl_fat, ΔTE, Δt, ϕ_scale, Δt2, iΔt, ty, w)
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