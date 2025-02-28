#= 
==================================================================  
Constrained VARPRO for water-fat mixtures
==================================================================  
=#

using LinearAlgebra, StaticArrays, Statistics, Compat
import VP4Optim as VP
@compat public ModParWF, check, GREMultiEchoWF, fat_fraction, coil_sensitivities

abstract type FatTrait end
struct AutoFat <: FatTrait end
struct ManualFat <: FatTrait end

"""
    ModParWF <: VP4Optim.ModPar

Parameters to setup an instance of `GREMultiEchoWF`

## Fields
- `ts::Vector{Float64}`: Echo times [ms]
- `B0::Float64`: Field strength [T]
- `ppm_fat::Vector{Float64}`: Chemical shift of fat peaks [ppm]
- `ampl_fat::Vector{Float64}`: Relative amplitudes of fat peaks (`≥ 0`, add up to one)
- `precession::Symbol`: Direction of `precession ∈ (:clockwise, :counterclockwise)`
- `n_coils::Int`: Number of coils
- `cov_mat::Matrix{ComplexF64}`: Coil covariance matrix
- `mode::Symbol`: Calculate fat (`mode == :auto_fat`) or explicitly set it (`mode == :manual_fat`)
- `x_sym::Vector{Symbol}`: Variable parameters
- `Δt::Float64`: Effective echo spacing (see docs), `Δt == 0` means `Δt = mean(ΔTE)`
"""
struct ModParWF <: VP.ModPar
    ts::Vector{Float64}
    B0::Float64
    ppm_fat::Vector{Float64}
    ampl_fat::Vector{Float64}
    precession::Symbol
    n_coils::Int
    cov_mat::Matrix{ComplexF64}
    mode::Symbol
    x_sym::Vector{Symbol}
    Δt::Float64
end

"""
    ModParWF()

Return default instance of `ModParWF`
"""
function ModParWF()
    ts = Float64[]
    B0 = 0.0
    ppm_fat = Float64[]
    ampl_fat = Float64[]
    precession = :unknown
    n_coils = 1
    cov_mat = ComplexF64[1;;]
    mode = :auto_fat
    x_sym = [:ϕ, :R2s]
    Δt = 0.0
    
    ModParWF(ts, B0, ppm_fat, ampl_fat, precession, n_coils, cov_mat, mode, x_sym, Δt)
end

"""
    VP.check(pars::ModParWF)

Throws an exception, if the fields in `pars` are defined inconsistently.
"""
function VP.check(pars::ModParWF)
    @assert length(pars.ts) > 1
    @assert pars.B0 > 0
    @assert length(pars.ppm_fat) == length(pars.ampl_fat) > 0
    @assert pars.precession ∈ [:clockwise, :counterclockwise]
    @assert pars.n_coils == size(pars.cov_mat, 1) == size(pars.cov_mat, 2)
    @assert pars.cov_mat' ≈ pars.cov_mat
    @assert pars.mode ∈ [:auto_fat, :manual_fat]
    sym = pars.mode == :auto_fat ? [:ϕ, :R2s] : [:ϕ, :R2s, :f]
    @assert all(sy -> sy ∈ sym, pars.x_sym)
    @assert pars.Δt ≥ 0.0
end

"""
[VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/) model

## Scope
- RF spoiled multi-echo GRE sequence
- Water-fat tissue model
## Specifics
- Water and fat *are constrained* to have equal phase at zero echo time.
- Support for multiple coil elements and coil noise covariance matrix.
- The fat fraction can either be calculated to minimize the cost function 
    ``\\chi^2`` for fixed `ϕ` and `R2s` (this is the default setting) or set manually.
"""
mutable struct GREMultiEchoWF{Ny,Nx,Nc,Nt} <: AbstractGREMultiEcho{Ny,Nx,Nc,ComplexF64}
    # how fat is treated
    fat_trait::FatTrait
    # common parameters of any model
    sym::Vector{Symbol}
    x_sym::Vector{Symbol}
    par_sym::Vector{Symbol}
    val::Vector{Float64}
    x_ind::SVector{Nx,Int}
    par_ind::Vector{Int}
    y::SVector{Ny,ComplexF64}
    y_mat::SMatrix{Nt,Nc,ComplexF64}
    y2::Float64
    #y_coils::SMatrix{Nt,Nc,ComplexF64}
    cov_mat::SMatrix{Nc,Nc,ComplexF64}
    U::SMatrix{Nc,Nc,ComplexF64}
    σ2::SVector{Nc,Float64}
    σ::SVector{Nc,Float64}

    # model specific information
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
    Δt1::Float64
    Δt2::Float64
    iΔt::ComplexF64
    ty::SMatrix{Nt,Nc,ComplexF64}
    w::SVector{Nt,ComplexF64}
    wp2::SVector{Nt,Float64}
    e::SVector{Nt,ComplexF64}
    u::SVector{Nt,ComplexF64}
    ey::SVector{Nc,ComplexF64}
    uy::SVector{Nc,ComplexF64}
    A::SMatrix{Ny,Nc,ComplexF64}
    A_vec::SVector{Nt,ComplexF64}
    tA::SVector{Nt,ComplexF64}
    # auxiliary element, specifically for automatic calculation of fat fraction 
    z2::Float64
    z1::Float64
    z0::Float64
    n2::Float64
    n1::Float64
    n0::Float64
    a2::Float64
    a1::Float64
    a0::Float64
    ε::Float64
    s::Float64
end

"""
    fat_fraction(gre::GREMultiEchoWF)

Just returns the actual value of `f`, whether calculated automatically or set manually.
"""
function fat_fraction(gre::GREMultiEchoWF)
    abs(gre.val[3])
end

"""
    coil_sensitivities(gre::GREMultiEchoWF)

Calculate and return the coil sensitivities.

## Remarks
- If the coil noise covariance matrix ``\\Psi`` differs from the unit matrix ``\\mathrm{I}``, the linear VARPRO coefficients, returned by the routine 
    `VP4Optim.c`, are *not* equal to the actual coil sensitivities.
- Otherwise they are.
"""
function coil_sensitivities(gre::GREMultiEchoWF)
    gre.U * (sqrt.(gre.σ) .* VP.c(gre))
end

"""
    fatTrait(gre::GREMultiEchoWF)

Auxiliary routine
"""
function fatTrait(gre::GREMultiEchoWF)
    gre.fat_trait
end

"""
    GREMultiEchoWF(pars::ModParWF)

Constructor 

# Arguments 
- `pars::ModParWF`: Model parameters to instantiate the model. See [ModParWF](@ref ModParWF).
"""
function GREMultiEchoWF(pars::ModParWF)
    # before doing anything else: check parameters
    VP.check(pars)
    
    # set parametric type parameters
    Nx, Nc, Nt = length(pars.x_sym), pars.n_coils, length(pars.ts)
    Ny = Nt * Nc

    GREMultiEchoWF(Val(Ny), Val(Nx), Val(Nc), Val(Nt), pars)
end

"""
    GREMultiEchoWF(::Val{Ny}, ::Val{Nx}, ::Val{Nc}, ::Val{Nt}, ts, B0, ppm_fat, ampl_fat, 
        precession, x_sym, Δt, mode, cov_mat) where {Ny,Nx,Nc,Nt}

Auxiliary routine
"""
function GREMultiEchoWF(::Val{Ny}, ::Val{Nx}, ::Val{Nc}, ::Val{Nt}, pars::ModParWF) where {Ny,Nx,Nc,Nt}
    # automatic or manual definition of fat fraction
    if pars.mode == :auto_fat
        fat_trait = AutoFat()
        # despite being stored in val[3], the fat fraction is treated like a hidden parameter
        # i.e. not accessible via par() or par!()
        sym = [:ϕ, :R2s]
    else
        fat_trait = ManualFat()
        sym = [:ϕ, :R2s, :f]
    end
    n_sym = length(sym)
    x_sym = deepcopy(pars.x_sym)
    # initialize storage and indexing of parameters
    x_ind = SVector{Nx,Int}(findfirst(x -> x == x_sym[i], sym) for i in 1:Nx) # indices of variable parameters
    par_ind = filter(x -> x ∉ x_ind, 1:n_sym)
    par_sym = sym[par_ind]
    # real vector of all parameters (variable and constant)
    val = zeros(3)
    # data
    y = SVector{Ny,ComplexF64}(zeros(ComplexF64, Ny))
    y_mat = SMatrix{Nt,Nc,ComplexF64}(zeros(ComplexF64, Nt, Nc))
    y2 = 0.0
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
    Δt1 = 1 / Δt
    Δt2 = Δt1^2
    iΔt = 1im * Δt1
    s = 1.0
    fac = im * 2π * 0.042577 * pars.B0
    pars.precession == :clockwise && (iΔt = -iΔt; fac = -fac; s = -s)
    ppm_fat = deepcopy(pars.ppm_fat)
    ampl_fat = deepcopy(pars.ampl_fat)
    ty = SMatrix{Nt,Nc,ComplexF64}(zeros(ComplexF64, Nt, Nc))
    w = SVector{Nt,ComplexF64}(sum(ampl_fat' .* exp.(fac * ppm_fat' .* ts), dims=2) .- 1)
    wp2 = abs2.(w)
    e = SVector{Nt,ComplexF64}(zeros(ComplexF64, Nt))
    u = SVector{Nt,ComplexF64}(zeros(ComplexF64, Nt))
    ey = SVector{Nc}(zeros(ComplexF64,Nc))
    uy = SVector{Nc}(zeros(ComplexF64,Nc))
    A = SMatrix{Ny,Nc,ComplexF64}(zeros(ComplexF64, Ny, Nc))
    A_vec = SVector{Nt,ComplexF64}(zeros(ComplexF64, Nt, 1))
    tA = SVector{Nt,ComplexF64}(zeros(ComplexF64, Nt, 1))
    z2 = z1 = z0 = 0.0
    n2 = n1 = n0 = 0.0
    a2 = a1 = a0 = 0.0
    ε = 0.0
    cov_mat = SMatrix{Nc,Nc,ComplexF64}(pars.cov_mat)
    (σ2, U) = eigen(cov_mat)
    σ = sqrt.(σ2)

    GREMultiEchoWF{Ny,Nx,Nc,Nt}(fat_trait, sym, x_sym, par_sym, val, x_ind, par_ind, y, y_mat, y2,
        cov_mat, U, σ2, σ, ts, pars.B0,
        pars.precession, ppm_fat, ampl_fat, ΔTE, Δt, ϕ_scale, Δt1, Δt2, iΔt, ty, w, wp2,
        e, u, ey, uy, A, A_vec, tA, z2, z1, z0, n2, n1, n0, a2, a1, a0, ε, s)
end

#=
==================================================================  
VP4Optim routines and specializations
==================================================================  
=#

"""
    VP.set_data!(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}, data::AbstractArray) where {Ny,Nx,Nc,Nt}

Sets new data and transforms them, if necessary.

# Arguments
- `data::AbstractArray`: Complex data for all time points and coils. The elements in `data`
    must be ordered such that the expression `SMatrix{Nt,Nc}(data)` produces the correct data matrix 
    ``y_{j\\gamma}``.
    
## Remarks
- The data are transformed based upon the coil noise covariance matrix, as described in the documentation.
- This is followed by a call of `VP4Optim.y!` with the transformed data as arguments.
"""
function VP.set_data!(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}, data::AbstractArray) where {Ny,Nx,Nc,Nt}
    y_mat = (SMatrix{Nt,Nc,ComplexF64}(data) * conj.(gre.U)) ./ transpose(gre.σ)
    VP.y!(gre, y_mat)
end

"""
    VP.x_changed!(gre::GREMultiEchoWF)

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.x_changed!(gre::GREMultiEchoWF)
    update!(gre)
end

"""
    VP.par_changed!(gre::GREMultiEchoWF)

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
TBW
"""
function VP.par_changed!(gre::GREMultiEchoWF)
    update!(gre)
end

"""
    VP.y!(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}, new_y::AbstractArray) where {Ny,Nx,Nc,Nt}

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.y!(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}, new_y::AbstractArray) where {Ny,Nx,Nc,Nt}
    gre.y = SVector{Ny,ComplexF64}(new_y)
    gre.y_mat = SMatrix{Nt,Nc,ComplexF64}(new_y)
    gre.y2 = sum(abs2.(gre.y_mat))
    gre.ty = gre.ts .* gre.y_mat
    update!(gre)
end

"""
    VP.Bb!(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.Bb!(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}
    B = gre.A_vec' * gre.A_vec
    b = SVector{Nc}(gre.A_vec' * gre.y_mat)

    return (B, b)
end

"""
    VP.∂Bb!(gre::GREMultiEchoWF)

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.∂Bb!(gre::GREMultiEchoWF)
    ∂Bb!(fatTrait(gre), gre)
end

"""
    VP.∂∂Bb!(gre::GREMultiEchoWF)

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.∂∂Bb!(gre::GREMultiEchoWF)
    ∂∂Bb!(fatTrait(gre), gre)
end

"""
    VP.y_model(gre::GREMultiEchoWF)

Method described in [VP4Optim](https://cganter.github.io/VP4Optim.jl/stable/)
"""
function VP.y_model(gre::GREMultiEchoWF)
    vec(gre.A_vec .* transpose(VP.c(gre)))
end

#=
==================================================================  
Auxiliary routines
==================================================================  
=#

"""
    update!(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}

Auxiliary routine
"""
function update!(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}
    gre.e = exp.((gre.iΔt * gre.val[1] - gre.val[2]) .* gre.ts)
    gre.u = gre.e .* gre.w
    gre.ey = SVector{Nc}(gre.e' * gre.y_mat)
    gre.uy = SVector{Nc}(gre.u' * gre.y_mat)

    update_fat_fraction!(fatTrait(gre), gre)

    update_A!(gre)
    gre.tA = gre.ts .* gre.A_vec
end

"""
    update_A!(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}

Auxiliary routine
"""
function update_A!(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}
    gre.A_vec = (1.0 .+ gre.val[3] .* gre.w) .* gre.e
    gre.A = StaticArrays.sacollect(SMatrix{Ny, Nc, ComplexF64}, (i-1) ÷ Nt == j-1 ? 
        gre.A_vec[mod.(i-1, Nt)+1] : 0.0 for i in 1:Ny, j in 1:Nc)
end

"""
    update_fat_fraction!(::ManualFat, ::GREMultiEchoWF)

Auxiliary routine
"""
function update_fat_fraction!(::ManualFat, ::GREMultiEchoWF) end

"""
    update_fat_fraction!(::AutoFat, gre::GREMultiEchoWF)

Auxiliary routine
"""
function update_fat_fraction!(::AutoFat, gre::GREMultiEchoWF)
    ep2 = abs2.(gre.e)
    gre.z2 = sum(abs2.(gre.uy))
    gre.z1 = real(gre.ey' * gre.uy)
    gre.z0 = sum(abs2.(gre.ey))
    gre.n2 = sum(abs2.(gre.u))
    gre.n1 = sum(ep2 .* real.(gre.w))
    gre.n0 = sum(ep2)
    gre.a2 = gre.z1 * gre.n2 - gre.z2 * gre.n1
    gre.a1 = 0.5(gre.z0 * gre.n2 - gre.z2 * gre.n0)
    gre.a0 = gre.z0 * gre.n1 - gre.z1 * gre.n0
    gre.ε = gre.a1^2 - gre.a2 * gre.a0

    gre.val[3] = -1.0

    if gre.a2 ≠ 0 && gre.ε ≥ 0
        gre.val[3] = (-gre.a1 + sqrt(gre.ε)) / gre.a2
    elseif gre.a2 == 0.0 && gre.a1 > 0.0
        gre.val[3] = -gre.a0 / 2gre.a1
    end

    if !(0 ≤ gre.val[3] ≤ 1)
        gre.val[3] = 0.0
        update_A!(gre)
        χ2_0 = VP.χ2(gre)
        gre.val[3] = 1.0
        update_A!(gre)
        χ2_1 = VP.χ2(gre)
        gre.val[3] = χ2_0 < χ2_1 ? 0.0 : 1.0
    end
end

"""
    ∂Bb!(::ManualFat, gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}

Auxiliary routine
"""
function ∂Bb!(::ManualFat, gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}
    (B, b) = VP.Bb!(gre)
    (∂B, ∂b) = calc_∂Bb(gre, SVector{Nx}(gre.x_sym))

    return (B, b, ∂B, ∂b)
end

"""
    ∂Bb!(::AutoFat, gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}

Auxiliary routine
"""
function ∂Bb!(::AutoFat, gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}
    (B, b) = VP.Bb!(gre)
    (∂B_, ∂b_) = calc_∂Bb(gre, SVector{Nx + 1}([gre.x_sym..., :f]))
    ∂f, _ = calc_∂f(gre)

    ∂B = SVector{Nx}(@views ∂B_[1:end-1] + ∂B_[end] * ∂f)
    ∂b = SVector{Nx}(@views _∂b_ .+ ∂b_[end] .* _∂f_ for (_∂b_, _∂f_) in zip(∂b_[1:end-1], ∂f))

    return (B, b, ∂B, ∂b)
end

"""
    ∂∂Bb!(::ManualFat, gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}

Auxiliary routine
"""
function ∂∂Bb!(::ManualFat, gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}
    (B, b) = VP.Bb!(gre)
    (∂B, ∂b) = calc_∂Bb(gre, SVector{Nx}(gre.x_sym))
    (∂∂B, ∂∂b) = calc_∂∂Bb(gre, SVector{Nx}(gre.x_sym))

    return (B, b, ∂B, ∂b, ∂∂B, ∂∂b)
end

"""
    ∂∂Bb!(::AutoFat, gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}

Auxiliary routine
"""
function ∂∂Bb!(::AutoFat, gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}
    (B, b) = VP.Bb!(gre)
    (∂B_, ∂b_) = calc_∂Bb(gre, SVector{Nx + 1}([gre.x_sym..., :f]))
    (∂∂B_, ∂∂b_) = calc_∂∂Bb(gre, SVector{Nx + 1}([gre.x_sym..., :f]))
    (∂f, ∂∂f, _, _) = calc_∂∂f(gre)
    ∂B = SVector{Nx}(@views ∂B_[1:end-1] + ∂B_[end] * ∂f)
    ∂b = SVector{Nx}(@views _∂b_ .+ ∂b_[end] .* _∂f_ for (_∂b_, _∂f_) in zip(∂b_[1:end-1], ∂f))

    ∂∂B = SMatrix{Nx,Nx,ComplexF64}(begin
        if i ≤ j
            ∂∂B_[i, j] + ∂∂B_[i, end] * ∂f[j] + ∂∂B_[end, j] * ∂f[i] +
            ∂∂B_[end, end] * ∂f[i] * ∂f[j] +
            ∂B_[end] * ∂∂f[i, j]
        else
            0.0
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂B = SMatrix{Nx,Nx,ComplexF64}(begin
        if i ≤ j
            ∂∂B[i, j]
        else
            ∂∂B[j, i]
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂b = SMatrix{Nx,Nx,SVector{Nc,ComplexF64}}(begin
        if i ≤ j
            ∂∂b_[i, j] .+ ∂∂b_[i, end] .* ∂f[j] .+ ∂∂b_[end, j] .* ∂f[i] .+
            ∂b_[end] .* ∂∂f[i, j]
        else
            SVector{Nc}(zeros(ComplexF64, Nc))
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂b = SMatrix{Nx,Nx,SVector{Nc,ComplexF64}}(begin
        if i ≤ j
            ∂∂b[i, j]
        else
            ∂∂b[j, i]
        end
    end for i in 1:Nx, j in 1:Nx)

    return (B, b, ∂B, ∂b, ∂∂B, ∂∂b)
end

"""
    calc_∂Bb(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}, sy::SVector{Nsy}) where {Ny,Nx,Nc,Nt,Nsy}

Auxiliary routine
"""
function calc_∂Bb(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}, sy::SVector{Nsy}) where {Ny,Nx,Nc,Nt,Nsy}
    ∂B = SVector{Nsy,ComplexF64}(begin
        if sy[i] == :ϕ
            0.0
        elseif sy[i] == :R2s
            -2(gre.A_vec' * gre.tA)
        else # :f
            2real(gre.A_vec' * gre.u)
        end
    end for i in 1:Nsy)

    (:ϕ ∈ sy || :R2s ∈ sy) && (Aty = SVector{Nc}(gre.tA' * gre.y_mat))
    
    ∂b = SVector{Nsy,SVector{Nc,ComplexF64}}(begin
        if sy[i] == :ϕ
            -gre.iΔt * Aty
        elseif sy[i] == :R2s
            -Aty
        else # :f
            gre.uy
        end
    end for i in 1:Nsy)

    (∂B, ∂b)
end

"""
    calc_∂∂Bb(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}, sy::SVector{Nsy}) where {Ny,Nx,Nc,Nt,Nsy}

Auxiliary routine
"""
function calc_∂∂Bb(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}, sy::SVector{Nsy}) where {Ny,Nx,Nc,Nt,Nsy}
    (:R2s ∈ sy && :f ∈ sy) && (Atu = -4real(gre.tA' * gre.u))

    ∂∂B = SMatrix{Nsy,Nsy,ComplexF64}(begin
        if sy[i] == :ϕ || sy[j] == :ϕ
            0.0
        elseif sy[i] == :R2s && sy[j] == :R2s
            4sum(abs2.(gre.tA))
        elseif (sy[i] == :R2s && sy[j] == :f) || (sy[i] == :f && sy[j] == :R2s)
            Atu
        else # (:f, :f)
            2sum(abs2.(gre.u))
        end
    end for i in 1:Nsy, j in 1:Nsy)

    if :ϕ ∈ sy || :R2s ∈ sy
        At2y = SVector{Nc}(gre.tA' * gre.ty)
        if :f ∈ sy
            uty = SVector{Nc}(gre.u' * gre.ty)
        end
    end

    ∂∂b = SMatrix{Nsy,Nsy,SVector{Nc,ComplexF64}}(begin
        if sy[i] == :ϕ && sy[j] == :ϕ
            -gre.Δt2 * At2y
        elseif (sy[i] == :ϕ && sy[j] == :R2s) || (sy[i] == :R2s && sy[j] == :ϕ)
            gre.iΔt * At2y
        elseif sy[i] == :R2s && sy[j] == :R2s
            At2y
        elseif (sy[i] == :ϕ && sy[j] == :f) || (sy[i] == :f && sy[j] == :ϕ)
            -gre.iΔt * uty
        elseif (sy[i] == :R2s && sy[j] == :f) || (sy[i] == :f && sy[j] == :R2s)
            - uty
        else # (:f, :f)
            SVector{Nc}(zeros(ComplexF64, Nc))
        end
    end for i in 1:Nsy, j in 1:Nsy)

    return (∂∂B, ∂∂b)
end

"""
    calc_∂f(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}

Auxiliary routine
"""
function calc_∂f(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}
    f = gre.val[3]
    !(0 < f < 1) && return (SVector{Nx,Float64}(zeros(Nx)), nothing)

    u1 = gre.ts .* gre.u
    e1 = gre.ts .* gre.e
    e1e = gre.e' * e1
    e1u = gre.e' * u1
    u1u = gre.u' * u1
    e1y = SVector{Nc}(gre.e' * gre.ty)
    u1y = SVector{Nc}(gre.u' * gre.ty)

    yee1y = gre.ey' * e1y
    yeu1y = gre.ey' * u1y
    yue1y = gre.uy' * e1y
    yuu1y = gre.uy' * u1y

    ∂z2 = SVector{Nx}(gre.x_sym[i] == :ϕ ? 2gre.s * gre.Δt1 * imag(yuu1y) : -2real(yuu1y)
                      for i in 1:Nx)
    ∂z1 = SVector{Nx}(gre.x_sym[i] == :ϕ ? gre.s * gre.Δt1 * imag(yue1y - yeu1y') : -real(yue1y + yeu1y')
                      for i in 1:Nx)
    ∂z0 = SVector{Nx}(gre.x_sym[i] == :ϕ ? 2gre.s * gre.Δt1 * imag(yee1y) : -2real(yee1y)
                      for i in 1:Nx)

    ∂n2 = SVector{Nx}(gre.x_sym[i] == :ϕ ? 0.0 : -2real(u1u)
                      for i in 1:Nx)
    ∂n1 = SVector{Nx}(gre.x_sym[i] == :ϕ ? 0.0 : -2real(e1u)
                      for i in 1:Nx)
    ∂n0 = SVector{Nx}(gre.x_sym[i] == :ϕ ? 0.0 : -2real(e1e)
                      for i in 1:Nx)

    ∂a2 = ∂z1 * gre.n2 + gre.z1 * ∂n2 - ∂z2 * gre.n1 - gre.z2 * ∂n1
    ∂a1 = 0.5(∂z0 * gre.n2 + gre.z0 * ∂n2 - ∂z2 * gre.n0 - gre.z2 * ∂n0)
    ∂a0 = ∂z0 * gre.n1 + gre.z0 * ∂n1 - ∂z1 * gre.n0 - gre.z1 * ∂n0

    if gre.a2 ≠ 0.0
        @assert gre.ε ≥ 0.0
        ∂ε = 2gre.a1 * ∂a1 - ∂a2 * gre.a0 - gre.a2 * ∂a0
        ∂f = (-1.0 / gre.a2) * (f * ∂a2 + ∂a1 - ∂ε / (2sqrt(gre.ε)))
    elseif gre.a1 > 0.0
        ∂f = (-1.0 / gre.a1) * (f * ∂a1 - 0.5∂a0)
    else
        ∂f = SVector{Nx,Float64}(zeros(Nx))
    end

    aux_∂f = (e1, u1, e1y, u1y, ∂z2, ∂z1, ∂z0, ∂n2, ∂n1, ∂n0, ∂a2, ∂a1, ∂a0, ∂ε)
    return (∂f, aux_∂f)
end

"""
    calc_∂∂f(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}

Auxiliary routine
"""
function calc_∂∂f(gre::GREMultiEchoWF{Ny,Nx,Nc,Nt}) where {Ny,Nx,Nc,Nt}
    f = gre.val[3]
    
    !(0 < f < 1) && return (SVector{Nx,Float64}(zeros(Nx)), SMatrix{Nx,Nx,Float64}(zeros(Nx, Nx)))

    ∂f, aux_∂f = calc_∂f(gre)
    e1, u1, e1y, u1y, ∂z2, ∂z1, ∂z0, ∂n2, ∂n1, ∂n0, ∂a2, ∂a1, ∂a0, ∂ε = aux_∂f

    e2e = sum(abs2.(e1))
    e2u = e1' * u1
    u2u = sum(abs2.(u1))
    e2y = SVector{Nc}(e1' * gre.ty)
    u2y = SVector{Nc}(u1' * gre.ty)

    yee2y = gre.ey' * e2y
    yeu2y = gre.ey' * u2y
    yue2y = gre.uy' * e2y
    yuu2y = gre.uy' * u2y
    y1ee1y = real(e1y' * e1y)
    y1eu1y = e1y' * u1y
    y1uu1y = real(u1y' * u1y)

    ∂∂z2 = SMatrix{Nx,Nx}(begin
        if gre.x_sym[i] == gre.x_sym[j] == :ϕ
            2gre.Δt2 * (y1uu1y - real(yuu2y))
        elseif gre.x_sym[i] == gre.x_sym[j] == :R2s
            2(y1uu1y + real(yuu2y))
        else
            -2gre.s * gre.Δt1 * imag(yuu2y)
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂z1 = SMatrix{Nx,Nx}(begin
        if gre.x_sym[i] == gre.x_sym[j] == :ϕ
            gre.Δt2 * real(2y1eu1y - yue2y - yeu2y')
        elseif gre.x_sym[i] == gre.x_sym[j] == :R2s
            real(2y1eu1y + yue2y + yeu2y)
        else
            - gre.s * gre.Δt1 * imag(yue2y - yeu2y')
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂z0 = SMatrix{Nx,Nx}(begin
        if gre.x_sym[i] == gre.x_sym[j] == :ϕ
            2gre.Δt2 * (y1ee1y - real(yee2y))
        elseif gre.x_sym[i] == gre.x_sym[j] == :R2s
            2(y1ee1y + real(yee2y))
        else
            -2gre.s * gre.Δt1 * imag(yee2y)
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂n2 = SMatrix{Nx,Nx}(begin
        if gre.x_sym[i] == :ϕ || gre.x_sym[j] == :ϕ
            0.0
        else
            4u2u
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂n1 = SMatrix{Nx,Nx}(begin
        if gre.x_sym[i] == :ϕ || gre.x_sym[j] == :ϕ
            0.0
        else
            4real(e2u)
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂n0 = SMatrix{Nx,Nx}(begin
        if gre.x_sym[i] == :ϕ || gre.x_sym[j] == :ϕ
            0.0
        else
            4 * e2e
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂a2 = SMatrix{Nx,Nx}(begin
        if i ≤ j
            ∂∂z1[i, j] * gre.n2 + ∂z1[i] * ∂n2[j] + ∂z1[j] * ∂n2[i] + gre.z1 * ∂∂n2[i, j] -
            ∂∂z2[i, j] * gre.n1 - ∂z2[i] * ∂n1[j] - ∂z2[j] * ∂n1[i] - gre.z2 * ∂∂n1[i, j]
        else
            0.0
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂a2 = SMatrix{Nx,Nx}(begin
        if i ≤ j
            ∂∂a2[i, j]
        else
            ∂∂a2[j, i]
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂a1 = SMatrix{Nx,Nx}(begin
        if i ≤ j
            0.5(∂∂z0[i, j] * gre.n2 + ∂z0[i] * ∂n2[j] + ∂z0[j] * ∂n2[i] + gre.z0 * ∂∂n2[i, j] -
            ∂∂z2[i, j] * gre.n0 - ∂z2[i] * ∂n0[j] - ∂z2[j] * ∂n0[i] - gre.z2 * ∂∂n0[i, j])
        else
            0.0
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂a1 = SMatrix{Nx,Nx}(begin
        if i ≤ j
            ∂∂a1[i, j]
        else
            ∂∂a1[j, i]
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂a0 = SMatrix{Nx,Nx}(begin
        if i ≤ j
            ∂∂z0[i, j] * gre.n1 + ∂z0[i] * ∂n1[j] + ∂z0[j] * ∂n1[i] + gre.z0 * ∂∂n1[i, j] -
            ∂∂z1[i, j] * gre.n0 - ∂z1[i] * ∂n0[j] - ∂z1[j] * ∂n0[i] - gre.z1 * ∂∂n0[i, j]
        else
            0.0
        end
    end for i in 1:Nx, j in 1:Nx)

    ∂∂a0 = SMatrix{Nx,Nx}(begin
        if i ≤ j
            ∂∂a0[i, j]
        else
            ∂∂a0[j, i]
        end
    end for i in 1:Nx, j in 1:Nx)

    if gre.a2 ≠ 0.0
        ∂∂ε = SMatrix{Nx,Nx}(begin
            if i ≤ j
                2(∂a1[i] * ∂a1[j] + gre.a1 * ∂∂a1[i, j]) -
                ∂∂a2[i, j] * gre.a0 - ∂a2[i] * ∂a0[j] - ∂a2[j] * ∂a0[i] - gre.a2 * ∂∂a0[i, j]
            else
                0.0
            end
        end for i in 1:Nx, j in 1:Nx)

        ∂∂ε = SMatrix{Nx,Nx}(begin
            if i ≤ j
                ∂∂ε[i, j]
            else
                ∂∂ε[j, i]
            end
        end for i in 1:Nx, j in 1:Nx)

        ∂∂f = SMatrix{Nx,Nx}(begin
            if i ≤ j
                (-1.0 / gre.a2) * (∂f[i] * ∂a2[j] + ∂f[j] * ∂a2[i] + f * ∂∂a2[i,j] + ∂∂a1[i,j] - 
                    ∂∂ε[i,j] / (2sqrt(gre.ε)) + 0.25∂ε[i] * ∂ε[j] / gre.ε^1.5)
            else
                0.0
            end
        end for i in 1:Nx, j in 1:Nx)

        ∂∂f = SMatrix{Nx,Nx}(begin
            if i ≤ j
                ∂∂f[i,j]
            else
                ∂∂f[j,i]
            end
        end for i in 1:Nx, j in 1:Nx)
    elseif gre.a1 > 0.0
        ∂∂f = SMatrix{Nx,Nx}(begin
            if i ≤ j
                (-1.0 / gre.a1) * (∂f[i] * ∂a1[j] + ∂f[j] * ∂a1[i] + f * ∂∂a1[i,j] + 0.5∂∂a0[i,j])
            else
                0.0
            end
        end for i in 1:Nx, j in 1:Nx)

        ∂∂f = SMatrix{Nx,Nx}(begin
            if i ≤ j
                ∂∂f[i,j]
            else
                ∂∂f[j,i]
            end
        end for i in 1:Nx, j in 1:Nx)
    else
        ∂∂f = SMatrix{Nx,Nx}(zeros(Nx, Nx))
    end

    aux_∂∂f = (∂∂z2, ∂∂z1, ∂∂z0, ∂∂n2, ∂∂n1, ∂∂n0, ∂∂a2, ∂∂a1, ∂∂a0, ∂∂ε)
    return (∂f, ∂∂f, aux_∂f, aux_∂∂f)
end
