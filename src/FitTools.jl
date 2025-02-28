using ChunkSplitters, Compat
@compat public FitPar, fitPar, FitOpt, fitOpt, set_num_phase_intervals, calc_par

"""
Data structure for fitting images.

## Content
- `gre::AbstractGREMultiEcho`: Specifics of the GRE acquisition and signal/tissue model
- `args::Tuple`: Arguments used to construct `gre` (not including keyword arguments)
- `data::Array`: Complex data
- `S::Array`: Mask to specify ROI
- `ϕ::Array`: Phase ``\\phi = \\omega \\cdot \\Delta t`` (fit parameter)
- `R2s::Array`: Relaxation rate ``R_2^\\ast`` (fit parameter)
- `c::Array`: Array of linear VARPRO coefficient vectors ``\\bm{c}(\\phi, R_2^\\ast)``
- `χ2::Array`: Least squares residual ``\\chi^2``
## Scope
- Local and non-local (regularized) fitting
"""
struct FitPar{T<:AbstractGREMultiEcho}
    gre::T
    args::Tuple
    data::Array
    S::Array
    ϕ::Array
    R2s::Array
    c::Array
    χ2::Array
end

"""
    fitPar(gre::AbstractGREMultiEcho{Ny,Nx,Nc,T}, data::AbstractArray, S::AbstractArray) where {Ny,Nx,Nc,T}

Constructor for [FitPar](@ref FitPar)

## Arguments
- `gre::AbstractGREMultiEcho`: Initialized structure with GRE sequence parameters and signal/tissue model.
- `data::AbstractArray`: Complex multi-echo (if available also multi-coil) GRE data for each location.
- `S::AbstractArray`: Mask to specify ROI.
## Remarks
- `size(S)` defines size and spatial dimensions (typically 2 or 3) of the data block.
- Mandatory: `size(data)[1:ndims(S)] == size(S)`
- The format of `data` must be such that `VP4Optim.setdata!(gre, reshape(data, size(S)..., :)[ci,:])` works for any `ci ∈ CartesianIndices(S)`
- For single-coil images, this means `ndims(data) == ndims(S) + 1` with the last index enumerating the echoes.
- For multi-coil data, we typically have `ndims(data) == ndims(S) + 2` with the last but one index enumerating echo times and the last index enumerating the coils.
"""
function fitPar(gre::AbstractGREMultiEcho{Ny,Nx,Nc,T}, data::AbstractArray, S::AbstractArray) where {Ny,Nx,Nc,T}
    szS = size(S)
    c = Array{Vector{T}}(undef, szS)
    fill!(c, zeros(T, Nc))
    FitPar{typeof(gre)}(
        gre,
        data,
        S,
        zeros(szS),
        zeros(szS),
        c,
        zeros(szS)
    )
end

"""
Data structure holding the fit parameters.

## Local Fit and PHASER
- `n_ϕ::Int`: number of golden section search (GSS) intervals for the initial phase search (with `R2s == 0`).
- `ϕ_rngs::Array`: GSS intervals, associated with `n_ϕ`.
- `R2s_rng::Array`: Search range for `R2s`.
- `ϕ_acc::Float64`: Required GSS accuracy for `ϕ_acc`
- `R2s_acc::Float64`: Required GSS accuracy for `R2s_acc`
## Local Fit only
- `optim::Bool`: Non-linear optimiztion in addition to GSS? (Requires gradients to be implemented for the GRE model.)
## PHASER only
- `λ_tikh::Float`: (Small) Tikhonov regularization parameter
## General
- `n_chunks::Int`: Number of chunks to profit from multi-threaded execution.
## Remark
- Use [`set_num_phase_intervals`](@ref set_num_phase_intervals) to modify `n_ϕ`, since only then `ϕ_rngs` will be set properly.
"""
mutable struct FitOpt
    n_ϕ::Int
    ϕ_rngs::Array
    R2s_rng::Array
    ϕ_acc::Float64
    R2s_acc::Float64
    optim::Bool
    λ_tikh::Float64
    n_chunks::Int
end

"""
    fitOpt(fitpar::FitPar)

Default constructor for [FitOpt](@ref FitOpt)

## Arguments
- `fitpar::FitPar`: Initialized `FitPar` instance
## Default values
- `n_ϕ == 3`
- `ϕ_rngs == phase_search_intervals(n_ϕ, gre.ϕ_scale)`
- `R2s_rng == [0.0, 1.0]`
- `ϕ_acc == 1.e-4`
- `R2s_acc == 1.e-4`
- `optim == true`
- `λ_tikh == 1.e-6`
- `n_chunks == 8Threads.nthreads()`
"""
function fitOpt(fitpar::FitPar)
    n_ϕ = 3
    ϕ_rngs = phase_search_intervals(fitpar.gre, n_ϕ)
    R2s_rng = [0.0, 1.0]
    ϕ_acc = 1.e-4
    R2s_acc = 1.e-4
    optim = true
    λ_tikh = 1.e-6
    n_chunks = 8Threads.nthreads()
    FitOpt(n_ϕ, ϕ_rngs, R2s_rng, ϕ_acc, R2s_acc, optim, λ_tikh, n_chunks)
end

"""
    phase_search_intervals(gre, n_ϕ)

Calulates the GSS intervals for the initial `ϕ` search (with `R2s == 0.0`)

## Arguments
- `gre::AbstractGREMultiEcho`: Initialized structure with GRE sequence parameters and signal/tissue model.
- `n_ϕ::Int`: number of golden section search (GSS) intervals
## Remarks
- Background: For non-equidistant echo times, the `2π`-periodicity with respect to `ϕ` no longer holds.
- `gre` contains a field `ϕ_scale == Δt / ΔTE` to define an effective periodicity (or better search range) `ϕ_scale * 2π` via the optional parameter `Δt`. 
- Returns empty vector `Any[]`, if `n_ϕ == 0`
- Should not be called directly. Use [`set_num_phase_intervals`](@ref set_num_phase_intervals) instead.
"""
function phase_search_intervals(gre, n_ϕ)
    n_ϕ == 0 && return []

    ϕ_period_2 = gre.ϕ_scale * π

    Δϕ2 = ϕ_period_2 / n_ϕ
    ϕs = range(-ϕ_period_2 + Δϕ2, ϕ_period_2 - Δϕ2, n_ϕ)

    return [[ϕ_ - Δϕ2, ϕ_ + Δϕ2] for ϕ_ in ϕs]
end

"""
    set_num_phase_intervals(fitpar, fitopt, n_ϕ)

Sets the GSS intervals for the initial `ϕ` search (with `R2s == 0.0`)
    
## Arguments
- `fitpar::FitPar`: Fit parameters
- `fitopt::FitOpt`: Fit options
- `n_ϕ`: Number of search intervals
## Remarks
- Resets field `n_ϕ` in `fitopt` and then calls [`phase_search_intervals`](@ref phase_search_intervals).
"""
function set_num_phase_intervals(fitpar, fitopt, n_ϕ)
    fitopt.n_ϕ = n_ϕ
    fitopt.ϕ_rngs = phase_search_intervals(fitpar.gre, n_ϕ)
end

"""
    calc_par(fitpar::FitPar{T}, fitopt::FitOpt, parfun::Function, res::AbstractArray) where {T<:AbstractGREMultiEcho}

Extract model specific information and store it in array `res`.

## Arguments
- `fitpar::FitPar`: Fit parameters
- `fitopt::FitOpt`: Fit options
- `parfun::Function`: Used to call `parfun(gre::T)` for each location in `fitpar.S`, based upon the local data and fit estimates. 
- `res::AbstractArray`: Allocated space for the results (`size(res) == size(fitpar.S)`).
"""
function calc_par(fitpar::FitPar{T}, fitopt::FitOpt, parfun::Function, res::AbstractArray) where {T<:AbstractGREMultiEcho}
    # Cartesian indices of valid data (defined by the mask S)
    cis = CartesianIndices(fitpar.S)[fitpar.S]
    cis_chunks = [view(cis, index_chunks(cis, n=fitopt.n_chunks)[i]) for i in 1:fitopt.n_chunks]

    # channel to prevent data races in case of multi-threaded execution
    ch_gre = Channel{T}(Threads.nthreads())

    for _ in 1:Threads.nthreads()
        put!(ch_gre, deepcopy(fitpar.gre))
    end

    # do the work
    @time Threads.@threads for cis_chunk in cis_chunks
        # take free models
        gre = take!(ch_gre)

        # work on actual chunk
        calc_par_chunk(gre, fitpar, parfun, res, cis_chunk)

        # put the model back
        put!(ch_gre, gre)
    end

    # close channel
    close(ch_gre)
end

"""
    calc_par_chunk(gre::AbstractGREMultiEcho, fitpar::FitPar, parfun::Function, res::AbstractArray, cis_chunk)

Auxiliary function
"""
function calc_par_chunk(gre::AbstractGREMultiEcho, fitpar::FitPar, parfun::Function, res::AbstractArray, cis_chunk)
    szS = size(fitpar.S)

    for ci in cis_chunk
        # set data
        @views VP.set_data!(gre, reshape(fitpar.data, szS..., :)[ci, :])

        # apply the argument
        VP.x!(gre, [fitpar.ϕ[ci], fitpar.R2s[ci]])

        # calculate parameter
        res[ci] = parfun(gre)
    end
end

"""
    fat_fraction_map(fitpar, fitopt)

Return the fat fraction map.

## Arguments
- `fitpar::FitPar`: Fit parameters
- `fitopt::FitOpt`: Fit options
## Remarks
- `size(returned map) == size(fitpar.S)`
- Avoids the necessity to allocate space.
- Relies on [`calc_par`](@ref calc_par).
"""
function fat_fraction_map(fitpar, fitopt)
    ff = zeros(size(fitpar.S))
    calc_par(fitpar, fitopt, fat_fraction, ff)
    return ff
end

"""
    freq_map(fitpar)

Return the frequency map [Hz].

## Arguments
- `fitpar::FitPar`: Fit parameters
## Remarks
- `size(returned map) == size(fitpar.S)`
- Simply rescales the phase map `fitpar.ϕ` instead of calling [`calc_par`](@ref calc_par).
"""
function freq_map(fitpar)
    (1000.0 / (2π * Δt(fitpar.gre))) * fitpar.ϕ
end
