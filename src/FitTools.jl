using ChunkSplitters, Compat
@compat public FitPar, fitPar, FitOpt, fitOpt, set_num_phase_intervals, calc_par

"""
Data structure for fitting images.

## Content
- Specifics of the GRE acquisition and signal/tissue model
- Mask to specify ROI
- Arrays for the fitted internal parameters `ϕ` and `R2s`
- Arrays for the resulting linear coil-dependent coefficients `c(ϕ, R2s)`
- Goodness of fit
## Scope
- Local and non-local (regularized) fitting
"""
struct FitPar{T<:AbstractGREMultiEcho}
    gre::T
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
- Mandatory: `size(data)[ndims(S)] == size(S)`
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

## Local Fit
- `n_ϕ::Int`: number of golden section search (GSS) intervals for the initial phase search (with `R2s == 0`).
- `ϕ_rngs::Array`: GSS intervals, associated with `n_ϕ`.
- `R2s_rng::Array`: Search range for `R2s`.
- `ϕ_acc::Float64`: Required GSS accuracy for `ϕ_acc`
- `R2s_acc::Float64`: Required GSS accuracy for `R2s_acc`
- `optim::Bool`: Non-linear optimiztion in addition to GSS? (Requires gradients to be implemented for the GRE model.)
## General
- `n_chunks::Int`: Number of chunks to profit from multi-threaded execution.
## Remark
- Use [ϕ_search_intervals](@ref ϕ_search_intervals) to modify `n_ϕ`, since only then `ϕ_rngs` will be set properly.
"""
mutable struct FitOpt
    n_ϕ::Int
    ϕ_rngs::Array
    R2s_rng::Array
    ϕ_acc::Float64
    R2s_acc::Float64
    optim::Bool
    n_chunks::Int
end

"""
    fitOpt(gre::AbstractGREMultiEcho)

Default constructor for [FitOpt](@ref FitOpt)

## Arguments
- `gre::AbstractGREMultiEcho`: Initialized structure with GRE sequence parameters and signal/tissue model.
## Default values
- `n_ϕ == 3`
- `ϕ_rngs == ϕ_search_intervals(n_ϕ, gre.ϕ_scale)`
- `R2s_rng == [0.0, 1.0]`
- `ϕ_acc == 1.e-4`
- `R2s_acc == 1.e-4`
- `optim == true`
- `n_chunks == 8Threads.nthreads()`
## Remarks
- `gre` is only needed, if `Δt != ΔTE` (to define an effective bandwidth in case of non-equidistant echo times)
"""
function fitOpt(gre::AbstractGREMultiEcho)
    n_ϕ = 3
    ϕ_rngs = ϕ_search_intervals(gre, n_ϕ)
    R2s_rng = [0.0, 1.0]
    ϕ_acc = 1.e-4
    R2s_acc = 1.e-4
    optim = true
    n_chunks = 8Threads.nthreads()
    FitOpt(n_ϕ, ϕ_rngs, R2s_rng, ϕ_acc, R2s_acc, optim, n_chunks)
end

"""
    ϕ_search_intervals(n_ϕ, ϕ_scale=1.0)

Calulates the GSS intervals for the initial `ϕ` search (with `R2s == 0.0`)

## Arguments
- `gre::AbstractGREMultiEcho`: Initialized structure with GRE sequence parameters and signal/tissue model.
- `n_ϕ::Int`: number of golden section search (GSS) intervals
## Remarks
- Background: For non-equidistant echo times, the `2π`-periodicity with respect to `ϕ` no longer holds.
- `gre` contains a field `ϕ_scale == Δt / ΔTE` to define an effective periodicity (or better search range) `ϕ_scale * 2π` via the optional parameter `Δt`. 
- Should not be called directly. Use [set_num_phase_intervals](@ref set_num_phase_intervals) instead.
"""
function ϕ_search_intervals(gre, n_ϕ)
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
- Resets field `n_ϕ` in `fitopt` and then calls [ϕ_search_intervals](@ref ϕ_search_intervals).
"""
function set_num_phase_intervals(fitpar, fitopt, n_ϕ)
    fitopt.n_ϕ = n_ϕ
    fitopt.ϕ_rngs = ϕ_search_intervals(fitpar.gre, n_ϕ)
end

"""
    calc_par(fitpar::FitPar{T}, parfun::Function, res::AbstractArray, n_chunks=8Threads.nthreads()) where {T<:AbstractGREMultiEcho}

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
