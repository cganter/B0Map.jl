using LinearAlgebra, ChunkSplitters, FFTW, StatsBase, Optim, Random, TimerOutputs, Compat
using Plots: plot, scatter!, heatmap
import VP4Optim as VP
@compat public BSmooth, calc, phase_map, phaser, smooth_projection!

"""
    BSmooth{N}

Supertype of smooth bases.

## Type parameter
- `N::Int`: Subspace dimensions
## Remark
- `N` does not necessarily equal the dimension of the data set. Specifically, `N == 2` makes sense for multi-slice data, if there are not enough slices for reasonable interpolation in that direction.
"""
abstract type BSmooth{N} end

"""
    Nfree(::BSmooth)

Return real degrees of freedom of subspace
"""
function Nfree(::BSmooth) end

"""
    ∇Bt∇B_∇Bty(::BSmooth, ::AbstractVector, ::AbstractVector)

Calculates and returns the tuple `(∇B' * ∇B, ∇B' * y)`.

Since efficient evaluation depends on the actual subtype of `BSmooth`,
no generic implementation is provided.
"""
function ∇Bt∇B_∇Bty(::BSmooth, ::AbstractVector, ::AbstractVector) end

"""
    phase_map(bs::BSmooth, b::Float64, c::AbstractVector)

Returns the phase map `φ = b + B' * c`.
"""
function phase_map(bs::BSmooth, b::Float64, c::AbstractVector)
    b .+ phase_map(bs, c)
end

"""
    phase_map(::BSmooth, ::AbstractVector)

Returns the phase map for zero constant offset ``c_0 = 0``.

## Remarks
- The cofficient ``c_0`` is *not* an element of the supplied coefficient vector. (instead of setting the element to zero)
- Since efficient evaluation depends on the actual subtype of `BSmooth`,
no generic implementation is provided.
"""
function phase_map(::BSmooth, ::AbstractVector) end

"""
    phaser(fitpar::FitPar, fitopt::FitOpt, bs::BSmooth{N}) where {N}

Take the data set and do the analysis.

## What it does
- First, a local fit is performed.
- If a smooth basis `bs` is supplied, the result is refined by subspace-based regularization.
- 
## Arguments
- `fitpar::FitPar`: Fit parameters
- `fitopt::FitOpt`: Fit options
- `bs::Union{Nothing, BSmooth}`: Smooth subspace for PHASER (default: `nothing`)
"""
function phaser(fitpar::FitPar, fitopt::FitOpt, bs::BSmooth{N}; locfit=true, diagnostics=false) where {N}
    # timing will always be monitored
    to = TimerOutput()

    # only GSS for PHASER
    optim = fitopt.optim
    fitopt.optim = false

    # generate subsampled mask
    @timeit to "subsample mask" S_PHASER = subsample_mask(fitpar, fitopt, bs)

    # do a local fit on S_PHASER only
    S_orig = deepcopy(fitpar.S)

    fitpar.S[:] .= S_PHASER[:]

    @timeit to "prep phaser" local_fit(fitpar, fitopt)

    # save the local fit results, if desired
    if diagnostics
        ϕ, R2s, c, χ2 = deepcopy(fitpar.ϕ), deepcopy(fitpar.R2s), deepcopy(fitpar.c), deepcopy(fitpar.χ2)
        ML = (; ϕ, R2s, c, χ2)
    end

    # PHASER
    @timeit to "calc phaser" if N < ndims(fitpar.S)
        @assert N == 2 && ndims(fitpar.S) == 3 # what else? (we rely on that below)
        n_sl = size(fitpar.S)[end]

        PHASER = []

        for j in 1:n_sl
            data_sl = @views reshape(fitpar.data, size(fitpar.S)..., :)[:, :, j, :]
            S_sl = @views fitpar.S[:, :, j]
            fitpar_sl = fitPar(fitpar.gre, data_sl, S_sl)
            fitpar_sl.ϕ[:, :] .= @views fitpar.ϕ[:, :, j]
            S_orig_sl = @views S_orig[:, :, j]

            push!(PHASER, pure_phaser(fitpar_sl, fitopt, bs, S_orig_sl, diagnostics))

            fitpar.ϕ[:, :, j] .= fitpar_sl.ϕ
            fitpar.R2s[:, :, j] .= fitpar_sl.R2s
            fitpar.c[:, :, j] .= fitpar_sl.c
            fitpar.χ2[:, :, j] .= fitpar_sl.χ2
        end
    elseif N == ndims(fitpar.S)
        PHASER = pure_phaser(fitpar, fitopt, bs, S_orig, diagnostics)
    else
        error(string("N == ", N, " and ndims(data) == ", ndims(fitpar.S), " not supported."))
    end

    # restore fitpar and fitopt
    fitpar.S[:] .= S_orig[:]
    fitopt.optim = optim

    if locfit
        # repeat the local fit, based upon the PHASER ϕ
        @timeit to "local fit" local_fit(fitpar, fitopt)
    end

    diagnostics ? (; to, ML, PHASER, S_PHASER) : (; to)
end

"""
    pure_phaser(fitpar::FitPar{T}, fitopt::FitOpt, bs::BSmooth) where {T<:AbstractGREMultiEcho}

TBW
"""
function pure_phaser(fitpar::FitPar{T}, fitopt::FitOpt, bs::BSmooth, S_orig, diagnostics=false) where {T<:AbstractGREMultiEcho}
    # some useful aliases
    data = fitpar.data
    ϕ = fitpar.ϕ
    S = fitpar.S
    szS = size(S)

    # monitor timing
    to = TimerOutput()

    # local ML phase factor estimate
    z = ones(ComplexF64, szS)
    @. z[S] = @views exp(im * ϕ[S])

    # ======================================================================
    # Smooth phase map (linearized gradients)
    # ======================================================================

    @timeit to "lin. grad." begin
        print("Smooth phase map (linearized gradients) ... ")

        # calculate local difference map ∇z and associated masks
        (y, Sj) = calc_y(z, S)

        # MPI estimate
        ∇Bt∇B = calc_∇Bt∇B(bs, Sj)
        ∇Bty = calc_∇Btx(bs, Sj, imag.(y))
        λ_tikh = fitopt.λ_tikh * mean(real.(diag(∇Bt∇B)))
        c = (∇Bt∇B + λ_tikh * I) \ ∇Bty

        # first guess of smooth phase map ...
        ϕ_0 = zeros(szS)
        ϕ_0[S_orig] .= @views phase_map(bs, c)[S_orig]

        # calculate generic global offset (for best match with phase factor z)
        calc_generic_offset!(ϕ_0, z, S, S_orig)

        # remaining deviations
        # since we now also look at local phase factors as such (instead on their gradients only)
        # we must be careful not to generate artificial jumps at phase wraps in the scaled map
        z_0 = ones(ComplexF64, szS)
        @. z_0[S] = @views exp(im * ϕ_0[S])
        Δz_0 = ones(ComplexF64, szS)
        @. Δz_0[S] = @views z[S] / z_0[S]   # smooth at 2π phase wraps
        Δϕ_0 = zeros(szS)
        @. Δϕ_0[S] = @views angle(Δz_0[S])

        Δy_0 = calc_y_(Δz_0, Sj)

        println("done")
    end

    # ======================================================================
    # Improved smooth phase map (correct gradients)
    # ======================================================================

    @timeit to "corr. grad." begin
        print("Smooth phase map (correct gradients) ... ")

        # to avoid linearization errors, we now use the logarithm of phase factor difference
        # (assuming that the magnitude of the latter is < π/2!)
        ly = [zeros(ComplexF64, szS) for _ in 1:length(Δy_0)]
        for (ly_, Sj_, y_) in zip(ly, Sj, Δy_0)
            @. ly_[Sj_] = @views log(y_[Sj_] + 1)
        end

        # MPI estimate
        ∇Btly = calc_∇Btx(bs, Sj, imag.(ly))
        c = (∇Bt∇B + λ_tikh * I) \ ∇Btly
        ϕ_1 = zeros(szS)
        ϕ_1[S_orig] .= @views ϕ_0[S_orig] .+ phase_map(bs, c)[S_orig]

        # calculate generic global offset (for best match with phase factor z)
        calc_generic_offset!(ϕ_1, z, S, S_orig)

        # remaining deviations
        # since we now also look at local phase factors as such (instead on their gradients only)
        # we must be careful not to generate artificial jumps at phase wraps in the scaled map
        z_1 = ones(ComplexF64, szS)
        @. z_1[S] = @views exp(im * ϕ_1[S])
        Δz_1 = ones(ComplexF64, szS)
        @. Δz_1[S] = @views z[S] / z_1[S]   # smooth at 2π phase wraps
        Δϕ_1 = zeros(szS)
        @. Δϕ_1[S] = @views angle(Δz_1[S])

        Δy_1 = calc_y_(Δz_1, Sj)

        println("done.")
    end

    # ======================================================================
    # Balancing of agreement with phase factor and derivative
    # ======================================================================

    @timeit to "balance cost" begin
        print("Search for best cost function to describe data ... ")

        BtB = calc_BtB(bs, S)

        lz = zeros(ComplexF64, szS)
        @. lz[S] = @views log(Δz_1[S])

        Btlz = calc_Btx(bs, S, imag.(lz))

        for (ly_, Sj_, y_) in zip(ly, Sj, Δy_1)
            @. ly_[Sj_] = @views log(y_[Sj_] + 1)
        end

        ∇Btly = calc_∇Btx(bs, Sj, imag.(ly))

        A, a, B, b = BtB, Btlz, ∇Bt∇B, ∇Btly

        fitparλ = fitPar(fitpar.gre, data, S)
        set_num_phase_intervals(fitparλ, fitopt, 0)

        λ_opt, χ2_opt, λs, χ2s = GSS(
            χ2_λ_fun(fitparλ, fitopt, bs, A, a, B, b, ϕ_1), (0.0, 1.0), 1e-4; show_all=true)

        iλ = sortperm(λs)
        λs = λs[iλ]
        χ2s = χ2s[iλ]

        # take the best match and calculate the solution on S_orig
        fitpar.S[:] .= S_orig[:]

        χ2_λ_fun(fitpar, fitopt, bs, A, a, B, b, ϕ_1)(λ_opt)

        println("done.")
    end

    # return diagnostic information, if desired
    if diagnostics
        ϕ = deepcopy(ϕ)
        (; to, ϕ, ϕ_0, ϕ_1, Δϕ_0, Δϕ_1, Δy_0, Δy_1, ly, lz, λ_opt, χ2_opt, λs, χ2s, y, Sj)
    else
        (; to)
    end
end

"""
    smooth_projection!(ϕ::AbstractArray, fitpar::FitPar, fitopt::FitOpt, bs::BSmooth)

Return projection of `ϕ` on smooth subspace, defined by `bs`.
The required agreement is restricted to the mask `S`.
"""
function smooth_projection!(ϕ::AbstractArray, fitpar::FitPar, fitopt::FitOpt, bs::BSmooth)
    # check that size is ok
    @assert size(ϕ) == size(fitpar.S)

    # prepare Moore-Penrose pseudoinverse
    BtB = BtB(bs, fitpar.S)
    Btϕ = Btx(bs, fitpar.S, ϕ)

    # Calculate the Moore-Penrose inverse.
    # To address potential ill-posedness of the matrix ∇Bt∇B, we make use of Tikhonov regularization.
    # The resulting bias, as long as not extremely large, should not affect the subsequent refinement step.
    λ_tikh = fitopt.λ_tikh * mean(real.(diag(BtB)))
    c_mpi = (BtB .+ λ_tikh .* I) \ Btϕ

    # calculate phase maps for b == 0
    fill!(fitpar.ϕ, 0.0)
    fitpar.ϕ[fitpar.S] = @views phase_map(bs, real(c_mpi[1]), c_mpi[2:end])[fitpar.S]
end

#= 
==================================================================  
Auxiliary routines
==================================================================  
=#

function subsample_mask(fitpar::FitPar, fitopt::FitOpt, bs::BSmooth{N}) where {N}
    S = fitpar.S
    S_PHASER = deepcopy(fitpar.S)

    # target number of locations in mask, which contain a derivative in every direction
    fitopt.redundancy * Nfree(bs)
    Nsub = ceil(Int, min(fitopt.redundancy * Nfree(bs), 0.99typemax(Int)))

    if N < ndims(S)
        @assert N == 2 && ndims(S) == 3 # what else? (we rely on that below)
        n_sl = size(S)[end]

        for j in 1:n_sl
            S_PHASER_sl = @views S_PHASER[:, :, j]
            S_sl = @views S[:, :, j]
            calc_subsample_mask!(S_PHASER_sl, S_sl, Nsub)
        end
    elseif N == ndims(S)
        calc_subsample_mask!(S_PHASER, S, Nsub)
    else
        error(string("N == ", N, " and ndims(data) == ", ndims(fitpar.S), " not supported."))
    end

    S_PHASER
end

"""
    calc_subsample_mask!(Ssub, S, Nsub)

TBW
"""
function calc_subsample_mask!(Ssub, S, Nsub)
    # set dimension
    ndS = ndims(S)

    # set index ranges 
    ciS = CartesianIndices(S)
    fiS, liS = first(ciS), last(ciS)

    # unit step in each direction
    ejs = [CartesianIndex(ntuple(k -> k == j ? 1 : 0, ndS)) for j in 1:ndS]

    # initialize
    fill!(Ssub, false)

    # determine subset of data points with direct neighbours in every direction
    ej = ejs[1]

    for iS in fiS:liS-ej
        S[iS] && S[iS+ej] && (Ssub[iS] = true)
    end

    for ej in ejs[2:end]
        for iS in fiS:liS-ej
            !Ssub[iS] && continue
            !(S[iS] && S[iS+ej]) && (Ssub[iS] = false)
        end
    end

    # number of candidate locations
    Ncand = @views sum(Ssub[S])

    # reduce mask, if possible
    if Nsub < Ncand
        ciSca = CartesianIndices(Ssub)[Ssub]
        iSubs = randperm(Ncand)[1:Nsub]

        Ssub[S] .= false

        for i in iSubs
            ci = ciSca[i]
            Ssub[ci] = true
            for ej in ejs
                Ssub[ci+ej] = true
            end
        end
    end
end

"""
    calc_generic_offset!(ϕ, z, S)

TBW
"""
function calc_generic_offset!(ϕ, z, S, S_orig)
    # coefficient b
    b = @views angle(sum(z[S] .* exp.(-im .* ϕ[S])))

    # calculate the median of ϕ over S
    ϕ_med = @views median(ϕ[S]) + b

    # limit the median phase to [-π,π]
    while ϕ_med > π
        ϕ_med -= 2π
        b -= 2π
    end

    while ϕ_med < -π
        ϕ_med += 2π
        b += 2π
    end

    # add offset to ϕ
    @. ϕ[S_orig] += b

    # return offset
    b
end

"""
    calc_y(z::AbstractArray, S::AbstractArray)

TBW
"""
function calc_y(z::AbstractArray, S::AbstractArray)
    # check for consistency of arguments
    (ndS = ndims(S)) != ndims(z) && throw(ArgumentError("z and S not compatible"))

    # compute difference map ∇z
    (∇z, Sj) = ∇j(z, S)

    # to compute y, we multiply ∇z with conj(z)
    y = [zeros(ComplexF64, size(z)) for _ in 1:length(∇z)]

    # ciS = CartesianIndices(S)
    for (∇z_, Sj_, y_) in zip(∇z, Sj, y)
        @. y_[Sj_] = @views conj(z[Sj_]) * ∇z_[Sj_]
    end

    # result
    return (y, Sj)
end

"""
    calc_y_(z::AbstractArray, Sj::AbstractVector)

TBW
"""
function calc_y_(z::AbstractArray, Sj::AbstractVector)
    # compute difference map ∇z
    ∇z = ∇j_(z, Sj)

    # to compute y, we multiply ∇z with conj(z)
    y = [zeros(ComplexF64, size(z)) for _ in 1:length(∇z)]

    # ciS = CartesianIndices(S)
    for (∇z_, Sj_, y_) in zip(∇z, Sj, y)
        @. y_[Sj_] = @views conj(z[Sj_]) * ∇z_[Sj_]
    end

    # result
    return y
end

"""
    ∇j(A::AbstractArray, S::AbstractArray)

Compute the local finite difference of array `A` along all dimensions of `S`.
as defined in the article.

# Arguments

- `A::AbstractArray`: arbitrary array, for which the difference shall be computed. `eltype(A)` is only restricted in the sense that it must support subtraction.
- `S::AbstractArray`: boolean mask, where values of `A` are meaningful. Can be a conventional Array or a BitArray.

# Boundary conditions

- The array `A` can have more dimensions than `S`, but the condition `size(A)[1:ndims(S)] == size(S)` must always be satisfied.

# Return values

The function returns a tuple `(∇A, Sj)`:

- `∇A`: difference array
- `Sj`: corresponding mask array

with dimensions:

- `size(∇A) == (size(S), ndims(S), size(A)[ndims(S)+1:end])`
- `size(Sj) == (size(S), ndims(S))`

# Example

```jldoctext
A = rand(5, 6, 7, 8)
S = A[:,:,:,1] .> 0.1   # ndims(S) < ndims(A) is allowed

# calculate differences along all directions
(∇A, Sj) = ∇j(A, S)
```
"""
function ∇j(A::AbstractArray, S::AbstractArray)
    # check for consistency of arguments
    ndA, ndS, szA, szS = ndims(A), ndims(S), size(A), size(S)
    ndA < ndS && throw(ArgumentError("ndims(A) < ndims(S)"))
    szA[1:ndS] != szS && throw(ArgumentError("A and S not compatible"))

    # set index ranges and allocate space
    ciS = CartesianIndices(S)
    fiS, liS = first(ciS), last(ciS)

    # compute Sj
    Sj = [falses(szS...) for _ in 1:ndS]

    for (j, Sj_) in zip(1:ndS, Sj)
        ej = CartesianIndex(ntuple(k -> k == j ? 1 : 0, ndS))

        for iS in fiS:liS-ej
            S[iS] && S[iS+ej] && (Sj_[iS] = true)
        end
    end

    # result
    return (∇j_(A, Sj), Sj)
end

"""
    ∇j_(A::AbstractArray, Sj::AbstractVector)

Helper function of `∇j`.

# Arguments

- `A::AbstractArray`: defined as in `∇j`
- `Sj::AbstractVector`: boolean mask vector (format as in `∇j`, but possibly with different content)

# Return values

- `∇A`: vector of difference arrays
"""
function ∇j_(A::AbstractArray, Sj::AbstractVector)
    # check for consistency of arguments
    ndA, ndS, szA, szS = ndims(A), ndims(Sj[1]), size(A), size(Sj[1])
    ndA < ndS && throw(ArgumentError("ndims(A) < ndims(S)"))
    szA[1:ndS] != szS && throw(ArgumentError("A and S not compatible"))
    szE = szA[ndS+1:ndA]   # possible extra dimensions, not affected by the gradient

    # set index ranges and allocate space
    ciS, ciE = CartesianIndices(Sj[1]), CartesianIndices(szE)
    fiS, liS = first(ciS), last(ciS)
    ∇A = [zeros(eltype(A), szA...) for _ in 1:ndS]

    # compute the local difference, where possible
    for (j, ∇A_, Sj_) in zip(1:ndS, ∇A, Sj)
        ej = CartesianIndex(ntuple(k -> k == j ? 1 : 0, ndS))

        for iE in ciE
            for iS in fiS:liS-ej
                Sj_[iS] && (∇A_[iS, iE] = A[iS+ej, iE] - A[iS, iE])
            end
        end
    end

    # result
    return ∇A
end

"""
    χ2_λ_fun(fitparλ, fitopt, z, A, a, B, b, ϕ_1)

TBW
"""
function χ2_λ_fun(fitparλ, fitopt, bs, A, a, B, b, ϕ_1)
    λ -> begin
        @assert λ ≠ 0

        mat_λ = λ .* A
        @. mat_λ[2:end, 2:end] += (1 - λ) * B
        λ_tikh = fitopt.λ_tikh * mean(real.(diag(mat_λ)))
        vec_λ = λ .* a
        @. vec_λ[2:end] += (1 - λ) * b
        c = (mat_λ + λ_tikh * I) \ vec_λ
        fitparλ.ϕ[fitparλ.S] .= @views phase_map(bs, real(c[1]), c[2:end])[fitparλ.S] .+ ϕ_1[fitparλ.S]

        local_fit(fitparλ, fitopt)
        sum(fitparλ.χ2[fitparλ.S])
    end
end