using LinearAlgebra, ChunkSplitters, StatsBase, Optim, Random, TimerOutputs, Compat
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
- If `fitopt.locfit == true`, a final local fit based upon PH is performed.
## Arguments
- `fitpar::FitPar`: Fit parameters
- `fitopt::FitOpt`: Fit options
- `bs::BSmooth{N}`: Smooth subspace for PH
"""
function phaser(fitpar::FitPar, fitopt::FitOpt, bs::BSmooth{N}) where {N}
    # timing will always be monitored
    to = TimerOutput()

    # nonlinear optimization for PH?
    optim = fitopt.optim
    fitopt.optim = fitopt.optim_phaser

    # save original value
    n_ϕ = fitopt.n_ϕ

    # generate subsampled mask
    @timeit to "subsample mask" S = subsample_mask(fitpar, fitopt, bs)

    # do a local fit on S_PH only
    S_orig = deepcopy(fitpar.S)

    fitpar.S[:] .= S[:]

    @timeit to "prep phaser" local_fit(fitpar, fitopt)

    # save the local fit results, if desired
    if fitopt.diagnostics
        ϕ, R2s, c, χ2 = deepcopy(fitpar.ϕ), deepcopy(fitpar.R2s), deepcopy(fitpar.c), deepcopy(fitpar.χ2)
        noS = (!).(S)
        ML = (; ϕ, R2s, c, χ2, S, noS)
    end

    # PH

    # disable nonlinear optimization, when balancing local vs gradient based solution
    fitopt.optim = false

    @timeit to "calc phaser" if N < ndims(fitpar.S)
        @assert N == 2 && ndims(fitpar.S) == 3 # what else? (we rely on that below)
        n_sl = size(fitpar.S)[end]

        PH = []

        for j in 1:n_sl
            data_sl = @views reshape(fitpar.data, size(fitpar.S)..., :)[:, :, j, :]
            S_sl = @views fitpar.S[:, :, j]
            fitpar_sl = fitPar(fitpar.grePar, data_sl, S_sl)
            fitpar_sl.ϕ[:, :] .= @views fitpar.ϕ[:, :, j]
            S_orig_sl = @views S_orig[:, :, j]

            push!(PH, pure_phaser(fitpar_sl, fitopt, bs, S_orig_sl))

            fitpar.ϕ[:, :, j] .= fitpar_sl.ϕ
            fitpar.R2s[:, :, j] .= fitpar_sl.R2s
            fitpar.c[:, :, j] .= fitpar_sl.c
            fitpar.χ2[:, :, j] .= fitpar_sl.χ2
        end
    elseif N == ndims(fitpar.S)
        PH = pure_phaser(fitpar, fitopt, bs, S_orig)
    else
        error(string("N == ", N, " and ndims(data) == ", ndims(fitpar.S), " not supported."))
    end

    # restore fitpar and fitopt
    fitpar.S[:] .= S_orig[:]
    fitopt.optim = optim

    if fitopt.locfit
        # repeat the local fit, based upon the PH ϕ
        @timeit to "local fit" local_fit(fitpar, fitopt)
    end

    # restore fitpar and fitopt
    set_num_phase_intervals(fitpar, fitopt, n_ϕ)

    fitopt.diagnostics ? (; to, ML, PH) : (; to)
end

"""
    pure_phaser(fitpar::FitPar{T}, fitopt::FitOpt, bs::BSmooth) where {T<:AbstractGREMultiEcho}

TBW
"""
function pure_phaser(fitpar::FitPar{T}, fitopt::FitOpt, bs::BSmooth, S_orig) where {T<:AbstractGREMultiEcho}
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
    # Gradient-based estimation
    # ======================================================================

    @timeit to "gradent-based estimate" begin
        print("Gradient-based phase map ... ")

        # calculate local difference map ∇z and associated masks
        (y, Sj) = calc_y(z, S)

        # calculate principle value of complex logarithm
        ly = [zeros(ComplexF64, szS) for _ in 1:length(y)]
        for (ly_, Sj_, y_) in zip(ly, Sj, y)
            @. ly_[Sj_] = @views log(y_[Sj_] + 1)
        end

        # if desired, remove outliers from Sj
        Sj_wo = Sj
        ly_wo = ly
        aily_hist = Histogram[]
        aily_max = Float64[]
        for (ly_, Sj_) in zip(ly, Sj)
            # setting (2n)^(1/3) for the number bins in the histogram was motivated in 
            # https://doi.org/10.2307/2288074
            nbins = ceil(Int, (2 * sum(Sj_))^(1 / 3))
            # store differences
            aily = abs.(imag.(ly_[Sj_]))
            # boundaries of bin intervals
            edges = @views range(0.0, max(aily...), nbins + 1)
            # median over Sj
            med = @views median(aily)
            # due to dimensional arguments, the median should be in the first peak (starting at zero)
            # and should therefore not correspond to phase jumps at region boundaries
            # the index iemin can therefore be used as a starting point
            iemin = findfirst(e -> e > med, edges)
            # generate the histogram curve based upon the bins defined above
            push!(aily_hist, fit(Histogram, aily, edges))
            # ideally, we want to fully include the first peak (related to actual ϕ variations) and 
            # ignore any other peak(s) (associated with phase jumps)
            # to do so, we search for the first index, when the |Δy| histogram starts to rise again
            # (ideally, this should more or less lie well between the first two peaks)
            fifi = @views findfirst(x -> x > 0, aily_hist[end].weights[iemin+1:end] - aily_hist[end].weights[iemin:end-1])
            # define cutoff value
            # (left edge of the bin, where the histogram has its first minimum)
            push!(aily_max, edges[fifi+iemin-1])
            
            # remove outliers
            if fitopt.remove_outliers
                remove = aily .> aily_max[end]
                ciSj = CartesianIndices(Sj_)[Sj_]
                Sj_[ciSj[remove]] .= false
                ly_[ciSj[remove]] .= 0.0
            end
        end

        # MPI estimate
        ∇Bt∇B = calc_∇Bt∇B(bs, Sj)
        ∇Btly = calc_∇Btx(bs, Sj, imag.(ly))
        λ_tikh = fitopt.λ_tikh * mean(real.(diag(∇Bt∇B)))

        c = (∇Bt∇B + λ_tikh * I) \ ∇Btly

        # calculate phase map with median limited to (-π, π]
        ϕ_0 = zeros(szS)
        ϕ_0[S_orig] .= @views phase_map(bs, c)[S_orig]

        calc_generic_offset!(ϕ_0, z, S, S_orig)

        # remaining deviations
        # since we now also look at local phase factors as such (instead on their gradients only)
        # we must be careful not to generate artificial jumps at phase wraps in the scaled map
        z_0 = ones(ComplexF64, szS)
        @. z_0[S] = @views z[S] * exp(-im * ϕ_0[S])
        Δϕ_0 = zeros(szS)
        @. Δϕ_0[S] = @views angle(z_0[S])

        y_0 = calc_y_(z_0, Sj)

        println("done.")
    end

    # ======================================================================
    # Balancing of agreement with phase factor and derivative
    # ======================================================================

    @timeit to "balance cost" begin
        print("Search for best cost function to describe data ... ")

        BtB = calc_BtB(bs, S)

        lz_0 = zeros(ComplexF64, szS)
        @. lz_0[S] = @views log(z_0[S])

        Btlz = calc_Btx(bs, S, imag.(lz_0))

        ly_0 = [zeros(ComplexF64, szS) for _ in 1:length(y)]

        for (ly_, Sj_, y_) in zip(ly_0, Sj, y_0)
            @. ly_[Sj_] = @views log(y_[Sj_] + 1)
        end

        ∇Btly = calc_∇Btx(bs, Sj, imag.(ly_0))

        A, a, B, b = BtB, Btlz, ∇Bt∇B, ∇Btly

        fitparλ = fitPar(fitpar.grePar, data, S)
        set_num_phase_intervals(fitparλ, fitopt, 0)

        λ_opt, χ2_opt, λs, χ2s = GSS(
            χ2_λ_fun(fitparλ, fitopt, bs, A, a, B, b, ϕ_0), (0.0, 1.0), 1e-4; show_all=true)

        iλ = sortperm(λs)
        λs = λs[iλ]
        χ2s = χ2s[iλ]

        # take the best match and calculate the solution on S_orig
        fitpar.S[:] .= S_orig[:]

        χ2_λ_fun(fitpar, fitopt, bs, A, a, B, b, ϕ_0)(λ_opt)

        # make sure that the phase median over S lies within [-π, π]
        median_shift!(fitpar.ϕ, fitpar.S)

        println("done.")
    end

    # return diagnostic information, if desired
    if fitopt.diagnostics
        ϕ = deepcopy(ϕ)
        (; to, ϕ, ϕ_0, Δϕ_0, y_0, ly, ly_wo, lz_0, ly_0,
            BtB, ∇Bt∇B, 
            λ_opt, χ2_opt, λs, χ2s, y, Sj, Sj_wo, aily_hist, aily_max)
    else
        (; to)
    end
end

"""
    smooth_projection!(ϕ::AbstractArray, S::AbstractArray, bs::BSmooth; λ_tikh = 1e-6)

Return projection of `ϕ` on smooth subspace, defined by `bs`.
The required agreement is restricted to the mask `S`.
"""
function smooth_projection!(ϕ::AbstractArray, S::AbstractArray, bs::BSmooth; λ_tikh=1e-6)
    # check that size is ok
    @assert size(ϕ) == size(S)

    # prepare Moore-Penrose pseudoinverse
    BtB = calc_BtB(bs, S)
    Btϕ = calc_Btx(bs, S, ϕ)

    # Calculate the Moore-Penrose inverse.
    # To address potential ill-posedness of the matrix ∇Bt∇B, we make use of Tikhonov regularization.
    # The resulting bias, as long as not extremely large, should not affect the subsequent refinement step.
    λ_tikh *= mean(real.(diag(BtB)))
    c_mpi = (BtB + λ_tikh .* I) \ Btϕ

    # calculate phase maps for b == 0
    ϕ[S] = @views phase_map(bs, real(c_mpi[1]), c_mpi[2:end])[S]
end

#= 
==================================================================  
Auxiliary routines
==================================================================  
=#

"""
    subsample_mask(fitpar::FitPar, fitopt::FitOpt, bs::BSmooth{N}) where {N}

TBW
"""
function subsample_mask(fitpar::FitPar, fitopt::FitOpt, bs::BSmooth{N}) where {N}
    S = fitpar.S

    fitopt.redundancy == Inf && return S

    S_PH = deepcopy(S)

    # target number of locations in mask, which contain a derivative in every direction
    fitopt.redundancy * Nfree(bs)
    Nsub = ceil(Int, min(fitopt.redundancy * Nfree(bs), 0.99typemax(Int)))

    if N < ndims(S)
        @assert N == 2 && ndims(S) == 3 # what else? (we rely on that below)
        n_sl = size(S)[end]

        for j in 1:n_sl
            S_PH_sl = @views S_PH[:, :, j]
            S_sl = @views S[:, :, j]
            calc_subsample_mask!(S_PH_sl, S_sl, Nsub, fitopt)
        end
    elseif N == ndims(S)
        calc_subsample_mask!(S_PH, S, Nsub, fitopt)
    else
        error(string("N == ", N, " and ndims(data) == ", ndims(fitpar.S), " not supported."))
    end

    S_PH
end

"""
    calc_subsample_mask!(Ssub, S, Nsub)

TBW
"""
function calc_subsample_mask!(Ssub, S, Nsub, fitopt)
    # check for correct setting
    @assert fitopt.subsampling ∈ (:fibonacci, :random)

    # set index ranges 
    ciS = CartesianIndices(S)
    fiS, liS = first(ciS), last(ciS)

    # unit step in each direction
    ndS, szS = ndims(S), size(S)
    ejs = [CartesianIndex(ntuple(k -> k == j ? 1 : 0, ndS)) for j in 1:ndS]

    # set Ssub equal to S except for last entries in each direction
    # (required, since we take derivatives below)
    fill!(Ssub, false)
    @views Ssub[fiS:(liS-fiS)] .= S[fiS:(liS-fiS)]

    # subset of S, from which derivatives in all directions can be taken
    for ej in ejs
        @views Ssub[fiS:(liS-fiS)] .&= S[fiS+ej:(liS-fiS+ej)]
    end

    # this yields the number of candidate locations
    Ncand = @views sum(Ssub[S])

    # reduce mask, if possible
    if Nsub < Ncand
        if fitopt.subsampling == :fibonacci
            # This approach reduces clustering, observed by conventional random sampling. 
            # One way to do so would be something like Poisson disk sampling, but this is
            # not easy to implement efficiently. We therefore use the multidimensional golden
            # means sampling, as proposed by Peter G. Anderson:
            # https://doi.org/10.1007/978-94-011-2058-6_1

            # candidate locations
            Scand = deepcopy(Ssub)

            # generate ndS-dimensional golden ratios
            (x, _) = GSS(x -> abs(x * (x + 1)^ndS - 1), (0, 1), 1e-10)
            z = [x * (x + 1)^n for n in 0:ndS-1]

            # size of the mask (apart from the outermost lines)
            szS1 = szS .- 1

            # number of found locations
            found = 0

            # location to look at
            loc = ones(ndS)

            # reset Ssub
            fill!(Ssub, false)

            # subsampling
            while found < Nsub
                # location to look at
                loc = mod.(loc .+ z, 1)
                iloc = ceil.(Int, loc .* szS1)
                iloc[iloc.==0] .= 1
                iloc = min.(iloc, szS1)
                ci = CartesianIndex(iloc...)

                if Scand[ci]
                    Ssub[ci] = true

                    for ej in ejs
                        Ssub[ci+ej] = true
                    end

                    found += 1
                end
            end
        elseif fitopt.subsampling == :random
            ciSca = CartesianIndices(Ssub)[Ssub]
            iSubs = randperm(fitopt.rng, Ncand)[1:Nsub]
            fill!(Ssub, false)

            for i in iSubs
                ci = ciSca[i]

                Ssub[ci] = true

                for ej in ejs
                    Ssub[ci+ej] = true
                end
            end
        else
            error(string("Unsupported argument: fitopt.subsampling == ", fitopt.subsampling))
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
    median_shift!(ϕ, S)

TBW
"""
function median_shift!(ϕ, S)
    # calculate the median of ϕ over S
    ϕ_med = @views median(ϕ[S])
    b = 0.0

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
    @. ϕ[S] += b

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
function χ2_λ_fun(fitparλ, fitopt, bs, A, a, B, b, ϕ_0)
    λ -> begin
        @assert λ ≠ 0

        mat_λ = λ .* A
        @. mat_λ[2:end, 2:end] += (1 - λ) * B
        λ_tikh = fitopt.λ_tikh * mean(real.(diag(mat_λ)))
        vec_λ = λ .* a
        @. vec_λ[2:end] += (1 - λ) * b
        c = (mat_λ + λ_tikh * I) \ vec_λ
        fitparλ.ϕ[fitparλ.S] .= @views phase_map(bs, real(c[1]), c[2:end])[fitparλ.S] .+ ϕ_0[fitparλ.S]

        local_fit(fitparλ, fitopt)
        sum(fitparλ.χ2[fitparλ.S])
    end
end