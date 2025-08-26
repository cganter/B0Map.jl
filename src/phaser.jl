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
function phase_map(bs::BSmooth, b::Float64, c::AbstractVector, to::TimerOutput=TimerOutput())
    b .+ phase_map(bs, c, to)
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
    phaser!(fitpar::FitPar, fitopt::FitOpt, bs::BSmooth{N}) where {N}

Take the data set and apply the PHASER algorithm.

## What it does
- First, a local fit is performed.
- If a smooth basis `bs` is supplied, the result is refined by subspace-based regularization.
- If `fitopt.locfit == true`, a final local fit based upon PH is performed.
## Arguments
- `fitpar::FitPar`: Fit parameters
- `fitopt::FitOpt`: Fit options
- `bs::BSmooth{N}`: Smooth subspace for PH
"""
function phaser!(fitpar::FitPar, fitopt::FitOpt, bs::BSmooth{N}) where {N}
    # timing will always be monitored
    to = TimerOutput()

    # 2d PHASER may be reasonable for 3d data sets (e.g. multi-slice imaging)
    @timeit to "calc phaser" if N < ndims(fitpar.S)
        @assert N == 2 && ndims(fitpar.S) == 3 # what else? (we rely on that below)
        n_sl = size(fitpar.S)[end]

        PH = []

        for j in 1:n_sl
            data_2d = @views reshape(fitpar.data, size(fitpar.S)..., :)[:, :, j, :]
            S_2d = @views fitpar.S[:, :, j]

            fitpar_2d = fitPar(fitpar.grePar, data_2d, S_2d)

            push!(PH, phaser_fire!(fitpar_2d, fitopt, bs, to))

            fitpar.ϕ[:, :, j] .= fitpar_2d.ϕ
            fitpar.R2s[:, :, j] .= fitpar_2d.R2s
            fitpar.c[:, :, j] .= fitpar_2d.c
            fitpar.χ2[:, :, j] .= fitpar_2d.χ2
        end
    elseif N == ndims(fitpar.S)
        PH = phaser_fire!(fitpar, fitopt, bs, to)
    else
        error(string("N == ", N, " and ndims(data) == ", ndims(fitpar.S), " not supported."))
    end

    # return timing and possibly further diagnostics
    fitopt.diagnostics ? (; to, PH) : (; to)
end

"""
    phaser_fire!(fitpar::FitPar{T}, fitopt::FitOpt, bs::BSmooth) where {T<:AbstractGREMultiEcho}

Actual implementation of PHASER
"""
function phaser_fire!(fitpar::FitPar{T}, fitopt::FitOpt, bs::BSmooth, to::TimerOutput) where {T<:AbstractGREMultiEcho}
    # ======================================================================
    # Generate subsampling masks
    # ======================================================================

    @timeit to "generate subsampling masks" begin
        print("Generate subsampling masks ... ")

        (S, S_c, S_t, Sj_wo, Sj_c_wo, Sj_t_wo) = subsample_mask(fitpar, fitopt, bs)
        szS = size(S)
        ndS = ndims(S)
        
        frac_S = sum(S) / sum(fitpar.S)

        println("done.")
    end

    # ======================================================================
    # Initial local optimization
    # ======================================================================

    @timeit to "initial local optimization" begin
        print("Initial local optimization ... ")

        fitpar_ML = fitPar(fitpar.grePar, fitpar.data, S)
        fitopt_ML = deepcopy(fitopt)
        fitopt_ML.optim = fitopt.optim_phaser

        @timeit to "local fit" begin
            local_fit(fitpar_ML, fitopt_ML)
        end

        ϕ_ML = fitpar_ML.ϕ
        R2s_ML = fitpar_ML.R2s
        c_ML = fitpar_ML.c
        χ2_ML = fitpar_ML.χ2

        # local ML phase factor estimate
        z = ones(ComplexF64, szS)
        @. z[S] = @views exp(im * ϕ_ML[S])

        # calculate u
        y = calc_y_(z, Sj_wo)
        u_wo = [zeros(szS) for _ in 1:length(y)]
        for (u_, Sj_, y_) in zip(u_wo, Sj_wo, y)
            @. u_[Sj_] = @views imag(log(y_[Sj_] + 1))
        end

        println("done.")
    end

    # ======================================================================
    # Remove outliers from Sj
    # ======================================================================

    @timeit to "remove outliers" begin
        print("Remove outliers ... ")

        # first we determine the maximally allowed u in each direction        
        au_hist = Histogram[]
        au_max = Float64[]

        for (u_, Sj_) in zip(u_wo, Sj_wo)
            # setting (2n)^(1/3) ("Rice rule") for the number bins in the histogram was motivated in 
            # https://doi.org/10.2307/2288074
            # we take this value (apart from dropping the factor 2 to get an even smoother histogram) 
            nbins = ceil(Int, (2 * sum(Sj_))^(1 / 3))
            # store differences
            au = abs.(u_[Sj_])
            # boundaries of bin intervals
            edges = @views range(0.0, max(au...), nbins + 1)
            # median over Sj
            med = @views median(au)
            # the first histogram peak (starting from zero) corresponds to ∇ξ = 0 (see article)
            # and due to dimensional arguments, the majority u's should be part of it.
            # the index `iemin` of the median can therefore be used as a starting point
            iemin = findfirst(e -> e > med, edges)
            # generate the histogram curve based upon the bins defined above
            push!(au_hist, fit(Histogram, au, edges))
            # it is difficult to formulate a general criterion, how to choose a max. allowed
            # value of `abs(u)`. Assuming that there is a relevant second peak, one can search
            # for the minimum of the histogram for `u > median(abs(u))`
            # if there is outlier, this should autmatically include the relevant part of the 
            # histogram (which a purely local derivative cannot provide with similar stability)
            fimi = @views findmin(au_hist[end].weights[iemin+1:end])[2]
            # define cutoff value
            push!(au_max, edges[fimi+iemin])
        end

        # remove outliers, if desired
        if fitopt.remove_outliers
            Sj = BitArray[]
            Sj_c = [falses(szS) for _ in 1:ndS]
            Sj_t = [falses(szS) for _ in 1:ndS]
            u_c = [zeros(szS) for _ in 1:ndS]
            u_t = [zeros(szS) for _ in 1:ndS]
            u = [zeros(szS) for _ in 1:ndS]

            for (Sj_c_, Sj_t_, Sj_c_wo_, Sj_t_wo_, u_, uc_, ut_, u_wo_, au_max_) in
                zip(Sj_c, Sj_t, Sj_c_wo, Sj_t_wo, u, u_c, u_t, u_wo, au_max)

                @. Sj_c_[Sj_c_wo_] = abs(u_wo_[Sj_c_wo_]) < au_max_
                @. Sj_t_[Sj_t_wo_] = abs(u_wo_[Sj_t_wo_]) < au_max_

                push!(Sj, Sj_c_ .| Sj_t_)

                @. u_[Sj_c_] = u_wo_[Sj_c_]
                @. uc_[Sj_c_] = u_wo_[Sj_c_]
                @. u_[Sj_t_] = u_wo_[Sj_t_]
                @. ut_[Sj_t_] = u_wo_[Sj_t_]
            end
        else
            Sj = Sj_wo
            Sj_c = Sj_c_wo
            Sj_t = Sj_t_wo
            u = u_wo
            u_c = [zeros(szS) for _ in 1:ndS]
            u_t = [zeros(szS) for _ in 1:ndS]

            for (Sj_c_wo_, Sj_t_wo_, u_, uc_, ut_, u_wo_) in
                zip(Sj_c_wo, Sj_t_wo, u, u_c, u_t, u_wo)

                @. u_[Sj_c_wo_] = u_wo_[Sj_c_wo_]
                @. uc_[Sj_c_wo_] = u_wo_[Sj_c_wo_]
                @. u_[Sj_t_wo_] = u_wo_[Sj_t_wo_]
                @. ut_[Sj_t_wo_] = u_wo_[Sj_t_wo_]
            end
        end

        println("done.")
    end

    # ======================================================================
    # Gradient-based estimation
    # ======================================================================

    @timeit to "gradient-based estimate" begin
        print("Gradient-based estimate ... ")

        @timeit to "prep matrices" begin
            # MPI estimate
            ∇Bt∇B = calc_∇Bt∇B(bs, Sj, to)
            ∇Btu = calc_∇Btx(bs, Sj, u, to)

            ∇Bt∇B_c = calc_∇Bt∇B(bs, Sj_c, to)
            ∇Btu_c = calc_∇Btx(bs, Sj_c, u_c, to)
            utu_c = sum([sum(abs2.(u_[Sj_])) for (u_, Sj_) in zip(u_c, Sj_c)])

            ∇Bt∇B_t = calc_∇Bt∇B(bs, Sj_t, to)
            ∇Btu_t = calc_∇Btx(bs, Sj_t, u_t, to)
            utu_t = sum([sum(abs2.(u_[Sj_])) for (u_, Sj_) in zip(u_t, Sj_t)])
        end

        tikh_grad = calc_μ_tikh(∇Bt∇B_c, ∇Btu_c, utu_c, ∇Bt∇B_t, ∇Btu_t, utu_t, to)

        c = (∇Bt∇B + tikh_grad.μ * I) \ ∇Btu

        # calculate phase map with median limited to (-π, π]
        ϕ0 = zeros(szS)
        ϕ0[fitpar.S] .= @views phase_map(bs, c, to)[fitpar.S]

        calc_generic_offset!(ϕ0, z, S, fitpar.S)

        # remaining deviations
        # since we now also look at local phase factors as such (instead on their gradients only)
        # we must be careful not to generate artificial jumps at phase wraps in the scaled map
        z0 = ones(ComplexF64, szS)
        @. z0[S] = @views z[S] * exp(-im * ϕ0[S])
        Δϕ0 = zeros(szS)
        @. Δϕ0[S] = @views angle(z0[S])

        y0 = calc_y_(z0, Sj)

        println("done.")
    end

    # ======================================================================
    # Balancing of agreement with phase factor and derivative
    # ======================================================================

    if fitopt.balance
        @timeit to "balance local vs gradient" begin
            print("Search for best cost function to describe data ... ")

            @timeit to "prep matrices" begin
                BtB = calc_BtB(bs, S, to)
                BtB_c = calc_BtB(bs, S_c, to)
                BtB_t = calc_BtB(bs, S_t, to)

                ilz0 = zeros(szS)
                ilz0_c = zeros(szS)
                ilz0_t = zeros(szS)

                @. ilz0[S] = @views imag(log(z0[S]))
                @. ilz0_c[S_c] = @views ilz0[S_c]
                @. ilz0_t[S_t] = @views ilz0[S_t]

                z0tz0_c = sum(abs2.(ilz0[S_c]))
                z0tz0_t = sum(abs2.(ilz0[S_t]))

                Btz0 = calc_Btx(bs, S, ilz0, to)
                Btz0_c = calc_Btx(bs, S_c, ilz0_c, to)
                Btz0_t = calc_Btx(bs, S_t, ilz0_t, to)

                u0 = [zeros(szS) for _ in 1:ndS]
                u0_c = [zeros(szS) for _ in 1:ndS]
                u0_t = [zeros(szS) for _ in 1:ndS]

                for (u_, uc_, ut_, Sj_, Sj_c_, Sj_t_, y_) in zip(u0, u0_c, u0_t, Sj, Sj_c, Sj_t, y0)
                    @. u_[Sj_] = @views imag(log(y_[Sj_] + 1))
                    @. uc_[Sj_c_] = @views u_[Sj_c_]
                    @. ut_[Sj_t_] = @views u_[Sj_t_]
                end

                u0tu0_c = sum([sum(abs2.(u_[Sj_])) for (u_, Sj_) in zip(u0_c, Sj_c)])
                u0tu0_t = sum([sum(abs2.(u_[Sj_])) for (u_, Sj_) in zip(u0_t, Sj_t)])

                ∇Btu0 = calc_∇Btx(bs, Sj, u0, to)
                ∇Btu0_c = calc_∇Btx(bs, Sj_c, u0_c, to)
                ∇Btu0_t = calc_∇Btx(bs, Sj_t, u0_t, to)

                fitparλ = fitPar(fitpar.grePar, fitpar.data, S)
                fitoptλ = deepcopy(fitopt)
                fitoptλ.optim = fitopt.optim_balance
                set_num_phase_intervals(fitparλ, fitoptλ, 0)

                tikh_λ = NamedTuple[]
            end

            χ2_λ_fun = create_χ2_λ_fun(fitparλ, fitoptλ, bs,
                BtB, Btz0, ∇Bt∇B, ∇Btu0,
                BtB_c, Btz0_c, z0tz0_c, ∇Bt∇B_c, ∇Btu0_c, u0tu0_c,
                BtB_t, Btz0_t, z0tz0_t, ∇Bt∇B_t, ∇Btu0_t, u0tu0_t,
                ϕ0, tikh_λ, to)

            @timeit to "GSS search λ" begin
                λ_opt, χ2_opt, λs, χ2s = GSS(χ2_λ_fun, (0.0, 1.0), 1e-4; show_all=true)
            end

            # take the best match and calculate the solution on fitpar.S
            create_χ2_λ_fun(fitpar, fitoptλ, bs,
                BtB, Btz0, ∇Bt∇B, ∇Btu0,
                BtB_c, Btz0_c, z0tz0_c, ∇Bt∇B_c, ∇Btu0_c, u0tu0_c,
                BtB_t, Btz0_t, z0tz0_t, ∇Bt∇B_t, ∇Btu0_t, u0tu0_t,
                ϕ0, tikh_λ, to)(λ_opt)

            @timeit to "global phase shift" begin
                # make sure that the phase median over S lies within [-π, π]
                median_shift!(fitpar.ϕ, fitpar.S)
            end

            println("done.")
        end
    else
        fitpar.ϕ[:] .= @views ϕ0[:]

        BtB = BtB_c = BtB_t = ilz0 = u0 =
            λ_opt = χ2_opt = λs = χ2s = tikh_λ = nothing
    end
    
    # Save the PHASER map
    ϕ = deepcopy(fitpar.ϕ)

    # ======================================================================
    # Final local fit, if desired
    # ======================================================================

    if fitopt.locfit
        @timeit to "final local fit" begin
            fitopt_loc = deepcopy(fitopt)
            set_num_phase_intervals(fitpar, fitopt_loc, 0)
            
            local_fit(fitpar, fitopt_loc)
            
            ϕ_loc = fitpar.ϕ
            R2s_loc = fitpar.R2s
        end
    else
        ϕ_loc = R2s_loc = nothing
    end

    # return diagnostic information, if desired
    if fitopt.diagnostics
        (; ϕ_ML, R2s_ML, c_ML, χ2_ML, ϕ_loc, R2s_loc,
            ϕ, ϕ0, Δϕ0, y0, u, u_c, u_t, u_wo, ilz0, u0,
            BtB, ∇Bt∇B, ∇Btu, ∇Bt∇B_c, ∇Btu_c, ∇Bt∇B_t, ∇Btu_t,
            λ_opt, χ2_opt, λs, χ2s, y,
            S, S_c, S_t,
            Sj_wo, Sj_c_wo, Sj_t_wo,
            Sj, Sj_c, Sj_t,
            au_hist, au_max,
            tikh_grad, tikh_λ, frac_S)
    else
        nothing
    end
end

"""
    smooth_projection!(ϕ::AbstractArray, S::AbstractArray, bs::BSmooth; μ_tikh = 1e-6)

Return projection of `ϕ` on smooth subspace, defined by `bs`.
The required agreement is restricted to the mask `S`.
"""
function smooth_projection!(ϕ::AbstractArray, S::AbstractArray, bs::BSmooth; μ_tikh=1e-6)
    # check that size is ok
    @assert size(ϕ) == size(S)

    # prepare Moore-Penrose pseudoinverse
    BtB = calc_BtB(bs, S)
    Btϕ = calc_Btx(bs, S, ϕ)

    # Calculate the Moore-Penrose inverse.
    # To address potential ill-posedness of the matrix ∇Bt∇B, we make use of Tikhonov regularization.
    # The resulting bias, as long as not extremely large, should not affect the subsequent refinement step.
    μ_tikh *= max(real.(diag(BtB))...)
    c_mpi = (BtB + μ_tikh .* I) \ Btϕ

    # calculate phase maps for b == 0
    ϕ[S] = @views phase_map(bs, real(c_mpi[1]), c_mpi[2:end], to)[S]
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
    # some initializations
    S = deepcopy(fitpar.S)
    ndS, szS = ndims(S), size(S)
    ciS = CartesianIndices(S)
    fiS, liS = first(ciS), last(ciS)

    # define Sj ⊂ S such that derivatives can be taken in direction j
    ejs = [CartesianIndex(ntuple(k -> k == j ? 1 : 0, ndS)) for j in 1:ndS]

    # candidates
    Sj_cand = falses(szS)
    Sj_cand[fiS:(liS-fiS)] .= @views S[fiS:(liS-fiS)]

    for ej in ejs
        Sj_cand[fiS:(liS-fiS)] .&= @views S[fiS+ej:(liS-fiS+ej)]
    end

    # target number of locations in mask, which contain a derivative in every direction
    N_sub = ceil(Int, min(fitopt.redundancy * Nfree(bs), 0.99typemax(Int)))
    N_cand = sum(Sj_cand)

    if N_sub < N_cand
        (Sj_, ciSj) = subsample_mask(N_sub, Sj_cand, fitopt)

        Nc = ceil(Int, (1.0 - fitopt.test_frac) * N_sub)

        Sj_c_ = falses(szS)
        Sj_c_[ciSj[1:Nc]] .= true

        Sj_t_ = falses(szS)
        Sj_t_[ciSj[Nc+1:end]] .= true
    else
        Sj_ = deepcopy(Sj_cand)
        Nt = ceil(Int, fitopt.test_frac * N_cand)
        (Sj_t_, _) = subsample_mask(Nt, Sj_cand, fitopt)
        Sj_c_ = deepcopy(Sj_cand)
        Sj_c_[Sj_t_] .= false
    end

    Sj_c = [Sj_c_ for _ in 1:N]
    Sj_t = [Sj_t_ for _ in 1:N]
    Sj = [Sj_ for _ in 1:N]

    S_c = falses(szS)
    S_t = falses(szS)

    for (Sj_c_, Sj_t_, ej) in zip(Sj_c, Sj_t, ejs)
        S_c .|= Sj_c_
        @views S_c[fiS+ej:(liS-fiS+ej)] .|= Sj_c_[fiS:(liS-fiS)]

        S_t .|= Sj_t_
        @views S_t[fiS+ej:(liS-fiS+ej)] .|= Sj_t_[fiS:(liS-fiS)]
    end

    S = S_c .| S_t

    (S, S_c, S_t, Sj, Sj_c, Sj_t)
end

"""
    subsample_mask(N, S_cand, subsampling)

TBW
"""
function subsample_mask(N, S_cand, fitopt)
    # check for correct setting
    @assert fitopt.subsampling ∈ (:fibonacci, :random)

    # some initializations
    ndS, szS = ndims(S_cand), size(S_cand)
    ciS = CartesianIndices(S_cand)

    # found locations
    S = falses(szS)
    ciSs = eltype(ciS)[]

    # reduce mask, if possible
    if fitopt.subsampling == :fibonacci
        # This approach reduces clustering, observed by conventional random sampling. 
        # One way to do so would be something like Poisson disk sampling, but this is
        # not easy to implement efficiently. We therefore use the multidimensional golden
        # means sampling, as proposed by Peter G. Anderson:
        # https://doi.org/10.1007/978-94-011-2058-6_1

        # generate ndS-dimensional golden ratios
        (x, _) = GSS(x -> abs(x * (x + 1)^ndS - 1), (0, 1), 1e-10)
        z = [x * (x + 1)^n for n in 0:ndS-1]

        # number of found locations
        found = 0

        # location to look at
        loc = ones(ndS)

        # subsampling
        while found < N
            # location to look at
            loc = mod.(loc .+ z, 1)
            iloc = ceil.(Int, loc .* szS)
            iloc[iloc.==0] .= 1
            iloc = min.(iloc, szS)
            ci = CartesianIndex(iloc...)

            if S_cand[ci]
                S[ci] = true
                push!(ciSs, ci)

                found += 1
            end
        end
    elseif fitopt.subsampling == :random
        ciSca = CartesianIndices(S_cand)[S_cand]
        iS = randperm(fitopt.rng, sum(S_cand))[1:N]

        for i in iS
            ci = ciSca[i]

            S[ci] = true
            push!(ciSs, ci)
        end
    else
        error(string("Unsupported argument: subsampling == ", fitopt.subsampling))
    end

    (S, ciSs)
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
    χ2_λ_fun(fitparλ, fitopt, bs, A, a, B, b, ϕ_0, μ_tikh)

TBW
"""
function create_χ2_λ_fun(fitparλ, fitopt, bs,
    A, a, B, b,
    A_c, a_c, xtx_c, B_c, b_c, yty_c,
    A_t, a_t, xtx_t, B_t, b_t, yty_t,
    ϕ_0, tikh_λ, to)

    @timeit to "create_χ2_λ_fun" begin
        λ -> let fitparλ = fitparλ, fitopt = fitopt, bs = bs,
            A = A, a = a, B = B, b = b,
            A_c = A_c, a_c = a_c, xtx_c = xtx_c, B_c = B_c, b_c = b_c, yty_c = yty_c,
            A_t = A_t, a_t = a_t, xtx_t = xtx_t, B_t = B_t, b_t = b_t, yty_t = yty_t,
            ϕ_0 = ϕ_0

            @assert λ ≠ 0

            @timeit to "prep matrices" begin
                mat_λ = λ .* A
                @. mat_λ[2:end, 2:end] += (1 - λ) * B
                vec_λ = λ .* a
                @. vec_λ[2:end] += (1 - λ) * b

                mat_λ_c = λ .* A_c
                @. mat_λ_c[2:end, 2:end] += (1 - λ) * B_c
                vec_λ_c = λ .* a_c
                @. vec_λ_c[2:end] += (1 - λ) * b_c
                xytxy_λ_c = λ * xtx_c + (1 - λ) * yty_c

                mat_λ_t = λ .* A_t
                @. mat_λ_t[2:end, 2:end] += (1 - λ) * B_t
                vec_λ_t = λ .* a_t
                @. vec_λ_t[2:end] += (1 - λ) * b_t
                xytxy_λ_t = λ * xtx_t + (1 - λ) * yty_t
            end

            push!(tikh_λ, calc_μ_tikh(mat_λ_c, vec_λ_c, xytxy_λ_c, mat_λ_t, vec_λ_t, xytxy_λ_t, to))

            c = (mat_λ + tikh_λ[end].μ * I) \ vec_λ
            fitparλ.ϕ[fitparλ.S] .= @views phase_map(bs, real(c[1]), c[2:end], to)[fitparλ.S] .+ ϕ_0[fitparλ.S]

            @timeit to "local fit" begin
                local_fit(fitparλ, fitopt)
            end

            sum(fitparλ.χ2[fitparλ.S])
        end
    end
end

"""
    χ2_λ_fun(fitparλ, fitopt, bs, A, a, B, b, ϕ_0, μ_tikh)

TBW
"""
function create_χ2_λ_fun_old(fitparλ, fitopt, bs, A, a, B, b, ϕ_0, μ_tikh)
    λ -> let fitparλ = fitparλ, fitopt = fitopt, bs = bs,
        A = A, a = a, B = B, b = b,
        ϕ_0 = ϕ_0, μ_tikh = μ_tikh

        @assert λ ≠ 0

        mat_λ = λ .* A
        @. mat_λ[2:end, 2:end] += (1 - λ) * B
        vec_λ = λ .* a
        @. vec_λ[2:end] += (1 - λ) * b

        #c = (mat_λ + (μ_tikh * max(real.(diag(mat_λ))...)) * I) \ vec_λ
        c = (mat_λ + μ_tikh * I) \ vec_λ
        fitparλ.ϕ[fitparλ.S] .= @views phase_map(bs, real(c[1]), c[2:end])[fitparλ.S] .+ ϕ_0[fitparλ.S]

        local_fit(fitparλ, fitopt)
        sum(fitparλ.χ2[fitparλ.S])
    end
end

"""
    split_OCV_cost(AtA_c, Atb_c, AtA_t, Atb_t, btb_t)

TBW
"""
function split_OCV_cost(AtA_c, Atb_c, μ_scale, AtA_t, Atb_t, btb_t)

    log10_μ -> let AtA_c = AtA_c, Atb_c = Atb_c,
        AtA_t = AtA_t, Atb_t = Atb_t, btb_t = btb_t,
        μ_scale = μ_scale

        μ = μ_scale * 10.0^log10_μ

        x_μ = (AtA_c + μ * I) \ Atb_c

        return real(btb_t + (x_μ' * AtA_t - 2Atb_t') * x_μ)
    end
end

"""
    calc_μ_tikh(AtA_c, Atb_c, btb_c, N_c, AtA_t, Atb_t, btb_t, N_t,
    μs_rel = logrange(1e-10, 1e3, 1000))

TBW
"""
function calc_μ_tikh(AtA_c, Atb_c, btb_c, AtA_t, Atb_t, btb_t, to,
    μ_log10_rng=(-10, 3), μ_log10_acc=0.01)

    @timeit to "calc Tikhonov factor" begin
        μ_scale_c = max(real.(diag(AtA_c))...)
        μ_scale_t = max(real.(diag(AtA_t))...)

        ocv_ct = split_OCV_cost(AtA_c, Atb_c, μ_scale_c, AtA_t, Atb_t, btb_t)
        ocv_tc = split_OCV_cost(AtA_t, Atb_t, μ_scale_t, AtA_c, Atb_c, btb_c)

        log10_μ_ct, χ2_opt_ct, log10_μs_ct, χ2s_ct =
            GSS(ocv_ct, μ_log10_rng, μ_log10_acc; show_all=true)

        log10_μ_tc, χ2_opt_tc, log10_μs_tc, χ2s_tc =
            GSS(ocv_tc, μ_log10_rng, μ_log10_acc; show_all=true)

        log10_μ = 0.5 * (log10_μ_ct + log10_μ_tc)
        μ = 10^log10_μ * 0.5 * (μ_scale_c + μ_scale_t)
    end

    (; μ, log10_μ, log10_μ_ct, χ2_opt_ct, log10_μs_ct, χ2s_ct,
        log10_μ_tc, χ2_opt_tc, log10_μs_tc, χ2s_tc)
end