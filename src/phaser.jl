using LinearAlgebra, LinearSolve, ChunkSplitters, StatsBase, Optim, Random, TimerOutputs, SavitzkyGolay, Compat
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
    B0map!(fitpar::FitPar, fitopt::FitOpt, bs::BSmooth{N}) where {N}

TBW
"""
function B0map!(fitpar::FitPar, fitopt::FitOpt)
    # timing will always be monitored
    to = TimerOutput()
    @timeit to "B0 map" begin

        # ======================================================================
        # set up Fourier kernel
        # ======================================================================

        d = length(fitopt.K)
        bs = fourier_lin(size(fitpar.data)[1:d], fitopt.K; os_fac=fitopt.os_fac)

        # ======================================================================
        # subsampling mask
        # ======================================================================

        (S, Sj) = subsample_mask(fitpar, fitopt, bs, to)
        R = fitpar.S
        noR = (!).(R)
        noS = (!).(S)

        # ======================================================================
        # local estimate
        # ======================================================================

        @timeit to "local ML estimate" begin
            print("Local ML estimate ... ")

            fitpar_ML = fitPar(fitpar.grePar, fitpar.data, S)

            fitopt_ML = deepcopy(fitopt)
            fitopt_ML.optim = fitopt.optim_phaser

            local_fit!(fitpar_ML, fitopt_ML)

            Φ_ML, R2s_ML = fitpar_ML.ϕ, fitpar_ML.R2s
            Φ_ML[noS] .= NaN
            R2s_ML[noS] .= NaN

            println("done.")
        end

        # ======================================================================
        # apply PHASER
        # ======================================================================

        println("PHASER Begin ...")

        PH = phaser!(Φ_ML, S, Sj, R, fitpar, fitopt, bs, to)

        fitpar.ϕ[R] .= @views PH.ϕ[end][R]

        println("... PHASER End")

        # ======================================================================
        # Final local fit, if desired
        # ======================================================================

        if fitopt.local_fit
            @timeit to "final local fit" begin
                print("Final local fit ... ")

                fitopt_loc = deepcopy(fitopt)
                set_num_phase_intervals(fitpar, fitopt_loc, 0)

                local_fit!(fitpar, fitopt_loc)

                ϕ_loc, R2s_loc = fitpar.ϕ, fitpar.R2s
                ϕ_loc[noR] .= NaN
                R2s_loc[noR] .= NaN

                println("done.")
            end
        else
            ϕ_loc = R2s_loc = nothing
        end
    end

    for ϕ in PH.ϕ
        ϕ[noR] .= NaN
    end

    # return results
    (; to, PH, Φ_ML, R2s_ML, ϕ_loc, R2s_loc, S, Sj, R)
end

"""
    phaser!(Φ_loc, S, Sj, R, fitopt::FitOpt, bs::BSmooth{N}, to::TimerOutput) where {N}

Actual implementation of PHASER
"""
function phaser!(Φ_ML, S, Sj, R, fitpar, fitopt, bs::BSmooth{N}, to::TimerOutput) where {N}

    @timeit to "phaser!" begin
        # ======================================================================
        # some preps
        # ======================================================================

        @assert ndims(S) == N
        info = Dict()
        T, Tj = [S], Vector{typeof(S)}[]
        ∇Φ_ML = map(map_2π, ∇j_(Φ_ML, Sj))
        ϕ, Φ, ∇Φ = typeof(Φ_ML)[], typeof(Φ_ML)[], typeof(∇Φ_ML)[]
        remove_gradient_outliers!(Tj, Sj, S, ∇Φ_ML, info, to)

        # ======================================================================
        # gradient-based estimate
        # ======================================================================

        n_grad = 0

        if fitopt.multi_scale
            K_ = ones(Int, N)
            @assert all(x -> x ≥ 0, fitopt.K - K_)

            while true
                bs_ = fourier_lin(size(fitpar.data)[1:N], K_; os_fac=fitopt.os_fac)
                gradient_based_estimate!(ϕ, T, Tj, S, Sj, R, Φ, ∇Φ, Φ_ML, ∇Φ_ML, fitopt.μ_tikh, bs_, info, to)
                n_grad += 1

                fitopt.K == K_ && break

                K_[fitopt.K.>K_] .+= 1
            end
        else
            gradient_based_estimate!(ϕ, T, Tj, S, Sj, R, Φ, ∇Φ, Φ_ML, ∇Φ_ML, fitopt.μ_tikh, bs, info, to)

            n_grad += 1
        end

        # ======================================================================
        # PHASER loop
        # ======================================================================

        i_data, n_bal = 1, 0

        while i_data <= fitopt.balance
            println("data: ", i_data, "/", fitopt.balance)

            balanced_estimate!(ϕ, T, Tj, S, Sj, R, Φ, ∇Φ, Φ_ML, ∇Φ_ML, fitpar, fitopt, bs, info, to)
            n_bal += 1

            masks_changed(T, Tj, i_data) || break

            i_data += 1
        end

        # ======================================================================
        # return results
        # ======================================================================

        (; ϕ, Φ_ML, ∇Φ_ML, Φ, ∇Φ, T, Tj, S, Sj, R, n_grad, n_bal, info)
    end
end

"""
    subsample_mask(fitpar::FitPar, fitopt::FitOpt, bs::BSmooth{N}) where {N}

TBW
"""
function subsample_mask(fitpar::FitPar, fitopt::FitOpt, bs::BSmooth{N}, to::TimerOutput) where {N}
    @timeit to "generate subsampling masks" begin
        print("Generate subsampling masks ... ")

        # some initializations
        R = deepcopy(fitpar.S)
        ndR, szR = ndims(R), size(R)
        ciR = CartesianIndices(R)
        fiR, liR = first(ciR), last(ciR)

        # define Sj ⊂ S such that derivatives can be taken in direction j
        ejs = [CartesianIndex(ntuple(k -> k == j ? 1 : 0, ndR)) for j in 1:ndR]

        # candidates
        Sj_ = falses(szR)
        Sj_[fiR:(liR-fiR)] .= @views R[fiR:(liR-fiR)]

        for ej in ejs
            Sj_[fiR:(liR-fiR)] .&= @views R[fiR+ej:(liR-fiR+ej)]
        end

        # target number of locations in mask, which contain a derivative in every direction
        N_sub = ceil(Int, min(fitopt.redundancy * Nfree(bs), 0.99typemax(Int)))

        N_sub < sum(Sj_) && ((Sj_, _) = subsample_mask(N_sub, Sj_, fitopt))

        S = deepcopy(Sj_)

        for ej in ejs
            S[fiR+ej:(liR-fiR+ej)] .|= @views Sj_[fiR:(liR-fiR)]
        end

        Sj = [deepcopy(Sj_) for _ in 1:N]

        println("done.")
    end

    (S, Sj)
end

"""
    subsample_mask(N, Scand, subsampling)

TBW
"""
function subsample_mask(N, Scand, fitopt, max_failed=1000)
    # check for correct setting
    @assert fitopt.subsampling ∈ (:fibonacci, :random)

    # some initializations
    ndS, szS = ndims(Scand), size(Scand)
    N_failed_max = max_failed * N
    ciS = CartesianIndices(Scand)

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
        found, failed = 0, 0

        # location to look at
        loc = ones(ndS)

        # subsampling
        while found < N && failed < N_failed_max
            # location to look at
            loc = mod.(loc .+ z, 1)
            iloc = ceil.(Int, loc .* szS)
            iloc[iloc.==0] .= 1
            iloc = min.(iloc, szS)
            ci = CartesianIndex(iloc...)

            if Scand[ci]
                if S[ci]
                    failed += 1
                else
                    S[ci] = true
                    push!(ciSs, ci)

                    found += 1
                end
            end
        end
    elseif fitopt.subsampling == :random
        ciSca = CartesianIndices(Scand)[Scand]
        iS = randperm(fitopt.rng, sum(Scand))[1:N]

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
    masks_changed(T, Tj)

TBW
"""
function masks_changed(T, Tj, n)
    # the local estimate should be calculated at least once
    n > 1 || return true

    for j in 1:n-1
        T[end-j] == T[end] && Tj[end-j] == Tj[end] && return false
    end

    return true
end

"""
    remove_gradient_outliers!(Tj, ∇Φ, msk, info, to)

TBW
"""
function remove_gradient_outliers!(Tj, Sj, S, ∇Φ, info, to)
    @timeit to "remove gradient outliers" begin
        haskey(info, :outliers) || (info[:outliers] = Dict())

        haskey(info[:outliers], :gradient) || (info[:outliers][:gradient] = Dict())
        ig = info[:outliers][:gradient]

        # first we determine the maximally allowed u in each direction        
        a∇Φ_hist = Histogram[]
        a∇Φ_max = Float64[]
        cntrs = Vector{Float64}[]
        wghts = Vector{Float64}[]

        for j in 1:ndims(S)
            # setting (2n)^(1/3) ("Rice rule") for the number bins in the histogram was motivated in 
            # https://doi.org/10.2307/2288074
            nbins = ceil(Int, (2sum(Sj[j]))^(1 / 3))
            # store differences
            a∇Φ = abs.(∇Φ[j][Sj[j]])
            # boundaries of bin intervals
            edges = @views range(0.0, max(a∇Φ...), nbins + 1)
            push!(cntrs, 0.5(edges[1:end-1] + edges[2:end]))
            # generate the histogram curve based upon the bins defined above
            push!(a∇Φ_hist, fit(Histogram, a∇Φ, edges))
            # in addition to the Rice conditition, we further smooth the histogram
            push!(wghts, savitzky_golay(a∇Φ_hist[end].weights, 3, 1).y)
            # search for the largest peak
            iemin = findmax(wghts[end])[2]
            # starting from zero, include the right flank until local minimum occurs
            fimi = iemin
            fimi_test = fimi + 1
            while fimi_test <= nbins && wghts[end][fimi_test] < wghts[end][fimi]
                fimi = fimi_test
                fimi_test += 1
            end
            # define cutoff value
            push!(a∇Φ_max, 0.5(edges[fimi-1] + edges[fimi]))
        end

        haskey(ig, :a∇Φ_hist) || (ig[:a∇Φ_hist] = typeof(a∇Φ_hist)[])
        push!(ig[:a∇Φ_hist], a∇Φ_hist)
        haskey(ig, :centers) || (ig[:centers] = typeof(cntrs)[])
        push!(ig[:centers], cntrs)
        haskey(ig, :weights) || (ig[:weights] = typeof(wghts)[])
        push!(ig[:weights], wghts)
        haskey(ig, :a∇Φ_max) || (ig[:a∇Φ_max] = typeof(a∇Φ_max)[])
        push!(ig[:a∇Φ_max], a∇Φ_max)

        push!(Tj, deepcopy(Sj))

        for j in 1:ndims(S)
            @. Tj[end][j][Sj[j]] = @views abs(∇Φ[j][Sj[j]]) < a∇Φ_max[j]
        end
    end
end

"""
    remove_local_outliers!(T, Φ, msk, info, to)

TBW
"""
function remove_local_outliers!(T, S, Φ, info, to)
    @timeit to "remove local outliers" begin
        haskey(info, :outliers) || (info[:outliers] = Dict())

        haskey(info[:outliers], :local) || (info[:outliers][:local] = Dict())
        ig = info[:outliers][:local]

        # setting (2n)^(1/3) ("Rice rule") for the number bins in the histogram was motivated in 
        # https://doi.org/10.2307/2288074
        nbins = ceil(Int, (2sum(S))^(1 / 3))
        # boundaries of bin intervals
        edges = @views range(min(Φ[S]...), max(Φ[S]...), nbins + 1)
        cntrs = 0.5(edges[1:end-1] + edges[2:end])
        # generate the histogram curve based upon the bins defined above
        Φ_hist = @views fit(Histogram, Φ[S], edges)
        # take the weights and apply an additional smoothing filter
        wghts = savitzky_golay(Φ_hist.weights, 3, 1).y
        # we assume the largest peak to correspond to the correct solution
        ip = argmax(wghts)
        # now we go down on both flanks of the peak until the next local minimum is reached
        ip_min = ip_max = ip
        ip_max_test = ip_max + 1
        while ip_max_test <= nbins &&
            (wghts[ip_max_test] < wghts[ip_max] ||
             wghts[ip_max_test] > 0.7 * wghts[ip])
            ip_max = ip_max_test
            ip_max_test += 1
        end
        ip_min_test = ip_min - 1
        while ip_min_test >= 1 &&
            (wghts[ip_min_test] < wghts[ip_min] ||
             wghts[ip_min_test] > 0.7 * wghts[ip])
            ip_min = ip_min_test
            ip_min_test -= 1
        end
        Φ_min = 0.5(edges[ip_min] + edges[ip_min+1])
        Φ_max = 0.5(edges[ip_max] + edges[ip_max-1])

        push!(T, deepcopy(S))

        # these local minima then define the locations to keep in S
        @. T[end][S] = @views Φ_min <= Φ[S] <= Φ_max

        haskey(ig, :Φ_hist) || (ig[:Φ_hist] = typeof(Φ_hist)[])
        push!(ig[:Φ_hist], Φ_hist)
        haskey(ig, :centers) || (ig[:centers] = typeof(cntrs)[])
        push!(ig[:centers], cntrs)
        haskey(ig, :weights) || (ig[:weights] = typeof(wghts)[])
        push!(ig[:weights], wghts)
        haskey(ig, :Φ_min) || (ig[:Φ_min] = typeof(Φ_min)[])
        push!(ig[:Φ_min], Φ_min)
        haskey(ig, :Φ_max) || (ig[:Φ_max] = typeof(Φ_max)[])
        push!(ig[:Φ_max], Φ_max)
    end
end

"""
    gradient_based_estimate!(ϕ, Tj, S, R, ∇Φ, Φ, μ, bs, info, to)

TBW
"""
function gradient_based_estimate!(ϕ, T, Tj, S, Sj, R, Φ, ∇Φ, Φ_ML, ∇Φ_ML, μ_tikh, bs, info, to)
    @timeit to "gradient-based estimate" begin
        print("Gradient-based estimate ... ")

        @timeit to "prep matrices" begin
            # MPI estimate
            ∇Bt∇B = calc_∇Bt∇B(bs, Tj[end], to)
            ∇Bt∇Φ = calc_∇Btx(bs, Tj[end], ∇Φ_ML, to)
        end

        haskey(info, :gradient) || (info[:gradient] = Dict())
        ib = info[:gradient]

        ib[:∇Bt∇B] = ∇Bt∇B
        ib[:∇Bt∇Φ] = ∇Bt∇Φ

        μ = μ_tikh * max(real.(diag(∇Bt∇B))...)
        prob = LinearSolve.LinearProblem(∇Bt∇B + μ * I, ∇Bt∇Φ)
        sol = LinearSolve.solve(prob)
        c = sol.u

        # calculate phase map with median limited to (-π, π]
        push!(ϕ, zeros(size(R)))
        ϕ[end][R] .= @views phase_map(bs, c, to)[R]

        calc_phase_offset!(ϕ[end], Φ_ML, S, R)

        # improve masks
        push!(Φ, map_2π(Φ_ML - ϕ[end]))
        push!(∇Φ, map(map_2π, ∇j_(Φ[end], Sj)))
        push!(T, S)
        remove_gradient_outliers!(Tj, Sj, S, ∇Φ[end], info, to)

        println("done.")
    end
end

"""
    balanced_estimate!(ϕ, T, Tj, S, R, Φ, ∇Φ, fitpar, fitopt, bs, info, to)

TBW
"""
function balanced_estimate!(ϕ, T, Tj, S, Sj, R, Φ, ∇Φ, Φ_ML, ∇Φ_ML, fitpar, fitopt, bs::BSmooth{N}, info, to) where {N}
    @timeit to "balanced estimate" begin
        print("balancing ... ")

        @timeit to "prep matrices" begin
            # MPI estimate
            BtB = calc_BtB(bs, T[end], to)
            BtΦ = calc_Btx(bs, T[end], Φ[end], to)
            ∇Bt∇B = calc_∇Bt∇B(bs, Tj[end], to)
            ∇Bt∇Φ = calc_∇Btx(bs, Tj[end], ∇Φ[end], to)
        end

        haskey(info, :balanced) || (info[:balanced] = Dict())
        ib = info[:balanced]

        haskey(ib, :BtB) || (ib[:BtB] = typeof(BtB)[])
        haskey(ib, :BtΦ) || (ib[:BtΦ] = typeof(BtΦ)[])
        haskey(ib, :∇Bt∇B) || (ib[:∇Bt∇B] = typeof(∇Bt∇B)[])
        haskey(ib, :∇Bt∇Φ) || (ib[:∇Bt∇Φ] = typeof(∇Bt∇Φ)[])

        push!(ib[:BtB], BtB)
        push!(ib[:BtΦ], BtΦ)
        push!(ib[:∇Bt∇B], ∇Bt∇B)
        push!(ib[:∇Bt∇Φ], ∇Bt∇Φ)

        push!(ϕ, zeros(size(R)))

        Φ2 = ∇Φ2 = 0.0
        fitoptλ = deepcopy(fitopt)

        fitparλ = fitPar(fitpar.grePar, fitpar.data, S)
        fitoptλ.optim = fitopt.optim_balance
        set_num_phase_intervals(fitparλ, fitoptλ, 0)
        fitoptλ.rapid_balance && (fitoptλ.R2s_rng = [0.0, 0.0])

        χ2_λ_fun = create_χ2_λ_fun(fitparλ, fitoptλ, bs, BtB, BtΦ, ∇Bt∇B, ∇Bt∇Φ, Φ2, ∇Φ2, ϕ, to)

        @timeit to "GSS search λ" begin
            λ_opt, χ2_opt, λs, χ2s = GSS(χ2_λ_fun, (0.0, 1.0), 1e-4; show_all=true)
        end

        haskey(ib, :λ_opt) || (ib[:λ_opt] = typeof(λ_opt)[])
        haskey(ib, :χ2_opt) || (ib[:χ2_opt] = typeof(χ2_opt)[])
        haskey(ib, :λs) || (ib[:λs] = typeof(λs)[])
        haskey(ib, :χ2s) || (ib[:χ2s] = typeof(χ2s)[])

        sumS = sum(S)

        push!(ib[:λ_opt], λ_opt)
        push!(ib[:χ2_opt], χ2_opt / sumS)
        push!(ib[:λs], λs)
        push!(ib[:χ2s], χ2s / sumS)

        # take the best match and calculate the solution on R
        create_χ2_λ_fun(fitpar, fitoptλ, bs, BtB, BtΦ, ∇Bt∇B, ∇Bt∇Φ, Φ2, ∇Φ2, ϕ, to; calc_ϕ=true)(λ_opt)

        ϕ[end][R] .= @views fitpar.ϕ[R]

        # make sure that the phase median over S lies within (-π, π]
        median_shift!(ϕ[end], R)

        # improve masks
        push!(Φ, map_2π(Φ_ML - ϕ[end]))
        push!(∇Φ, map(map_2π, ∇j_(Φ[end], Sj)))
        remove_local_outliers!(T, S, Φ[end], info, to)
        remove_gradient_outliers!(Tj, Sj, S, ∇Φ[end], info, to)

        println("done.")
    end
end

"""
    calc_phase_offset!(ϕ, Φ, S, R)

TBW
"""
function calc_phase_offset!(ϕ, Φ, S, R)
    # coefficient b
    b = @views angle(sum(exp.(im .* (Φ[S] .- ϕ[S]))))

    # calculate the median of ϕ over S
    ϕ_med = @views median(ϕ[S]) + b

    # limit the median phase to [-π,π)
    while ϕ_med >= π
        ϕ_med -= 2π
        b -= 2π
    end

    while ϕ_med < -π
        ϕ_med += 2π
        b += 2π
    end

    # add offset to ϕ
    @. ϕ[R] += b

    nothing
end

"""
    median_shift!(ϕ, S)

TBW
"""
function median_shift!(ϕ, S)
    # calculate the median of ϕ over S
    ϕ_med = @views median(ϕ[S])
    b = 0.0

    # limit the median phase to [-π,π)
    while ϕ_med >= π
        ϕ_med -= 2π
        b -= 2π
    end

    while ϕ_med < -π
        ϕ_med += 2π
        b += 2π
    end

    # add offset to ϕ
    @. ϕ[S] += b

    nothing
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
function create_χ2_λ_fun(fitparλ, fitoptλ, bs, A, a, B, b, Φ2, ∇Φ2, ϕ, to; calc_ϕ=false)

    @timeit to "create_χ2_λ_fun" begin
        λ -> let fitparλ = fitparλ, fitoptλ = fitoptλ, bs = bs,
            A = A, a = a, B = B, b = b, ϕ = ϕ, to = to

            (c, _) = c_MPI(A, a, B, b, λ, fitoptλ.μ_tikh)

            fitparλ.ϕ[fitparλ.S] .= @views phase_map(bs, real(c[1]), c[2:end], to)[fitparλ.S] .+ ϕ[end-1][fitparλ.S]

            @timeit to "local fit" begin
                local_fit!(fitparλ, fitoptλ)
            end

            sum(fitparλ.χ2[fitparλ.S])
        end
    end
end

"""
    c_MPI(A, a, B, b, λ, μ_tikh)

TBW
"""
function c_MPI(A, a, B, b, λ, μ_tikh)
    @assert λ ≠ 0

    mat_λ = λ .* A
    @. mat_λ[2:end, 2:end] += (1 - λ) * B
    vec_λ = λ .* a
    @. vec_λ[2:end] += (1 - λ) * b

    μ = μ_tikh * max(real.(diag(mat_λ))...)
    prob = LinearSolve.LinearProblem(mat_λ + μ * I, vec_λ)
    sol = LinearSolve.solve(prob)

    # return coefficients
    (sol.u, mat_λ)
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
    ϕ[S] = @views phase_map(bs, real(c_mpi[1]), c_mpi[2:end])[S]
end

#= 
==================================================================  
Auxiliary routines
==================================================================  
=#

"""
    map_2π(ϕ)

Limit the argument to the interval `[-π, π)` by shifting with integer multiples of `2π`.

## Remarks
- The argument `ϕ` can be a scalar or an array. The result is of the same type.
"""
function map_2π(ϕ)
    mod.(ϕ .+ π, 2π) .- π
end
