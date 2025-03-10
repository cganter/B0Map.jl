using ChunkSplitters, Optim, TimerOutputs, Compat
import VP4Optim as VP
@compat public local_fit, GSS

"""
    local_fit(fitpar::FitPar{T}, fitopt::FitOpt) where {T<:AbstractGREMultiEcho}

Fit data to multi-echo GRE model locally.

## Arguments
- `fitpar::FitPar`: Fit parameters (see [FitPar](@ref FitPar))
- `fitopt::FitOpt`: Fit options (see [FitOpt](@ref FitOpt))
## Remarks
- Fits `fitpar.data` to all points specified by `fitpar.S`.
- Makes use of multi-threading. Just start julia with as many workers as you like to use.
- If `isempty(fitopt.ϕ_rngs) == true`, the GSS fit is only with respect to `R2s`, based upon the local phase, specified in `fitpar.ϕ`.
- Otherwise, for each interval in `fitopt.ϕ_rngs` a GSS search with respect to `ϕ` is performed, each time for `R2s == 0`. For each of these phases fixed, a subsequent GSS search with respect to `R2s` is performed. The combination `[ϕ, R2s]`, which produces the best fit is selected.
- Optionally (`fitopt.optim == true`), for the best GSS estimate `[ϕ, R2s]` is refined with a nonlinear fit.
- The final estimates `[ϕ, R2s]` are stored in `fitpar`, together with the linear coefficients `VP4Optim.c` and the goodness of fit `VP4Optim.χ2`.
"""
function local_fit(fitpar::FitPar{T}, fitopt::FitOpt; verbose = false) where {T<:AbstractGREMultiEcho}
    # Cartesian indices of valid data (defined by the mask S)
    cis = CartesianIndices(fitpar.S)[fitpar.S]
    cis_chunks = [view(cis, index_chunks(cis, n=fitopt.n_chunks)[i]) for i in 1:fitopt.n_chunks]

    # channel to prevent data races in case of multi-threaded execution
    ch_gre = Channel{T}(Threads.nthreads())

    for _ in 1:Threads.nthreads()
        put!(ch_gre, deepcopy(fitpar.gre))
    end

    verbose && print("Local fit ... ")

    # do the work
    Threads.@threads for cis_chunk in cis_chunks
        # take free models
        gre = take!(ch_gre)

        # work on actual chunk
        local_fit_chunk(gre, fitpar, fitopt, cis_chunk)

        # put the model back
        put!(ch_gre, gre)
    end

    verbose && println("done.")

    # close channel
    close(ch_gre)
end

"""
    local_fit_chunk(gre, fitpar, fitopt, cis_chunk)

Auxiliary function
"""
function local_fit_chunk(gre, fitpar, fitopt, cis_chunk)
    szS = size(fitpar.S)

    for ci in cis_chunk
        # set data
        @views VP.set_data!(gre, reshape(fitpar.data, szS..., :)[ci, :])

        if isempty(fitopt.ϕ_rngs)
            # just search for the best R2s
            fitpar.R2s[ci], fitpar.χ2[ci] = GSS_fit_R2s(gre, fitpar.ϕ[ci], fitopt.R2s_rng, fitopt.R2s_acc)
        else
            # init χ2_opt
            fitpar.χ2[ci] = Inf

            for ϕ_rng in fitopt.ϕ_rngs
                # search for best ϕ, assuming R2s == 0
                ϕ_, _ = GSS_fit_ϕ(gre, 0.0, ϕ_rng, fitopt.ϕ_acc)

                # search for the best R2s
                R2s_, χ2_ = GSS_fit_R2s(gre, ϕ_, fitopt.R2s_rng, fitopt.R2s_acc)

                # did we improve?
                if χ2_ < fitpar.χ2[ci]
                    fitpar.ϕ[ci], fitpar.R2s[ci], fitpar.χ2[ci] = ϕ_, R2s_, χ2_
                end
            end
        end

        if fitopt.optim
            # starting values and bounds
            ϕ_period_2 = gre.ϕ_scale * π
            x0 = [fitpar.ϕ[ci], fitpar.R2s[ci]]
            lx = [fitpar.ϕ[ci] - ϕ_period_2, fitopt.R2s_rng[1]]
            ux = [fitpar.ϕ[ci] + ϕ_period_2, fitopt.R2s_rng[2]]

            # search for optimum
            res = optimize(Optim.only_fg!(VP.fg!(gre)), lx, ux, x0, Fminbox(LBFGS()))

            # store result
            fitpar.ϕ[ci], fitpar.R2s[ci] = res.minimizer
            fitpar.χ2[ci] = VP.f(gre)([fitpar.ϕ[ci], fitpar.R2s[ci]])
        end

        # linear coefficients
        VP.x!(gre, [fitpar.ϕ[ci], fitpar.R2s[ci]])
        fitpar.c[ci] = VP.c(gre)
    end
end

"""
    GSS_fit_ϕ(gre::AbstractGREMultiEcho, R2s, ϕ_rng; ϕ_acc=1e-4)

Auxiliary function
"""
function GSS_fit_ϕ(gre::AbstractGREMultiEcho, R2s, ϕ_rng, ϕ_acc)
    #  range must be meaningful
    @assert ϕ_rng[1] ≤ ϕ_rng[2]

    if ϕ_rng[1] == ϕ_rng[2]
        # handle the trivial case
        return (ϕ_rng[1], VP.f(gre)([ϕ_rng[1], R2s]))
    else
        # let block to avoid boxing of variables
        χ2 = let gre = gre, R2s = R2s
            ϕ -> VP.f(gre)([ϕ, R2s])
        end

        # GSS search for best R2s
        GSS(χ2, ϕ_rng, ϕ_acc)
    end
end

"""
    GSS_fit_R2s(gre::AbstractGREMultiEcho, ϕ, R2s_rng; R2s_acc=1e-4)

Calculate the best `R2s` for given `ϕ` with GSS
Auxiliary function
"""
function GSS_fit_R2s(gre::AbstractGREMultiEcho, ϕ, R2s_rng, R2s_acc)
    # R2* range must be meaningful
    @assert 0 ≤ R2s_rng[1] ≤ R2s_rng[2]

    if R2s_rng[1] == R2s_rng[2]
        # handle the trivial case
        return (R2s_rng[1], VP.f(gre)([ϕ, R2s_rng[1]]))
    else
        # let block to avoid boxing of variables
        χ2 = let gre = gre, ϕ = ϕ
            R2s -> VP.f(gre)([ϕ, R2s])
        end

        # GSS search for best R2s
        GSS(χ2, R2s_rng, R2s_acc)
    end
end


"""
    GSS(fun::Function, var_rng, acc; show_all=false)

Search for minimum of given function `fun` with golden section search (GSS).

# Arguments
- `fun::Function`: real valued function of a single real valued argument.
- `var_rng::T <: Union{Vector, Tuple}`: search interval boundaries, `length(var_rng) == 2`
- `acc::Float64`: Required accuracy of *location*.
- `show_all::Bool`: Also return visited locations and their values (default: `false`)

# Output
- `show_all ? ((x_opt, fun(x_opt)), xs, fs) : (x_opt, fun(x_opt))`, where
- `x_opt` is the location of the minimum with value `fun(x_opt)`
- `xs::Vector{Float64}` is the vector of all tested locations and `fs = map(x -> fun(x), xs)`
"""
function GSS(fun::Function, var_rng, acc; show_all=false)
    # the (absolute) accuracy must be positive
    @assert acc > 0.0

    # save and return all value/function pairs, if required 
    show_all && (xs = Vector{Float64}(undef, 0); fs = Vector{Float64}(undef, 0))

    # golden section ratio
    r = 0.5(sqrt(5) - 1)

    # outer bounds
    a, b = var_rng

    # initial intermediate points
    c, d = r * a + (1 - r) * b, (1 - r) * a + r * b

    # ... and the corresponding values of χ²
    f_c, f_d = fun(c), fun(d)

    if show_all
        push!(xs, c)
        push!(fs, f_c)
        push!(xs, d)
        push!(fs, f_d)
    end

    # execute the GSS loop
    while abs(d - c) > acc
        if f_c < f_d
            b = d
            d, f_d = c, f_c
            c = r * a + (1 - r) * b
            f_c = fun(c)

            if show_all
                push!(xs, c)
                push!(fs, f_c)
            end
        else
            a = c
            c, f_c = d, f_d
            d = (1 - r) * a + r * b
            f_d = fun(d)

            if show_all
                push!(xs, d)
                push!(fs, f_d)
            end
        end
    end

    # return best result
    best_result = f_c < f_d ? (c, f_c) : (d, f_d)

    return show_all ? (best_result..., xs, fs) : best_result
end