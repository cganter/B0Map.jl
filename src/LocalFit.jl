using ChunkSplitters, Optim
import VP4Optim as VP

mutable struct B0_map_varpro_results
    ϕ_GSS::Array
    R2s_GSS::Array
    f_GSS::Array
    c_GSS::Array
    χ2_GSS::Array
    ϕ::Array
    R2s::Array
    f::Array
    c::Array
    coil_phase_GSS::Array
    χ2::Array
end

"""
    B0_map(ge_con::Function, geFW_con::Function, args, data, S, bs::BSmooth;
    λ_tikh=1e-6,
    n_ϕ_0=6,
    R2s_rng=(0.0, 1.0),
    abs_acc_R2s=1e-3,
    n_chunks=8Threads.nthreads())

TBW
"""
function B0_map_varpro(gre_con::Function, args, data, S;
    n_ϕ=3,
    R2s_rng=(0.0, 1.0),
    ϕ_acc=1e-4, R2s_acc=1e-4,
    n_chunks=8Threads.nthreads(),
    Δt=nothing,
    optim=true)

    # use the constructor to create a model and extract information about the number of coefficients
    gre_ = gre_con(args..., Δt=Δt)
    Nc = VP.N_coeff(gre_)

    # size of data block
    sz = size(S)

    # allocate return parameters
    ϕ_GSS = zeros(sz)
    R2s_GSS = zeros(sz)
    f_GSS = zeros(sz)
    c_GSS = zeros(ComplexF64, sz..., Nc)
    coil_phase_GSS = zeros(Float64, sz...)
    χ2_GSS = zeros(sz)
    χ2 = zeros(sz)

    # Cartesian indices of valid data (defined by the mask S)
    cis = CartesianIndices(S)[S]
    cis_chunks = [view(cis, index_chunks(cis, n=n_chunks)[i]) for i in 1:n_chunks]

    # starting values for search
    ϕ_scale = gre_.ϕ_scale
    ϕ_period_2 = π * ϕ_scale

    Δϕ2 = ϕ_period_2 / n_ϕ
    ϕs = range(-ϕ_period_2 + Δϕ2, ϕ_period_2 - Δϕ2, n_ϕ)
    ϕ_rngs = [[ϕ_ - Δϕ2, ϕ_ + Δϕ2] for ϕ_ in ϕs]

    # channel to prevent data races in case of multi-threaded execution
    ch_gre = Channel{typeof(gre_)}(Threads.nthreads())

    for _ in 1:Threads.nthreads()
        put!(ch_gre, gre_con(args..., Δt=Δt))
    end

    println("Local VARPRO fit ... ")

    # We do this for every point in S
    @time Threads.@threads for cis_chunk in cis_chunks
        # take free models
        gre = take!(ch_gre)

        # work on actual chunk
        fit_chunk_GSS(gre, cis_chunk, data, ϕ_rngs, R2s_rng, ϕ_acc, R2s_acc, ϕ_GSS, R2s_GSS, χ2_GSS)
        calc_f_c_chunk(gre, cis_chunk, data, ϕ_GSS, R2s_GSS, f_GSS, c_GSS, coil_phase_GSS)

        # put the model back
        put!(ch_gre, gre)
    end

    ϕ = deepcopy(ϕ_GSS)
    R2s = deepcopy(R2s_GSS)
    f = deepcopy(f_GSS)
    c = deepcopy(c_GSS)
    χ2 = deepcopy(χ2_GSS)

    if optim
        # We do this for every point in S
        @time Threads.@threads for cis_chunk in cis_chunks
            # take free models
            gre = take!(ch_gre)

            # work on actual chunk
            fit_chunk_optim(gre, cis_chunk, data, R2s_rng, ϕ, R2s, χ2)
            calc_f_c_chunk(gre, cis_chunk, data, ϕ, R2s, f, c, coil_phase_GSS)

            # put the model back
            put!(ch_gre, gre)
        end
    end

    println("done.")

    # close all channels
    close(ch_gre)

    # return results
    return B0_map_varpro_results(
        ϕ_GSS,
        R2s_GSS,
        f_GSS,
        c_GSS,
        χ2_GSS,
        ϕ,
        R2s,
        f,
        c,
        coil_phase_GSS,
        χ2)
end

"""
    calc_f_c_chunk(gre, cis_chunk, data, ϕ, R2s, f, c)

TBW
"""
function calc_f_c_chunk(gre, cis_chunk, data, ϕ, R2s, f, c, c_phase)
    for ci in cis_chunk
        # set data
        VP.set_data!(gre, data[ci, :])

        # apply the argument
        VP.x!(gre, [ϕ[ci], R2s[ci]])

        # calculate fat fraction
        f[ci] = fat_fraction(gre)

        # save linear factor(s)
        c[ci, :] = VP.c(gre)
        
        # phase of real model
        gre isa GREMultiEchoWFRW &&
            (c_phase[ci] = coil_phase(gre))
    end
end

"""
    fit_chunk_GSS(gre, cis_chunk, data, ϕ_rngs, R2s_rng, ϕ_acc, R2s_acc, ϕ, R2s, χ2)

TBW
"""
function fit_chunk_GSS(gre, cis_chunk, data, ϕ_rngs, R2s_rng, ϕ_acc, R2s_acc, ϕ, R2s, χ2)
    for ci in cis_chunk
        # set data
        VP.set_data!(gre, data[ci, :])

        # init χ2_opt
        χ2[ci] = Inf

        for ϕ_rng in ϕ_rngs
            # first search for best ϕ, assuming R2s == 0
            ϕ_, _ = GSS_fit_ϕ(gre, 0.0, ϕ_rng; ϕ_acc=ϕ_acc)

            # then search for the best R2s
            R2s_, χ2_ = GSS_fit_R2s(gre, ϕ_, R2s_rng; R2s_acc=R2s_acc)

            # did we improve?
            if χ2_ < χ2[ci]
                ϕ[ci], R2s[ci], χ2[ci] = ϕ_, R2s_, χ2_
            end
        end
    end
end

"""
    fit_chunk_GSS(gre, cis_chunk, data, R2s_rng, R2s_acc, ϕ, R2s, χ2)

TBW
"""
function fit_chunk_GSS(gre, cis_chunk, data, R2s_rng, R2s_acc, ϕ, R2s, χ2)
    for ci in cis_chunk
        # set data
        VP.set_data!(gre, data[ci, :])

        # search for the best R2s
        R2s[ci], χ2[ci] = GSS_fit_R2s(gre, ϕ[ci], R2s_rng; R2s_acc=R2s_acc)
    end
end

"""
    fit_chunk_optim(gre, cis_chunk, data, R2s_rng, ϕ, R2s, χ2)

TBW
"""
function fit_chunk_optim(gre, cis_chunk, data, R2s_rng, ϕ, R2s, χ2)
    for ci in cis_chunk
        # set data
        VP.set_data!(gre, data[ci, :])

        # starting values and bounds
        ϕ_period_2 = gre.ϕ_scale * π
        x0, lx, ux = [ϕ[ci], R2s[ci]], [ϕ[ci] - ϕ_period_2, R2s_rng[1]], [ϕ[ci] + ϕ_period_2, R2s_rng[2]]

        # search for optimum
        res = optimize(Optim.only_fg!(VP.fg!(gre)), lx, ux, x0, Fminbox(LBFGS()))

        # store result
        ϕ[ci], R2s[ci] = res.minimizer
        χ2[ci] = VP.f(gre)([ϕ[ci], R2s[ci]])
    end
end

"""
    GSS_fit_ϕ(gre::AbstractGREMultiEcho, R2s, ϕ_rng; ϕ_acc=1e-4)

TBW
"""
function GSS_fit_ϕ(gre::AbstractGREMultiEcho, R2s, ϕ_rng; ϕ_acc=1e-4)
    #  range must be meaningful
    @assert ϕ_rng[1] < ϕ_rng[2]

    # let block to avoid boxing of variables
    χ2 = let gre = gre, R2s = R2s
        ϕ -> VP.f(gre)([ϕ, R2s])
    end

    # GSS search for best R2s
    GSS(χ2, ϕ_rng, ϕ_acc)
end

"""
    GSS_fit_R2s(gre::AbstractGREMultiEcho, ϕ, R2s_rng; R2s_acc=1e-4)

TBW
"""
function GSS_fit_R2s(gre::AbstractGREMultiEcho, ϕ, R2s_rng; R2s_acc=1e-4)
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
    GSS(fun::Function, var_rng; rel_acc=1e-4, abs_acc=nothing)

Search for minimum of given function with golden section search (GSS).

Returns location of minimum with absolute accuracy `abs_acc`.

# Arguments
- `fun::Function`: real valued function of a single real valued argument.
- `var_rng::T <: Union{Vector, Tuple}`: search interval boundaries, `length(var_rng) == 2`
- `rel_acc::Float64 = 1e-4`: compute `abs_acc = rel_acc * (var_rng[2] - var_rng[1])`.
- `abs_acc::T <: Union{Nothing, Float64} = nothing`: Overrules `rel_acc`, if supplied.
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

    if show_all
        return (best_result..., xs, fs)
    else
        return best_result
    end
end
