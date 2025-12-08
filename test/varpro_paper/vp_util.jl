using Statistics, Random, LaTeXStrings, CairoMakie, LinearAlgebra
import VP4Optim as VP
import B0Map as BM

# utility functions

"""
    set_up_GREs(; TEs, B0, ppm_fat, ampl_fat, precession=:counterclockwise)
    
Generate the three different VARPRO models and return in a `Dict()`
"""
function set_up_GREs(; TEs, B0, ppm_fat, ampl_fat, precession=:counterclockwise, models=(:FW, :RW, :FC))
    grePar, gre = Dict(), Dict()

    dictGRE = Dict(
        :FW => BM.GREMultiEchoWFFW,
        :RW => BM.GREMultiEchoWFRW,
        :FC => BM.GREMultiEchoWF,
    )

    for m in models
        grePar[m] = VP.modpar(dictGRE[m];
            ts=TEs,
            B0=B0,
            ppm_fat=ppm_fat,
            ampl_fat=ampl_fat,
            precession=precession)
        gre[m] = VP.create_model(grePar[m])
    end

    (; gre, grePar)
end

"""
    MC_sim_loc(gre, ϕ_t, R2s_t, f_t, c_t, σ, n_σ, rng, ϕs, R2ss)
"""
function MC_sim_loc(rng, gre, ϕ_t, R2s_t, f_t, c_t, σ, n_σ, ϕs, R2ss)
    VP.x!(gre[:FW], [ϕ_t, R2s_t])
    y_t = Vector{ComplexF64}(undef, BM.nTE(gre[:FW]))

    cw, cf = (1 - f_t) * c_t, f_t * c_t
    y_t = VP.A(gre[:FW]) * [cw, cf]

    n_ϕ, n_R2s = length(ϕs), length(R2ss)

    χ2 = Dict()
    f = Dict()
    ϕ_min = Dict()
    R2s_min = Dict()
    f_min = Dict()

    for m in keys(gre)
        χ2[m] = Matrix{Float64}(undef, n_ϕ, n_R2s)
        f[m] = Matrix{Float64}(undef, n_ϕ, n_R2s)
        ϕ_min[m] = Vector{Float64}(undef, n_σ)
        R2s_min[m] = Vector{Float64}(undef, n_σ)
        f_min[m] = Vector{Float64}(undef, n_σ)
    end

    wf_par = Matrix{Bool}(undef, n_ϕ, n_R2s)
    wf_opp = Matrix{Bool}(undef, n_ϕ, n_R2s)
    wf_par_min = Vector{Bool}(undef, n_σ)
    wf_opp_min = Vector{Bool}(undef, n_σ)

    for iσ in 1:n_σ
        rand_data = y_t .+ σ .* randn(rng, ComplexF64, size(y_t)...)

        for m in keys(gre)
            VP.set_data!(gre[m], deepcopy(rand_data))
        end

        for (iϕ, ϕ) in enumerate(ϕs)
            for (iR2s, R2s) in enumerate(R2ss)
                for m in keys(gre)
                    VP.x!(gre[m], [ϕ, R2s])
                    χ2[m][iϕ, iR2s] = VP.χ2(gre[m])
                    f[m][iϕ, iR2s] = BM.fat_fraction(gre[m])
                end

                c = VP.c(gre[:RW])
                wf_par[iϕ, iR2s] = c[1] * c[2] ≥ 0
                wf_opp[iϕ, iR2s] = !wf_par[iϕ, iR2s]
            end
        end

        for m in keys(gre)
            argmin_χ2 = argmin(χ2[m])

            if m == :RW
                wf_par_min[iσ] = wf_par[argmin_χ2]
                wf_opp_min[iσ] = !wf_par_min[iσ]
            end

            ϕ_min[m][iσ] = ϕs[argmin_χ2[1]]
            R2s_min[m][iσ] = R2ss[argmin_χ2[2]]
            f_min[m][iσ] = f[m][argmin_χ2]
        end
    end

    ϕ_min_par = Dict()
    ϕ_min_opp = Dict()
    R2s_min_par = Dict()
    R2s_min_opp = Dict()
    f_min_par = Dict()
    f_min_opp = Dict()

    for m in keys(gre)
        ϕ_min_par[m] = ϕ_min[m][wf_par_min]
        ϕ_min_opp[m] = ϕ_min[m][wf_opp_min]
        R2s_min_par[m] = R2s_min[m][wf_par_min]
        R2s_min_opp[m] = R2s_min[m][wf_opp_min]
        f_min_par[m] = f_min[m][wf_par_min]
        f_min_opp[m] = f_min[m][wf_opp_min]
    end

    frac_wf_par_min = sum(wf_par_min) / length(wf_par_min)
    frac_wf_opp_min = 1 - frac_wf_par_min

    (; χ2, f, wf_par, wf_opp,
        wf_par_min, wf_opp_min,
        frac_wf_par_min, frac_wf_opp_min,
        ϕ_min, ϕ_min_par, ϕ_min_opp,
        R2s_min, R2s_min_par, R2s_min_opp,
        f_min, f_min_par, f_min_opp,
    )
end

"""
    MC_sim(;
    rng=MersenneTwister(),
    TEs,
    B0,
    ppm_fat=[-3.80, -3.40, -2.60, -1.95, -0.5, 0.60],
    ampl_fat=[0.0875, 0.6998, 0.1206, 0.0062, 0.0389, 0.0471],
    ϕ_t,
    R2ss_t,
    fs_t,
    c_t,
    σ,
    n_σ,
    ϕs,
    R2ss,
)

Perform MC simulation for matrix of true `R2s` and fat fraction values.
"""
function MC_sim(;
    rng=MersenneTwister(),
    TEs,
    B0,
    ppm_fat=[-3.80, -3.40, -2.60, -1.95, -0.5, 0.60],
    ampl_fat=[0.0875, 0.6998, 0.1206, 0.0062, 0.0389, 0.0471],
    ϕ_t,
    R2ss_t,
    fs_t,
    c_t,
    σ,
    n_σ,
    ϕs,
    R2ss,
)
    # finetune fat model amplitudes
    ampl_fat /= sum(ampl_fat)
    @assert all(ampl_fat .> 0) && sum(ampl_fat) ≈ 1

    # generate GRE models
    gre_info = set_up_GREs(;
        TEs=TEs,
        B0=B0,
        ppm_fat=ppm_fat,
        ampl_fat=ampl_fat,
    )

    # number of true parameters and other convenience settings
    n_R2s_t = length(R2ss_t)
    n_f_t = length(fs_t)
    sz_t = (n_R2s_t, n_f_t)
    gre_keys = keys(gre_info.gre)

    if n_R2s_t == 1 && n_f_t == 1
        MC_sim_loc(rng, gre_info.gre, ϕ_t, R2ss_t[1], fs_t[1], c_t, σ, n_σ, ϕs, R2ss)
    else
        ϕ_min = Dict()
        R2s_min = Dict()
        f_min = Dict()
        ϕ_min_par = Dict()
        ϕ_min_opp = Dict()
        R2s_min_par = Dict()
        R2s_min_opp = Dict()
        f_min_par = Dict()
        f_min_opp = Dict()

        for m in gre_keys
            ϕ_min[m] = Array{Float64,3}(undef, sz_t..., n_σ)
            R2s_min[m] = Array{Float64,3}(undef, sz_t..., n_σ)
            f_min[m] = Array{Float64,3}(undef, sz_t..., n_σ)
            ϕ_min_par[m] = Matrix{Vector{Float64}}(undef, sz_t...)
            ϕ_min_opp[m] = Matrix{Vector{Float64}}(undef, sz_t...)
            R2s_min_par[m] = Matrix{Vector{Float64}}(undef, sz_t...)
            R2s_min_opp[m] = Matrix{Vector{Float64}}(undef, sz_t...)
            f_min_par[m] = Matrix{Vector{Float64}}(undef, sz_t...)
            f_min_opp[m] = Matrix{Vector{Float64}}(undef, sz_t...)
        end

        wf_par_min = Array{Bool,3}(undef, sz_t..., n_σ)
        wf_opp_min = Array{Bool,3}(undef, sz_t..., n_σ)
        frac_wf_par_min = Matrix{Float64}(undef, sz_t...)
        frac_wf_opp_min = Matrix{Float64}(undef, sz_t...)

        # MC simulation results
        MC_sim_res = Array{Any}(undef, n_R2s_t, n_f_t)

        # create channels
        ch_gre = Channel{Dict{Any,Any}}(Threads.nthreads())

        for _ in 1:Threads.nthreads()
            put!(ch_gre, deepcopy(gre_info.gre))
        end

        Threads.@threads for (iR2s_t, if_t) in collect(Iterators.product(1:n_R2s_t, 1:n_f_t))
            println("threadid: ", Threads.threadid(), ", ", (iR2s_t, if_t))
            gre = take!(ch_gre)

            MC_sim_res = MC_sim_loc(rng, gre, ϕ_t, R2ss_t[iR2s_t], fs_t[if_t], c_t, σ, n_σ, ϕs, R2ss)

            for m in gre_keys
                ϕ_min[m][iR2s_t, if_t, :] = MC_sim_res.ϕ_min[m]
                R2s_min[m][iR2s_t, if_t, :] = MC_sim_res.R2s_min[m]
                f_min[m][iR2s_t, if_t, :] = MC_sim_res.f_min[m]
                ϕ_min_par[m][iR2s_t, if_t] = MC_sim_res.ϕ_min_par[m]
                ϕ_min_opp[m][iR2s_t, if_t] = MC_sim_res.ϕ_min_opp[m]
                R2s_min_par[m][iR2s_t, if_t] = MC_sim_res.R2s_min_par[m]
                R2s_min_opp[m][iR2s_t, if_t] = MC_sim_res.R2s_min_opp[m]
                f_min_par[m][iR2s_t, if_t] = MC_sim_res.f_min_par[m]
                f_min_opp[m][iR2s_t, if_t] = MC_sim_res.f_min_opp[m]
            end

            wf_par_min[iR2s_t, if_t, :] = MC_sim_res.wf_par_min
            wf_opp_min[iR2s_t, if_t, :] = MC_sim_res.wf_opp_min
            frac_wf_par_min[iR2s_t, if_t] = MC_sim_res.frac_wf_par_min
            frac_wf_opp_min[iR2s_t, if_t] = MC_sim_res.frac_wf_opp_min

            put!(ch_gre, gre)
        end

        close(ch_gre)

        (; wf_par_min, wf_opp_min,
            frac_wf_par_min, frac_wf_opp_min,
            ϕ_min, ϕ_min_par, ϕ_min_opp,
            R2s_min, R2s_min_par, R2s_min_opp,
            f_min, f_min_par, f_min_opp,
        )
    end
end

"""
    ismrm_challenge(
    greType::Type{<:BM.AbstractGREMultiEcho},
    fitopt::BM.FitOpt;
    data_set::Int,
    ic_dir="test/data/ISMRM_challenge_2012/",
    nTE=0)
    
TBW
"""
function ismrm_challenge(
    greType::Type{<:BM.AbstractGREMultiEcho},
    fitopt::BM.FitOpt;
    data_set::Int,
    ic_dir="test/data/ISMRM_challenge_2012/",
    nTE=0)

    # check that data set exists
    @assert 1 <= data_set <= 17

    # IRMRM challenge fat specification
    ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
    ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

    # read data set
    nmb_str = data_set < 10 ? string("0", data_set) : string(data_set)
    file_str = ic_dir * nmb_str * "_ISMRM.mat"

    datPar = matread(file_str)["imDataParams"]
    TEs = 1000.0 * datPar["TE"][:]
    nTE == 0 && (nTE = length(TEs))
    TEs = TEs[1:nTE]
    B0 = datPar["FieldStrength"]
    precession = (datPar["PrecessionIsClockwise"] != 1.0) ? :clockwise : :counterclockwise

    # set up GRE sequence model
    grePar = VP.modpar(greType;
        ts=TEs,
        B0=B0,
        ppm_fat=ppm_fat,
        ampl_fat=ampl_fat,
        precession=precession)

    # read data and mask
    Nρ = size(datPar["images"])[1:3]
    data = zeros(ComplexF64, Nρ..., nTE)
    copy!(data, reshape(datPar["images"][:, :, :, 1, 1:nTE], Nρ..., nTE))
    data ./= max(abs.(data)...)
    S = datPar["eval_mask"] .!= 0.0

    # generate instance of FitPar
    fitpar = BM.fitPar(grePar, data, S)

    # if ϕ_scale ≠ 1, we need this
    BM.set_num_phase_intervals(fitpar, fitopt, fitopt.n_ϕ)

    # do the work
    BM.local_fit!(fitpar, fitopt)

    # return results
    return (; fitpar)
end
