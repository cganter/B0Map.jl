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
    MC_data_S(
    rng=MersenneTwister(),
    gre,
    ϕ_t,
    R2ss_t,
    fs_t,
    c_t,
    σ,
    n_σ,
)

Generate data and mask as expected by routine `local_fit`.
"""
function MC_data_S(
    rng,
    gre,
    ϕ_t,
    R2ss_t,
    fs_t,
    c_t,
    σ,
    n_σ,
)
    # number of true parameters and other convenience settings
    n_R2s_t = length(R2ss_t)
    n_f_t = length(fs_t)

    # allocate data, set mask
    data = Array{ComplexF64,4}(undef, n_R2s_t, n_f_t, n_σ, BM.nTE(gre))
    S = trues(n_R2s_t, n_f_t, n_σ)

    # generate data
    for (ift, f_t) in enumerate(fs_t)
        cw, cf = (1 - f_t) * c_t, f_t * c_t
        for (iR2st, R2s_t) in enumerate(R2ss_t)
            VP.x!(gre, [ϕ_t, R2s_t])
            data[iR2st, ift, :, :] .=
                reshape(VP.A(gre) * [cw, cf], 1, BM.nTE(gre)) .+ σ .* randn(rng, ComplexF64, n_σ, BM.nTE(gre))
        end
    end

    # return result
    (data, S)
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
    MC_sim_LocFit(;
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
)

TBW
"""
function MC_sim_LocFit(;
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

    # generate data and mask
    (data, S) = MC_data_S(
        rng,
        gre_info.gre[:FW],
        ϕ_t,
        R2ss_t,
        fs_t,
        c_t,
        σ,
        n_σ,
    )

    # GRE VARPRO models
    gre_keys = keys(gre_info.gre)
    fitpar = Dict()
    ϕ_min = Dict()
    R2s_min = Dict()
    f_min = Dict()
    # number of true parameters 
    n_R2s_t = length(R2ss_t)
    n_f_t = length(fs_t)
    sz_t = (n_R2s_t, n_f_t)

    # perform local fits
    for m in gre_keys
        # generate instance of fit parameter and options
        fitpar[m] = BM.fitPar(gre_info.grePar[m], deepcopy(data), deepcopy(S))
        fitopt = BM.fitOpt()
        fitopt.optim = false

        # do local fit (automatically supports multi-threading)
        BM.local_fit(fitpar[m], fitopt)

        # extract maps
        ϕ_min[m] = fitpar[m].ϕ
        R2s_min[m] = fitpar[m].R2s
        f_min[m] = BM.fat_fraction_map(fitpar[m], fitopt)
    end

    # minima, where water and fat are parallel ..
    wf_par_min = map(c -> c[1] * c[2] ≥ 0, fitpar[:RW].c)
    # .. or not
    wf_opp_min = (!).(wf_par_min)
    # fraction of minima with parallel water and fat ..
    frac_wf_par_min = sum(wf_par_min, dims=3) ./ n_σ
    # .. and the complement
    frac_wf_opp_min = 1 .- frac_wf_par_min

    # split information based upon alignment
    ϕ_min_par = Dict()
    ϕ_min_opp = Dict()
    R2s_min_par = Dict()
    R2s_min_opp = Dict()
    f_min_par = Dict()
    f_min_opp = Dict()

    for m in gre_keys
        ϕ_min_par[m] = Matrix{Vector{Float64}}(undef, sz_t...)
        ϕ_min_opp[m] = Matrix{Vector{Float64}}(undef, sz_t...)
        R2s_min_par[m] = Matrix{Vector{Float64}}(undef, sz_t...)
        R2s_min_opp[m] = Matrix{Vector{Float64}}(undef, sz_t...)
        f_min_par[m] = Matrix{Vector{Float64}}(undef, sz_t...)
        f_min_opp[m] = Matrix{Vector{Float64}}(undef, sz_t...)

        for iR2s_t in 1:n_R2s_t
            for if_t in 1:n_f_t
                ϕ_min_par[m][iR2s_t, if_t] = ϕ_min[m][iR2s_t, if_t, :][wf_par_min[iR2s_t, if_t, :]]
                ϕ_min_opp[m][iR2s_t, if_t] = ϕ_min[m][iR2s_t, if_t, :][wf_opp_min[iR2s_t, if_t, :]]
                R2s_min_par[m][iR2s_t, if_t] = R2s_min[m][iR2s_t, if_t, :][wf_par_min[iR2s_t, if_t, :]]
                R2s_min_opp[m][iR2s_t, if_t] = R2s_min[m][iR2s_t, if_t, :][wf_opp_min[iR2s_t, if_t, :]]
                f_min_par[m][iR2s_t, if_t] = f_min[m][iR2s_t, if_t, :][wf_par_min[iR2s_t, if_t, :]]
                f_min_opp[m][iR2s_t, if_t] = f_min[m][iR2s_t, if_t, :][wf_opp_min[iR2s_t, if_t, :]]
            end
        end
    end

    # return everything
    (; wf_par_min, wf_opp_min,
        frac_wf_par_min, frac_wf_opp_min,
        ϕ_min, ϕ_min_par, ϕ_min_opp,
        R2s_min, R2s_min_par, R2s_min_opp,
        f_min, f_min_par, f_min_opp,
    )
end

"""
    MC_sim_LocFit_Δppm(;
    rng=MersenneTwister(),
    TEs,
    B0,
    ppm_fat=[-3.80, -3.40, -2.60, -1.95, -0.5, 0.60],
    ampl_fat=[0.0875, 0.6998, 0.1206, 0.0062, 0.0389, 0.0471],
    ϕ_t,
    R2ss_t,
    fs_t,
    Δppms_t,
    c_t,
    σ,
)

TBW
"""
function MC_sim_LocFit_Δppm(;
    rng=MersenneTwister(),
    TEs,
    B0,
    ppm_fat=[-3.80, -3.40, -2.60, -1.95, -0.5, 0.60],
    ampl_fat=[0.0875, 0.6998, 0.1206, 0.0062, 0.0389, 0.0471],
    ϕ_t,
    R2ss_t,
    fs_t,
    Δppms_t,
    c_t,
    σ,
)
    # finetune fat model amplitudes
    ampl_fat /= sum(ampl_fat)
    @assert all(ampl_fat .> 0) && sum(ampl_fat) ≈ 1

    # number of true parameters 
    n_R2s_t = length(R2ss_t)
    n_f_t = length(fs_t)
    n_Δppms_t = length(Δppms_t)
    sz_t = (n_R2s_t, n_f_t, n_Δppms_t)

    nTE = length(TEs)

    # allocate data and generate mask
    data = Array{ComplexF64,4}(undef, sz_t..., nTE)
    S = trues(sz_t...)

    for (iΔ, Δppm) in enumerate(Δppms_t)
        # generate GRE models with ppm shift
        gre_info = set_up_GREs(;
            TEs=TEs,
            B0=B0,
            ppm_fat=ppm_fat .+ Δppm,
            ampl_fat=ampl_fat,
        )

        # generate data
        (data[:, :, iΔ, :], _) = MC_data_S(
            rng,
            gre_info.gre[:FW],
            ϕ_t,
            R2ss_t,
            fs_t,
            c_t,
            σ,
            1,
        )
    end

    # generate GRE models without ppm shift
    gre_info = set_up_GREs(;
        TEs=TEs,
        B0=B0,
        ppm_fat=ppm_fat,
        ampl_fat=ampl_fat,
    )

    # GRE VARPRO models
    gre_keys = keys(gre_info.gre)
    fitpar = Dict()
    χ2_min = Dict()
    ϕ_min = Dict()
    R2s_min = Dict()
    f_min = Dict()

    # perform local fits
    for m in gre_keys
        # generate instance of fit parameter and options
        fitpar[m] = BM.fitPar(gre_info.grePar[m], deepcopy(data), deepcopy(S))
        fitopt = BM.fitOpt()
        fitopt.optim = false

        # do local fit (automatically supports multi-threading)
        BM.local_fit(fitpar[m], fitopt)

        # extract maps
        χ2_min[m] = fitpar[m].χ2
        ϕ_min[m] = fitpar[m].ϕ
        R2s_min[m] = fitpar[m].R2s
        f_min[m] = BM.fat_fraction_map(fitpar[m], fitopt)
    end

    # minima, where water and fat are parallel ..
    wf_par_min = map(c -> c[1] * c[2] ≥ 0, fitpar[:RW].c)
    # .. or not
    wf_opp_min = (!).(wf_par_min)

    # return everything
    (; wf_par_min, wf_opp_min,
        χ2_min, ϕ_min, R2s_min, f_min, 
    )
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
    BM.local_fit(fitpar, fitopt)

    # return results
    return (; fitpar)
end

"""
    plot_χ2_maps(;
    rng=MersenneTwister(),
    nTE,
    t0,
    ΔTE,
    B0,
    ppm_fat=[-3.80, -3.40, -2.60, -1.95, -0.5, 0.60],
    ampl_fat=[0.0875, 0.6998, 0.1206, 0.0062, 0.0389, 0.0471],
    ϕ_t,
    R2s_t,
    f_t,
    c_t,
    σ=0,
    n_σ=1,
    n_ϕ=181,
    n_R2s=100,
    R2s_max=0.1,
)

Plot `χ²` maps for all GRE models
"""
function plot_χ2_maps(;
    rng=MersenneTwister(),
    nTE,
    t0,
    ΔTE,
    B0,
    ppm_fat=[-3.80, -3.40, -2.60, -1.95, -0.5, 0.60],
    ampl_fat=[0.0875, 0.6998, 0.1206, 0.0062, 0.0389, 0.0471],
    ϕ_t,
    R2s_t,
    f_t,
    c_t,
    σ=0,
    n_σ=1,
    n_ϕ=181,
    n_R2s=100,
    R2s_max=0.1,
)

    # set up parameters
    TEs = [range(t0, t0 + (nTE - 1) * ΔTE, nTE);]
    ampl_fat /= sum(ampl_fat)
    @assert all(ampl_fat .> 0) && sum(ampl_fat) ≈ 1
    ϕs = [range(-π, π, n_ϕ);]
    R2ss = [range(0.0, R2s_max, n_R2s);]

    # generate GRE models
    gre_info = set_up_GREs(;
        TEs=TEs,
        B0=B0,
        ppm_fat=ppm_fat,
        ampl_fat=ampl_fat,
    )

    # calculate χ² maps
    res = χ2_ϕ_R2s(gre_info.gre, ϕ_t, R2s_t, f_t, c_t, σ, n_σ, rng, ϕs, R2ss)

    # plot figure
    fig = Figure(size=(900, 660),
        fontsize=16)

    lim = (0.0, max(res.χ2[:RW][:, :, 1]...))

    # ============= axes =============

    ax_χ2_1 = Axis(fig[1, 1],
        title=L"$\chi^2$ (RW)",
        titlesize=20,
        ylabel=L"$R_2^\ast$ [1/ms]",
        ylabelsize=20,
        xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    )

    ax_χ2_2 = Axis(fig[1, 2],
        title=L"$\chi^2$ (FC)",
        titlesize=20,
        xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    )

    ax_χ2_3 = Axis(fig[1, 4],
        title=L"$\log(\Delta\chi^2)$",
        titlesize=20,
        xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    )

    hidexdecorations!(ax_χ2_1)
    hidexdecorations!(ax_χ2_2)
    hidexdecorations!(ax_χ2_3)
    hideydecorations!(ax_χ2_2)
    hideydecorations!(ax_χ2_3)

    ax_f_1 = Axis(fig[2, 1],
        title=L"$$fat fraction (RW)",
        titlesize=20,
        xlabel=L"$\varphi$",
        xlabelsize=20,
        ylabel=L"$R_2^\ast$ [1/ms]",
        ylabelsize=20,
        xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    )

    ax_f_2 = Axis(fig[2, 2],
        title=L"$$fat fraction (FC)",
        titlesize=20,
        xlabel=L"$\varphi$",
        xlabelsize=20,
        xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    )

    Colorbar(fig[2, 3],
        colormap=:batlowW)

    ax_f_3 = Axis(fig[2, 4],
        title=L"$\log(|\Delta f\,|)$",
        titlesize=20,
        xlabel=L"$\varphi$",
        xlabelsize=20,
        xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    )

    hideydecorations!(ax_f_2)
    hideydecorations!(ax_f_3)

    # ============= plots =============

    heatmap!(ax_χ2_1,
        ϕs, R2ss, res.χ2[:RW][:, :, 1],
        colorrange=lim,
        colormap=:roma,
    )

    heatmap!(ax_χ2_2,
        ϕs, R2ss, res.χ2[:FC][:, :, 1],
        colorrange=lim,
        colormap=:roma,
    )

    Colorbar(fig[1, 3],
        colorrange=lim,
        colormap=:roma)

    heatmap!(ax_χ2_3,
        ϕs, R2ss, res.χ2[:FC][:, :, 1] - res.χ2[:RW][:, :, 1],
        #ϕs, R2ss, log.(abs.(res.χ2[:FC][:, :, 1] - res.χ2[:RW][:, :, 1])),
        colormap=:roma,
    )

    heatmap!(ax_f_1,
        ϕs, R2ss, res.f[:RW][:, :, 1],
        colormap=:batlowW,
    )

    heatmap!(ax_f_2,
        ϕs, R2ss, res.f[:FC][:, :, 1],
        colormap=:batlowW,
    )

    heatmap!(ax_f_3,
        ϕs, R2ss, log.(abs.(res.f[:FC][:, :, 1] - res.f[:RW][:, :, 1])),
        colormap=:batlowW,
    )

    scatter!(ax_χ2_1, [res.ϕ_min[:RW][1];], [res.R2s_min[:RW][1];], color=:lime)
    scatter!(ax_χ2_2, [res.ϕ_min[:FC][1];], [res.R2s_min[:FC][1];], color=:lime)
    scatter!(ax_χ2_1, [ϕ_t;], [R2s_t;], color=:yellow)
    scatter!(ax_χ2_2, [ϕ_t;], [R2s_t;], color=:yellow)
    scatter!(ax_χ2_3, [ϕ_t;], [R2s_t;], color=:yellow)
    scatter!(ax_f_1, [ϕ_t;], [R2s_t;], color=:yellow)
    scatter!(ax_f_2, [ϕ_t;], [R2s_t;], color=:yellow)
    scatter!(ax_f_3, [ϕ_t;], [R2s_t;], color=:yellow)

    display(fig)

    (; res, fig)
end

"""
    β_fat(B0, ppm_fat, ampl_fat, precession=:counterclockwise)

TBW
"""
function β_fat(B0, ppm_fat, ampl_fat, precession=:counterclockwise)
    @assert precession ∈ (:clockwise, :counterclockwise)

    fac = im * 2π * 0.042577 * B0
    precession == :clockwise && (fac = -fac)

    t -> let fac = fac, ppm_fat = ppm_fat, ampl_fat = ampl_fat
        ampl_fat' * exp.(fac * ppm_fat * t)
    end
end

#=
ax = Axis(f[1, 1],
    title=L"$\chi^2$",
    titlesize=20,
    xlabel=L"$\varphi$",
    xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)
lines!(ax, ϕs, res_MC.χ2[:FC][:, iR2st])
lines!(ax, ϕs, res_MC.χ2[:RW][:, iR2st])

"""
Show differences between real-valued and fully-constrained VARPRO model
"""
function plot_differences(MC_sim)
    fig = Figure(size=(900, 660),
        fontsize=16)

    lim = (0.0, max(res.χ2[:RW][:, :, 1]...))

    # ============= axes =============

    ax_χ2_1 = Axis(fig[1, 1],
        title=L"$\chi^2$ (RW)",
        titlesize=20,
        ylabel=L"$R_2^\ast$ [1/ms]",
        ylabelsize=20,
        xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    )

    ax_χ2_2 = Axis(fig[1, 2],
        title=L"$\chi^2$ (FC)",
        titlesize=20,
        xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    )

    ax_χ2_3 = Axis(fig[1, 4],
        title=L"$\log(\Delta\chi^2)$",
        titlesize=20,
        xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    )

    hidexdecorations!(ax_χ2_1)
    hidexdecorations!(ax_χ2_2)
    hidexdecorations!(ax_χ2_3)
    hideydecorations!(ax_χ2_2)
    hideydecorations!(ax_χ2_3)

    ax_f_1 = Axis(fig[2, 1],
        title=L"$$fat fraction (RW)",
        titlesize=20,
        xlabel=L"$\varphi$",
        xlabelsize=20,
        ylabel=L"$R_2^\ast$ [1/ms]",
        ylabelsize=20,
        xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    )

    ax_f_2 = Axis(fig[2, 2],
        title=L"$$fat fraction (FC)",
        titlesize=20,
        xlabel=L"$\varphi$",
        xlabelsize=20,
        xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    )

    Colorbar(fig[2, 3],
        colormap=:batlowW)

    ax_f_3 = Axis(fig[2, 4],
        title=L"$\log(|\Delta f\,|)$",
        titlesize=20,
        xlabel=L"$\varphi$",
        xlabelsize=20,
        xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    )

    hideydecorations!(ax_f_2)
    hideydecorations!(ax_f_3)

    # ============= plots =============

    heatmap!(ax_χ2_1,
        ϕs, R2ss, res.χ2[:RW][:, :, 1],
        colorrange=lim,
        colormap=:roma,
    )

    heatmap!(ax_χ2_2,
        ϕs, R2ss, res.χ2[:FC][:, :, 1],
        colorrange=lim,
        colormap=:roma,
    )

    Colorbar(fig[1, 3],
        colorrange=lim,
        colormap=:roma)

    heatmap!(ax_χ2_3,
        ϕs, R2ss, res.χ2[:FC][:, :, 1] - res.χ2[:RW][:, :, 1],
        #ϕs, R2ss, log.(abs.(res.χ2[:FC][:, :, 1] - res.χ2[:RW][:, :, 1])),
        colormap=:roma,
    )

    heatmap!(ax_f_1,
        ϕs, R2ss, res.f[:RW][:, :, 1],
        colormap=:batlowW,
    )

    heatmap!(ax_f_2,
        ϕs, R2ss, res.f[:FC][:, :, 1],
        colormap=:batlowW,
    )

    heatmap!(ax_f_3,
        ϕs, R2ss, log.(abs.(res.f[:FC][:, :, 1] - res.f[:RW][:, :, 1])),
        colormap=:batlowW,
    )

    scatter!(ax_χ2_1, [res.ϕ_min[:RW][1];], [res.R2s_min[:RW][1];], color=:lime)
    scatter!(ax_χ2_2, [res.ϕ_min[:FC][1];], [res.R2s_min[:FC][1];], color=:lime)
    scatter!(ax_χ2_1, [ϕ_t;], [R2s_t;], color=:yellow)
    scatter!(ax_χ2_2, [ϕ_t;], [R2s_t;], color=:yellow)
    scatter!(ax_χ2_3, [ϕ_t;], [R2s_t;], color=:yellow)
    scatter!(ax_f_1, [ϕ_t;], [R2s_t;], color=:yellow)
    scatter!(ax_f_2, [ϕ_t;], [R2s_t;], color=:yellow)
    scatter!(ax_f_3, [ϕ_t;], [R2s_t;], color=:yellow)

    display(fig)
    (; fig)
end

rng = MersenneTwister()

res = plot_differences(;
    rng=rng,
    nTE=3,
    t0=1.5,
    ΔTE=1.5,
    B0=3,
    ϕ_t=0.5π,
    n_R2s_t=5,
    n_f_t=11,
    c_t=1,
    σ=0.05,
    n_σ=100,
    n_ϕ=37,
    n_R2s=11,
    R2s_max=1,
);

## =============== generate χ² maps =======================

rng = MersenneTwister(42)

ϕs = [range(-π, π, 361);]
R2ss = [range(0, 0.2, 101);]

res_MC = MC_sim(;
    TEs=[1.2 .+ 1.2(0:2);],
    B0=3,
    ppm_fat=[-3.80, -3.40, -2.60, -1.95, -0.5, 0.60],
    ampl_fat=[0.0875, 0.6998, 0.1206, 0.0062, 0.0389, 0.0471],
    ϕ_t=0,
    R2ss_t = [0.1],
    fs_t = [0.05],
    c_t=randn(rng, ComplexF64),
    σ=0.05,
    n_σ= 1,
    ϕs = ϕs,
    R2ss = R2ss,
)

nrows, ncols = 1, 1
wi = he = 500
width, height = wi * ncols, he * nrows

f = Figure(size = (width, height))

#ax = [Axis(f[i,j]) for i in 1:nrows, j in 1:ncols]

ax = Axis(f[1, 1],
    title=L"$\chi^2$",
    titlesize=20,
    xlabel=L"$\varphi$",
    xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)
lines!(ax, ϕs, res_MC.χ2[:FC][:,1])
lines!(ax, ϕs, res_MC.χ2[:RW][:,1])
display(f)
##
lines!(ax[1,2], ϕs, res_MC.f[:FC][:,1])
lines!(ax[1,2], ϕs, res_MC.f[:RW][:,1])
heatmap!(ax[2, 1], log.(abs.(res_MC.χ2[:RW])), colormap=:roma)
heatmap!(ax[2, 2], log.(abs.(res_MC.χ2[:FC])), colormap=:roma)
heatmap!(ax[3, 1], log.(abs.(res_MC.χ2[:FC] .- res_MC.χ2[:RW])), colormap=:roma)
heatmap!(ax[3, 2], res_MC.wf_opp)
f01 = map(f -> f ∈ (0.0, 1.0), res_MC.f[:FC])
heatmap!(ax[3, 3], f01)

hidexdecorations!(ax_χ2_1)
hidexdecorations!(ax_χ2_2)
hidexdecorations!(ax_χ2_3)
hideydecorations!(ax_χ2_2)
hideydecorations!(ax_χ2_3)

Colorbar(f[1, 4],
    colorrange=lim,
    colormap=:batlowW)

##

save("test.svg", f)
run(`/home/cganter/bin/svg2eps test`)

## =============== generate χ² maps =======================

res_MC = MC_sim_LocFit(;
    TEs=[1.5 .+ 1.5(0:2);],
    B0=3,
    ppm_fat=[-3.80, -3.40, -2.60, -1.95, -0.5, 0.60],
    ampl_fat=[0.0875, 0.6998, 0.1206, 0.0062, 0.0389, 0.0471],
    ϕ_t=0,
    R2ss_t=[range(0, 0.2, 51);],
    fs_t=[range(0, 1, 51);],
    c_t=randn(ComplexF64),
    σ=0.05,
    n_σ=100,
)

## =============== generate χ² maps =======================

res = plot_χ2_maps(;
    nTE=3,
    t0=1.5,
    ΔTE=1.5,
    B0=3,
    ϕ_t=0.5π,
    R2s_t=0.05,
    f_t=0.1,
    c_t=1,
    σ=0.05,
    n_σ=1,
    n_ϕ=181,
    n_R2s=101,
    R2s_max=1,
);



## ====================================================================================================================


f

#
fig = Figure(size=(800, 800),
    fontsize=16)

min_RW = [(χ2 = res.χ2[:RW][i, :, 1]; f = res.f[:RW][i, :, 1]; wfp = res.wf_par[i, :, 1]; j = argmin(χ2); (R2ss[j], f[j], χ2[j], wfp[j])) for i in 1:n_ϕ]
min_R2s_RW = map(x -> x[1], min_RW)
min_f_RW = map(x -> x[2], min_RW)
min_χ2_RW = map(x -> x[3], min_RW)
msk_wfp = map(x -> x[4], min_RW)
msk_wfo = (!).(msk_wfp)

min_FC = [(χ2 = res.χ2[:FC][i, :, 1]; f = res.f[:FC][i, :, 1]; j = argmin(χ2); (R2ss[j], f[j], χ2[j])) for i in 1:n_ϕ]
min_R2s_FC = map(x -> x[1], min_FC)
min_f_FC = map(x -> x[2], min_FC)
min_χ2_FC = map(x -> x[3], min_FC)

msk_min_RW = min_χ2_RW .< 1e-4
msk_min_FC = min_χ2_FC .< 1e-4

ax_R2s = Axis(fig[1, 1],
    title=L"$R_2^\ast$",
    titlesize=20,
    xlabel=L"$\varphi$",
    xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)
lines!(ax_R2s, ϕs, min_R2s_FC)
scatter!(ax_R2s, ϕs[msk_min_FC.&msk_wfp], min_R2s_FC[msk_min_FC.&msk_wfp], color=:green)
scatter!(ax_R2s, ϕs[msk_min_FC.&msk_wfo], min_R2s_FC[msk_min_FC.&msk_wfo], color=:red)

ax_f = Axis(fig[1, 2],
    title=L"$$fat fraction",
    titlesize=20,
    xlabel=L"$\varphi$",
    xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)
lines!(ax_f, ϕs, min_f_FC)
scatter!(ax_f, ϕs[msk_min_FC.&msk_wfp], min_f_FC[msk_min_FC.&msk_wfp], color=:green)
scatter!(ax_f, ϕs[msk_min_FC.&msk_wfo], min_f_FC[msk_min_FC.&msk_wfo], color=:red)

ax_χ2 = Axis(fig[2, 1],
    title=L"$\chi^2$",
    titlesize=20,
    xlabel=L"$\varphi$",
    xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"])
)

#lines!(ax_χ2, ϕs, log10.(min_χ2_RW))
#lines!(ax_χ2, ϕs, log10.(min_χ2_FC))

lines!(ax_χ2, ϕs, log10.(abs.(min_χ2_FC)))
lines!(ax_χ2, ϕs, log10.(abs.(res.χ2[:FC][:, 1, 1])), color=:red)
#scatter!(ax_χ2, ϕs[msk_min_FC .& msk_wfp], min_χ2_FC[msk_min_FC .& msk_wfp], color=:green)
#scatter!(ax_χ2, ϕs[msk_min_FC .& msk_wfo], min_χ2_FC[msk_min_FC .& msk_wfo], color=:red)

ax_f_R2s = Axis(fig[2, 2],
    title=L"$R_2^\ast$",
    titlesize=20,
    xlabel=L"$$fat fraction",
    xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"])
)

#lines!(ax_f_R2s, min_R2s_RW[msk_min_RW], min_f_RW[msk_min_RW])
scatter!(ax_f_R2s, min_f_FC[msk_min_FC.&msk_wfp], min_R2s_FC[msk_min_FC.&msk_wfp], color=:green)
scatter!(ax_f_R2s, min_f_FC[msk_min_FC.&msk_wfo], min_R2s_FC[msk_min_FC.&msk_wfo], color=:red)

# two echo fit

# sample values
n_ϕ, n_R2s = 181, 100
ϕs = [range(-π, π, n_ϕ);]
R2ss = [range(0.0, 1.0, 100);]
fs = collect(range(0, 1, 100))

ϕ_t = 0.5π
R2s_t = 0.2
f_t = 0.9

res = two_echo_fit(gre_info.gre, ϕ_t, R2s_t, ff_t, 0, rng, fs);

eiϕ = exp.(im .* ϕs)
res_fit = exp.(-res.R2ss .* ΔTE) .* exp.(im .* res.ϕs)

f = Figure(size=(800, 400))
ax11 = Axis(f[1, 1])
ax12 = Axis(f[1, 2])

lines!(ax11, real.(eiϕ), imag.(eiϕ))
scatter!(ax11, real.(res_fit), imag.(res_fit), color=:green)

lines!(ax12, real.(eiϕ), imag.(eiϕ))
scatter!(ax12, real.(res.d21fw12), imag.(res.d21fw12), color=:green)

f

##

# sample values
n_ϕ, n_R2s = 181, 100
ϕs = [range(-π, π, n_ϕ);]
R2ss = [range(0.0, 1.0, 100);]

# true values
ϕ_t = 0
R2s_t = 0.1
ff_t = 0.1
c_t = randn(rng, ComplexF64, n_coils)  #exp.(im * 2π * (rand(rng, n_coils) .- 0.5))
#R2s_t, ff_t = min_R2s_FC[6], min_ff_FC[6]
# no noise
σ, n_σ = 0.01, 1

# perform simulation
res = χ2_ϕ_R2s(gre_info.gre, ϕ_t, R2s_t, ff_t, c_t, σ, n_σ, rng, ϕs, R2ss)

## generate plots 


##

ts = [0.5, 1]
#ts = range(0, 10, 100)
fs = collect(range(0, 1, 100))

function wts(ts, ppm_fat, ampl_fat; precession=:counterclockwise, B0=3.0)
    fac = im * 2π * 0.042577 * B0
    precession == :clockwise && (fac = -fac)
    (sum(ampl_fat' .* exp.(fac * ppm_fat' .* ts), dims=2).-1)[:]
end

ws = wts(ts, ppm_fat, ampl_fat)

f12s = map(f -> (1 + f * ws[1]) / (1 + f * ws[2]), fs)


#lines(real.(f12s), imag.(f12s))
lines(fs, angle.(f12s))
lines(fs, abs.(f12s))

##

f = Figure(size=(900, 660),
    fontsize=16)

lim = (0.0, max(res.χ2[:RW][:, :, 1]...))

ax_χ2_1 = Axis(f[1, 1],
    title=L"$\chi^2$ (FW)",
    titlesize=20,
    #xlabel=L"$\varphi$",
    #xlabelsize=20,
    ylabel=L"$R_2^\ast$ [1/ms]",
    ylabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

ax_χ2_2 = Axis(f[1, 2],
    title=L"$\chi^2$ (RW)",
    titlesize=20,
    #xlabel=L"$\varphi$",
    #xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

ax_χ2_3 = Axis(f[1, 3],
    title=L"$\chi^2$ (FC)",
    titlesize=20,
    #xlabel=L"$\varphi$",
    #xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

hidexdecorations!(ax_χ2_1)
hidexdecorations!(ax_χ2_2)
hidexdecorations!(ax_χ2_3)
hideydecorations!(ax_χ2_2)
hideydecorations!(ax_χ2_3)

Colorbar(f[1, 4],
    colorrange=lim,
    colormap=:batlowW)

heatmap!(ax_χ2_1,
    ϕs, R2ss, res.χ2[:FW][:, :, 1],
    colorrange=lim,
    colormap=:batlowW,
)

heatmap!(ax_χ2_2,
    ϕs, R2ss, res.χ2[:RW][:, :, 1],
    colorrange=lim,
    colormap=:batlowW,
)

heatmap!(ax_χ2_3,
    ϕs, R2ss, res.χ2[:FC][:, :, 1],
    colorrange=lim,
    colormap=:batlowW,
)

ax_ff_1 = Axis(f[2, 1],
    title=L"$$fat fraction (FW)",
    titlesize=20,
    xlabel=L"$\varphi$",
    xlabelsize=20,
    ylabel=L"$R_2^\ast$ [1/ms]",
    ylabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

ax_ff_2 = Axis(f[2, 2],
    title=L"$$fat fraction (RW)",
    titlesize=20,
    xlabel=L"$\varphi$",
    xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

ax_ff_3 = Axis(f[2, 3],
    title=L"$$fat fraction (FC)",
    titlesize=20,
    xlabel=L"$\varphi$",
    xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

hideydecorations!(ax_ff_2)
hideydecorations!(ax_ff_3)

Colorbar(f[2, 4],
    colormap=:batlowW)

heatmap!(ax_ff_1,
    ϕs, R2ss, res.ff[:FW][:, :, 1],
    colormap=:batlowW,
)

heatmap!(ax_ff_2,
    ϕs, R2ss, res.ff[:RW][:, :, 1],
    colormap=:batlowW,
)

heatmap!(ax_ff_3,
    ϕs, R2ss, res.ff[:FC][:, :, 1],
    colormap=:batlowW,
)

f

##

save("chi2.svg", f)
#run(`inkscape -o chi2.eps chi2.svg`)

##


# generate plot

plts = Matrix(undef, 3, 1)

cmp = :batlow
dpi = 300
subplt_size = (330, 220)
margin = 2mm

plts[1, 1] = heatmap(ϕs, R2ss, res.χ2[:FW][:, :, 1]',
    color=cmp,
    margin=margin,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    title=L"$χ^2$ (FW)",
    titlefontsize=12)

xlabel!(L"$\phi$ [deg]")
ylabel!(L"$R_2^\ast$ [1/ms]")

plts[2, 1] = heatmap(ϕs, R2ss, res.χ2[:RW][:, :, 1]',
    color=cmp,
    margin=margin,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    title=L"$χ^2$ (RV)",
    titlefontsize=12)

xlabel!(L"$\phi$ [deg]")

plts[3, 1] = heatmap(ϕs, R2ss, res.χ2[:FC][:, :, 1]',
    color=cmp,
    margin=margin,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    title=L"$χ^2$ (FC)",
    titlefontsize=12)

xlabel!(L"$\phi$ [deg]")

display(plot(plts...,
    layout=size(plts'),
    size=size(plts) .* subplt_size))

##

savefig("/home/cganter/tmp/chi2.eps")

##

# noise
σ = 0.05
n_σ = 1000

# sample values
n_ϕ = 180
ϕs = [range(-π, π, n_ϕ + 1);][1:end-1]
R2ss = [0.0;]

# true values
ϕ_t = 0.0
n_R2s_t, n_ff_t = 10, 11
R2ss_t = [range(0.0, 0.1, n_R2s_t);]
ffs_t = [range(0.0, 1.0, n_ff_t);]
θ_t = 2π * (rand(rng) - 0.5)

# MC simulation
res = Array{Any}(undef, n_R2s_t, n_ff_t)

# create channels
ch_gre = Channel{Dict{Any,Any}}(Threads.nthreads())

for _ in 1:Threads.nthreads()
    put!(ch_gre, deepcopy(gre_info.gre))
end


Threads.@threads for (iR2s_t, iff_t) in collect(Iterators.product(1:n_R2s_t, 1:n_ff_t))
    println("threadid: ", Threads.threadid(), ", ", (iR2s_t, iff_t))
    gre = take!(ch_gre)

    R2s_t, ff_t = R2ss_t[iR2s_t], ffs_t[iff_t]
    R2ss = [0.0]
    #R2ss = [R2s_t;]
    res[iR2s_t, iff_t] = χ2_ϕ_R2s(gre, ϕ_t, R2s_t, ff_t, θ_t, σ, n_σ, rng, ϕs, R2ss)

    put!(ch_gre, gre)
end

# regroup information

msk_par = map(r -> r.wf_par_min, res)
msk_opp = map(mp -> (!).(mp), msk_par)
frac_par = map(mp -> sum(mp) / n_σ, msk_par)
frac_opp = 1 .- frac_par

ϕ, ϕ_par, ϕ_opp = Dict(), Dict(), Dict()
mean_ϕ, mean_ϕ_par, mean_ϕ_opp = Dict(), Dict(), Dict()
std_ϕ, std_ϕ_par, std_ϕ_opp = Dict(), Dict(), Dict()

ff, ff_par, ff_opp = Dict(), Dict(), Dict()
mean_ff, mean_ff_par, mean_ff_opp = Dict(), Dict(), Dict()
std_ff, std_ff_par, std_ff_opp = Dict(), Dict(), Dict()

for m in keys(gre)
    ϕ[m] = map(r -> r.ϕ_min[m], res)
    ϕ_par[m] = map((mp, ϕ_) -> ϕ_[mp], msk_par, ϕ[m])
    ϕ_opp[m] = map((mo, ϕ_) -> ϕ_[mo], msk_opp, ϕ[m])

    mean_ϕ[m] = mean.(ϕ[m])
    mean_ϕ_par[m] = map(ϕp -> isempty(ϕp) ? 0.0 : mean(ϕp), ϕ_par[m])
    mean_ϕ_opp[m] = map(ϕo -> isempty(ϕo) ? 0.0 : mean(ϕo), ϕ_opp[m])

    std_ϕ[m] = std.(ϕ[m])
    std_ϕ_par[m] = map(ϕp -> isempty(ϕp) ? 0.0 : std(ϕp), ϕ_par[m])
    std_ϕ_opp[m] = map(ϕo -> isempty(ϕo) ? 0.0 : std(ϕo), ϕ_opp[m])

    ff[m] = map(r -> r.ff_min[m], res)
    ff_par[m] = map((mp, ff_) -> ff_[mp], msk_par, ff[m])
    ff_opp[m] = map((mo, ff_) -> ff_[mo], msk_opp, ff[m])

    mean_ff[m] = mean.(ff[m])
    mean_ff_par[m] = map(ffp -> isempty(ffp) ? 0.0 : mean(ffp), ff_par[m])
    mean_ff_opp[m] = map(ffo -> isempty(ffo) ? 0.0 : mean(ffo), ff_opp[m])

    std_ff[m] = std.(ff[m])
    std_ff_par[m] = map(ffp -> isempty(ffp) ? 0.0 : std(ffp), ff_par[m])
    std_ff_opp[m] = map(ffo -> isempty(ffo) ? 0.0 : std(ffo), ff_opp[m])
end

# differences between FC and RW, split with respect to whether water and fat are parallel in RW

ϕ_par_FC_RW = map((ϕpFC, ϕpRW) -> ϕpFC - ϕpRW, ϕ_par[:FC], ϕ_par[:RW])
ff_par_FC_RW = map((ffpFC, ffpRW) -> ffpFC - ffpRW, ff_par[:FC], ff_par[:RW])
ϕ_opp_FC_RW = map((ϕoFC, ϕoRW) -> ϕoFC - ϕoRW, ϕ_opp[:FC], ϕ_opp[:RW])
ff_opp_FC_RW = map((ffoFC, ffoRW) -> ffoFC - ffoRW, ff_opp[:FC], ff_opp[:RW])

maxabs_ϕ_par_FC_RW = map(x -> isempty(x) ? 0.0 : max(abs.(x)...), ϕ_par_FC_RW)
maxabs_ff_par_FC_RW = map(x -> isempty(x) ? 0.0 : max(abs.(x)...), ff_par_FC_RW)
maxabs_ϕ_opp_FC_RW = map(x -> isempty(x) ? 0.0 : max(abs.(x)...), ϕ_opp_FC_RW)
maxabs_ff_opp_FC_RW = map(x -> isempty(x) ? 0.0 : max(abs.(x)...), ff_opp_FC_RW)



##

#=

    Generate GRE models for given sequence and tissue parameters

## Return (as `@NamedTuple`)
- `gre::Dict`: GRE models (see docs for `χ2_ϕ_R2s` above)
- `grePar::Dict`: associated parameters
For given true values `ϕ_t`, `R2s_t`, `ff_t` and `θ_t`, analyze the three GRE models at `ϕ ∈ ϕs` and `R2s ∈ R2ss`.

## Arguments
- `gre::Dict`: Dictionary with all GRE models, specified by keys `m ∈ (:FW, :RW, :FC)` (see below)
- `ϕ_t::Float64`: local off-resonance phase `ϕ_t = ω_t * TR`.
- `R2ss_t::Float64`: Relaxation rate [ms].
- `ff_t::Float64`: local fat fraction
- `θ_t::`: local coil phase
- `σ::Float64`: Gaussian data noise standard deviation
- `n_σ::Int`: Number of noise samples for MC simulation
- `rng::MersenneTwister`: Seed to generte repeatable random numbers, if desired
- `ϕs::Vector{Float64}`: Phase values to be sampled
- `R2ss::Vector{Float64}`: `R2s` values to be sampled

## Returns a named tuple with fields
- `χ2::Dict`: Contains `χ2[m]::Array{Float64, 3}`, with elements defined as `χ²(ϕ, R2s, iσ)` with `ϕ ∈ ϕs` and `R2s ∈ R2ss` and `iσ` specifying a specific MC noise instance.
- `ff::Dict`: Contains fat fractions `ff[m]::Array{Float64, 3}`
- `wf_par::Array{Bool, 3}`: `true`, iff water and fat are parallel in the real-valued (`:RW`) model
- `argmin_χ2::Dict`: with `argmin_χ2[m]::Vector{CartesianIndex{3}}` providing the location of the minimal `χ²` for each tested `σ_`
- `ϕ_min::Dict`: with `ϕ_min[m]::Vector{Float64}` containing the phase `ϕ` at the found minima in each MC run
- `R2s_min::Dict`: with `R2s_min[m]::Vector{Float64}` containing the transverse relaxation rate at the found minima in each MC run
- `ff_min::Dict`: with `ff_min[m]::Vector{Float64}` containing the fat fraction at the found minima in each MC run
- `wf_par_min::Vector{Bool}`: For each MC run, shows whether water and fat are parallel inthe best fit 
## Remarks
- VARPRO variants in `gre`: `:FW` (unconstrained), `:RW` (real-valued weights), `:FC` (fully constrained)
"""
=#

#=

function show_fat_fraction(
    f_FW,
    f_RW,
    f_FC,
    subplt_size=(330, 220),
    dpi=100,
    margin=2mm,
    errbnds=(-0.11, 0.11),
    ncats=11,
    cmp=:batlow)

    plts = Matrix(undef, 2, length(seqs))



    for (is, s) in enumerate(seqs)
        if typeof(s) == Ernst
            title = "Ernst formula"
        elseif typeof(s) == Coherent
            title = string("configuration (n = ", f(:n, s), ")")
        elseif typeof(s) == RFSpoiled
            title = string("RF spoiled, (ϕ = ", f(:ϕ_inc_deg, s), "°)")
        elseif typeof(s) == Balanced
            title = "bSSFP"
        end

        plts[1, is] = plot()

        for (iβ, β) in enumerate(βs)
            plot!(
                rad2deg.(αs[:]),
                reshape(abs.(f(:m, s)[iβ, :]), length(αs)),
                label=string(β),
                margin=margin,
                title=title,
                titlefontsize=12,
                legend=:outerright,
                legend_title="β"
            )
        end

        is == length(seqs) && xlabel!(L"$\alpha$ [deg]")
        ylabel!(L"$\beta\cdot |m(R_1/\beta^2,\alpha/\beta)|$")

        par!(s, Dict(
            :ρ => ρ,
            :R1 => R1,
            :α_nom => αs))

        m_1 = f(:m, s)

        par!(s, Dict(
            :ρ => βr .* ρ,
            :R1 => R1 ./ βr .^ 2,
            :α_nom => αs ./ βr))

        m_β = f(:m, s)

        plts[2, is] = heatmap(rad2deg.(αs[:]),
            βr[:],
            errfak .* (abs.(m_β ./ m_1) .- 1),
            yticks=βs,
            clim=errfak .* errbnds,
            color=ncats == 0 ? cmp : cgrad(cmp, ncats, categorical=true),
            margin=margin,
            title="rel. deviation [%]",
            titlefontsize=12)

        is == length(seqs) && xlabel!(L"$\alpha$ [deg]")
        ylabel!(L"\beta")
    end

    display(plot(plts...,
        layout=size(plts'),
        dpi=dpi,
        size=size(plts) .* subplt_size))
end
=#

##

heatmap(ϕ_FC)

##

cmp = :batlow
cmpO = :romaO
cmpdiv = :vik

plts = Matrix(undef, 2,)



plts[2, is] = heatmap(rad2deg.(αs[:]),
    βr[:],
    errfak .* (abs.(m_β ./ m_1) .- 1),
    yticks=βs,
    clim=errfak .* errbnds,
    color=cmp,
    margin=margin,
    title="rel. deviation [%]",
    titlefontsize=12)

display(plot(plts...,
    layout=size(plts'),
    dpi=dpi,
    size=size(plts) .* subplt_size))

=#