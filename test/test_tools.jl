using Random, LinearAlgebra, ChunkSplitters, Statistics
import VP4Optim as VP
import B0Map as BM

mutable struct SimPhaPar
    # GRE acquisition parameters
    TEs::Vector{Float64}
    B0::Float64
    precession::Symbol
    # noise covariance matrix
    cov_mat::Matrix{Float64}
    # add noise to data
    add_noise::Bool
    # FOV (without oversampling)
    Nρ::Vector{Int}
    # subspace kernel
    K::Vector{Int} # PHASER
    K_pha::Vector{Int} # phantom
    # oversampling factor(s)
    os_fac::Vector{Float64} # PHASER
    os_fac_pha::Vector{Float64} # phantom
    # fat model
    ppm_fat::Vector{Float64} # PHASER
    ampl_fat::Vector{Float64} # PHASER
    ppm_fat_pha::Vector{Float64} # phantom
    ampl_fat_pha::Vector{Float64} # phantom
    # parameter ranges
    S_rng::Vector{Float64}
    freq_rng::Vector{Float64}
    ϕ_rng::Vector{Float64}
    R2s_rng::Vector{Float64} # PHASER
    R2s_rng_pha::Vector{Float64} # phantom
    f_rng::Vector{Float64} # PHASER
    f_rng_pha::Vector{Float64} # phantom
    coils_rng::Vector{Float64}
    # number of sincs 
    S_nSinc::Int
    ϕ_nSinc::Int
    R2s_nSinc::Int
    f_nSinc::Int
    coils_nSinc::Int
    # zero crossings of sincs over FOV
    S_zc::Float64
    ϕ_zc::Float64
    R2s_zc::Float64
    f_zc::Float64
    coils_zc::Float64
    # fraction of voids in ROI
    S_holes::Float64
    # use voids (:out) or not (:in)
    S_io::Symbol
    # project phase to smooth subspace
    ϕ_proj::Bool
    # phase median over ROI (abs(ϕ_med) < π)
    ϕ_med::Float64
    # PHASER settings
    redundancy::Float64
    subsampling::Symbol
    remove_outliers::Bool
    locfit::Bool
    optim::Bool
    optim_phaser::Bool
    λ_tikh::Float64
    # miscellaneous settings
    n_chunks::Int
    rng::MersenneTwister
end

function SimPhaPar()
    # GRE acquisition parameters
    TEs = Float64[]
    B0 = 3.0
    precession = :counterclockwise
    # noise covariance matrix
    cov_mat = Float64[1;;]
    # add noise to data
    add_noise = false
    # FOV (without oversampling)
    Nρ = Int[]
    # subspace kernel
    K = Int[] # PHASER
    K_pha = Int[] # phantom
    # oversampling factor(s)
    os_fac = [1.5] # PHASER
    os_fac_pha = Float64[] # phantom
    # fat model (ISMRM challenge)
    ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60] # PHASER
    ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048] # PHASER
    ppm_fat_pha = [] # phantom
    ampl_fat_pha = [] # phantom
    # parameter ranges
    S_rng = [-1.0, 1.0]
    freq_rng = Float64[] # frequency limits [kHz]
    ϕ_rng = Float64[]
    R2s_rng = [0.0, 1.0] # PHASER
    R2s_rng_pha = Float64[] # phantom
    f_rng = [0.0, 1.0] # PHASER
    f_rng_pha = Float64[] # phantom
    coils_rng = [1e-3, 1.0]
    # number of sincs 
    S_nSinc = 3
    ϕ_nSinc = 5
    R2s_nSinc = 5
    f_nSinc = 5
    coils_nSinc = 2
    # zero crossings of sincs over FOV
    S_zc = 4.0
    ϕ_zc = 5.0
    R2s_zc = 7.0
    f_zc = 7.0
    coils_zc = 3.0
    # fraction of voids in ROI
    S_holes = 0.7
    # use voids (:out) or not (:in)
    S_io = :out
    # project phase to smooth subspace
    ϕ_proj = true
    # phase median over ROI (abs(ϕ_med) < π)
    ϕ_med = 0.0
    # PHASER settings
    redundancy = Inf
    subsampling = :fibonacci
    remove_outliers = true
    locfit = true
    optim = true
    optim_phaser = false
    λ_tikh = 100eps()
    # miscellaneous settings
    n_chunks = 8Threads.nthreads()
    rng = MersenneTwister()

    SimPhaPar(
        TEs, B0, precession, cov_mat, add_noise,
        Nρ, K, K_pha, os_fac, os_fac_pha, 
        ppm_fat, ampl_fat, ppm_fat_pha, ampl_fat_pha,
        S_rng, freq_rng, ϕ_rng, R2s_rng, R2s_rng_pha, f_rng, f_rng_pha, coils_rng,
        S_nSinc, ϕ_nSinc, R2s_nSinc, f_nSinc, coils_nSinc,
        S_zc, ϕ_zc, R2s_zc, f_zc, coils_zc,
        S_holes, S_io,
        ϕ_proj, ϕ_med,
        redundancy, subsampling, remove_outliers, locfit, optim, optim_phaser, λ_tikh, n_chunks, rng
    )
end

"""
    create_sinc_map(Nρ, nSincs, zc, mima, rng=MersenneTwister())

TBW
"""
function create_sinc_map(S, nSincs, zc, mima;
    n_chunks=8Threads.nthreads(), rng=MersenneTwister())
    # map size
    Nρ = size(S)
    sinc_map = zeros(Nρ)
    ciS = CartesianIndices(S)[S]
    ciS_chunks = [view(ciS, index_chunks(ciS, n=n_chunks)[i]) for i in 1:n_chunks]

    # generate sinc map
    x0s = [(n -> rand(rng, 1:n)).(Nρ) for _ in 1:nSincs]
    ampls = randn(rng, nSincs)

    # do the work
    Threads.@threads for ciS_chunk in ciS_chunks
        create_sinc_chunk(sinc_map, ciS_chunk, x0s, ampls, Nρ, zc)
    end

    # adjust to the desired range
    min_sm = @views min(sinc_map[S]...)
    max_sm = @views max(sinc_map[S]...)
    a = (mima[1] - mima[2]) / (min_sm - max_sm)
    b = (min_sm * mima[2] - max_sm * mima[1]) / (min_sm - max_sm)

    sinc_map[S] = @views a * sinc_map[S] .+ b

    # return map
    sinc_map
end

"""
    create_sinc_chunk(sinc_map, ciS_chunk, x0s, ampls, Nρ, zc)

TBW
"""
function create_sinc_chunk(sinc_map, ciS_chunk, x0s, ampls, Nρ, zc)
    den = 1 ./ (Nρ .÷ zc)
    for ci in ciS_chunk
        x = Tuple(ci)
        for (ampl, x0) in zip(ampls, x0s)
            sinc_map[ci] += ampl * (sinc ∘ norm)((x .- x0) .* den)
        end
    end
end

"""
    create_msk(in, holes)

TBW
"""
function create_msk(data, holes, msk_io)
    @assert msk_io ∈ (:in, :out)

    qtl = quantile(data, (0.5holes, 1 - 0.5holes))

    msk = (data .>= qtl[1]) .& (data .<= qtl[2])
    msk_io == :out && (msk = (!).(msk))

    return msk
end

function create_wf_phantom_and_data(spp::SimPhaPar)
    # set phase range, if necessary
    if isempty(spp.ϕ_rng)
        @assert !isempty(spp.freq_rng)
        ΔTE = mean(spp.TEs[2:end] - spp.TEs[1:end-1])
        spp.ϕ_rng = 2π * spp.freq_rng / ΔTE
    end

    # set unspecified parameters, if necessary

    K_pha = isempty(spp.K_pha) ? spp.K : spp.K_pha
    os_fac_pha = isempty(spp.os_fac_pha) ? spp.os_fac : spp.os_fac_pha
    ppm_fat_pha = isempty(spp.ppm_fat_pha) ? spp.ppm_fat : spp.ppm_fat_pha
    ampl_fat_pha = isempty(spp.ampl_fat_pha) ? spp.ampl_fat : spp.ampl_fat_pha
    R2s_rng_pha = isempty(spp.R2s_rng_pha) ? spp.R2s_rng : spp.R2s_rng_pha
    f_rng_pha = isempty(spp.f_rng_pha) ? spp.f_rng : spp.f_rng_pha
    
    # ------------ ROI ------------ 

    S_data = create_sinc_map(trues(spp.Nρ...), spp.S_nSinc, spp.S_zc, spp.S_rng;
        n_chunks=spp.n_chunks, rng=spp.rng)

    S = create_msk(S_data, spp.S_holes, spp.S_io)
    noS = (!).(S)

    # Cartesian indices of mask and its complement
    ciS = CartesianIndices(S)[S]
    cinoS = CartesianIndices(noS)[noS]

    # split cartesian indices into chunks for multi-threading
    ciS_chunks = [view(ciS, index_chunks(ciS, n=spp.n_chunks)[i]) for i in 1:spp.n_chunks]

    # ------------ phase ------------ 

    ϕ = create_sinc_map(S, spp.ϕ_nSinc, spp.ϕ_zc, spp.ϕ_rng;
        n_chunks=spp.n_chunks, rng=spp.rng)

    if spp.ϕ_proj
        bs_pha = BM.fourier_lin(spp.Nρ, K_pha; os_fac=os_fac_pha)
        BM.smooth_projection!(ϕ, S, bs_pha; λ_tikh=spp.λ_tikh)
    end

    ϕ[S] .+= @views spp.ϕ_med - median(ϕ[S])

    ϕ[noS] .= NaN

    # ------------ R2s ------------ 

    R2s = create_sinc_map(S, spp.R2s_nSinc, spp.R2s_zc, R2s_rng_pha;
        n_chunks=spp.n_chunks, rng=spp.rng)

    R2s[noS] .= NaN

    # ------------ fat fraction ------------ 

    f = create_sinc_map(S, spp.f_nSinc, spp.f_zc, f_rng_pha;
        n_chunks=spp.n_chunks, rng=spp.rng)

    f[noS] .= NaN

    # ------------ coil map(s) ------------ 

    n_coils = size(spp.cov_mat, 1)
    coils = zeros(ComplexF64, spp.Nρ..., n_coils)

    for j in 1:n_coils
        abs_c = create_sinc_map(S, spp.coils_nSinc, spp.coils_zc, spp.coils_rng,
            n_chunks=spp.n_chunks, rng=spp.rng)
        phs_c = create_sinc_map(S, spp.coils_nSinc, spp.coils_zc, spp.ϕ_rng,
            n_chunks=spp.n_chunks, rng=spp.rng)

        coils[ciS, j] .= @views abs_c[ciS] .* exp.(im * phs_c[ciS])
    end

    for ci in ciS
        coils[ci, :] /= norm(coils[ci, :])
    end

    coils[cinoS, :] .= NaN

    # ------------ generate GRE data ------------  

    # allocate data
    nTE = length(spp.TEs)
    data = zeros(ComplexF64, spp.Nρ..., nTE, n_coils)

    # GRE parameters
    grePar = VP.modpar(BM.GREMultiEchoWF;
        ts=spp.TEs,
        B0=spp.B0,
        ppm_fat=ppm_fat_pha,
        ampl_fat=ampl_fat_pha,
        precession=spp.precession,
        mode=:manual_fat,
        x_sym=[:ϕ, :R2s, :f],
        n_coils=size(spp.cov_mat, 1),
        cov_mat=spp.cov_mat)

    # create GRE instance
    gre = VP.create_model(grePar)

    # create channels
    ch_gre = Channel{BM.GREMultiEchoWF}(Threads.nthreads())

    for _ in 1:Threads.nthreads()
        put!(ch_gre, deepcopy(gre))
    end

    Threads.@threads for ciS_chunk in ciS_chunks
        gre_ = take!(ch_gre)

        create_phantom_data_chunk(gre_, ϕ, R2s, f, coils, data, ciS_chunk)

        put!(ch_gre, gre_)
    end

    close(ch_gre)

    # if desired, add some noise according to the noise covariance matrix
    if spp.add_noise == true
        perfect_data = deepcopy(data)
        C = cholesky(spp.cov_mat)
        nS = length(ciS)
        data[ciS, :, :] .+= reshape(randn(spp.rng, ComplexF64, nS * nTE, n_coils) * C.L, nS, nTE, n_coils)
    else
        perfect_data = data
    end

    # ------------ return everything ------------

    (; data, S, noS, ϕ, R2s, f, coils, perfect_data)
end

function create_phantom_data_chunk(gre, ϕ, R2s, f, coils, data, cis_chunk)
    for ci in cis_chunk
        VP.x!(gre, [ϕ[ci], R2s[ci], f[ci]])
        data[ci, :, :] = VP.A(gre) * vec(coils[ci, :])
    end
end

function simulate_phantom(spp::SimPhaPar)
    # ------------ generate phantom ------------

    phantom = create_wf_phantom_and_data(spp)

    # ------------ PHASER ------------

    # GRE parameters
    grePar = VP.modpar(BM.GREMultiEchoWF;
        ts=spp.TEs,
        B0=spp.B0,
        ppm_fat=spp.ppm_fat,
        ampl_fat=spp.ampl_fat,
        precession=spp.precession,
        n_coils=size(spp.cov_mat, 1),
        cov_mat=spp.cov_mat)

    # fit parameters
    fitpar = BM.fitPar(grePar, deepcopy(phantom.data), deepcopy(phantom.S))

    # fit options
    fitopt = BM.fitOpt()
    fitopt.K = spp.K
    fitopt.R2s_rng = spp.R2s_rng
    fitopt.redundancy = spp.redundancy
    fitopt.subsampling = spp.subsampling
    fitopt.remove_outliers = spp.remove_outliers
    fitopt.locfit = spp.locfit
    fitopt.optim = spp.optim
    fitopt.optim_phaser = spp.optim_phaser
    fitopt.os_fac = spp.os_fac
    fitopt.λ_tikh = spp.λ_tikh
    fitopt.rng = spp.rng
    fitopt.diagnostics = true

    # smooth subspace
    bs = BM.fourier_lin(spp.Nρ, spp.K; os_fac=spp.os_fac)

    # apply PHASER
    rp = BM.phaser(fitpar, fitopt, bs)

    # ------------ preparing results ------------

    ML = rp.ML
    PH = rp.PH
    to = rp.to

    noS = phantom.noS

    ML.ϕ[ML.noS] .= NaN

    PH.ϕ[noS] .= NaN
    PH.ϕ_0[noS] .= NaN
    PH.Δϕ_0[noS] .= NaN
    PH.lz_0[noS] .= NaN

    noSj = [(!).(PH.Sj[j]) for j in 1:2]
    for j in 1:2
        PH.y[j][noSj[j]] .= NaN
        PH.ly[j][noSj[j]] .= NaN
        PH.ly_0[j][noSj[j]] .= NaN
    end

    fitpar.ϕ[noS] .= NaN

    # ------------ return everything ------------

    (; phantom, fitpar, fitopt, bs, ML, PH, to)
end