using Statistics, Random, LaTeXStrings, CairoMakie, LinearAlgebra
import VP4Optim as VP
import B0Map as BM

# PHASER utility functions

"""
    ismrm_challenge(
    greType::Type{<:BM.AbstractGREMultiEcho},
    fitopt::BM.FitOpt;
    data_set::Int,
    ic_dir="test/data/ISMRM_challenge_2012/",
    nTE=0)
    
Apply PHASER to specified data set and slice.
"""
function ismrm_challenge(
    fitopt::BM.FitOpt;
    data_set::Int,
    slice::Int,
    ic_dir="test/data/ISMRM_challenge_2012/")

    # check that data set exists
    @assert 1 <= data_set <= 17

    # IRMRM challenge fat specification
    ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
    ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

    # read data set
    nmb_str = data_set < 10 ? string("0", data_set) : string(data_set)
    file_str = ic_dir * nmb_str * "_ISMRM.mat"

    datPar = matread(file_str)["imDataParams"]

    # set up GRE sequence model
    TEs = 1000.0 * datPar["TE"][:]
    nTE = length(TEs)
    B0 = datPar["FieldStrength"]
    precession = (datPar["PrecessionIsClockwise"] != 1.0) ? :clockwise : :counterclockwise

    grePar = VP.modpar(BM.GREMultiEchoWF;
        ts=TEs,
        B0=B0,
        ppm_fat=ppm_fat,
        ampl_fat=ampl_fat,
        precession=precession)

    # read data and mask
    Nρ = size(datPar["images"])[1:2]
    data = zeros(ComplexF64, Nρ[1:2]..., nTE)
    copy!(data, reshape(datPar["images"][:, :, slice, 1, 1:nTE], Nρ..., nTE))
    data ./= max(abs.(data)...)
    S = datPar["eval_mask"][:, :, slice] .!= 0.0

    # generate instance of FitPar
    fitpar = BM.fitPar(grePar, data, S)

    # if ϕ_scale ≠ 1, we need this
    BM.set_num_phase_intervals(fitpar, fitopt, fitopt.n_ϕ)

    # smooth subspace
    bs = BM.fourier_lin(Nρ[1:length(fitopt.K)], fitopt.K; os_fac=fitopt.os_fac)

    # do the work
    bm = BM.B0map!(fitpar, fitopt, bs)
    # ---------
    #fp, fo = deepcopy(fitpar), deepcopy(fitopt)
    PH = nothing #BM.phaser!(fp, fo, bs)
    # ---------
    pdff = BM.fat_fraction_map(fitpar, fitopt)

    fitpar_ML = deepcopy(fitpar)
    fo_ML = deepcopy(fitopt)

    locfit_0 = deepcopy(fitpar)
    fo_0 = deepcopy(fitopt)

    fitpar_ML.ϕ[:, :] = bm.Φ_ML
    fitpar_ML.R2s[:, :] = bm.R2s_ML
    locfit_0.ϕ[:, :] = bm.PH.ϕ[1]

    BM.set_num_phase_intervals(locfit_0, fo_0, 0)
    BM.local_fit!(locfit_0, fo_0)

    pdff_ML = BM.fat_fraction_map(fitpar_ML, fo_ML)
    pdff_0 = BM.fat_fraction_map(locfit_0, fo_0)
    pdff_ref = datPar["ref"][:, :, slice]

    # return results
    return (; fitpar, PH, locfit_0, pdff_ML, pdff_0, pdff, pdff_ref, datPar, data_set, bm)
end

function gen_fig_ISMRM(cal; width, height, cm_phase, cm_fat)
    dax = Dict()

    pt = 4 / 3

    fig = Figure(size=(width, height), fontsize=12pt)

    oi = orient_ISMRM(cal.data_set)

    S = cal.fitpar.S
    noS = (!).(S)

    PH = cal.bm.PH

    Φ_ML = PH.Φ_ML
    Φ_ML[noS] .= NaN

    ϕ_PH = PH.ϕ
    map(ϕ -> ϕ[noS] .= NaN, ϕ_PH)

    bal_rng = (min(ϕ_PH[end][S]..., -π), max(ϕ_PH[end][S]..., π))

    ϕ1_PH_loc = cal.locfit_0.ϕ
    ϕ1_PH_loc[noS] .= NaN

    ϕ_PH_loc = cal.fitpar.ϕ
    ϕ_PH_loc[noS] .= NaN

    pdff_ML = cal.pdff_ML
    pdff_0 = cal.pdff_0
    pdff = cal.pdff
    pdff_ref = cal.pdff_ref

    pdff_ML[noS] .= NaN
    pdff_0[noS] .= NaN
    pdff[noS] .= NaN
    pdff_ref[noS] .= NaN

    #λs_1 = PH.λs_1
    #λs_2 = PH.λs_2
    #χ2s_1 = cal.PH.PH.χ2s_1 ./ cal.PH.PH.sumS_wo
    #χ2s_2 = cal.PH.PH.χ2s_2 ./ cal.PH.PH.sumS

    # -------------------------------------------------

    dax[:Φ_ML] = Axis(fig[1, 1],
        title=L"$\Phi$",
    )

    heatmap!(dax[:Φ_ML],
        oi(Φ_ML),
        colormap=cm_phase,
        colorrange=bal_rng,
        nan_color=:black
    )

    Label(fig[1, 1, TopLeft()], "A",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)

    # -------------------------------------------------

    dax[:ϕ1] = Axis(fig[1, 2],
        title=L"$\varphi^{(1)}$",
    )

    heatmap!(dax[:ϕ1],
        oi(ϕ_PH[1]),
        colormap=cm_phase,
        colorrange=bal_rng,
        nan_color=:black
    )

    Label(fig[1, 2, TopLeft()], "B",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)

    # -------------------------------------------------

    n = length(ϕ_PH)

    dax[:ϕ] = Axis(fig[1, 3],
        title=L"$\varphi^{(%$n)}$",
    )

    heatmap!(dax[:ϕ],
        oi(ϕ_PH[end]),
        colormap=cm_phase,
        colorrange=bal_rng,
        nan_color=:black
    )

    Label(fig[1, 3, TopLeft()], "C",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)

    # -------------------------------------------------

    #=
    dax[:λ_opt] = Axis(fig[2, 1],
        title=L"$\langle \chi^{2}\rangle$",
        xlabel=L"$\lambda$",
        xticklabelsize=8pt,
        yticklabelsize=8pt,
    )

    lines!(dax[:λ_opt], λs_2, χ2s_2 ./ χ2s_2[1], color=:red)
    scatter!(dax[:λ_opt], λs_2, χ2s_2 ./ χ2s_2[1], color=:blue)

    Label(fig[2, 1, TopLeft()], "D",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)
    =#

    # -------------------------------------------------

    dax[:ϕ1_loc] = Axis(fig[2, 2],
        title=L"$\Phi\left(\varphi^{(1)}\right)$",
    )

    heatmap!(dax[:ϕ1_loc],
        oi(ϕ1_PH_loc),
        colormap=cm_phase,
        colorrange=bal_rng,
        nan_color=:black
    )

    Label(fig[2, 2, TopLeft()], "E",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)

    # -------------------------------------------------

    dax[:ϕ_loc] = Axis(fig[2, 3],
        title=L"$\Phi\left(\varphi^{(%$n)}\right)$",
    )

    heatmap!(dax[:ϕ_loc],
        oi(ϕ_PH_loc),
        colormap=cm_phase,
        colorrange=bal_rng,
        nan_color=:black
    )

    Label(fig[2, 3, TopLeft()], "F",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)

    # -------------------------------------------------

    dax[:pdff_ML] = Axis(fig[3, 1],
        title=L"PDFF: $\Phi$",
    )

    heatmap!(dax[:pdff_ML],
        oi(pdff_ML),
        colormap=cm_fat,
        colorrange=(0, 1),
        nan_color=:black
    )

    Label(fig[3, 1, TopLeft()], "G",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)

    # -------------------------------------------------

    dax[:pdff_0] = Axis(fig[3, 2],
        title=L"PDFF: $\Phi\left(φ^{(1)}\right)$",
    )

    heatmap!(dax[:pdff_0],
        oi(pdff_0),
        colormap=cm_fat,
        colorrange=(0, 1),
        nan_color=:black
    )

    Label(fig[3, 2, TopLeft()], "H",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)

    # -------------------------------------------------

    dax[:pdff] = Axis(fig[3, 3],
        title=L"PDFF: $\Phi\left(φ^{(%$n)}\right)$",
    )

    heatmap!(dax[:pdff],
        oi(pdff),
        colormap=cm_fat,
        colorrange=(0, 1),
        nan_color=:black
    )

    Label(fig[3, 3, TopLeft()], "I",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)

    Colorbar(fig[1, 4],
        colorrange=bal_rng,
        colormap=cm_phase,
        ticklabelsize=8pt,
    )

    Colorbar(fig[2, 4],
        colorrange=bal_rng,
        colormap=cm_phase,
        ticklabelsize=8pt,
    )

    Colorbar(fig[3, 4],
        colorrange=(0, 1),
        colormap=cm_fat,
        ticklabelsize=8pt,
    )

    for a in (:Φ_ML, :ϕ1, :ϕ, :ϕ1_loc, :ϕ_loc, :pdff_ML, :pdff_0, :pdff)
        hidedecorations!(dax[a])
    end

    return (fig, dax)
end

"""
    orient_ISMRM(data_set::Int)

Rotate data set properly.
"""
function orient_ISMRM(data_set::Int)
    if data_set ∈ (1:12..., 14:15,)
        x -> rotr90(x)
    elseif data_set ∈ (13, 17,)
        x -> rot180(x)
    elseif data_set ∈ (16,)
        x -> rotl90(x)
    else
        x -> x
    end
end

function show_outliers_tikh(PH; slice=1, width=800, height=800)
    pt = 4 / 3
    fig = Figure(size=(width, height), fontsize=12pt)

    axs = [Axis(fig[i, j], xticklabelsize=10pt, yticklabelsize=10pt)
           for i in 1:2, j in 1:2]

    show_balancing!(axs[1, 1], PH)
    show_tikh_grad!(axs[1, 2], PH)
    show_tikh_balance_opt!(axs[2, 1], PH)
    show_tikh_balance!(axs[2, 2], PH, ())

    display(fig)
end

function show_phase_maps(PH; slice=1, width=800, height=800, cm=:roma, cm_2π=:romaO)
    pt = 4 / 3
    fig = Figure(size=(width, height), fontsize=12pt)

    axs = [Axis(fig[i, j], xticklabelsize=10pt, yticklabelsize=10pt)
           for i in 1:2, j in 1:2]

    show_map!(axs[1, 1], PH.ϕ_ML, L"$\varphi_{ML}$", PH.S; cm=:romaO, cr=(-π, π), slice=slice)
    show_map!(axs[1, 2], PH.ϕ0, L"$\varphi_0$", cm=:romaO, PH.fitpar.S; cr=(-π, π), slice=slice)
    show_map!(axs[2, 2], PH.Δϕ0, L"$\Delta\varphi_0$", PH.fitpar.S; cm=:romaO, cr=(-π, π), slice=slice)
    show_map!(axs[2, 1], PH.ϕ1, L"$\varphi$", PH.fitpar.S; cm=:romaO, cr=(-π, π), slice=slice)

    display(fig)
end

function show_balancing!(ax, PH)
    λs = PH.λs
    χ2s = PH.χ2s

    ax.title = L"Balancing local vs. gradient fit$$"
    ax.xlabel = L"$\lambda$"
    lines!(ax, λs, χ2s, color=:red)
    scatter!(ax, λs, χ2s, color=:blue)

    text!(ax, 0.3, 0.5, align=(:left, :top), space=:relative,
        text=latexstring(L"$\lambda_{opt} =$ ", latexstring(round(PH.λ_opt, digits=4))))
end

function show_tikh_grad!(ax, PH)
    log10_μs_ct = PH.tikh_grad.log10_μs_ct
    log10_μs_tc = PH.tikh_grad.log10_μs_tc
    χ2s_ct = PH.tikh_grad.χ2s_ct
    χ2s_ct .-= min(χ2s_ct...)
    χ2s_tc = PH.tikh_grad.χ2s_tc
    χ2s_tc .-= min(χ2s_tc...)

    pt = 4 / 3

    ax.title = L"$\langle \chi^{2}\rangle$ (Tikhonov gradient)"
    ax.xlabel = L"$\log_{10}\,\mu$"

    lines!(ax, log10_μs_ct, χ2s_ct, color=:red)
    scatter!(ax, log10_μs_ct, χ2s_ct, color=:blue)
    lines!(ax, log10_μs_tc, χ2s_tc, color=:red)
    scatter!(ax, log10_μs_tc, χ2s_tc, color=:blue)

    text!(ax, 0.3, 0.5, align=(:left, :top), space=:relative,
        text=latexstring(L"$\log_{10}\,\mu$ = ", round(PH.tikh_grad.log10_μ, digits=3)))
end

function show_tikh_balance_opt!(ax, PH)
    λs = [t.λ for t in PH.tikh_λ]
    log10_μs = [t.log10_μ for t in PH.tikh_λ]

    iλ = sortperm(λs)
    λs = λs[iλ]
    log10_μs = log10_μs[iλ]

    ax.title = L"$\log_{10}\,\mu$"
    ax.xlabel = L"$\lambda$"

    lines!(ax, λs, log10_μs, color=:red)
    scatter!(ax, λs, log10_μs, color=:blue)

    text!(ax, 0.3, 0.5, align=(:left, :top), space=:relative,
        text=latexstring(L"$\log_{10}\,\mu\left(\lambda_{opt}\right)$ = ",
            round(PH.tikh_λ[end].log10_μ, digits=3)))
end

function show_tikh_balance!(ax, PH, is)
    isempty(is) && (is = eachindex(PH.tikh_λ))

    ax.title = L"$\langle \chi^{2}\rangle$ (Tikhonov balance)"
    ax.xlabel = L"$\log_{10}\,\mu$"

    for i in is
        log10_μs_ct = PH.tikh_λ[i].log10_μs_ct
        log10_μs_tc = PH.tikh_λ[i].log10_μs_tc
        χ2s_ct = PH.tikh_λ[i].χ2s_ct
        χ2s_ct .-= min(χ2s_ct...)
        χ2s_tc = PH.tikh_λ[i].χ2s_tc
        χ2s_tc .-= min(χ2s_tc...)

        lines!(ax, log10_μs_ct, χ2s_ct, color=:red)
        scatter!(ax, log10_μs_ct, χ2s_ct, color=:blue)
        lines!(ax, log10_μs_tc, χ2s_tc, color=:red)
        scatter!(ax, log10_μs_tc, χ2s_tc, color=:blue)
    end
end

function show_map!(ax, map_in, name, S; slice=1, cm, cr=Makie.automatic)
    map_sl = @views map_in[:, :, slice]
    nanS = @views (!).(S[:, :, slice])
    map_sl[nanS] .= NaN

    ax.title = name
    hidedecorations!(ax)

    heatmap!(ax, map_sl,
        colormap=cm,
        colorrange=cr,
        nan_color=:black,
    )
end

function show_hist_phase_grad!(ax, PH, name; j=1, nbins=100, col_in=:blue, col_out=:red, alpha_out=0.3)
    ax.title = name
    hideydecorations!(ax)

    bins = range(0, π, nbins + 1)
    clrs = [mean((lb, rb)) < PH.au_max[j] ? col_in : (col_out, alpha_out) for (lb, rb) in zip(bins[1:end-1], bins[2:end])]

    hist!(ax, abs.(PH.u_wo[j][PH.Sj_wo[j]]), bins=bins, color=clrs)
end

function show_hist_Δϕ1!(ax, PH, name; nbins=100, col_in=:blue, col_out=:red, alpha_out=0.3)
    ax.title = name
    hideydecorations!(ax)

    bins = range(-π, π, nbins + 1)
    clrs = [PH.Δϕ1_min <= mean((lb, rb)) <= PH.Δϕ1_max ? col_in : (col_out, alpha_out) for (lb, rb) in zip(bins[1:end-1], bins[2:end])]

    hist!(ax, PH.Δϕ1_wo[PH.S_wo], bins=bins, color=clrs)
end

function phaser_phase_histograms(PH;
    width=800,
    height_per_plot=200,
    nbins=100,
    j=1,
    col_in=:blue, col_out=:red, alpha_out=0.3,
    cm=:roma,
    font_pt=12,
    slice=1,
    oi=x -> x,
)
    nΦ = length(PH.Φ)
    
    pt = 4 / 3
    height = height_per_plot * (nΦ + 1)
    fig = Figure(size=(width, height), fontsize=font_pt * pt)

    Φ_col, T_col, ∇Φ_col, Tj_col = 1, 2, 3, 4

    ttls = Matrix{LaTeXString}(undef, nΦ + 1, 4)
    for n in 0:nΦ
        if n == 0
            ttls[1, Φ_col] = L"$\Phi_{ML}$"
            ttls[1, ∇Φ_col] = L"$\nabla_1\Phi_{ML}$"
            ttls[1, T_col] = L"$S$"
            ttls[1, Tj_col] = L"$S_{1}$"
        else
            ttls[n+1, Φ_col] = L"$\Phi^{(%$n)}$"
            ttls[n+1, ∇Φ_col] = L"$\nabla_1\Phi^{(%$n)}$"
            ttls[n+1, T_col] = L"$T^{(%$n)}$"
            ttls[n+1, Tj_col] = L"$T_{1}^{(%$n)}$"
        end
    end

    axs = [Axis(fig[i+1, j], title=ttls[i+1, j]) for i in 0:nΦ, j in 1:4]

    for n in 0:nΦ
        hideydecorations!(axs[n+1, Φ_col])
        hideydecorations!(axs[n+1, ∇Φ_col])
        hidedecorations!(axs[n+1, T_col])
        hidedecorations!(axs[n+1, Tj_col])
    end

    mi, ma = min(PH.Φ_ML[PH.S]...), max(PH.Φ_ML[PH.S]...)
    bins = range(mi, ma, nbins + 1)
    hist!(axs[1, Φ_col], PH.Φ_ML[PH.S], bins=bins, scale_to=1, color=(col_out, alpha_out))

    mi, ma = min(PH.∇Φ_ML[1][PH.Sj[1]]...), max(PH.∇Φ_ML[1][PH.Sj[1]]...)
    bins = range(mi, ma, nbins + 1)
    hist!(axs[1, ∇Φ_col], PH.∇Φ_ML[1][PH.Sj[1]], bins=bins, scale_to=1, color=(col_out, alpha_out))
    hist!(axs[1, ∇Φ_col], PH.∇Φ_ML[1][PH.Tj[1][1]], bins=bins, scale_to=1, color=:blue)
    heatmap!(axs[1, T_col], oi(PH.S))
    heatmap!(axs[1, Tj_col], oi(PH.Sj[1]))


    for i in 1:nΦ
        mi, ma = min(PH.Φ[i][PH.S]...), max(PH.Φ[i][PH.S]...)
        bins = range(mi, ma, nbins + 1)
        hist!(axs[i+1, Φ_col], PH.Φ[i][PH.S], bins=bins, scale_to=1, color=(col_out, alpha_out))
        hist!(axs[i+1, Φ_col], PH.Φ[i][PH.T[i+1]], bins=bins, scale_to=1, color=:blue)

        mi, ma = min(PH.∇Φ[i][1][PH.Sj[1]]...), max(PH.∇Φ[i][1][PH.Sj[1]]...)
        bins = range(mi, ma, nbins + 1)
        hist!(axs[i+1, ∇Φ_col], PH.∇Φ[i][1][PH.Sj[1]], bins=bins, scale_to=1, color=(col_out, alpha_out))
        hist!(axs[i+1, ∇Φ_col], PH.∇Φ[i][1][PH.Tj[i][1]], bins=bins, scale_to=1, color=:blue)

        heatmap!(axs[i+1, T_col], oi(PH.T[i+1]))
        heatmap!(axs[i+1, Tj_col], oi(PH.Tj[i+1][1]))
    end

    (fig, axs)
end

function phaser_workflow!(PH;
    width=800,
    height=500,
    nbins=100,
    j=1,
    col_in=:blue, col_out=:red, alpha_out=0.3,
    cm=:roma,
    font_pt=12,
    slice=1,
    oi=x -> x,
)
    pt = 4 / 3
    fig = Figure(size=(width, height), fontsize=font_pt * pt)

    # map every phase into interval [-π, π)
    f_2π = ϕ -> mod.(ϕ .+ π, 2π) .- π

    # several shorthands
    S = @views PH.S[:, :, slice]
    S_wo = @views PH.S_wo[:, :, slice]
    Sj_wo = @views PH.Sj_wo[j][:, :, slice]

    ϕ_ML = @views PH.ϕ_ML[:, :, slice]
    ϕ0 = @views PH.ϕ0[:, :, slice]
    ϕ1 = @views PH.ϕ1[:, :, slice]
    ϕ1_wo = @views PH.ϕ1_wo[:, :, slice]
    u_wo = @views PH.u_wo[j][:, :, slice]

    dax = Dict()

    dax[:ϕ_ML] = Axis(fig[1, 1])
    dax[:au_hist] = Axis(fig[1, 2], title=L"abs$\left(\nabla_%$j\,\Phi\right)$")
    dax[:Φ_ϕ0] = Axis(fig[1, 3], title=L"$\Phi - \varphi_0$")
    dax[:bal] = Axis(fig[2, 1],
        title=L"balancing $\langle \chi^{2}\rangle$",
        xlabel=L"$\lambda$",
        xticklabelsize=8pt,
        yticklabelsize=8pt,
    )
    dax[:Φ_ϕ1_wo] = Axis(fig[2, 2], title=L"$\Phi - \varphi_1$ (after step 1)")
    dax[:Φ_ϕ1] = Axis(fig[2, 3], title=L"$\Phi - \varphi_1$ (final)")

    # local guess of phase map
    show_map!(dax[:ϕ_ML], oi(ϕ_ML), L"$\Phi$", oi(S_wo); cm=:roma, cr=(-π, π))

    # derivative histogram
    bins = range(0, π, nbins + 1)
    clrs = [mean((lb, rb)) < PH.au_max[j] ? col_in : (col_out, alpha_out) for (lb, rb) in zip(bins[1:end-1], bins[2:end])]
    hist!(dax[:au_hist], abs.(u_wo[Sj_wo]), bins=bins, color=clrs)
    hideydecorations!(dax[:au_hist])

    # deviation after gradient based estimate 
    Φ_ϕ0 = f_2π(ϕ_ML - ϕ0)
    mi, ma = min(Φ_ϕ0[S_wo]...), max(Φ_ϕ0[S_wo]...)
    bins = range(mi, ma, nbins + 1)

    hist!(dax[:Φ_ϕ0], Φ_ϕ0[S_wo], bins=bins, scale_to=1)
    #hist!(dax[:Φ_ϕ0], Φ_ϕ0[S], bins=bins, color=:blue, scale_to=1)
    hideydecorations!(dax[:Φ_ϕ0])

    # balancing local and gradient
    mi, ma = min(PH.χ2s_1...), max(PH.χ2s_1...)
    bal_1 = (PH.χ2s_1 .- mi) ./ (ma - mi)
    lines!(dax[:bal], PH.λs_1, bal_1, color=:red, label=L"step $1$")
    scatter!(dax[:bal], PH.λs_1, bal_1, color=:red)
    mi, ma = min(PH.χ2s_2...), max(PH.χ2s_2...)
    bal_2 = (PH.χ2s_2 .- mi) ./ (ma - mi)
    lines!(dax[:bal], PH.λs_2, bal_2, color=:blue, label=L"step $2$")
    scatter!(dax[:bal], PH.λs_2, bal_2, color=:blue)
    hideydecorations!(dax[:bal])
    axislegend(dax[:bal], merge=true, unique=true, labelsize=10pt, position=:ct)

    # deviation histogram after first balancing step
    bins = range(-π, π, nbins + 1)
    clrs = [PH.Δϕ1_min <= mean((lb, rb)) <= PH.Δϕ1_max ? col_in : (col_out, alpha_out) for (lb, rb) in zip(bins[1:end-1], bins[2:end])]
    hideydecorations!(dax[:Φ_ϕ1_wo])
    hist!(dax[:Φ_ϕ1_wo], (ϕ_ML-ϕ1_wo)[S_wo], bins=bins, color=clrs)

    # (final) deviation histogram after second balancing step
    Φ_ϕ1 = f_2π(ϕ_ML - ϕ1)
    mi, ma = min(Φ_ϕ1[S_wo]...), max(Φ_ϕ1[S_wo]...)
    bins = range(mi, ma, nbins + 1)

    hist!(dax[:Φ_ϕ1], Φ_ϕ1[S_wo], bins=bins, scale_to=1, color=(col_out, alpha_out))
    hist!(dax[:Φ_ϕ1], Φ_ϕ1[S], bins=bins, color=:blue, scale_to=1)
    hideydecorations!(dax[:Φ_ϕ1])

    # return figure and axes
    (fig, dax)
end