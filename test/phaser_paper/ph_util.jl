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
    
Apply PHASER to all slices of specified data set.
"""
function ismrm_challenge(
    fitopt::BM.FitOpt;
    data_set::Int,
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
    TEs = 1000.0 * datPar["TE"][:]
    nTE = length(TEs)
    B0 = datPar["FieldStrength"]
    precession = (datPar["PrecessionIsClockwise"] != 1.0) ? :clockwise : :counterclockwise

    # set up GRE sequence model
    grePar = VP.modpar(BM.GREMultiEchoWF;
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

    # smooth subspace
    bs = BM.fourier_lin(Nρ[1:length(fitopt.K)], fitopt.K; os_fac=fitopt.os_fac)

    # do the work
    PH = BM.phaser!(fitpar, fitopt, bs)
    pdff = BM.fat_fraction_map(fitpar, fitopt)

    fitpar_ML = deepcopy(fitpar)
    fo_ML = deepcopy(fitopt)
    
    locfit_0 = deepcopy(fitpar)
    fo_0 = deepcopy(fitopt)

    for i in 1:size(fitpar.S, 3)
        fitpar_ML.ϕ[:, :, i] = PH.PH[i].ϕ_ML
        fitpar_ML.R2s[:, :, i] = PH.PH[i].R2s_ML
        locfit_0.ϕ[:, :, i] = PH.PH[i].ϕ0
    end

    BM.set_num_phase_intervals(locfit_0, fo_0, 0)
    BM.local_fit(locfit_0, fo_0)

    pdff_ML = BM.fat_fraction_map(fitpar_ML, fo_ML)
    pdff_0 = BM.fat_fraction_map(locfit_0, fo_0)
    
    # return results
    return (; fitpar, PH, locfit_0, pdff_ML, pdff_0, pdff, datPar, data_set)
end

function gen_fig_ISMRM(cal; width, height, slice, cm_phase_2π, cm_phase, cm_fat)
    dax = Dict()

    pt = 4 / 3

    fig = Figure(size=(width, height), fontsize=12pt)

    oi = orient_ISMRM(cal.data_set)

    S = cal.fitpar.S[:, :, slice]
    noS = (!).(S)

    ϕ_ML = cal.PH.PH[slice].ϕ_ML
    ϕ_ML[noS] .= NaN

    ϕ0_PH = cal.PH.PH[slice].ϕ0
    ϕ0_PH[noS] .= NaN

    ϕ_PH = cal.PH.PH[slice].ϕ
    ϕ_PH[noS] .= NaN

    bal_rng = (min(ϕ_PH[S]..., -π), max(ϕ_PH[S]..., π))

    ϕ0_PH_loc = cal.locfit_0.ϕ[:, :, slice]
    ϕ0_PH_loc[noS] .= NaN

    ϕ_PH_loc = cal.fitpar.ϕ[:, :, slice]
    ϕ_PH_loc[noS] .= NaN

    pdff_ML = cal.pdff_ML
    pdff_0 = cal.pdff_0
    pdff = cal.pdff

    pdff_ML = pdff_ML[:, :, slice]
    pdff_ML[noS] .= NaN

    pdff_0 = pdff_0[:, :, slice]
    pdff_0[noS] .= NaN

    pdff = pdff[:, :, slice]
    pdff[noS] .= NaN

    pdff_ref = cal.datPar["ref"][:, :, slice]
    pdff_ref[noS] .= NaN

    λs = cal.PH.PH[slice].λs
    χ2s = cal.PH.PH[slice].χ2s

    # -------------------------------------------------

    dax[:ϕ_ML] = Axis(fig[1, 1],
        title=L"$\varphi$ (local fit)",
    )

    heatmap!(dax[:ϕ_ML],
        oi(ϕ_ML),
        #colormap=cm_phase_2π,
        colormap=cm_phase,
        colorrange=bal_rng,
        nan_color=:black
    )

    Label(fig[1, 1, TopLeft()], "A",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)

    # -------------------------------------------------

    dax[:ϕ0] = Axis(fig[1, 2],
        title=L"$\varphi_0$",
    )

    heatmap!(dax[:ϕ0],
        oi(ϕ0_PH),
        colormap=cm_phase,
        colorrange=bal_rng,
        nan_color=:black
    )

    Label(fig[1, 2, TopLeft()], "B",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)

    # -------------------------------------------------

    dax[:ϕ] = Axis(fig[1, 3],
        title=L"$\varphi$",
    )

    heatmap!(dax[:ϕ],
        oi(ϕ_PH),
        colormap=cm_phase,
        colorrange=bal_rng,
        nan_color=:black
    )

    Label(fig[1, 3, TopLeft()], "C",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)

    # -------------------------------------------------

    dax[:λ_opt] = Axis(fig[2, 1],
        title=L"$\langle \chi^{2}\rangle$",
        xlabel=L"$\lambda$",
        xticklabelsize=8pt,
        yticklabelsize=8pt,
    )

    lines!(dax[:λ_opt], λs, χ2s, color=:red)
    scatter!(dax[:λ_opt], λs, χ2s, color=:blue)

    Label(fig[2, 1, TopLeft()], "D",
        font=:bold,
        padding=(0, -20, 5, 0),
        halign=:right)

    # -------------------------------------------------

    dax[:ϕ0_loc] = Axis(fig[2, 2],
        title=L"$\varphi_0$ + local fit",
    )

    heatmap!(dax[:ϕ0_loc],
        oi(ϕ0_PH_loc),
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
        title=L"$\varphi$ + local fit",
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
        title=L"$$PDFF (local fit)",
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
        title=L"$$PDFF ($φ_0$ + local fit)",
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
        title=L"$$PDFF ($φ$ + local fit)",
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
    
    for a in (:ϕ_ML, :ϕ0, :ϕ, :ϕ0_loc, :ϕ_loc, :pdff_ML, :pdff_0, :pdff)
        hidedecorations!(dax[a])
    end

    return (fig, dax)
end

"""
    orient_ISMRM(data_set::Int)

Rotate data set properly.
"""
function orient_ISMRM(data_set::Int)
    @assert data_set ∈ 1:17

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
