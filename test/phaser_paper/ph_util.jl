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

    # do the work
    bm = BM.B0map!(fitpar, fitopt)
    PH = bm.PH

    # reference PDFF
    pdff_ref = datPar["ref"][:, :, slice]

    # return results
    return (; fitpar, PH, pdff_ref, datPar, data_set, bm, data)
end

"""
    ismrm_challenge(
    greType::Type{<:BM.AbstractGREMultiEcho},
    fitopt::BM.FitOpt;
    data_set::Int,
    ic_dir="test/data/ISMRM_challenge_2012/",
    nTE=0)
    
Apply PHASER to specified data set and slice.
"""
function ismrm_challenge_score(
    fitopt::BM.FitOpt;
    data_sets=1:17,
    n_digits=2,
    ic_dir="test/data/ISMRM_challenge_2012/")

    # data set anatomies
    ds_anatomies = [
        "knee (tra)",
        "torso (cor)",
        "foot, (sag)",
        "knee (sag)",
        "2 legs (tra)",
        "2 legs (tra)",
        "foot (sag)",
        "thorax (tra)",
        "head (cor)",
        "hand (cor)",
        "abdomen (tra)",
        "abdomen (tra)",
        "thorax (tra)",
        "head/neck (cor)",
        "breast (tra)",
        "torso (sag)",
        "shoulder (cor)"
    ]

    # dictionary with the results
    d = Dict()

    # IRMRM challenge fat specification
    ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
    ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

    for (ids, data_set) in enumerate(data_sets)
        println(ids, " / ", length(data_sets), ": data_set = ", data_set)

        d[data_set] = Dict()
        d_ = d[data_set]

        d_[:set] = data_set
        d_[:anatomy] = ds_anatomies[data_set]

        # read data set
        nmb_str = data_set < 10 ? string("0", data_set) : string(data_set)
        file_str = ic_dir * nmb_str * "_ISMRM.mat"

        datPar = matread(file_str)["imDataParams"]

        # set up GRE sequence model
        d_[:TEs] = TEs = 1000.0 * datPar["TE"][:]
        d_[:ΔTEs] = TEs[2:end] .- TEs[1:end-1]
        d_[:ΔTE] = mean(d_[:ΔTEs])
        d_[:min_ΔTE] = round(min(d_[:ΔTEs]...), digits=n_digits)
        d_[:max_ΔTE] = round(max(d_[:ΔTEs]...), digits=n_digits)
        d_[:ΔTE] = round(d_[:ΔTE], digits=n_digits)
        d_[:ΔTEs] = round.(d_[:ΔTEs], digits=n_digits)

        d_[:nTE] = nTE = length(TEs)
        B0 = datPar["FieldStrength"]
        d_[:B0] = round(B0, digits=1)
        d_[:precession] = precession = (datPar["PrecessionIsClockwise"] != 1.0) ? :clockwise : :counterclockwise

        grePar = VP.modpar(BM.GREMultiEchoWF;
            ts=TEs,
            B0=B0,
            ppm_fat=ppm_fat,
            ampl_fat=ampl_fat,
            precession=precession)

        d_[:n_slices] = n_slices = size(datPar["images"], 3)

        # read data and mask
        Nρ = size(datPar["images"])[1:2]
        data = zeros(ComplexF64, Nρ[1:2]..., nTE)

        d_[:slice_score_ML] = []
        d_[:slice_score_grad] = []
        d_[:slice_score_PH] = []
        totW_ML = 0
        totW_grad = 0
        totW_PH = 0
        totS = 0

        for slice in 1:n_slices
            copy!(data, reshape(datPar["images"][:, :, slice, 1, 1:nTE], Nρ..., nTE))
            data ./= max(abs.(data)...)
            S = datPar["eval_mask"][:, :, slice] .!= 0.0
            pdff_ref = datPar["ref"][:, :, slice]

            sumS = sum(S)
            totS += sumS

            # generate instance of FitPar
            fitpar = BM.fitPar(grePar, data, S)

            # do the work
            bm = BM.B0map!(fitpar, fitopt)

            # score PHASER
            pdff = BM.fat_fraction_map(fitpar, fitopt)
            nW_PH = sum(abs.(pdff[S] .- pdff_ref[S]) .>= 0.1)
            totW_PH += nW_PH

            # score ML
            fp = deepcopy(fitpar)
            fo = deepcopy(fitopt)
            fp.ϕ[S] .= bm.PH.Φ_ML[S]
            pdff_ML = BM.fat_fraction_map(fp, fo)
            nW_ML = sum(abs.(pdff_ML[S] .- pdff_ref[S]) .>= 0.1)
            totW_ML += nW_ML

            fp.ϕ[S] .= bm.PH.ϕ[1][S]
            BM.set_num_phase_intervals(fp, fo, 0)
            BM.local_fit!(fp, fo)
            pdff_grad = BM.fat_fraction_map(fp, fo)
            nW_grad = sum(abs.(pdff_grad[S] .- pdff_ref[S]) .>= 0.1)
            totW_grad += nW_grad

            push!(d_[:slice_score_ML], 100.0(1 - nW_ML / sumS))
            push!(d_[:slice_score_grad], 100.0(1 - nW_grad / sumS))
            push!(d_[:slice_score_PH], 100.0(1 - nW_PH / sumS))
        end

        d_[:score_ML] = round(100.0(1 - totW_ML / totS), digits=1)
        d_[:score_grad] = round(100.0(1 - totW_grad / totS), digits=1)
        d_[:score_PH] = round(100.0(1 - totW_PH / totS), digits=1)
    end

    # return results
    return d
end

function export_score_table(d, data_sets, export_table)
    # table headers
    table_headers = Dict(
        :set => ("Set", ""),
        :score_PH => ("", "PH"),
        :score_grad => ("", "grad"),
        :score_ML => ("", "ML"),
        :score => ("Score", ""),
        :B0 => ("\\(B_0\\)", "[T]"),
        :nTE => ("TEs", ""),
        :ΔTE => ("\\(\\Delta \\mathrm{TE}\\)", "[ms]"),
        :anatomy => ("Anatomy", ""),
    )

    table_lines = String[]

    if !isempty(export_table)
        export_table_2 = []
        for e in export_table
            if e == :score 
                map(x -> push!(export_table_2, x), (:score_ML, :score_grad, :score_PH))
            else
                push!(export_table_2, e)
            end
        end

        ncols = length(export_table)
        ncols2 = length(export_table_2)

        s = "\\begin{tabular}{"
        for _ = 1:ncols2
            s *= "|c"
        end
        s *= "|}"

        push!(table_lines, s)
        push!(table_lines, "\\hline")

        s = ""
        for (ie, entry) in enumerate(export_table)
            if entry == :score
                s *= "\\multicolumn{3}{|c|}{" * table_headers[entry][1] * "}"
            else
                s *= table_headers[entry][1]
            end
            if ie < ncols
                s *= " & "
            else
                s *= " \\\\ "
            end
        end
        push!(table_lines, s)

        s = ""
        for (ie, entry) in enumerate(export_table_2)
                s *= table_headers[entry][2]
            if ie < ncols2
                s *= " & "
            else
                s *= " \\\\ "
            end
        end
        push!(table_lines, s)

        push!(table_lines, "\\hline")

        for data_set in data_sets
            s = ""

            for (ie, entry) in enumerate(export_table_2)
                s *= string(d[data_set][entry])
                if ie < ncols2
                    s *= " & "
                else
                    s *= " \\\\ "
                end
            end

            push!(table_lines, s)
        end

        push!(table_lines, "\\hline")
        push!(table_lines, "\\end{tabular}")
    end

    # return results
    return table_lines
end

"""
    orient_ISMRM(data_set::Int)

Rotate data set properly.
"""
function orient_ISMRM(data_set::Int)
    if data_set ∈ (1:12..., 14,)
        x -> rotr90(x)
    elseif data_set ∈ (13, 17,)
        x -> rot180(x)
    elseif data_set ∈ (16,)
        x -> rotl90(x)
    else
        x -> x
    end
end

"""
    phaser_plots(plots, PH, fitpar, fitopt;
    width_per_plot=200,
    height_per_plot=200,
    j=1,
    col_in=:blue, col_out=:red, alpha_out=0.3,
    font_pt=12, label_pt=8,
    slice=1,
    oi=x -> x,
    ϕns=nothing,
    letters=false,
    ϕ_loc=nothing,
    pdff=nothing,
)

TBW
"""
function phaser_plots(plots, PH, fitpar, fitopt;
    width_per_plot=200,
    height_per_plot=200,
    j=1,
    col_in=:blue, col_out=:red, alpha_out=0.3,
    font_pt=12, label_pt=8,
    slice=1,
    oi=x -> x,
    ϕns=nothing,
    letters=false,
    ϕ_loc=nothing,
    pdff=nothing,
)
    nrows, ncols = size(plots)

    Φ_ML = @views PH.Φ_ML[:, :, slice]
    ∇Φ_ML = @views PH.∇Φ_ML[j][:, :, slice]
    a∇Φ_ML = @views abs.(PH.∇Φ_ML[j][:, :, slice])
    Φ = @views [Φ_[:, :, slice] for Φ_ in PH.Φ]
    ϕ = @views [ϕ_[:, :, slice] for ϕ_ in PH.ϕ]
    ϕ_Φ = @views [ϕ_ - Φ_ML for ϕ_ in ϕ]
    ∇Φ = @views [∇Φ_[j][:, :, slice] for ∇Φ_ in PH.∇Φ]
    a∇Φ = @views [abs.(∇Φ_[j][:, :, slice]) for ∇Φ_ in PH.∇Φ]
    S = @views PH.S[:, :, slice]
    R = @views fitpar.S[:, :, slice]
    noR = (!).(R)
    Sj = @views PH.Sj[j][:, :, slice]
    noSj = (!).(Sj)
    ∇Φ_ML[noSj] .= NaN
    a∇Φ_ML[noSj] .= NaN
    map(x -> x[noSj] .= NaN, a∇Φ)
    T = @views [T_[:, :, slice] for T_ in PH.T]
    Tj = @views [Tj_[j][:, :, slice] for Tj_ in PH.Tj]
    Φ_red = @views [(Φ_red_ = deepcopy(Φ_); Φ_red_[(!).(T_)] .= NaN; Φ_red_) for (Φ_, T_) in zip(Φ, T[2:end])]
    ∇Φ_red = @views [(∇Φ_red_ = deepcopy(∇Φ_); ∇Φ_red_[(!).(Tj_)] .= NaN; ∇Φ_red_) for (∇Φ_, Tj_) in zip(∇Φ, Tj[2:end])]
    a∇Φ_red = @views [(a∇Φ_red_ = deepcopy(a∇Φ_); a∇Φ_red_[(!).(Tj_)] .= NaN; a∇Φ_red_) for (a∇Φ_, Tj_) in zip(a∇Φ, Tj[2:end])]
    data = (ndims(fitpar.data) == 3 || size(fitpar.data, 4) == 1) ? fitpar.data : @views fitpar.data[:, :, slice, :]
    grePar = fitpar.grePar
    fp = BM.fitPar(grePar, data, R)
    fo = deepcopy(fitopt)
    BM.set_num_phase_intervals(fp, fo, 0)
    fo.optim = true

    nΦ = length(Φ)
    ϕns == nothing && (ϕns = 1:nΦ)

    if ϕ_loc == nothing
        if S == R
            ϕ_loc = [Φ_ML]
            fp.ϕ[R] = @views Φ_ML[R]
        else
            BM.local_fit!(fp, fitopt)
            ϕ_loc = [deepcopy(fp.ϕ)]
            ϕ_loc[1][noR] .= NaN
        end

        for i in 1:nΦ
            fp.ϕ[R] .= @views ϕ[i][R]

            BM.local_fit!(fp, fo)
            push!(ϕ_loc, deepcopy(fp.ϕ))

            ϕ_loc[end][noR] .= NaN
        end
    end

    if pdff == nothing
        for ϕ_loc_ in ϕ_loc
            fp.ϕ[R] = @views ϕ_loc_[R]

            if pdff == nothing
                pdff = [BM.fat_fraction_map(fp, fo)]
            else
                push!(pdff, BM.fat_fraction_map(fp, fo))
            end
            pdff[end][noR] .= NaN
        end
    end

    width = width_per_plot * ncols
    height = height_per_plot * nrows

    pt = 4 / 3
    fig = Figure(size=(width, height), fontsize=font_pt * pt)

    i_col = [1:ncols;]

    for ic in 1:ncols-1
        any(x -> hasproperty(x, :colbar) && x.colbar == true, plots[:, ic]) && (i_col[ic+1:end] .+= 1)
    end

    dax = Matrix{Any}(undef, nrows, ncols)

    az = ['A':'Z';]
    nrows * ncols <= length(az) || (letters = false)
    letters && (maz = reshape(az[1:nrows*ncols], ncols, nrows))

    for ir in 1:nrows
        for ic in 1:ncols
            plt = plots[ir, ic]
            dax[ir, ic] = ax = Axis(fig[ir, i_col[ic]])

            # --------------------------------------------------------------------

            if plt.val == :Φ
                n = plt.n
                ax.title = n == 0 ? L"$\Phi$" : L"$\mathcal{P}\,\left[\,\Phi - \varphi^{(%$n)}\,\right]$"
                hidedecorations!(ax)

                heatmap!(ax,
                    n == 0 ? oi(Φ_ML) : oi(Φ[n]),
                    colormap=plt.cm,
                    colorrange=(-π, π),
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=(-π, π),
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                        ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :∇Φ
                n = plt.n
                ax.title = n == 0 ? L"$\nabla_{%$j}\,\Phi$" : L"$\nabla_{%$j}\,\Phi^{(%$n)}$"
                hidedecorations!(ax)

                heatmap!(ax,
                    n == 0 ? oi(∇Φ_ML) : oi(∇Φ[n]),
                    colormap=plt.cm,
                    colorrange=plt.cm_rng,
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=plt.cm_rng,
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :a∇Φ
                n = plt.n
                ax.title = n == 0 ? L"$|\,\nabla_{%$j}\,\Phi\,|$" : L"$|\,\nabla_{%$j}\,\Phi^{(%$n)}\,|$"
                hidedecorations!(ax)

                heatmap!(ax,
                    n == 0 ? oi(a∇Φ_ML) : oi(a∇Φ[n]),
                    colormap=plt.cm,
                    colorrange=plt.cm_rng,
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=plt.cm_rng,
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :Φ_red
                n = plt.n
                ax.title = L"$\mathcal{P}\,\left[\,\Phi - \varphi^{(%$n)}\,\right]$"
                hidedecorations!(ax)

                heatmap!(ax,
                    oi(Φ_red[n]),
                    colormap=plt.cm,
                    colorrange=(-π, π),
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=(-π, π),
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                        ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :∇Φ_red
                n = plt.n
                ax.title = L"$\nabla_%$j\,\Phi^{(%$n)}$"
                hidedecorations!(ax)

                heatmap!(ax,
                    oi(∇Φ_red[n]),
                    colormap=plt.cm,
                    colorrange=plt.cm_rng,
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=plt.cm_rng,
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :a∇Φ_red
                n = plt.n
                ax.title = L"$|\,\nabla_%$j\,\Phi^{(%$n)}\,|$"
                hidedecorations!(ax)

                heatmap!(ax,
                    oi(a∇Φ_red[n]),
                    colormap=plt.cm,
                    colorrange=plt.cm_rng,
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=plt.cm_rng,
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :ϕ
                n = plt.n
                ax.title = plt.rng_2π ?
                           L"$\mathcal{P}\,\left[\,\varphi^{(%$n)}\,\right]$" :
                           L"$\varphi^{(%$n)}$"
                hidedecorations!(ax)

                rng_ϕ = plt.rng_2π ? (-π, π) : (min(ϕ[end][R]..., -π), max(ϕ[end][R]..., π))

                heatmap!(ax,
                    plt.rng_2π ? oi(BM.map_2π(ϕ[n])) : oi(ϕ[n]),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if plt.colbar
                    if plt.rng_2π
                        Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=(-π, π),
                            colormap=plt.cm,
                            ticklabelsize=label_pt,
                            ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                        )
                    else
                        Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=rng_ϕ,
                            colormap=plt.cm,
                            ticklabelsize=label_pt,
                        )
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :ϕ_loc
                n = plt.n

                if n == 0
                    ax.title = L"$\Phi$"
                else
                    ax.title = plt.rng_2π ?
                               L"$\mathcal{P}\,\left[\,\Phi\left(\varphi^{(%$n)}\right)\,\right]$" :
                               L"$\Phi\left(\varphi^{(%$n)}\right)$"
                end
                hidedecorations!(ax)

                rng_ϕ = plt.rng_2π ? (-π, π) : (min(ϕ[end][R]..., -π), max(ϕ[end][R]..., π))

                heatmap!(ax,
                    plt.rng_2π ? oi(BM.map_2π(ϕ_loc[n+1])) : oi(ϕ_loc[n+1]),
                    colormap=plt.cm,
                    colorrange=rng_ϕ,
                    nan_color=:black,
                )

                if plt.colbar
                    if plt.rng_2π || n == 0
                        Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=(-π, π),
                            colormap=plt.cm,
                            ticklabelsize=label_pt,
                            ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                        )
                    else
                        Colorbar(fig[ir, i_col[ic]+1],
                            colorrange=rng_ϕ,
                            colormap=plt.cm,
                            ticklabelsize=label_pt,
                        )
                    end
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :pdff
                n = plt.n

                ax.title = n == 0 ? L"PDFF: $\Phi$" : L"PDFF: $\Phi\left(\varphi^{(%$n)}\right)$"
                hidedecorations!(ax)

                heatmap!(ax,
                    oi(pdff[n+1]),
                    colormap=plt.cm,
                    colorrange=(0, 1),
                    nan_color=:black,
                )

                if plt.colbar
                    Colorbar(fig[ir, i_col[ic]+1],
                        colorrange=(0, 1),
                        colormap=plt.cm,
                        ticklabelsize=label_pt,
                        ticks=([0, 1], ["0", "1"]),
                    )
                end
            end

            # --------------------------------------------------------------------

            if plt.val == :hist_ϕ_Φ
                n = plt.n

                ax.title = L"$\varphi^{(%$n)} - \Phi$"
                hideydecorations!(ax)

                bins = range(min(ϕ_Φ[end][S]..., -π), max(ϕ_Φ[end][S]..., π), plt.bin_mode == :fixed ? plt.nbins + 1 :
                                                                              ceil(Int, (2sum(S))^(1 / 3) + 1))

                @views hist!(ax, ϕ_Φ[n][S], bins=bins, scale_to=1, color=(col_out, alpha_out))
                @views hist!(ax, ϕ_Φ[n][T[n+1]], bins=bins, scale_to=1, color=col_in)
                #ax.xticks = ([-π, 0.0, π], ["-π", "0", "π"])
                ax.xticklabelsize = label_pt
            end

            # --------------------------------------------------------------------

            if plt.val == :hist_Φ
                n = plt.n

                ax.title = n == 0 ? L"$\Phi$" : L"$\mathcal{P}\,\left[\,\Phi - \varphi^{(%$n)}\,\right]$"
                hideydecorations!(ax)

                bins = range(-π, π, plt.bin_mode == :fixed ? plt.nbins + 1 :
                                    ceil(Int, (2sum(PH.S))^(1 / 3) + 1))
                if n == 0
                    @views hist!(ax, PH.Φ_ML[PH.S], bins=bins, scale_to=1, color=(col_out, alpha_out))
                else
                    @views hist!(ax, PH.Φ[n][PH.S], bins=bins, scale_to=1, color=(col_out, alpha_out))
                    @views hist!(ax, PH.Φ[n][PH.T[n+1]], bins=bins, scale_to=1, color=col_in)
                end
                ax.xticks = ([-π, 0.0, π], ["-π", "0", "π"])
                ax.xticklabelsize = label_pt
            end

            # --------------------------------------------------------------------

            if plt.val == :hist_a∇Φ
                n = plt.n

                ax.title = n == 0 ? L"$\left|\,\nabla_%$j\,\Phi\,\right|$" :
                           L"$\left|\,\nabla_%$j\,\Phi^{(%$n)}\,\right|$"
                hideydecorations!(ax)

                bins = range(0, π, plt.bin_mode == :fixed ? plt.nbins + 1 :
                                   ceil(Int, (2sum(PH.Sj[j]))^(1 / 3)) + 1)
                if n == 0
                    @views hist!(ax, abs.(PH.∇Φ_ML[j][PH.Sj[j]]), bins=bins, scale_to=1, color=(col_out, alpha_out))
                    @views hist!(ax, abs.(PH.∇Φ_ML[j][PH.Tj[1][j]]), bins=bins, scale_to=1, color=col_in)
                else
                    @views hist!(ax, abs.(PH.∇Φ[n][j][PH.Sj[j]]), bins=bins, scale_to=1, color=(col_out, alpha_out))
                    @views hist!(ax, abs.(PH.∇Φ[n][j][PH.Tj[n+1][j]]), bins=bins, scale_to=1, color=col_in)
                end
                ax.xticks = ([0, π], ["0", "π"])
                ax.xticklabelsize = label_pt
            end

            # --------------------------------------------------------------------

            if plt.val == :χ2λ
                n = plt.n

                l = round(cal.PH.info[:balanced][:λ_opt][n], digits=3)
                ax.title = L"$\chi^2\,(\lambda)$"
                hideydecorations!(ax)

                lbl = L"$\lambda^{(%$n)} = %$l$"
                λs = PH.info[:balanced][:λs][n]
                χ2s = PH.info[:balanced][:χ2s][n]
                ax.xticklabelsize = label_pt
                #ax.xlabelsize = label_pt
                #ax.xlabel = L"$\lambda$"
                scatterlines!(ax, λs, χ2s, label=lbl)
                axislegend(ax)
            end

            # --------------------------------------------------------------------

            if letters
                Label(fig[ir, i_col[ic], TopLeft()], string(maz[ic, ir]),
                    font=:bold,
                    padding=(0, -20, 5, 0),
                    halign=:right)
            end
        end
    end

    #=


    for col in columns
        push!(v_col, i_col)
        dax[col] = axs = [Axis(fig[i, i_col]) for i in 1:nrows]

        # --------------------------------------------------------------------

        if col == :hist
            axs[1].title = L"$\Phi_{ML}$"
            hideydecorations!(axs[1])

            bins = range(-π, π, bin_mode == :fixed ? nbins + 1 : ceil(Int, (2sum(S))^(1 / 3) + 1))
            hist!(axs[1], Φ_ML[S], bins=bins, scale_to=1, color=(col_out, alpha_out))
            axs[1].xticks = ([-π, 0.0, π], ["-π", "0", "π"])
            axs[1].xticklabelsize = label_pt

            for (i, n) in enumerate(ϕns)
                if n == 1
                    axs[i+1].title = L"$\left|\,\nabla_%$j\,\Phi_{ML}\,\right|$"
                    hideydecorations!(axs[i+1])

                    bins_abs = range(0, π, bin_mode == :fixed ? nbins + 1 : ceil(Int, (2sum(Sj))^(1 / 3)) + 1)
                    @views hist!(axs[i+1], abs.(∇Φ_ML[Sj]), bins=bins_abs, scale_to=1, color=(col_out, alpha_out))
                    @views hist!(axs[i+1], abs.(∇Φ_ML[Tj[1]]), bins=bins_abs, scale_to=1, color=col_in)
                    axs[i+1].xticks = ([0, π], ["0", "π"])
                    axs[i+1].xticklabelsize = label_pt
                else
                    n1 = n - 1
                    axs[i+1].title = L"$\Phi^{(%$n1)}$"
                    hideydecorations!(axs[i+1])

                    hist!(axs[i+1], Φ[n1][S], bins=bins, scale_to=1, color=(col_out, alpha_out))
                    hist!(axs[i+1], Φ[n1][T[n]], bins=bins, scale_to=1, color=col_in)
                    axs[i+1].xticks = ([-π, 0.0, π], ["-π", "0", "π"])
                    axs[i+1].xticklabelsize = label_pt
                end
            end

            i_col += 1
        end

        # --------------------------------------------------------------------

        if col == :Φ_hist
            axs[1].title = L"$\left|\,\nabla_%$j\,\Phi_{ML}\,\right|$"
            hideydecorations!(axs[1])

            bins = range(0, π, bin_mode == :fixed ? nbins + 1 : ceil(Int, (2sum(Sj))^(1 / 3)) + 1)
            @views hist!(axs[1], abs.(∇Φ_ML[Sj]), bins=bins, scale_to=1, color=(col_out, alpha_out))
            @views hist!(axs[1], abs.(∇Φ_ML[Tj[1]]), bins=bins, scale_to=1, color=col_in)
            axs[1].xticks = ([0, π], ["0", "π"])
            axs[1].xticklabelsize = label_pt

            #axs[1].title = L"$\Phi_{ML}$"
            #hideydecorations!(axs[1])

            bins = range(-π, π, bin_mode == :fixed ? nbins + 1 : ceil(Int, (2sum(S))^(1 / 3) + 1))
            #hist!(axs[1], Φ_ML[S], bins=bins, scale_to=1, color=(col_out, alpha_out))
            #axs[1].xticks = ([-π, 0.0, π], ["-π", "0", "π"])
            #axs[1].xticklabelsize = label_pt

            for (i, n) in enumerate(ϕns)
                axs[i+1].title = L"$\Phi^{(%$n)}$"
                hideydecorations!(axs[i+1])

                hist!(axs[i+1], Φ[n][S], bins=bins, scale_to=1, color=(col_out, alpha_out))
                hist!(axs[i+1], Φ[n][T[n+1]], bins=bins, scale_to=1, color=col_in)
                axs[i+1].xticks = ([-π, 0.0, π], ["-π", "0", "π"])
                axs[i+1].xticklabelsize = label_pt
            end

            i_col += 1
        end

        # --------------------------------------------------------------------

        if col == :∇Φ_hist
            axs[1].title = L"$\nabla_%$j\,\Phi_{ML}$"
            hideydecorations!(axs[1])

            bins = range(-π, π, bin_mode == :fixed ? nbins + 1 : ceil(Int, (2sum(Sj))^(1 / 3)) + 1)
            hist!(axs[1], ∇Φ_ML[Sj], bins=bins, scale_to=1, color=(col_out, alpha_out))
            hist!(axs[1], ∇Φ_ML[Tj[1]], bins=bins, scale_to=1, color=col_in)

            for (i, n) in enumerate(ϕns)
                axs[i+1].title = L"$\nabla_%$j\,\Phi^{(%$n)}$"
                hideydecorations!(axs[i+1])

                hist!(axs[i+1], ∇Φ[n][Sj], bins=bins, scale_to=1, color=(col_out, alpha_out))
                hist!(axs[i+1], ∇Φ[n][Tj[n+1]], bins=bins, scale_to=1, color=col_in)
                axs[i+1].xticks = ([-π, 0.0, π], ["-π", "0", "π"])
                axs[i+1].xticklabelsize = label_pt
            end

            i_col += 1
        end

        # --------------------------------------------------------------------

        if col == :abs_∇Φ_hist
            axs[1].title = L"$\left|\,\nabla_%$j\,\Phi_{ML}\,\right|$"
            hideydecorations!(axs[1])

            bins = range(0, π, bin_mode == :fixed ? nbins + 1 : ceil(Int, (2sum(Sj))^(1 / 3)) + 1)
            @views hist!(axs[1], abs.(∇Φ_ML[Sj]), bins=bins, scale_to=1, color=(col_out, alpha_out))
            @views hist!(axs[1], abs.(∇Φ_ML[Tj[1]]), bins=bins, scale_to=1, color=col_in)
            axs[1].xticks = ([0, π], ["0", "π"])
            axs[1].xticklabelsize = label_pt

            for (i, n) in enumerate(ϕns)
                axs[i+1].title = L"$\left|\,\nabla_%$j\,\Phi^{(%$n)}\,\right|$"
                hideydecorations!(axs[i+1])

                @views hist!(axs[i+1], abs.(∇Φ[n][Sj]), bins=bins, scale_to=1, color=(col_out, alpha_out))
                @views hist!(axs[i+1], abs.(∇Φ[n][Tj[n+1]]), bins=bins, scale_to=1, color=col_in)
                axs[i+1].xticks = ([0, π], ["0", "π"])
                axs[i+1].xticklabelsize = label_pt
            end

            i_col += 1
        end

        # --------------------------------------------------------------------

        if col == :T
            axs[1].title = L"$S$"
            hidedecorations!(axs[1])

            heatmap!(axs[1], oi(S))

            for (i, n) in enumerate(ϕns)
                axs[i+1].title = L"$T^{(%$n)}$"
                hidedecorations!(axs[i+1])

                heatmap!(axs[i+1], oi(T[n]))
            end

            i_col += 1
        end

        # --------------------------------------------------------------------

        if col == :Tj
            axs[1, i_col].title = L"$S_{%$j}$"
            hidedecorations!(axs[1])

            heatmap!(axs[1], oi(Sj))

            for (i, n) in enumerate(ϕns)
                axs[i+1].title = L"$T^{(%$n)}_{%$j}$"
                hidedecorations!(axs[i+1])

                heatmap!(axs[i+1], oi(Tj[n]))
            end

            i_col += 1
        end

        # --------------------------------------------------------------------

        if col == :balance
            axs[1].title = L"$\Phi_{ML}$"
            hidedecorations!(axs[1])

            heatmap!(axs[1],
                oi(ϕ_loc[1]),
                colormap=cmO,
                colorrange=(-π, π),
                nan_color=:black,
            )

            #=
            if :ϕ ∈ colbars
                Colorbar(fig[1, i_col+1],
                    colorrange=(-π, π),
                    colormap=cmO,
                    ticklabelsize=label_pt,
                    ticks=([-π, 0.0, π], ["-π", "0", "π"]),
                )
            end
            =#
            #axs[1].title = L"$\left|\,\nabla_%$j\,\Phi_{ML}\,\right|$"
            #hideydecorations!(axs[1])

            #bins = range(0, π, bin_mode == :fixed ? nbins + 1 : ceil(Int, (2sum(Sj))^(1 / 3)) + 1)
            #@views hist!(axs[1], abs.(∇Φ_ML[Sj]), bins=bins, scale_to=1, color=(col_out, alpha_out))
            #@views hist!(axs[1], abs.(∇Φ_ML[Tj[1]]), bins=bins, scale_to=1, color=col_in)
            #axs[1].xticks = ([0, π], ["0", "π"])
            #axs[1].xticklabelsize = label_pt

            #axs[1].title = L"$\chi^2$"
            #hideydecorations!(axs[1])

            #for n = 2:nΦ
            #    λs = PH.info[:balanced][:data][:λs][n-1]
            #    χ2s = PH.info[:balanced][:data][:χ2s][n-1]
            #    mi, ma = min(χ2s...), max(χ2s...)
            #    scatterlines!(axs[1], λs, (χ2s .- mi) ./ (ma .- mi), label=L"$n = %$n$")
            #end

            #axs[1].xticks = ([0, 1], ["0", "1"])
            #axs[1].xticklabelsize = label_pt
            #axs[1].xlabelsize = label_pt
            #axs[1].xlabel = L"$\lambda$"
            #axislegend(axs[1])
            for (i, n) in enumerate(ϕns)
                if n ≥ 2
                    axs[i+1].title = L"$\chi^2$"
                    hideydecorations!(axs[1])
                    λs = PH.info[:balanced][:data][:λs][n-1]
                    χ2s = PH.info[:balanced][:data][:χ2s][n-1]
                    mi, ma = min(χ2s...), max(χ2s...)
                    scatterlines!(axs[i+1], λs, (χ2s .- mi) ./ (ma .- mi))
                end
            end

            i_col += 1
        end

        # --------------------------------------------------------------------

    end
    =#

    (fig, dax, ϕ_loc, pdff)
end
