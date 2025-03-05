using MAT, Plots, PlotThemes, BenchmarkTools, LinearAlgebra
import VP4Optim as VP
import B0Map as BM

theme(:dark, color=:batlow)
BLAS.set_num_threads(1)

# ISMRM challenge 2012 data sets:

# 1: tibia, tra
# 2: upper body, cor
# 3: foot, sag
# 4: knee, sag
# 5: 2 lower legs, tra
# 6: 2 lower legs, tra
# 7: foot, sag
# 8: thorax, tra (strong gradient)
# 9: head, cor (strong gradient)
# 10: hand, cor
# 11: liver, lung, spleen, tra
# 12: liver, lung, tra
# 13: thorax, tra (motion artifacts)
# 14: head & shoulders, cor
# 15: breast, tra (strong gradient)
# 16: torso, sag
# 17: shoulder, cor

function ismrm_challenge(
    greType::Type{<:BM.AbstractGREMultiEcho},
    parType::Type{<:VP.ModPar},
    fitopt::BM.FitOpt;
    fit,
    data_set::Int,
    ic_dir="test/ISMRM_challenge_2012/",
    diagnostics=true)

    # check what to do
    @assert fit ∈ (:local_fit, :PHASER)
    
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
    pars = VP.modpar(parType;
        ts=TEs,
        B0=B0,
        ppm_fat=ppm_fat,
        ampl_fat=ampl_fat,
        precession=precession)
    
    gre = greType(pars)

    # read data and mask
    Nρ = size(datPar["images"])[1:3]
    data = zeros(ComplexF64, Nρ..., nTE)
    copy!(data, reshape(datPar["images"][:, :, :, 1, :], Nρ..., nTE))
    data ./= max(abs.(data)...)
    S = datPar["eval_mask"] .!= 0.0
    
    # generate instance of FitPar
    fitpar = BM.fitPar(gre, data, S)

    # if ϕ_scale ≠ 1, we need this
    BM.set_num_phase_intervals(fitpar, fitopt, fitopt.n_ϕ)

    # do the work
    if fit == :local_fit
        d = nothing
        BM.local_fit(fitpar, fitopt)
    else
        bs = BM.fourier_lin(Nρ[1:2], fitopt.K; os_fac=fitopt.os_fac)
        d = BM.phaser(fitpar, fitopt, bs; diagnostics=diagnostics)
    end
    
    # return results
    return (; fitpar, d)
end

fitopt = BM.fitOpt()
fitopt.K = [10, 10]
data_set = 1

res = ismrm_challenge(BM.GREMultiEchoWF, BM.ModParWF, fitopt; fit = :PHASER, data_set = data_set)

#=
function ismrm_challenge(data_set=[];
    ic_dir="ISMRM_challenge/",
    K,
    n_chunks=8Threads.nthreads(),
    os_fak=2,
    n_ϕ=4)

    data_set == [] && (data_set = 1:17)
    @assert all(i -> 1 <= i <= 17, data_set)

    # IRMRM challenge fat specification
    ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
    ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

    res = Dict()

    for nmbr in data_set
        nmb_str = nmbr < 10 ? string("0", nmbr) : string(nmbr)
        file_str = ic_dir * nmb_str * "_ISMRM.mat"

        res[nmbr] = Dict()

        # extract acquisition parameters
        d = matread(file_str)["imDataParams"]
        n_slices = size(d["images"], 3)
        TEs = 1000.0 * d["TE"][:]
        nTE = length(TEs)
        B0 = d["FieldStrength"]
        precession = (d["PrecessionIsClockwise"] != 1.0) ? :clockwise : :counterclockwise
        args = (TEs, B0, ppm_fat, ampl_fat, precession)
        res[nmbr]["args"] = args
        res[nmbr]["gre"] = MR.greMultiEchoWF(args...)

        Nρ = size(d["images"])[1:3]
        data = zeros(ComplexF64, Nρ..., nTE)
        copy!(data, reshape(d["images"][:, :, :, 1, :], Nρ..., nTE))
        data ./= max(abs.(data)...)
        res[nmbr]["data"] = data
        S = d["eval_mask"] .!= 0.0
        sz = size(S)
        res[nmbr]["S"] = S

        pdff_ML = zeros(sz)
        pdff_phaser = zeros(sz)
        pdff_ref = d["ref"]
        ϕ_ML = zeros(sz)
        ϕ_0 = zeros(sz)
        ϕ_1 = zeros(sz)
        ϕ_2 = zeros(sz)
        ϕ_phaser = zeros(sz)
        Δϕ_0 = zeros(sz)
        Δϕ_1 = zeros(sz)
        Δϕ_2 = zeros(sz)
        R2s_ML = zeros(sz)
        c_ML = zeros(ComplexF64, sz)
        χ2_ML = zeros(sz)
        λs = []
        χ2s = []
        lz = zeros(ComplexF64, sz)
        ly = [zeros(ComplexF64, sz) for _ in 1:2]

        v_S = []
        v_ciS = []
        for i in 1:n_slices
            push!(v_S, @views S[:, :, i])
            push!(v_ciS, @views CartesianIndices(v_S[i])[v_S[i]])
        end

        @time begin
            Nρ_ = Nρ[1:2]
            res[nmbr]["Phaser"] = Vector{Ph.PurePhaserResults}(undef, n_slices)
            res[nmbr]["bf"] = bf
            for i in 1:n_slices
                println("slice ", i, "/", n_slices)

                S_ = v_S[i]
                ciS_ = v_ciS[i]

                data_ = @views reshape(data[:, :, i, :], Nρ_..., nTE)
                res[nmbr]["Phaser"][i] = Ph.phaser(MR.greMultiEchoWF, args, data_, S_, bf, n_ϕ=n_ϕ, n_chunks=n_chunks)
                #pdff_ML[ciS_, i] = res[nmbr]["Phaser"][i].f_ML[S_]
                #pdff_phaser[ciS_, i] = res[nmbr]["Phaser"][i].f_phaser[S_]
                ϕ_ML[ciS_, i] = res[nmbr]["Phaser"][i].ϕ_ML[S_]
                ϕ_0[ciS_, i] = res[nmbr]["Phaser"][i].ϕ_0[S_]
                ϕ_1[ciS_, i] = res[nmbr]["Phaser"][i].ϕ_1[S_]
                ϕ_2[ciS_, i] = res[nmbr]["Phaser"][i].ϕ_2[S_]
                ϕ_phaser[ciS_, i] = res[nmbr]["Phaser"][i].ϕ_phaser[S_]
                Δϕ_0[ciS_, i] = res[nmbr]["Phaser"][i].Δϕ_0[S_]
                Δϕ_1[ciS_, i] = res[nmbr]["Phaser"][i].Δϕ_1[S_]
                Δϕ_2[ciS_, i] = res[nmbr]["Phaser"][i].Δϕ_2[S_]
                #R2s_ML[ciS_, i] = res[nmbr]["Phaser"][i].R2s_ML[S_]
                #c_ML[ciS_, i] = res[nmbr]["Phaser"][i].c_ML[S_]
                χ2_ML[ciS_, i] = res[nmbr]["Phaser"][i].χ2_ML[S_]
                push!(λs, res[nmbr]["Phaser"][i].λs)
                push!(χ2s, res[nmbr]["Phaser"][i].χ2s)
                lz[ciS_, i] = res[nmbr]["Phaser"][i].lz[S_]
                for j in 1:2
                    ly[j][ciS_, i] = @views res[nmbr]["Phaser"][i].ly[j][S_]
                end
            end

            res[nmbr]["pdff_ML"] = pdff_ML
            res[nmbr]["pdff_phaser"] = pdff_phaser
            res[nmbr]["pdff_ref"] = pdff_ref

            pdff_ref[(!).(S)] .= 0

            #succ_ML = sum(abs.(pdff_ML[S] - pdff_ref[S]) .< 0.1)
            #fail_ML = sum(S) - succ_ML
            #succ_phaser = sum(abs.(pdff_phaser[S] - pdff_ref[S]) .< 0.1)
            #fail_phaser = sum(S) - succ_phaser

            subplt_size_22 = (510, 320)
            subplt_size_23 = (510, 320)
            subplt_size_34 = (340, 280)

            plts_mat_22 = Matrix(undef, 2, 2)
            plts_mat_23 = Matrix(undef, 2, 3)
            plts_mat_34 = Matrix(undef, 3, 4)
            plts_mat_24 = Matrix(undef, 2, 4)

            for i in 1:n_slices
                ciS_ = v_ciS[i]
                #pdff_ML_ = @views pdff_ML[:, :, i]
                #pdff_phaser_ = @views pdff_phaser[:, :, i]
                #pdff_ref_ = @views pdff_ref[:, :, i]
                ϕ_ML_ = @views ϕ_ML[:, :, i]
                ϕ_0_ = @views ϕ_0[:, :, i]
                ϕ_1_ = @views ϕ_1[:, :, i]
                ϕ_2_ = @views ϕ_2[:, :, i]
                ϕ_phaser_ = @views ϕ_phaser[:, :, i]
                Δϕ_0_ = @views Δϕ_0[:, :, i]
                Δϕ_1_ = @views Δϕ_1[:, :, i]
                Δϕ_2_ = @views Δϕ_2[:, :, i]
                #R2s_ML_ = @views R2s_ML[:, :, i]
                #c_ML_ = @views c_ML[:, :, i]
                #χ2_ML_ = @views χ2_ML[:, :, i]
                #lz_ = @views lz[:, :, i]
                #ly_ = @views [_ly[:, :, i] for _ly in ly]
                #λs_ = λs[i]
                #χ2s_ = χ2s[i]

                #=
                plts_mat_23[1, 1] = @views heatmap(pdff_ML_, title="pdff: ML")
                plts_mat_23[1, 2] = @views heatmap(pdff_ref_, title="pdff: REF")
                iλ = sortperm(λs_)
                plts_mat_23[1, 2] = scatter(λs_[iλ], χ2s_[iλ]./max(χ2s_...), mc=:red, label="", ms=3)
                plts_mat_23[1, 3] = @views heatmap(pdff_phaser_, title="pdff: OPT")
                plts_mat_23[2, 1] = plot()
                plts_mat_23[2, 2] = plot()
                plts_mat_23[2, 3] = plot()

                display(plot(plts_mat_23...,
                    layout= (3, 2),
                    size= (2, 3) .* subplt_size_23, plot_title=string("data set ", nmbr, ", slice ", i)))
                =#

                #plts_mat_34[1, 1] = @views heatmap(ϕ_ML_, title="ϕ_ML", showaxis=false)
                mi = min(ϕ_0_[ciS_]..., ϕ_2_[ciS_]..., ϕ_2_[ciS_]...)
                ma = max(ϕ_0_[ciS_]..., ϕ_2_[ciS_]..., ϕ_2_[ciS_]...)
                cls = (mi, ma)
                @show cls ./ π
                plts_mat_34[1, 1] = @views heatmap(ϕ_0_, title="ϕ_0", showaxis=false, clims=cls)
                plts_mat_34[2, 1] = @views heatmap(Δϕ_0_, title="Δϕ_0", showaxis=false)
                plts_mat_34[3, 1] = @views histogram(Δϕ_0_[ciS_], title="Δϕ_0", lc=:red)
                plts_mat_34[1, 2] = @views heatmap(ϕ_1_, title="ϕ_1", showaxis=false, clims=cls)
                plts_mat_34[2, 2] = @views heatmap(Δϕ_1_, title="Δϕ_1", showaxis=false)
                plts_mat_34[3, 2] = @views histogram(Δϕ_1_[ciS_], title="Δϕ_1", lc=:red)
                plts_mat_34[1, 3] = @views heatmap(ϕ_2_, title="ϕ_2", showaxis=false, clims=cls)
                plts_mat_34[2, 3] = @views heatmap(Δϕ_2_, title="Δϕ_2", showaxis=false)
                plts_mat_34[3, 3] = @views histogram(Δϕ_2_[ciS_], title="Δϕ_2", lc=:red)
                plts_mat_34[1, 4] = plot() # @views heatmap(ϕ_3_, title="ϕ_3", showaxis=false, clims=cls)
                plts_mat_34[2, 4] = plot() #@views heatmap(Δϕ_3_, title="Δϕ_3", showaxis=false)
                plts_mat_34[3, 4] = plot() #@views histogram(Δϕ_3_[ciS_], title="Δϕ_3", lc = :red)
                @show median(Δϕ_0_[ciS_])
                #plts_mat_34[1, 2] = @views heatmap(ϕ_1_, title="ϕ_1", showaxis=false, clims=(-π,π))
                #plts_mat_34[2, 2] = @views heatmap(Δϕ_1_, title="Δϕ_1", showaxis=false)
                #plts_mat_34[3, 2] = @views heatmap(ϕ_2_, title="ϕ_2", showaxis=false, clims=(-π,π))
                #plts_mat_34[1, 3] = @views heatmap(Δϕ_2_, title="Δϕ_2", showaxis=false)
                #plts_mat_34[2, 3] = @views heatmap(ϕ_3_, title="ϕ_3", showaxis=false)
                #plts_mat_34[3, 3] = @views heatmap(Δϕ_3_, title="Δϕ_3", showaxis=false)
                #plts_mat_34[1, 4] = @views heatmap(ϕ_phaser_, title="ϕ_phaser", showaxis=false)
                #plts_mat_34[2, 4] = @views heatmap(angle.(exp.(im * (ϕ_ML_ - ϕ_phaser_))), title="Δϕ_phaser", showaxis=false)
                #plts_mat_34[3, 4] = @views heatmap(ϕ_ML_, title="ϕ_ML", showaxis=false)

                display(plot(plts_mat_34...,
                    layout=(4, 3),
                    size=(3, 4) .* subplt_size_34, plot_title=string("data set ", nmbr, ", slice ", i)))

                #=
                plts_mat_22[1, 1] = @views heatmap(Δϕ_0_, title="Δϕ_0", showaxis=false)
                plts_mat_22[2, 1] = @views heatmap(Δϕ_1_, title="Δϕ_1", showaxis=false)
                plts_mat_22[1, 2] = @views heatmap(Δϕ_2_, title="Δϕ_2", showaxis=false)
                plts_mat_22[2, 2] = @views heatmap(Δϕ_3_, title="Δϕ_3", showaxis=false)

                display(plot(plts_mat_22...,
                    layout= (2, 2),
                    size= (2, 2) .* subplt_size_22, plot_title=string("data set ", nmbr, ", slice ", i)))
                plts_mat_23[1, 1] = @views histogram(abs.(lz_[ciS_]), title="lz")
                plts_mat_23[1, 2] = @views histogram(abs.(ly_[1][ciS_]), title="ly_1")
                plts_mat_23[1, 3] = @views histogram(abs.(ly_[2][ciS_]), title="ly_2")

                display(plot(plts_mat_23...,
                    layout= (3, 2),
                    size= (2, 3) .* subplt_size_23, plot_title=string("data set ", nmbr, ", slice ", i)))

                plts_mat_23[1, 1] = @views heatmap(abs.(lz_), title="lz")
                plts_mat_23[1, 2] = @views heatmap(abs.(ly_[1]), title="ly_1")
                plts_mat_23[1, 3] = @views heatmap(abs.(ly_[2]), title="ly_2")

                display(plot(plts_mat_23...,
                    layout= (3, 2),
                    size= (2, 3) .* subplt_size_23, plot_title=string("data set ", nmbr, ", slice ", i)))
                    =#
                #=
                                mi, ma = min(Δϕ_1_[ciS_]..., Δϕ_2_[ciS_]...), max(Δϕ_1_[ciS_]..., Δϕ_2_[ciS_]...)
                                bins = range(mi, ma, 100)

                                calc_clims = plts_mat_32[1, 1].subplots[1].attr[:clims_calculated]
                                #plts_mat_22[2, 1] = @views heatmap(R2s_ML_, title="R2s_ML", clim=(0,0.3))
                                plts_mat_32[2, 1] = @views heatmap(ϕ_0_, title="ϕ_0")
                                #plts_mat_32[1, 2] = @views heatmap(imag.(lz_), title="imag(lz)", clims=(-0.5π, 0.5π))
                                #plts_mat_32[2, 2] = @views histogram(rad2deg.(imag.(lz_))[ciS_], bins=100, title="imag(lz)[S]")
                                plts_mat_32[1, 2] = @views heatmap(ϕ_1_, title="ϕ_1")
                                plts_mat_32[2, 2] = @views heatmap(ϕ_2_, title="ϕ_2")
                                plts_mat_32[3, 1] = @views heatmap(ϕ_phaser_, title="ϕ_phaser")
                                plts_mat_32[2, 3] = @views heatmap(ϕ_phaser_, title="ϕ_phaser")
                                #plts_mat_32[2, 3] = @views heatmap(Δϕ_2_, title="Δϕ_2", clims = (-1,1))
                                #plts_mat_32[1, 3] = @views histogram(Δϕ_0_[ciS_], bins=bins, la=0, fa=0.5, label="Δϕ_0", fillcolor=:red)
                                #plts_mat_32[1, 3] = @views stephist!(Δϕ_1_[ciS_], bins=bins, color=:blue, label="Δϕ_1", title="Δϕx")
                                #plts_mat_32[2, 3] = @views stephist(Δϕ_0_[ciS_], bins=bins, color=:black, label="Δϕ_0", fillcolor=:red)
                                #plts_mat_32[2, 3] = @views stephist!(Δϕ_1_[ciS_], bins=bins, color=:green, label="Δϕ_1", fillcolor=:red)
                                #plts_mat_32[2, 3] = @views stephist!(Δϕ_2_[ciS_], bins=bins, color=:blue, label="Δϕ_2")
                                #plts_mat_32[2, 3] = @views stephist!(Δϕ_phaser_[ciS_], bins=bins, color=:red, label="Δϕ_phaser", title="Δϕx")
                                #plts_mat_32[1, 3] = @views histogram(Δϕ_1_[ciS_], title="Δϕ_1")
                                #plts_mat_32[2, 3] = @views histogram(Δϕ_2_[ciS_], title="Δϕ_2")
                                #plts_mat_22[1, 2] = @views heatmap(abs.(c_ML_), title="abs(c_ML)")
                                #plts_mat_22[2, 2] = @views heatmap(angle.(c_ML_), title="angle(c_ML)")

                                display(plot(plts_mat_32...,
                                    layout=size(plts_mat_32'),
                                    size=size(plts_mat_32) .* subplt_size, plot_title=string("data set ", nmbr, ", slice ", i)))
                =#
            end
            println("======================================")
            println("Total time spent for dataset: ")
        end
        #score_ML = 100 * succ_ML / (fail_ML + succ_ML)
        #score_phaser = 100 * succ_phaser / (fail_phaser + succ_phaser)
        #println("ML: score = ", score_ML, ", (succ, fail) = (", succ_ML, ", ", fail_ML, ")")
        #println("PHASER: score = ", score_phaser, ", (succ, fail) = (", succ_phaser, ", ", fail_phaser, ")")
        #println("======================================")
    end

    return res
end

res = ismrm_challenge([12], K=(10, 10))

=#