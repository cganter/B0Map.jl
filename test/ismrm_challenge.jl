using MAT, Plots
import VP4Optim as VP
import VP4MRI as MR

function ismrm_challenge(data_set=[];
    ic_dir="test/ISMRM_challenge_2012/",
    n_ϕ=3,
    n_chunks=8Threads.nthreads(),
    optim=true,
    visual=true,
    fmt=".pdf",
    R2s_rng=(0.0, 1.0))

    data_set == [] && (data_set = 1:17)
    @assert all(i -> 1 <= i <= 17, data_set)

    # IRMRM challenge fat specification
    ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
    ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

    res = Dict()

    for nmbr in data_set
        set_str = nmbr < 10 ? "0" * string(nmbr) : string(nmbr)

        println("Set ", nmbr)
        nmb_str = nmbr < 10 ? string("0", nmbr) : string(nmbr)
        file_str = ic_dir * nmb_str * "_ISMRM.mat"

        res[nmbr] = Dict()

        # extract acquisition parameters
        d = matread(file_str)["imDataParams"]
        n_slices = size(d["images"], 3)
        TEs = 1000.0 * d["TE"][:]
        #TEs = 1000.0 * d["TE"][1:2]
        nTE = length(TEs)
        B0 = d["FieldStrength"]
        precession = (d["PrecessionIsClockwise"] != 1.0) ? :clockwise : :counterclockwise
        args = (TEs, B0, ppm_fat, ampl_fat, precession)
        res[nmbr]["args"] = args
        res[nmbr]["gre_af"] = MR.greMultiEchoWF(args...)
        res[nmbr]["gre_mf"] = MR.greMultiEchoWF(args..., :manual_fat)
        #res[nmbr]["gre_fw"] = MR.greMultiEchoWFFW(args...)
        res[nmbr]["gre_fw"] = MR.greMultiEchoWFRW(args...)

        Nρ = size(d["images"])[1:3]
        data = zeros(ComplexF64, Nρ..., nTE)
        copy!(data, reshape(d["images"][:, :, :, 1, 1:nTE], Nρ..., nTE))
        data ./= max(abs.(data)...)
        res[nmbr]["data"] = data
        S = d["eval_mask"] .!= 0.0
        res[nmbr]["S"] = S

        pdff_af = zeros(size(S)...)
        pdff_fw = zeros(size(S)...)
        pdff_ref = d["ref"]
        ϕ_af = zeros(size(S)...)
        ϕ_fw = zeros(size(S)...)
        R2s_af = zeros(size(S)...)
        R2s_fw = zeros(size(S)...)
        c_af = zeros(ComplexF64, size(S)...)
        c_fw = zeros(ComplexF64, size(S)..., 2)
        χ2_af = zeros(size(S)...)
        χ2_fw = zeros(size(S)...)

        v_S = []
        v_ciS = []
        for i in 1:n_slices
            push!(v_S, @views S[:, :, i])
            push!(v_ciS, @views CartesianIndices(v_S[i])[v_S[i]])
        end

        @time begin
            res_af = res[nmbr]["res_af"] = MR.B0_map_varpro(
                MR.greMultiEchoWF, args, data, S; n_ϕ=n_ϕ, n_chunks=n_chunks, optim=optim, R2s_rng=R2s_rng)
            #res_fw = res[nmbr]["res_fw"] = MR.B0_map_varpro(
            #    MR.greMultiEchoWFFW, args, data, S; n_ϕ=n_ϕ, n_chunks=n_chunks)
            res_fw = res[nmbr]["res_fw"] = MR.B0_map_varpro(
                MR.greMultiEchoWFRW, args, data, S; n_ϕ=n_ϕ, n_chunks=n_chunks, optim=optim, R2s_rng=R2s_rng)

            ϕ_af[S] = @views res_af.ϕ[S]
            ϕ_fw[S] = @views res_fw.ϕ[S]

            R2s_af[S] = @views res_af.R2s[S]
            R2s_fw[S] = @views res_fw.R2s[S]

            pdff_af[S] = @views res_af.f[S]
            pdff_fw[S] = @views res_fw.f[S]

            c_af[S] = @views res_af.c[S]
            c_fw[S, :] = @views res_fw.c[S, :]

            χ2_af[S] = @views res_af.χ2[S]
            χ2_fw[S] = @views res_fw.χ2[S]

            res[nmbr]["pdff_af"] = pdff_af
            res[nmbr]["pdff_fw"] = pdff_fw
            res[nmbr]["pdff_ref"] = pdff_ref

            pdff_ref[(!).(S)] .= 0
            succ_af = sum(abs.(pdff_af[S] - pdff_ref[S]) .< 0.1)
            fail_af = sum(S) - succ_af
            succ_fw = sum(abs.(pdff_fw[S] - pdff_ref[S]) .< 0.1)
            fail_fw = sum(S) - succ_fw

            if visual
                subplt_size = (480, 330)

                plts_mat_32 = Matrix(undef, 2, 3)
                plts_mat_12 = Matrix(undef, 2, 1)

                for i in 1:n_slices
                    slice_str = i < 10 ? "0" * string(i) : string(i)

                    ciS_ = v_ciS[i]
                    pdff_af_ = @views reverse(pdff_af[:, :, i], dims=(1,))
                    pdff_fw_ = @views reverse(pdff_fw[:, :, i], dims=(1,))
                    pdff_ref_ = @views reverse(pdff_ref[:, :, i], dims=(1,))
                    ϕ_af_ = @views reverse(ϕ_af[:, :, i], dims=(1,))
                    ϕ_fw_ = @views reverse(ϕ_fw[:, :, i], dims=(1,))
                    R2s_af_ = @views reverse(R2s_af[:, :, i], dims=(1,))
                    R2s_fw_ = @views reverse(R2s_fw[:, :, i], dims=(1,))
                    abs_c_af_ = @views reverse(abs.(c_af[:, :, i]), dims=(1,))
                    angle_c_af_ = @views reverse(angle.(c_af[:, :, i]), dims=(1,))
                    abs_cw_fw_ = @views reverse(abs.(c_fw[:, :, i, 1]), dims=(1,))
                    angle_cw_fw_ = @views reverse(angle.(c_fw[:, :, i, 1]), dims=(1,))
                    abs_cf_fw_ = @views reverse(abs.(c_fw[:, :, i, 2]), dims=(1,))
                    angle_cf_fw_ = @views reverse(angle.(c_fw[:, :, i, 2]), dims=(1,))
                    χ2_af_ = @views reverse(χ2_af[:, :, i], dims=(1,))
                    χ2_fw_ = @views reverse(χ2_fw[:, :, i], dims=(1,))

                    Δpdff_fw = @views pdff_fw_[ciS_] - pdff_ref_[ciS_]
                    Δpdff_af = @views pdff_af_[ciS_] - pdff_ref_[ciS_]
                    mi, ma = min(Δpdff_fw..., Δpdff_af...), max(Δpdff_fw..., Δpdff_af...)
                    bins = range(mi, ma, 100)

                    plts_mat_32[1, 1] = @views heatmap(pdff_fw_, title="Free Weights (FW)", clim=(0, 1))
                    plts_mat_32[2, 1] = @views heatmap(pdff_fw_ - pdff_ref_, title="FW - REF")
                    plts_mat_32[1, 2] = @views heatmap(pdff_ref_, title="Reference (REF)", clim=(0, 1))
                    plts_mat_32[2, 2] = @views histogram(pdff_fw_[ciS_] - pdff_ref_[ciS_], bins=bins, la=0, fa=0.5, label="FW", fillcolor=:red)
                    plts_mat_32[2, 2] = @views stephist!(pdff_af_[ciS_] - pdff_ref_[ciS_], bins=bins, color=:blue, label="ML", title="Deviation from REF")
                    plts_mat_32[1, 3] = @views heatmap(pdff_af_, title="Maximum Likelihood (ML)", clim=(0, 1))
                    plts_mat_32[2, 3] = @views heatmap(pdff_af_ - pdff_ref_, title="ML - REF")

                    display(plot(plts_mat_32...,
                        layout=size(plts_mat_32'),
                        size=size(plts_mat_32) .* subplt_size, plot_title=string("PDFF: ", "data set ", nmbr, ", slice ", i)))

                    !isempty(fmt) && savefig("pdff_" * set_str * "_" * slice_str * fmt)
                    !isempty(fmt) && savefig("challenge_" * set_str * "_" * slice_str * "_1" * fmt)

                    plts_mat_32[1, 1] = @views heatmap(ϕ_af_, title="ϕ (ML)")
                    plts_mat_32[2, 1] = @views heatmap(ϕ_fw_, title="ϕ (FW)")

                    plts_mat_32[1, 2] = @views heatmap(R2s_af_, title="R2s (ML)", clim=(0, 0.15))
                    plts_mat_32[2, 2] = @views heatmap(R2s_fw_, title="R2s (FW)", clim=(0, 0.15))

                    plts_mat_32[1, 3] = @views heatmap(abs_c_af_, title="abs(c) (ML)")
                    plts_mat_32[2, 3] = @views heatmap(angle_c_af_, title="angle(c) (ML)")

                    display(plot(plts_mat_32...,
                        layout=size(plts_mat_32'),
                        size=size(plts_mat_32) .* subplt_size, plot_title=string("data set ", nmbr, ", slice ", i)))

                    !isempty(fmt) && savefig("phase_" * set_str * "_" * slice_str * fmt)
                    !isempty(fmt) && savefig("challenge_" * set_str * "_" * slice_str * "_2" * fmt)

                    plts_mat_32[1, 1] = @views heatmap(pdff_fw_ - pdff_ref_, title="FW - REF")
                    plts_mat_32[2, 1] = @views heatmap(angle_cw_fw_, title="angle(w) (FW)")

                    Δ_angle_wf = angle.(exp.(im * (angle_cw_fw_ .- angle_cf_fw_)))
                    plts_mat_32[1, 2] = @views heatmap(Δ_angle_wf, title="angle(w - f) (FW)")
                    plts_mat_32[2, 2] = @views heatmap(angle_cf_fw_, title="angle(f) (FW)")

                    plts_mat_32[1, 3] = @views heatmap(abs_c_af_, title="abs(c) (ML)")
                    plts_mat_32[2, 3] = @views heatmap(angle_c_af_, title="angle(c) (ML)")

                    display(plot(plts_mat_32...,
                        layout=size(plts_mat_32'),
                        size=size(plts_mat_32) .* subplt_size, plot_title=string("data set ", nmbr, ", slice ", i)))

                    !isempty(fmt) && savefig("coeff_" * set_str * "_" * slice_str * fmt)
                    !isempty(fmt) && savefig("challenge_" * set_str * "_" * slice_str * "_3" * fmt)

                    plts_mat_12[1, 1] = @views heatmap(χ2_af_ - χ2_fw_, clim=(0, 0.01), title="chi2(ML) - chi2(FW)")
                    plts_mat_12[2, 1] = @views histogram(χ2_af_[ciS_] - χ2_fw_[ciS_], title="chi2(ML) - chi2(FW)")

                    display(plot(plts_mat_12...,
                        layout=size(plts_mat_12'),
                        size=size(plts_mat_12) .* subplt_size, plot_title=string("data set ", nmbr, ", slice ", i)))

                    !isempty(fmt) && savefig("chi2_" * set_str * "_" * slice_str * fmt)
                    !isempty(fmt) && savefig("challenge_" * set_str * "_" * slice_str * "_4" * fmt)
                end
            end
            println("======================================")
            println("Total time spent for dataset ", nmbr, ":")
        end
        score_af = 100 * succ_af / (fail_af + succ_af)
        score_fw = 100 * succ_fw / (fail_fw + succ_fw)
        println("auto fat score = ", score_af)
        println("free weights score = ", score_fw)
        println("======================================")
    end

    return res
end

res = ismrm_challenge(1:1; visual=true, n_ϕ=4, fmt="", optim=false, R2s_rng=(0.0, 0.0))