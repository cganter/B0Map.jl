using MAT, Plots, Statistics
import VP4Optim as VP
import VP4MRI as MR

function low_field(;
    ic_dir="test/low_field/",
    n_ϕ=3,
    n_chunks=8Threads.nthreads(),
    optim=true,
    visual=true,
    fmt=".pdf",
    R2s_rng=(0.0, 1.0))

    # IRMRM challenge fat specification
    ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
    ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

#    ppm_fat = [0.59, 0.49, -0.5, -1.95, -2.46, -2.68, -3.1, -3.4, -3.8]
#    ampl_fat = 0.01 * [3.7, 1, 3.9, 0.6, 5.8, 6.2, 5.8, 64.2, 8.8]

    file_str = ic_dir * "imDataParams_small.mat"
    res = Dict()

    # extract acquisition parameters
    d = matread(file_str)["imDataParams"]
    n_slices = size(d["images"], 3)
    TEs = 1000.0 * d["TE"][:]
    nTE = length(TEs)
    B0 = d["FieldStrength"]
    precession = (d["PrecessionIsClockwise"] != 1.0) ? :clockwise : :counterclockwise

    Nρ = size(d["images"])[1:3]
    data = zeros(ComplexF64, Nρ..., nTE)
    copy!(data, reshape(d["images"][:, :, :, 1, 1:nTE], Nρ..., nTE))
    data ./= max(abs.(data)...)
    a_data = abs.(data)
    ph_data = angle.(data)
    data = a_data .* exp.((im * π) .* ph_data)
    res["data"] = data
    S = abs.(d["images"][:, :, :, 1, 1]) .> 125.0
    res["S"] = S

    pdwf_af = zeros(size(S)...)
    pdwf_rw = zeros(size(S)...)
    pdff_af = zeros(size(S)...)
    pdff_rw = zeros(size(S)...)
    ϕ_af = zeros(size(S)...)
    ϕ_rw = zeros(size(S)...)
    R2s_af = zeros(size(S)...)
    R2s_rw = zeros(size(S)...)
    c_af = zeros(ComplexF64, size(S)...)
    c_rw = zeros(Float64, size(S)..., 2)
    χ2_af = zeros(size(S)...)
    χ2_rw = zeros(size(S)...)
    coil_phase_rw = zeros(size(S)...)

    v_S = []
    v_ciS = []
    for i in 1:n_slices
        push!(v_S, @views S[:, :, i])
        push!(v_ciS, @views CartesianIndices(v_S[i])[v_S[i]])
    end

    @time begin
        args = (TEs, B0, ppm_fat, ampl_fat, precession)
        res = Dict()

        res["args"] = args
        res["gre_af"] = MR.greMultiEchoWF(args...)
        res["gre_rw"] = MR.greMultiEchoWFRW(args...)

        res_af = res["res_af"] = MR.B0_map_varpro(
            MR.greMultiEchoWF, args, data, S; n_ϕ=n_ϕ, n_chunks=n_chunks, optim=optim, R2s_rng=R2s_rng)
        res_rw = res["res_rw"] = MR.B0_map_varpro(
            MR.greMultiEchoWFRW, args, data, S; n_ϕ=n_ϕ, n_chunks=n_chunks, optim=optim, R2s_rng=R2s_rng)

        ϕ_af[S] = @views res_af.ϕ[S]
        ϕ_rw[S] = @views res_rw.ϕ[S]

        R2s_af[S] = @views res_af.R2s[S]
        R2s_rw[S] = @views res_rw.R2s[S]

        pdwf_af[S] = @views 1.0 .- res_af.f[S]
        pdwf_rw[S] = @views 1.0 .- res_rw.f[S]
        pdff_af[S] = @views res_af.f[S]
        pdff_rw[S] = @views res_rw.f[S]

        c_af[S] = @views res_af.c[S]
        c_rw[S, :] = @views real.(res_rw.c[S, :])

        χ2_af[S] = @views res_af.χ2_GSS[S]
        χ2_rw[S] = @views res_rw.χ2_GSS[S]

        coil_phase_rw[S] = res_rw.coil_phase_GSS[S]
        
        if visual
            subplt_size = (480, 330)

            plts_mat_12 = Matrix(undef, 2, 1)
            plts_mat_22 = Matrix(undef, 2, 2)
            plts_mat_32 = Matrix(undef, 2, 3)

            for i in 1:n_slices
                slice_str = i < 10 ? "0" * string(i) : string(i)
                ciS_ = v_ciS[i]

                plts_mat = Matrix(undef, 2, 6)
                
                

                pdwf_af_ = @views reverse(pdwf_af[:, :, i], dims=(1,))
                pdwf_rw_ = @views reverse(pdwf_rw[:, :, i], dims=(1,))
                pdff_af_ = @views reverse(pdff_af[:, :, i], dims=(1,))
                pdff_rw_ = @views reverse(pdff_rw[:, :, i], dims=(1,))

                plts_mat[1, 1] = @views heatmap(pdwf_rw_, title="wf (RW)", clim=(0, 1))
                plts_mat[2, 1] = @views heatmap(pdwf_af_, title="wf (ML)", clim=(0, 1))
                plts_mat[1, 2] = @views heatmap(pdff_rw_, title="ff (RW)", clim=(0, 1))
                plts_mat[2, 2] = @views heatmap(pdff_af_, title="ff (ML)", clim=(0, 1))


                ϕ_af_ = @views reverse(ϕ_af[:, :, i], dims=(1,))
                ϕ_rw_ = @views reverse(ϕ_rw[:, :, i], dims=(1,))

                plts_mat[1, 3] = @views heatmap(ϕ_rw_, title="ϕ(B0) (RW)", clim=(-π, π))
                plts_mat[2, 3] = @views heatmap(ϕ_af_, title="ϕ(B0) (ML)", clim=(-π, π))

                c_af_ = @views reverse(angle.(c_af[:, :, i]), dims=(1,))
                cw_rw_ = @views reverse(c_rw[:, :, i, 1], dims=(1,))
                cf_rw_ = @views reverse(c_rw[:, :, i, 2], dims=(1,))
                in_phase = cw_rw_ .* cf_rw_ .> 0
                opp_phase = cw_rw_ .* cf_rw_ .< 0
                coil_phase_rw_ = @views reverse(coil_phase_rw[:, :, i, 1], dims=(1,))

                plts_mat[1, 4] = @views heatmap(coil_phase_rw_, title="ϕ(coil) (RW)", clim=(-π, π))
                plts_mat[2, 4] = @views heatmap(c_af_, title="ϕ(coil) (ML)", clim=(-π, π))

                plts_mat[1, 5] = @views heatmap(cw_rw_ .* cf_rw_ .> 0.0, title="w * f > 0")
                plts_mat[2, 5] = @views heatmap(cw_rw_ .> 0.0, title="w > 0")
                
                χ2_af_ = @views reverse(χ2_af[:, :, i], dims=(1,))
                χ2_rw_ = @views reverse(χ2_rw[:, :, i], dims=(1,))
                
                cw_ip = deepcopy(pdwf_rw_)
                cw_ip[opp_phase] .= 0.0
                cw_op = deepcopy(pdwf_rw_)
                cw_op[in_phase] .= 0.0
                cf_ip = deepcopy(pdff_rw_)
                cf_ip[opp_phase] .= 0.0
                cf_op = deepcopy(pdff_rw_)
                cf_op[in_phase] .= 0.0


                plts_mat[1, 5] = @views heatmap(abs.(cw_ip), title="wf[in_phase] (RW)")
                plts_mat[2, 5] = @views heatmap(abs.(cw_op), title="wf[opp_phase] (RW)")
                plts_mat[1, 6] = @views heatmap(abs.(cf_ip), title="ff[in_phase] (RW)")
                plts_mat[2, 6] = @views heatmap(abs.(cf_op), title="ff[opp_phase] (RW)")


                display(plot(plts_mat...,
                    layout=size(plts_mat'),
                    size=size(plts_mat) .* subplt_size, plot_title=string("slice ", i)))

                !isempty(fmt) && savefig("maps_" * slice_str * fmt)

                #=
                plts_mat= Matrix(undef, length(precs), 2)

                median_χ2_af = zeros(length(precs))
                median_χ2_rw = zeros(length(precs))

                for (j, prec) in enumerate(precs)
                    p_str = string(prec)

                    χ2_af_ = @views reverse(χ2_af[:, :, i, j], dims=(1,))
                    χ2_rw_ = @views reverse(χ2_rw[:, :, i, j], dims=(1,))

                    median_χ2_af[j] = median(χ2_af[ciS_, i, j])
                    median_χ2_rw[j] = median(χ2_rw[ciS_, i, j])

                    plts_mat[j, 1] = @views heatmap(χ2_rw_, title= p_str * " (RW)")
                    plts_mat[j, 2] = @views heatmap(χ2_af_, title= p_str * " (ML)")
                end

                println("ML: median(", precs[1], ") = ", median_χ2_af[1])
                println("ML: median(", precs[2], ") = ", median_χ2_af[2])
                println("RW: median(", precs[1], ") = ", median_χ2_rw[1])
                println("RW: median(", precs[2], ") = ", median_χ2_rw[2])

                display(plot(plts_mat...,
                    layout=size(plts_mat'),
                    size=size(plts_mat) .* subplt_size, plot_title=string("χ2, slice ", i)))

                !isempty(fmt) && savefig("chi2_" * slice_str * fmt)


                ϕ_af_ = @views reverse(ϕ_af[:, :, i], dims=(1,))
                ϕ_rw_ = @views reverse(ϕ_rw[:, :, i], dims=(1,))
                R2s_af_ = @views reverse(R2s_af[:, :, i], dims=(1,))
                R2s_rw_ = @views reverse(R2s_rw[:, :, i], dims=(1,))
                abs_c_af_ = @views reverse(abs.(c_af[:, :, i]), dims=(1,))
                angle_c_af_ = @views reverse(angle.(c_af[:, :, i]), dims=(1,))
                abs_cw_rw_ = @views reverse(abs.(c_rw[:, :, i, 1]), dims=(1,))
                angle_cw_rw_ = @views reverse(angle.(c_rw[:, :, i, 1]), dims=(1,))
                abs_cf_rw_ = @views reverse(abs.(c_rw[:, :, i, 2]), dims=(1,))
                angle_cf_rw_ = @views reverse(angle.(c_rw[:, :, i, 2]), dims=(1,))
                χ2_af_ = @views reverse(χ2_af[:, :, i], dims=(1,))
                χ2_rw_ = @views reverse(χ2_rw[:, :, i], dims=(1,))


                plts_mat_22[1, 1] = @views heatmap(pdff_rw_, title="pdff (FW)", clim=(0, 1))
                plts_mat_22[2, 1] = @views heatmap(pdff_af_, title="pdff (ML)", clim=(0, 1))
                plts_mat_22[1, 2] = @views heatmap(ϕ_rw_, title="ϕ (FW)")
                plts_mat_22[2, 2] = @views heatmap(ϕ_af_, title="ϕ (ML)")

                display(plot(plts_mat_22...,
                    layout=size(plts_mat_22'),
                    size=size(plts_mat_22) .* subplt_size, plot_title=string("slice ", i)))


                !isempty(fmt) && savefig("pdff_" * set_str * "_" * slice_str * fmt)
                !isempty(fmt) && savefig("challenge_" * set_str * "_" * slice_str * "_1" * fmt)


                plts_mat_32[1, 3] = @views heatmap(abs_c_af_, title="abs(c) (ML)")
                plts_mat_32[2, 3] = @views heatmap(angle_c_af_, title="angle(c) (ML)")

                display(plot(plts_mat_32...,
                    layout=size(plts_mat_32'),
                    size=size(plts_mat_32) .* subplt_size, plot_title=string("data set ", nmbr, ", slice ", i)))

                !isempty(fmt) && savefig("phase_" * set_str * "_" * slice_str * fmt)
                !isempty(fmt) && savefig("challenge_" * set_str * "_" * slice_str * "_2" * fmt)

                plts_mat_32[1, 1] = @views heatmap(pdff_rw_ - pdff_ref_, title="FW - REF")
                plts_mat_32[2, 1] = @views heatmap(angle_cw_rw_, title="angle(w) (FW)")

                Δ_angle_wf = angle.(exp.(im * (angle_cw_rw_ .- angle_cf_rw_)))
                plts_mat_32[1, 2] = @views heatmap(Δ_angle_wf, title="angle(w - f) (FW)")
                plts_mat_32[2, 2] = @views heatmap(angle_cf_rw_, title="angle(f) (FW)")

                plts_mat_32[1, 3] = @views heatmap(abs_c_af_, title="abs(c) (ML)")
                plts_mat_32[2, 3] = @views heatmap(angle_c_af_, title="angle(c) (ML)")

                display(plot(plts_mat_32...,
                    layout=size(plts_mat_32'),
                    size=size(plts_mat_32) .* subplt_size, plot_title=string("data set ", nmbr, ", slice ", i)))

                !isempty(fmt) && savefig("coeff_" * set_str * "_" * slice_str * fmt)
                !isempty(fmt) && savefig("challenge_" * set_str * "_" * slice_str * "_3" * fmt)

                plts_mat_12[1, 1] = @views heatmap(χ2_af_ - χ2_rw_, clim=(0, 0.01), title="chi2(ML) - chi2(FW)")
                plts_mat_12[2, 1] = @views histogram(χ2_af_[ciS_] - χ2_rw_[ciS_], title="chi2(ML) - chi2(FW)")

                display(plot(plts_mat_12...,
                    layout=size(plts_mat_12'),
                    size=size(plts_mat_12) .* subplt_size, plot_title=string("data set ", nmbr, ", slice ", i)))

                !isempty(fmt) && savefig("chi2_" * set_str * "_" * slice_str * fmt)
                !isempty(fmt) && savefig("challenge_" * set_str * "_" * slice_str * "_4" * fmt)
                =#
            end
        end
        println("======================================")
        println("Total time spent for dataset:")
        println("======================================")
    end

    return res
end

res = low_field(; visual=true, n_ϕ=4, fmt=".pdf", optim=false, R2s_rng=(0.05, 0.05))