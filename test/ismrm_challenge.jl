using MAT, Plots, PlotThemes, BenchmarkTools, LinearAlgebra
import VP4Optim as VP
import B0Map as BM

#theme(:dark, color=:batlow)
BLAS.set_num_threads(1)

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

data_set = 17
fitopt.redundancy = Inf

@time res = ismrm_challenge(BM.GREMultiEchoWF, BM.ModParWF, fitopt; fit = :PHASER, data_set = data_set);