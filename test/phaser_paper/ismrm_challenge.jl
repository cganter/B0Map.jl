using MAT, CairoMakie, LaTeXStrings, LinearAlgebra
import VP4Optim as VP
import B0Map as BM

include("ph_util.jl")

BLAS.set_num_threads(1)

# ISMRM challenge 2012 data sets:

data_set = 5

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

fitopt = BM.fitOpt()
fitopt.K = [4, 4]
fitopt.redundancy = Inf
fitopt.diagnostics = true
fitopt.balance = true
fitopt.os_fac = [1.3]

cal = ismrm_challenge(fitopt; data_set=data_set);

##

(fig, dax) = gen_fig_ISMRM(cal;
    slice = 3,
    width = 800,
    height = 800,
    cm_phase_2π = :romaO,
    cm_phase = :roma,
    cm_fat = :imola,
)

display(fig)

##

fig_name = "ismrm_ds_5_sl_3"
#fig_name = "ismrm_ds_12_sl_2"
save(fig_name * ".svg", fig)
run(`/home/cganter/bin/svg2eps $fig_name`)
run(`epspdf $fig_name".eps"`)
