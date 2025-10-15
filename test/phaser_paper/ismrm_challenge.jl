using MAT, CairoMakie, LaTeXStrings, LinearAlgebra
import VP4Optim as VP
import B0Map as BM

include("ph_util.jl")

BLAS.set_num_threads(1)

# ISMRM challenge 2012 data sets:
#data_set, slice = 5, 5
data_set, slice = 12, 3

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

# set PHASER parameters
fitopt = BM.fitOpt()
fitopt.K = [5, 5]
fitopt.redundancy = Inf
fitopt.diagnostics = true
fitopt.os_fac = [1.3]

# apply PHASER
cal = ismrm_challenge(fitopt; data_set=data_set, slice=slice);

# generate image
(fig, dax) = gen_fig_ISMRM(cal;
    width = 800,
    height = 800,
    cm_phase = :roma,
    cm_fat = :imola,
)

# show it
display(fig)

(fig_wf, _) = phaser_workflow!(cal.PH.PH, oi=orient_ISMRM(data_set))
display(fig_wf)

## save results

# file name
fig_name = "ismrm_ds_" * string(data_set) * "_sl_" * string(slice)
# save svg
save(fig_name * ".svg", fig)
# to generate an eps file, svg2eps can be used (on Linux)
run(`/home/cganter/bin/svg2eps $fig_name`) # just set the path which applies to you
# generate a pdf for rapid complilation in Overleaf.
run(`epspdf $fig_name".eps"`)
