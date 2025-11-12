using MAT, CairoMakie, LaTeXStrings, LinearAlgebra
import VP4Optim as VP
import B0Map as BM

include("ph_util.jl")

BLAS.set_num_threads(1)

# ISMRM challenge 2012 data sets:
data_set, slice = 5, 3
#data_set, slice = 12, 2
oi = orient_ISMRM(data_set)

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
fitopt.os_fac = [1.3]
fitopt.balance = 2

# apply PHASER
cal = ismrm_challenge(fitopt; data_set=data_set, slice=slice);

ϕ_loc = pdff = nothing

##

n_max = fitopt.balance + 1

_Φ = [(val=:Φ, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
_Φ_red = [(val=:Φ_red, cm=:roma, n=n, colbar=true) for n in 1:n_max]
_ϕ = [(val=:ϕ, rng_2π=false, cm=:roma, n=n, colbar=true) for n in 1:n_max]
_ϕ_loc = [(val=:ϕ_loc, rng_2π=true, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
_pdff = [(val=:pdff, cm=:imola, n=n, colbar=true) for n in 0:n_max]
_hist_Φ = [(val=:hist_Φ, n=n, nbins=50, bin_mode=:fixed) for n in 0:n_max]
_hist_a∇Φ = [(val=:hist_a∇Φ, n=n, nbins=50, bin_mode=:fixed) for n in 0:n_max]

plots = [_Φ[1] _ϕ[1] _ϕ[2] _ϕ[end];
         _hist_a∇Φ[1] _Φ_red[1] _Φ_red[2] _Φ_red[end];
         _hist_Φ[1] _hist_Φ[2] _hist_Φ[3] _hist_Φ[end];
         _pdff[1] _pdff[2] _pdff[3] _pdff[end]]

(fig, dax, ϕ_loc, pdff) = phaser_plots(plots, cal.bm.PH, cal.fitpar, fitopt;
    width_per_plot=260,
    height_per_plot=200,
    col_in=:blue, col_out=:red, alpha_out=0.3,
    font_pt=12, label_pt=8,
    slice=1,
    j=1,
    oi=oi,
    letters=true,
    ϕ_loc=ϕ_loc,
    pdff=pdff,
)

display(fig)

## save results

# file name
fig_name = "ismrm_ds_" * string(data_set) * "_sl_" * string(slice)
# save svg
save(fig_name * ".svg", fig)
# to generate an eps file, svg2eps can be used (on Linux)
run(`/home/cganter/bin/svg2eps $fig_name`) # just set the path which applies to you
# generate a pdf for rapid complilation in Overleaf.
run(`epspdf $fig_name".eps"`)
