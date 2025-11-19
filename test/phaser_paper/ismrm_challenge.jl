using MAT, CairoMakie, LaTeXStrings, LinearAlgebra
import VP4Optim as VP
import B0Map as BM

include("ph_util.jl")

BLAS.set_num_threads(1)

# ISMRM challenge 2012 data sets:
data_set, slice = 17, 2
#data_set, slice = 5, 2
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
fitopt.K = [7, 7]
fitopt.redundancy = Inf
fitopt.os_fac = [1.3]
fitopt.balance = 5
fitopt.rapid_balance = true
fitopt.multi_scale = false

# calculate score

d = ismrm_challenge_score(fitopt; data_sets = (1:2...,4:17...));

##

# apply PHASER
cal = ismrm_challenge(fitopt; data_set=data_set, slice=slice);

ϕ_loc = pdff = nothing

##

n_grad, n_bal = cal.PH.n_grad, cal.PH.n_bal
n_max = n_grad + n_bal

_Φ = [(val=:Φ, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
_Φ_red = [(val=:Φ_red, cm=:romaO, n=n, colbar=true) for n in 1:n_max]
_∇Φ = [(val=:∇Φ, cm=:managua, cm_rng=(-0.2, 0.2), n=n, colbar=true) for n in 0:n_max]
_a∇Φ = [(val=:a∇Φ, cm=:imola, cm_rng=(0, 1), n=n, colbar=true) for n in 0:n_max]
_∇Φ_red = [(val=:∇Φ_red, cm=:managua, cm_rng=(-0.2, 0.2), n=n, colbar=true) for n in 1:n_max]
_a∇Φ_red = [(val=:a∇Φ_red, cm=:imola, cm_rng=(0, 0.2), n=n, colbar=true) for n in 1:n_max]
_ϕ = [(val=:ϕ, rng_2π=false, cm=:roma, n=n, colbar=true) for n in 1:n_max]
_ϕ_loc = [(val=:ϕ_loc, rng_2π=false, cm=:roma, n=n, colbar=true) for n in 0:n_max]
_pdff = [(val=:pdff, cm=:imola, n=n, colbar=true) for n in 0:n_max]
_hist_Φ = [(val=:hist_Φ, n=n, nbins=100, bin_mode=:fixed) for n in 0:n_max]
_hist_ϕ_Φ = [(val=:hist_ϕ_Φ, n=n, nbins=100, bin_mode=:fixed) for n in 1:n_max]
_hist_a∇Φ = [(val=:hist_a∇Φ, n=n, nbins=50, bin_mode=:fixed) for n in 0:n_max]
_χ2λ = [(val=:χ2λ, n=n) for n in 1:n_bal]

#=
plots = [_hist_Φ[9] _ϕ[1] _ϕ[2] _ϕ[3];
        _hist_Φ[end] _ϕ_loc[end] _pdff[end] _ϕ[end]]

plots = [_Φ[1] _hist_a∇Φ[1] _hist_Φ[1] _pdff[1];
    _Φ_red[n_grad] _ϕ[n_grad] _hist_Φ[n_grad+1] _pdff[n_grad+1];
    _Φ_red[end] _ϕ[end] _hist_Φ[end] _pdff[end]]

plots = [_∇Φ[1] _∇Φ_red[1];
    _ϕ[1] _a∇Φ_red[1]]
=#
plots = [_Φ[1] _Φ_red[1] _hist_Φ[1] _pdff[1];
    _Φ_red[n_grad] _ϕ[n_grad] _hist_Φ[n_grad+1] _pdff[n_grad+1];
    _Φ_red[end] _ϕ[end] _hist_Φ[end] _pdff[end]]


(fig, dax, ϕ_loc, pdff) = phaser_plots(plots, cal.PH, cal.fitpar, fitopt;
    width_per_plot=300,
    height_per_plot=230,
    col_in=:blue, col_out=:red, alpha_out=0.3,
    font_pt=12, label_pt=10,
    slice=1,
    j=2,
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
