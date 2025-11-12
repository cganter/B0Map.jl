using CairoMakie, Random, StatsBase
import VP4Optim as VP
import B0Map as BM

include("../test_tools.jl")
include("ph_util.jl")

spp = SimPhaPar()

spp.TEs = 1.15 * [1, 2, 3]
spp.B0 = 1.5
spp.freq_rng = [-2, 2]
spp.Nρ = [256, 256]
spp.K = [2, 2]
spp.local_fit = false
spp.S_holes = 0.5
spp.S_io = :out
spp.cov_mat = 0.05^2 * [1;;]
spp.subsampling = :fibonacci
spp.balance = 2
spp.add_noise = true
spp.os_fac = [1.3]
spp.ppm_fat_pha = spp.ppm_fat .- 0.1
spp.rng = MersenneTwister(42)
spp.redundancy = Inf
sim = simulate_phantom(spp);

ϕ_loc = pdff = nothing

##

n_max = sim.fitopt.balance + 1

_Φ = [(val=:Φ, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
_Φ_red = [(val=:Φ_red, cm=:roma, n=n, colbar=true) for n in 1:n_max]
_ϕ = [(val=:ϕ, rng_2π=true, cm=:romaO, n=n, colbar=true) for n in 1:n_max]
_ϕ_loc = [(val=:ϕ_loc, rng_2π=true, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
_pdff = [(val=:pdff, cm=:imola, n=n, colbar=true) for n in 0:n_max]
_hist_Φ = [(val=:hist_Φ, n=n, nbins=50, bin_mode=:fixed) for n in 0:n_max]
_hist_a∇Φ = [(val=:hist_a∇Φ, n=n, nbins=50, bin_mode=:fixed) for n in 0:n_max]

plots = [_Φ[1] _ϕ[1] _ϕ[2] _ϕ[end];
         _hist_a∇Φ[1] _Φ_red[1] _Φ_red[2] _Φ_red[end];
         _hist_Φ[1] _hist_Φ[2] _hist_Φ[3] _hist_Φ[end];
         _pdff[1] _pdff[2] _pdff[3] _pdff[end]]

(fig, dax, ϕ_loc, pdff) = phaser_plots(plots, sim.PH, sim.fitpar, sim.fitopt;
    width_per_plot=260,
    height_per_plot=200,
    col_in=:blue, col_out=:red, alpha_out=0.3,
    font_pt=12, label_pt=8,
    slice=1,
    j=1,
    letters=true,
    ϕ_loc=ϕ_loc,
    pdff=pdff,
)

display(fig)
