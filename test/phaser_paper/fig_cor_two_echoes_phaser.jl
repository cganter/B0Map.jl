using HDF5, CairoMakie, Random, LinearAlgebra, StatsBase
import VP4Optim as VP
import B0Map as BM

BLAS.set_num_threads(1)

include("ph_util.jl")

# read the HDF5 file
fid = h5open("test/data/two_echoes/20241024_171954_702_ImDataParamsBMRR_subspace2comp_wfi.h5", "r");
obj_data = read(fid["ImDataParams"]);

signal = obj_data["signal"];
ss = size(signal);
data = zeros(ComplexF64, ss[3:5]..., ss[1])
for i in 1:2
    data[:, :, :, i] .= signal[i, 1, :, :, :]
end

# the supplied mask is too inclusive
# the following choice is better but far from perfect..
# the choice is insofar important as 
S = abs.(data[:, :, :, 2]) .> 0.25; # 0.5

# echo times
TEs = 1000obj_data["TE_s"];  # the expected unit is [ms]

# field strength
B0 = Float64(attrs(fid["ImDataParams"])["fieldStrength_T"])

# fat model
ppm_fat = read(fid["AlgoParams"]["FatModel"]["freqs_ppm"])
ampl_fat = read(fid["AlgoParams"]["FatModel"]["relAmps"])

# close the HDF5 file
close(fid)

# scanner-dependent convention for the orientation of precession
precession = :counterclockwise

# set up GRE parameters
grePar = VP.modpar(BM.GREMultiEchoWF;
    ts=TEs,
    B0=B0,
    ppm_fat=ppm_fat,
    ampl_fat=ampl_fat,
    precession=precession)

# generate instance of FitPar ...
fitpar = BM.fitPar(grePar, deepcopy(data), deepcopy(S))

# ... and of FitOpt
fitopt = BM.fitOpt()
fitopt.K = [3, 3, 3]
fitopt.R2s_rng = [0.0, 0.0]   # R2* ≡ 0 for two-echo GRE
fitopt.redundancy = 100
fitopt.subsampling = :random
fitopt.local_fit = false # we only want to reconstruct a single slice
fitopt.os_fac = [1.3]
fitopt.rng = MersenneTwister(42)
fitopt.μ_tikh = 1e-5
fitopt.balance = 2

# set up Fourier Kernel
Nρ = size(data)[1:3]
bs = BM.fourier_lin(Nρ[1:length(fitopt.K)], fitopt.K; os_fac=fitopt.os_fac)

cal = BM.B0map!(fitpar, fitopt, bs);

# to reset diagnostics
ϕ_loc = pdff = nothing

##

n_max = fitopt.balance + 1

_Φ = [(val=:Φ, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
_Φ_red = [(val=:Φ_red, cm=:roma, n=n, colbar=true) for n in 1:n_max]
_ϕ = [(val=:ϕ, rng_2π=true, cm=:romaO, n=n, colbar=true) for n in 1:n_max]
_ϕ_loc = [(val=:ϕ_loc, rng_2π=true, cm=:romaO, n=n, colbar=true) for n in 0:n_max]
_pdff = [(val=:pdff, cm=:imola, n=n, colbar=true) for n in 0:n_max]
_hist_Φ = [(val=:hist_Φ, n=n, nbins=40, bin_mode=:rice) for n in 0:n_max]
_hist_a∇Φ = [(val=:hist_a∇Φ, n=n, nbins=40, bin_mode=:rice) for n in 0:n_max]

plots = [_Φ[1] _ϕ[1] _ϕ[2] _ϕ[3];
         _hist_a∇Φ[1] _Φ_red[1] _Φ_red[2] _Φ_red[3];
         _hist_Φ[1] _hist_Φ[2] _hist_Φ[3] _hist_Φ[4];
         _pdff[1] _pdff[2] _pdff[3] _pdff[4]]

(fig, dax, ϕ_loc, pdff) = phaser_plots(plots, cal.PH, fitpar, fitopt;
    width_per_plot=230,
    height_per_plot=210,
    col_in=:blue, col_out=:red, alpha_out=0.3,
    font_pt=12, label_pt=8,
    slice=64,
    j=2,
    oi=x -> rotl90(x[:, end:-1:1]),
    letters=true,
    ϕ_loc=ϕ_loc,
    pdff=pdff,
)

display(fig)

##

fig_name = "two_echo_cor"
save(fig_name * ".svg", fig)
run(`/home/cganter/bin/svg2eps $fig_name`)
run(`epspdf $fig_name".eps"`)