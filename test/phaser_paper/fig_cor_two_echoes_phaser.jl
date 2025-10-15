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
fitopt.R2s_rng = [0.0, 0.0]
fitopt.redundancy = 100
fitopt.subsampling = :fibonacci
fitopt.balance = true
fitopt.locfit = false # we only want to reconstruct a single slice
fitopt.os_fac = [1.3]
fitopt.rng = MersenneTwister(42)
fitopt.diagnostics = true
fitopt.remove_gradient_outliers = true
fitopt.remove_local_outliers = true

# set up Fourier Kernel
Nρ = size(data)[1:3]
bs = BM.fourier_lin(Nρ[1:length(fitopt.K)], fitopt.K; os_fac=fitopt.os_fac)

# apply PHASER
res = BM.phaser!(fitpar, fitopt, bs)

# we select a slice to show
cor_sl = 64

# full 2d local fit
fitpar_loc = BM.fitPar(grePar, deepcopy(data[:,:,cor_sl,:]), deepcopy(S[:,:,cor_sl]))
BM.local_fit(fitpar_loc, fitopt)
ϕ_loc = fitpar_loc.ϕ
f_loc = BM.fat_fraction_map(fitpar_loc, fitopt)
freq_loc = BM.freq_map(fitpar_loc)

fitpar_loc_phs = BM.fitPar(grePar, deepcopy(data[:,:,cor_sl,:]), deepcopy(S[:,:,cor_sl]))
# set PHASER as starting value
fitpar_loc_phs.ϕ[:,:] .= @views fitpar.ϕ[:,:,cor_sl]
BM.set_num_phase_intervals(fitpar_loc_phs, fitopt, 0)
fitopt.optim = true
BM.local_fit(fitpar_loc_phs, fitopt)
f_loc_phs = BM.fat_fraction_map(fitpar_loc_phs, fitopt)
freq_loc_phs = BM.freq_map(fitpar_loc_phs)

#

# size definitions
# these are relative to 1 CSS px
inch = 96
pt = 4 / 3

#width, height = 6.9inch, 4.6inch
width, height = 6.9inch, 4.6inch #6.6inch
colmapO = :romaO
colmap = :roma
colmap_f = :imola

S_sl = deepcopy(S[:,:,cor_sl])
noS_sl = (!).(S_sl)
S_phs_sl = deepcopy(res.PH.S[:,:,cor_sl])
noS_phs_sl = (!).(S_phs_sl)
c_loc_phs = zeros(ComplexF64, size(S_sl))
BM.calc_par(fitpar_loc_phs, fitopt, x -> BM.coil_sensitivities(x)[1], c_loc_phs)

#

ϕ_ML_sl = deepcopy(res.PH.ϕ_ML[:,:,cor_sl])
ϕ_ML_sl[noS_phs_sl] .= NaN
ϕ_loc[noS_sl] .= NaN
ϕ_phs_sl = @views fitpar.ϕ[:,:,cor_sl]
ϕ_phs_sl[noS_sl] .= NaN
ϕ0_phs_sl = @views res.PH.ϕ0[:,:,cor_sl]
ϕ0_phs_sl[noS_sl] .= NaN
ϕ1_phs_sl = @views res.PH.ϕ1[:,:,cor_sl]
ϕ1_phs_sl[noS_sl] .= NaN
ϕ_phs_loc_sl = fitpar_loc_phs.ϕ
ϕ_phs_loc_sl[noS_sl] .= NaN
c_loc_phs[noS_sl] .= NaN
lim_ϕ = (-π, π)
f_loc[noS_sl] .= NaN
f_loc_phs[noS_sl] .= NaN
cw_loc_phs = abs.(c_loc_phs) .* (1 .- f_loc_phs)
cf_loc_phs = abs.(c_loc_phs) .* f_loc_phs
c_angle_loc_phs = angle.(c_loc_phs)

fig = Figure(size = (width, height))

# -------------------------------------------------

ax = Axis(fig[1, 1],
    title=L"$$PDFF: $\Phi$",
)

heatmap!(ax,
    rotl90(f_loc[:,end:-1:1]),
    colormap=colmap_f,
    colorrange=(0,1),
    nan_color=:black,
)

hidedecorations!(ax)
Label(fig[1, 1, TopLeft()], "A",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

ax = Axis(fig[1, 2],
    title=L"$\Phi$",
)

heatmap!(ax,
    rotl90(ϕ_loc[:,end:-1:1]),
    colormap=colmapO,
    nan_color=:black,
)

hidedecorations!(ax)
Label(fig[1, 2, TopLeft()], "B",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

ax = Axis(fig[1, 3],
    title=L"$\Phi \in S$",
)

heatmap!(ax,
    rotl90(ϕ_ML_sl[:,end:-1:1]),
    colormap=colmapO,
    nan_color=:black,
)

hidedecorations!(ax)
Label(fig[1, 3, TopLeft()], "C",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

Colorbar(fig[1, 4],
    colorrange=(-π, π),
    colormap=colmapO,
    ticks=([-π, 0.0, π], ["-π", "0", "π"]),
    ticklabelsize=8pt,
)

# -------------------------------------------------

ax = Axis(fig[2, 1],
    title=L"PDFF: $\Phi\left(\varphi^{(1)}\right)$",
)

heatmap!(ax,
    rotl90(f_loc_phs[:,end:-1:1]),
    colormap=colmap_f,
    colorrange=(0,1),
    nan_color=:black,
)

hidedecorations!(ax)
Label(fig[2, 1, TopLeft()], "D",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

ax = Axis(fig[2, 3],
    title=L"$\varphi^{(1)}$",
)

heatmap!(ax,
    rotl90(ϕ_phs_sl[:,end:-1:1]),
    colormap=colmapO,
    nan_color=:black,
    colorrange=lim_ϕ,
)

hidedecorations!(ax)
Label(fig[2, 3, TopLeft()], "F",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

ax = Axis(fig[2, 2],
    title=L"$\Phi\left(\varphi^{(1)}\right)$",
)

heatmap!(ax,
    rotl90(ϕ_phs_loc_sl[:,end:-1:1]),
    colormap=colmapO,
    nan_color=:black,
    colorrange=lim_ϕ,
)

hidedecorations!(ax)
Label(fig[2, 2, TopLeft()], "E",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

Colorbar(fig[2, 4],
    colorrange=(-π, π),
    colormap=colmapO,
    ticks=([-π, 0.0, π], ["-π", "0", "π"]),
    ticklabelsize=8pt,
)

#=
# -------------------------------------------------

ax = Axis(fig[3, 1],
    title=L"$$water",
)

heatmap!(ax,
    rotl90(cw_loc_phs[:,end:-1:1]),
    colormap=colmap_f,
    nan_color=:black,
)

hidedecorations!(ax)
Label(fig[3, 1, TopLeft()], "G",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

ax = Axis(fig[3, 2],
    title=L"$$fat",
)

heatmap!(ax,
    rotl90(cf_loc_phs[:,end:-1:1]),
    colormap=colmap_f,
    nan_color=:black,
)

hidedecorations!(ax)
Label(fig[3, 2, TopLeft()], "H",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

ax = Axis(fig[3, 3],
    title=L"$$coil phase",
)

heatmap!(ax,
    rotl90(c_angle_loc_phs[:,end:-1:1]),
    colormap=colmapO,
    colorrange=(-π, π),
    nan_color=:black,
)

hidedecorations!(ax)
Label(fig[3, 3, TopLeft()], "I",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

Colorbar(fig[3, 4],
    colorrange=(-π, π),
    colormap=colmapO,
    ticks=([-π, 0.0, π], ["-π", "0", "π"]),
    ticklabelsize=8pt,
)

# -------------------------------------------------
=#

display(fig)

## show workflow

(fig_wf, _) = phaser_workflow!(res.PH, slice = cor_sl, oi= x -> rotl90(x[:,end:-1:1]))
display(fig_wf)

##

fig_name = "two_echo_cor"
save(fig_name * ".svg", fig)
run(`/home/cganter/bin/svg2eps $fig_name`)
run(`epspdf $fig_name".eps"`)