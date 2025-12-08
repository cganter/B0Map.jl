using HDF5, CairoMakie, Random, LinearAlgebra
import VP4Optim as VP
import B0Map as BM

BLAS.set_num_threads(1)

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
S = abs.(data[:, :, :, 2]) .> 0.5;

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
grePar_FC = VP.modpar(BM.GREMultiEchoWF;
    ts=TEs,
    B0=B0,
    ppm_fat=ppm_fat,
    ampl_fat=ampl_fat,
    precession=precession)

grePar_RW = VP.modpar(BM.GREMultiEchoWFRW;
    ts=TEs,
    B0=B0,
    ppm_fat=ppm_fat,
    ampl_fat=ampl_fat,
    precession=precession)

# we select a single slice ...
sl = 64
S_ = S[:, :, sl]
not_S = (!).(S_)
data_ = data[:, :, sl, :]

# generate fit parameters
fitpar_FC = BM.fitPar(grePar_FC, deepcopy(data_), deepcopy(S_))
fitpar_RW = BM.fitPar(grePar_RW, deepcopy(data_), deepcopy(S_))

# generate fit options
fitopt = BM.fitOpt()
fitopt.R2s_rng = [0.0, 0.0]
fitopt.optim = false # not needed, since R2s = 0 is fixed

fitopt_FC = deepcopy(fitopt)
fitopt_RW = deepcopy(fitopt)

# local fit
res_FC = BM.local_fit!(fitpar_FC, fitopt_FC)
res_RW = BM.local_fit!(fitpar_RW, fitopt_RW)

# extract maps
ϕ_RW = fitpar_RW.ϕ
ϕ_RW[not_S] .= NaN
f_RW = BM.fat_fraction_map(fitpar_RW, fitopt_RW)
f_RW[not_S] .= NaN
freq_RW = BM.freq_map(fitpar_RW);

ϕ_FC = fitpar_FC.ϕ
ϕ_FC[not_S] .= NaN
f_FC = BM.fat_fraction_map(fitpar_FC, fitopt_FC)
f_FC[not_S] .= NaN
freq_FC = BM.freq_map(fitpar_FC);

coil_FC = map(x -> x[1], fitpar_FC.c)
in_phase_FC = abs.(coil_FC)
coil_phase_FC = angle.(coil_FC);

# =============== generate phase maps =======================

# size definitions
# these are relative to 1 CSS px
inch = 96
pt = 4 / 3

width, height = 6.9inch, 6inch
colmapO = :romaO

fig = Figure(size=(width, height), fontsize=12pt)

ax = Axis(fig[1, 1],
    title=L"$\varphi_{RW}$",
)

heatmap!(ax,
    ϕ_RW',
    colormap=colmapO,
    nan_color=:black,
)
hidedecorations!(ax)
Label(fig[1, 1, TopLeft()], "A",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

ax = Axis(fig[1, 2],
    title=L"$\varphi_{FC}$",
)

heatmap!(ax,
    ϕ_FC',
    colormap=colmapO,
    nan_color=:black,
)
hidedecorations!(ax)
Label(fig[1, 2, TopLeft()], "B",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

Colorbar(fig[1, 3],
    colorrange=(-π, π),
    colormap=colmapO,
    ticks=([-π, 0.0, π], ["-π", "0", "π"]),
    ticklabelsize=8pt,
)

# =============== generate fat fraction maps =======================

colmap = :imola

ax = Axis(fig[2, 1],
    title=L"$f_{RW}$",
)

heatmap!(ax,
    f_RW',
    colormap=colmap,
    nan_color=:black,
)
hidedecorations!(ax)
Label(fig[2, 1, TopLeft()], "C",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

ax = Axis(fig[2, 2],
    title=L"$f_{FC}$",
)

heatmap!(ax,
    f_FC',
    colormap=colmap,
    nan_color=:black,
)

hidedecorations!(ax)
Label(fig[2, 2, TopLeft()], "D",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

Colorbar(fig[2, 3],
    colorrange=(0, 1),
    colormap=colmap,
    ticklabelsize=8pt,
)

display(fig)

##

fig_name = "fig_5"
save(fig_name * ".svg", fig)
run(`/home/cganter/bin/svg2eps $fig_name`)
run(`epspdf $fig_name".eps"`)