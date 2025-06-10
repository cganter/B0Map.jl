include("vp_util.jl")

# =============== MC simulation =======================

B0 = 1.5
ppm_fat = [-3.80, -3.40, -2.60, -1.95, -0.5, 0.60]
ampl_fat = [0.0875, 0.6998, 0.1206, 0.0062, 0.0389, 0.0471]

rng = MersenneTwister(42)
# echo times for data set 14
TEs = [1.352, 3.656, 5.96, 8.264]
c_t = randn(rng, ComplexF64)
c_t /= abs(c_t)
σ = 0.0
n_σ = 1
ϕ_t = 0
R2ss_t = [range(0, 1, 26);]
fs_t = [range(0, 1, 21);]
Δppms_t = [range(-5, 5, 31);] #[range(-0.1, 0.2, 31);]

res_MC_Δppm = MC_sim_LocFit_Δppm(;
    rng=rng,
    TEs=TEs,
    B0=B0,
    ppm_fat=ppm_fat,
    ampl_fat=ampl_fat,
    ϕ_t=ϕ_t,
    R2ss_t=R2ss_t,
    fs_t=fs_t,
    Δppms_t=Δppms_t,
    c_t=c_t,
    σ=σ,
)

# size definitions
# these are relative to 1 CSS px
inch = 96
pt = 4 / 3

##

# =============== generate fat fraction maps =======================

width, height = 6.9inch, 2.5inch
colmap = :batlowW

f_f = Figure(size=(width, height), fontsize=10pt)

f_FC = res_MC.f[:FC]
f_RW = res_MC.f[:RW]
la_Δf = log.(abs.(f_RW .- f_FC) .+ eps())

ax = Axis(f_f[1, 1],
    title=L"$f_{RW}$",
    titlesize=12pt,
    xlabel=L"$\varphi$ [rad]",
    ylabel=L"$R_2^\ast$ \;[1/ms$\,$]",
    yticklabelsize=8pt,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    xticklabelsize=8pt,
)

heatmap!(ax, ϕs, R2ss, f_RW, colormap=colmap, colorrange=(0, 1))
scatter!(ax, 0.0, R2ss_t, color=:lime)
Label(f_f[1, 1, BottomLeft()], "A",
    font=:bold,
    padding=(0, -10, -30, 0),
    halign=:right)

ax = Axis(f_f[1, 2],
    title=L"$f_{FC}$",
    titlesize=12pt,
    xlabel=L"$\varphi$ [rad]",
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    xticklabelsize=8pt,
)

heatmap!(ax, ϕs, R2ss, f_FC, colormap=colmap, colorrange=(0, 1))
scatter!(ax, 0.0, R2ss_t, color=:lime)
hideydecorations!(ax)
Label(f_f[1, 2, BottomLeft()], "B",
    font=:bold,
    padding=(0, -10, -30, 0),
    halign=:right)

Colorbar(f_f[1, 3],
    colorrange=(0, 1),
    colormap=colmap,
    ticklabelsize=8pt
)

ax = Axis(f_f[1, 4],
    title=L"$\log_{10}\,|\,f_{RW} \,- \,f_{FC}\,|$",
    titlesize=12pt,
    xlabel=L"$\varphi$ [rad]",
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    xticklabelsize=8pt,
)

lim = (min(la_Δf...), max(la_Δf...))
heatmap!(ax, ϕs, R2ss, la_Δf, colormap=colmap, colorrange=lim)
scatter!(ax, 0.0, R2ss_t, color=:lime)
hideydecorations!(ax)
Label(f_f[1, 4, BottomLeft()], "C",
    font=:bold,
    padding=(0, -10, -30, 0),
    halign=:right)

Colorbar(f_f[1, 5],
    colorrange=lim,
    colormap=colmap,
    ticklabelsize=8pt,
)

display(f_f)

# =============== generate χ² maps =======================

width, height = 6.9inch, 2.5inch
colmap = :batlowW

f_χ2 = Figure(size=(width, height), fontsize=10pt)

χ2_FC = res_MC.χ2[:FC]
χ2_RW = res_MC.χ2[:RW]
laχ2_FC = log10.(abs.(χ2_FC) .+ eps())
laχ2_RW = log10.(abs.(χ2_RW) .+ eps())
laχ2_RW_FC = log10.(abs.(χ2_RW .- χ2_FC) .+ eps())

lim = (0.6min(laχ2_FC..., laχ2_RW...), 0.8max(laχ2_FC..., laχ2_RW...))

ax = Axis(f_χ2[1, 1],
    title=L"$\log_{10}\,\chi^2_{RW}$",
    titlesize=12pt,
    xlabel=L"$\varphi$ [rad]",
    ylabel=L"$R_2^\ast$ \;[1/ms$\,$]",
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    xticklabelsize=8pt,
    yticklabelsize=8pt,
)

heatmap!(ax, ϕs, R2ss, laχ2_RW, colormap=colmap, colorrange=lim)
scatter!(ax, 0.0, R2ss_t, color=:lime)
Label(f_χ2[1, 1, BottomLeft()], "A",
    font=:bold,
    padding=(0, -10, -30, 0),
    halign=:right)

ax = Axis(f_χ2[1, 2],
    title=L"$\log_{10}\,\chi^2_{FC}$",
    titlesize=12pt,
    xlabel=L"$\varphi$ [rad]",
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    xticklabelsize=8pt,
)

#heatmap!(ax, ϕs, R2ss, χ2_FC, colormap=colmap, colorrange=lim)
heatmap!(ax, ϕs, R2ss, laχ2_FC, colormap=colmap, colorrange=lim)
scatter!(ax, 0.0, R2ss_t, color=:lime)
hideydecorations!(ax)
Label(f_χ2[1, 2, BottomLeft()], "B",
    font=:bold,
    padding=(0, -10, -30, 0),
    halign=:right)

Colorbar(f_χ2[1, 3],
    colorrange=lim,
    colormap=colmap,
    ticklabelsize=8pt,
)

ax = Axis(f_χ2[1, 4],
    title=L"$\log_{10}\,(\,\chi^2_{FC} \,-\, \chi^2_{RW}\,)$",
    titlesize=12pt,
    xlabel=L"$\varphi$ [rad]",
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
    xticklabelsize=8pt,
)

lim = (min(laχ2_RW_FC...), max(laχ2_RW_FC...))
heatmap!(ax, ϕs, R2ss, laχ2_RW_FC, colormap=colmap, colorrange=lim)
scatter!(ax, 0.0, R2ss_t, color=:lime)
hideydecorations!(ax)
Label(f_χ2[1, 4, BottomLeft()], "C",
    font=:bold,
    padding=(0, -10, -30, 0),
    halign=:right)

Colorbar(f_χ2[1, 5],
    colorrange=lim,
    colormap=colmap,
    ticklabelsize=8pt,
   )

display(f_χ2)

##

fig_name = "fig_1"
save(fig_name * ".svg", f_χ2)
run(`/home/cganter/bin/svg2eps $fig_name`)
run(`epspdf $fig_name".eps"`)

fig_name = "fig_2"
save(fig_name * ".svg", f_f)
run(`/home/cganter/bin/svg2eps $fig_name`)
run(`epspdf $fig_name".eps"`)
