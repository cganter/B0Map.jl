include("vp_util.jl")

# =============== generate χ² maps =======================

B0 = 3
ppm_fat = [-3.80, -3.40, -2.60, -1.95, -0.5, 0.60]
ampl_fat = [0.0875, 0.6998, 0.1206, 0.0062, 0.0389, 0.0471]

# approximate opposed-phase echo time
Δt_opp = (3.0 / B0) * 1.12

rng = MersenneTwister(42)
nTE = 2
t0 = 1.02 # 0.9Δt_opp # 1.02
ΔTE = 1.07 # 0.9Δt_opp # 1.07
TEs = t0 .+ [ΔTE .* (0:nTE-1);]
c_t = randn(rng, ComplexF64)
c_t /= abs(c_t)
ϕ_t = 0
R2ss_t = [range(0, 1, 101);]
fs_t = [range(0, 1, 101);]
σ = 0.1
n_σ = 1

res_MC = MC_sim_LocFit(;
    rng=rng,
    TEs=TEs,
    B0=B0,
    ppm_fat=ppm_fat,
    ampl_fat=ampl_fat,
    ϕ_t=ϕ_t,
    R2ss_t=R2ss_t,
    fs_t=fs_t,
    c_t=c_t,
    σ=σ,
    n_σ=n_σ,
)

##

width, height = 1000, 400

f_f = Figure(size=(width, height))

f_FC = res_MC.f[:FC]
f_RW = res_MC.f[:RW]
la_Δf = log.(abs.(f_RW .- f_FC) .+ eps())

ax = Axis(f_f[1, 1],
    title=L"$f_{RW}$",
    titlesize=20,
    xlabel=L"$\varphi$ [rad]",
    xlabelsize=20,
    ylabel=L"$R_2^\ast$ \;[1/ms$\,$]",
    ylabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

heatmap!(ax, ϕs, R2ss, f_RW, colormap=:roma, colorrange=(0, 1))
scatter!(ax, 0.0, R2ss_t, color=:yellow)
Label(f_f[1, 1, TopLeft()], "A",
    fontsize=20,
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

ax = Axis(f_f[1, 2],
    title=L"$f_{FC}$",
    titlesize=20,
    xlabel=L"$\varphi$ [rad]",
    xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

heatmap!(ax, ϕs, R2ss, f_FC, colormap=:roma, colorrange=(0, 1))
scatter!(ax, 0.0, R2ss_t, color=:yellow)
hideydecorations!(ax)
Label(f_f[1, 2, TopLeft()], "B",
    fontsize=20,
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

Colorbar(f_f[1, 3],
    colorrange=(0, 1),
    colormap=:roma)

ax = Axis(f_f[1, 4],
    title=L"$\log_{10}\,|\,f_{RW} \,- \,f_{FC}\,|$",
    titlesize=20,
    xlabel=L"$\varphi$ [rad]",
    xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

lim = (min(la_Δf...), max(la_Δf...))
heatmap!(ax, ϕs, R2ss, la_Δf, colormap=:roma, colorrange=lim)
scatter!(ax, 0.0, R2ss_t, color=:yellow)
hideydecorations!(ax)
Label(f_f[1, 4, TopLeft()], "C",
    fontsize=20,
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

Colorbar(f_f[1, 5],
    colorrange=lim,
    colormap=:roma)

display(f_f)

# ==========================================================================

width, height = 1000, 400

f_χ2 = Figure(size=(width, height))

χ2_FC = res_MC.χ2[:FC]
χ2_RW = res_MC.χ2[:RW]
laχ2_FC = log10.(abs.(χ2_FC) .+ eps())
laχ2_RW = log10.(abs.(χ2_RW) .+ eps())
laχ2_RW_FC = log10.(abs.(χ2_RW .- χ2_FC) .+ eps())


#lim = (min(laχ2_FC..., laχ2_RW...), max(laχ2_FC..., laχ2_RW...))
lim = (0, max(χ2_FC..., χ2_RW...))

ax = Axis(f_χ2[1, 1],
    #title=L"$\log_{10}\,\chi^2_{RW}$",
    title=L"$\chi^2_{RW}$",
    titlesize=20,
    xlabel=L"$\varphi$ [rad]",
    xlabelsize=20,
    ylabel=L"$R_2^\ast$ \;[1/ms$\,$]",
    ylabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

#heatmap!(ax, ϕs, R2ss, laχ2_RW, colormap=:roma, colorrange=lim)
heatmap!(ax, ϕs, R2ss, χ2_RW, colormap=:roma, colorrange=lim)
scatter!(ax, 0.0, R2ss_t, color=:yellow)
Label(f_χ2[1, 1, TopLeft()], "A",
    fontsize=20,
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

ax = Axis(f_χ2[1, 2],
    title=L"$\chi^2_{FC}$",
    titlesize=20,
    xlabel=L"$\varphi$ [rad]",
    xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

#heatmap!(ax, ϕs, R2ss, laχ2_FC, colormap=:roma, colorrange=lim)
heatmap!(ax, ϕs, R2ss, χ2_FC, colormap=:roma, colorrange=lim)
scatter!(ax, 0.0, R2ss_t, color=:yellow)
hideydecorations!(ax)
Label(f_χ2[1, 2, TopLeft()], "B",
    fontsize=20,
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

Colorbar(f_χ2[1, 3],
    colorrange=lim,
    colormap=:roma)

ax = Axis(f_χ2[1, 4],
    title=L"$\log_{10}\,(\,\chi^2_{FC} \,-\, \chi^2_{RW}\,)$",
    titlesize=20,
    xlabel=L"$\varphi$ [rad]",
    xlabelsize=20,
    xticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

lim = (min(laχ2_RW_FC...), max(laχ2_RW_FC...))
heatmap!(ax, ϕs, R2ss, laχ2_RW_FC, colormap=:roma, colorrange=lim)
scatter!(ax, 0.0, R2ss_t, color=:yellow)
hideydecorations!(ax)
Label(f_χ2[1, 4, TopLeft()], "C",
    fontsize=20,
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

Colorbar(f_χ2[1, 5],
    colorrange=lim,
    colormap=:roma)

display(f_χ2)

##

save("fig_1.svg", f_χ2)
run(`/home/cganter/bin/svg2eps fig_1`)

save("fig_2.svg", f_f)
run(`/home/cganter/bin/svg2eps fig_2`)
