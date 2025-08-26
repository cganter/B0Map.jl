using MAT, CairoMakie, LinearAlgebra
import VP4Optim as VP
import B0Map as BM

include("vp_util.jl")

BLAS.set_num_threads(1)


# ISMRM challenge 2012 data sets:

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

data_set = 17
sl = 4

# local fit
fitopt_FW = BM.fitOpt()
@time res_FW = ismrm_challenge(BM.GREMultiEchoWFFW, fitopt_FW; data_set = data_set);
fitopt_RW = BM.fitOpt()
@time res_RW = ismrm_challenge(BM.GREMultiEchoWFRW, fitopt_RW; data_set = data_set);
fitopt_FC = BM.fitOpt()
@time res_FC = ismrm_challenge(BM.GREMultiEchoWF, fitopt_FC; data_set = data_set);

not_S = (!).(res_FW.fitpar.S[:,:,sl])

# extract maps
ϕ_FW = res_FW.fitpar.ϕ[:,:,sl]
ϕ_FW[not_S] .= NaN
f_FW = BM.fat_fraction_map(res_FW.fitpar, fitopt_FW)[:,:,sl]
f_FW[not_S] .= NaN

ϕ_RW = res_RW.fitpar.ϕ[:,:,sl]
ϕ_RW[not_S] .= NaN
f_RW = BM.fat_fraction_map(res_RW.fitpar, fitopt_RW)[:,:,sl]
f_RW[not_S] .= NaN

ϕ_FC = res_FC.fitpar.ϕ[:,:,sl]
ϕ_FC[not_S] .= NaN
f_FC = BM.fat_fraction_map(res_FC.fitpar, fitopt_FC)[:,:,sl]
f_FC[not_S] .= NaN

# =============== generate phase maps =======================

# size definitions
# these are relative to 1 CSS px
inch = 96
pt = 4 / 3

width, height = 6.9inch, 4.6inch
colmapO = :romaO

fig = Figure(size=(width, height), fontsize=12pt)

ax = Axis(fig[1, 1],
    title=L"$\varphi_{FW}$",
)

heatmap!(ax,
    rot180(ϕ_FW),
    colormap=colmapO,
    nan_color=:black,
)
hidedecorations!(ax)
Label(fig[1, 1, TopLeft()], "A",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

ax = Axis(fig[1, 2],
    title=L"$\varphi_{RW}$",
)

heatmap!(ax,
    rot180(ϕ_RW),
    colormap=colmapO,
    nan_color=:black,
)
hidedecorations!(ax)
Label(fig[1, 2, TopLeft()], "B",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

ax = Axis(fig[1, 3],
    title=L"$\varphi_{FC}$",
)

heatmap!(ax,
    rot180(ϕ_FC),
    colormap=colmapO,
    nan_color=:black,
)
hidedecorations!(ax)
Label(fig[1, 3, TopLeft()], "C",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

Colorbar(fig[1, 4],
    colorrange=(-π, π),
    colormap=colmapO,
    ticks=([-π, 0.0, π], ["-π", "0", "π"]),
    ticklabelsize=8pt,
)

# =============== generate fat fraction maps =======================

colmap = :imola

ax = Axis(fig[2, 1],
    title=L"$f_{FW}$",
)

heatmap!(ax,
    rot180(f_FW),
    colormap=colmap,
    nan_color=:black,
)
hidedecorations!(ax)
Label(fig[2, 1, TopLeft()], "D",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

ax = Axis(fig[2, 2],
    title=L"$f_{RW}$",
)

heatmap!(ax,
    rot180(f_RW),
    colormap=colmap,
    nan_color=:black,
)
hidedecorations!(ax)
Label(fig[2, 2, TopLeft()], "E",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

ax = Axis(fig[2, 3],
    title=L"$f_{FC}$",
)

heatmap!(ax,
    rot180(f_FC),
    colormap=colmap,
    nan_color=:black,
)

hidedecorations!(ax)
Label(fig[2, 3, TopLeft()], "F",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

Colorbar(fig[2, 4],
    colorrange=(0, 1),
    colormap=colmap,
    ticklabelsize=8pt,
)


display(fig)

##

fig_name = "fig_3"
save(fig_name * ".svg", fig)
run(`/home/cganter/bin/svg2eps $fig_name`)
run(`epspdf $fig_name".eps"`)