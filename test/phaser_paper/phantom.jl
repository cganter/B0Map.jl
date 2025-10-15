using CairoMakie, Random, StatsBase
import VP4Optim as VP
import B0Map as BM

include("../test_tools.jl")
include("ph_util.jl")

spp = SimPhaPar()

spp.TEs = 1.15 * [1, 2, 3] #, 4]
#spp.TEs = 1.15 * [1, 2, 3, 4]
spp.B0 = 1.5
spp.freq_rng = [-2, 2]
spp.Nρ = [256, 256]
spp.K = [2, 2]
#spp.rng = MersenneTwister()
spp.locfit = false
spp.S_holes = 0.5
spp.S_io = :out
spp.cov_mat = 0.05^2 * [1;;]
spp.subsampling = :fibonacci
spp.balance = true
spp.remove_gradient_outliers = true
spp.remove_local_outliers = true
spp.add_noise = true
spp.os_fac = [1.3]
spp.ppm_fat_pha = spp.ppm_fat .- 0.1

spp.rng = MersenneTwister(42)
spp.redundancy = 100
sim = simulate_phantom(spp);

##

include("ph_util.jl")
(fig, _) = phaser_workflow!(sim.PH)
display(fig)

## phases 

PH = sim.PH

width = 1000
height = 900
cm = :roma
cm2π = :romaO

pt = 4 / 3
fig = Figure(size=(width, height), fontsize=12pt)

f_2π = ϕ -> mod.(ϕ .+ π, 2π) .- π

Sp = sim.phantom.S
S = PH.S
S_wo = PH.S_wo

ϕ_ML = PH.ϕ_ML
ϕ0 = PH.ϕ0
ϕ1 = PH.ϕ1
ϕ1_wo = PH.ϕ1_wo
Δϕ0 = PH.Δϕ0
Δϕ1 = PH.Δϕ1
u_wo = PH.u_wo[1]
u = PH.u[1]

dax = Dict()

#dax[:Δϕ] = Axis(fig[2, 2])

#cr = (min(ϕt[S_wo]...), max(ϕt[S_wo]...))

j = 1

dax[:ϕ_ML] = Axis(fig[1, 1])
show_map!(dax[:ϕ_ML], ϕ_ML, L"$\Phi$", S_wo; cm=:roma, cr=(-π,π))

dax[:u_wo] = Axis(fig[1, 2])
show_map!(dax[:u_wo], u_wo, L"$\nabla_%$j\,\Phi$\quad(\ast)", PH.Sj_wo[1]; cm=:roma)

dax[:au_hist] = Axis(fig[1, 3])
show_hist_phase_grad!(dax[:au_hist], PH, L"abs\,$(\nabla_%$j\,\Phi)$\quad(\ast)")

dax[:u] = Axis(fig[1, 4])#, title = L"$S_1$ (without outliers)")
show_map!(dax[:u], u, L"$\nabla_%$j\,\Phi$", PH.Sj[1]; cm=:roma)

dax[:ϕ0] = Axis(fig[2, 1])
show_map!(dax[:ϕ0], f_2π(ϕ_ML - ϕ0), L"$\Phi - \varphi_{0}$", S; cm=:roma, cr=(-π,π))

dax[:Φ_ϕ0] = Axis(fig[2, 2], title=L"$\Phi - \varphi_0$ \quad(\ast)")
hist!(dax[:Φ_ϕ0], f_2π(ϕ_ML - ϕ0)[S_wo], bins=100, color=(:red, 0.2), scale_to=1)
hist!(dax[:Φ_ϕ0], f_2π(ϕ_ML - ϕ0)[S], bins=100, color=:blue, scale_to=1)

dax[:λ_opt_1] = Axis(fig[2, 3],
    title=L"$\langle \chi^{2}\rangle$ (\ast)",
    xlabel=L"$\lambda$",
    xticklabelsize=8pt,
    yticklabelsize=8pt,
)

lines!(dax[:λ_opt_1], PH.λs_1, PH.χ2s_1 ./ PH.χ2s_1[1], color=:red)
scatter!(dax[:λ_opt_1], PH.λs_1, PH.χ2s_1 ./ PH.χ2s_1[1], color=:blue)
hideydecorations!(dax[:λ_opt_1])

dax[:Δϕ1_hist] = Axis(fig[2, 4])
show_hist_Δϕ1!(dax[:Δϕ1_hist], PH, L"$\Phi - \varphi_1$\quad(\ast)")

dax[:λ_opt_2] = Axis(fig[3, 1],
    title=L"$\langle \chi^{2}\rangle$",
    xlabel=L"$\lambda$",
    xticklabelsize=8pt,
    yticklabelsize=8pt,
)

lines!(dax[:λ_opt_2], PH.λs_2, PH.χ2s_2 ./ PH.χ2s_2[1], color=:red)
scatter!(dax[:λ_opt_2], PH.λs_2, PH.χ2s_2 ./ PH.χ2s_2[1], color=:blue)
hideydecorations!(dax[:λ_opt_2])

dax[:ϕ1] = Axis(fig[3, 2])
show_map!(dax[:ϕ1], f_2π(ϕ_ML - ϕ1), L"$\Phi - \varphi_{1}$", S; cm=:roma, cr=(-π,π))

dax[:Φ_ϕ1] = Axis(fig[3, 3], title=L"$\Phi - \varphi_j$")
hist!(dax[:Φ_ϕ1], f_2π(ϕ_ML - ϕ1)[S_wo], bins=100, color=:blue, label=L"$j=1$", scale_to = 1)
stephist!(dax[:Φ_ϕ1], f_2π(ϕ_ML - ϕ0)[S_wo], bins=100, color= :red, label=L"$j=0$", scale_to = 1)
axislegend(dax[:Φ_ϕ1], merge=true, unique=true, labelsize=10pt, position = :lt)
#dax[:ϕt] = Axis(fig[3, 3])
#show_map!(dax[:ϕt], f_2π(ϕt), L"$\varphi_{true}$", Sp; cm=:romaO, cr=(-π,π))

dax[:δϕ] = Axis(fig[3, 4])
dax[:δϕ].title = L"$\varphi_j - \varphi_{true}$"

stephist!(dax[:δϕ], δϕ0[Sp], color=:red, bins=100, label=L"$j = 0$")
stephist!(dax[:δϕ], δϕ1[Sp], color=:blue, bins=100, label=L"$j = 1$")
hideydecorations!(dax[:δϕ])
axislegend(dax[:δϕ], merge=true, unique=true, labelsize=10pt, position = :ct)
#fig[3,4] = Legend(fig, dax[:δϕ])

display(fig);


## phases 

PH = sim.PH

width = 1000
height = 900
cm = :roma
cm2π = :romaO

pt = 4 / 3
fig = Figure(size=(width, height), fontsize=12pt)

f_2π = ϕ -> mod.(ϕ .+ π, 2π) .- π

Sp = sim.phantom.S
S = PH.S
S_wo = PH.S_wo

ϕ_ML = PH.ϕ_ML
ϕ0 = PH.ϕ0
ϕ1 = PH.ϕ1
ϕ1_wo = PH.ϕ1_wo
Δϕ0 = PH.Δϕ0
Δϕ1 = PH.Δϕ1
ϕt = sim.phantom.ϕ
δϕ0 = angle.(exp.(im .* (ϕ0 .- ϕt)))
δϕ1 = angle.(exp.(im .* (ϕ1 .- ϕt)))
u_wo = PH.u_wo[1]
u = PH.u[1]

dax = Dict()

#dax[:Δϕ] = Axis(fig[2, 2])

#cr = (min(ϕt[S_wo]...), max(ϕt[S_wo]...))

j = 1

dax[:ϕ_ML] = Axis(fig[1, 1])
show_map!(dax[:ϕ_ML], ϕ_ML, L"$\Phi$", S_wo; cm=:roma, cr=(-π,π))

dax[:u_wo] = Axis(fig[1, 2])
show_map!(dax[:u_wo], u_wo, L"$\nabla_%$j\,\Phi$\quad(\ast)", PH.Sj_wo[1]; cm=:roma)

dax[:au_hist] = Axis(fig[1, 3])
show_hist_phase_grad!(dax[:au_hist], PH, L"abs\,$(\nabla_%$j\,\Phi)$\quad(\ast)")

dax[:u] = Axis(fig[1, 4])#, title = L"$S_1$ (without outliers)")
show_map!(dax[:u], u, L"$\nabla_%$j\,\Phi$", PH.Sj[1]; cm=:roma)

dax[:ϕ0] = Axis(fig[2, 1])
show_map!(dax[:ϕ0], f_2π(ϕ_ML - ϕ0), L"$\Phi - \varphi_{0}$", S; cm=:roma, cr=(-π,π))

dax[:Φ_ϕ0] = Axis(fig[2, 2], title=L"$\Phi - \varphi_0$ \quad(\ast)")
hist!(dax[:Φ_ϕ0], f_2π(ϕ_ML - ϕ0)[S_wo], bins=100, color=(:red, 0.2), scale_to=1)
hist!(dax[:Φ_ϕ0], f_2π(ϕ_ML - ϕ0)[S], bins=100, color=:blue, scale_to=1)

dax[:λ_opt_1] = Axis(fig[2, 3],
    title=L"$\langle \chi^{2}\rangle$ (\ast)",
    xlabel=L"$\lambda$",
    xticklabelsize=8pt,
    yticklabelsize=8pt,
)

lines!(dax[:λ_opt_1], PH.λs_1, PH.χ2s_1 ./ PH.χ2s_1[1], color=:red)
scatter!(dax[:λ_opt_1], PH.λs_1, PH.χ2s_1 ./ PH.χ2s_1[1], color=:blue)
hideydecorations!(dax[:λ_opt_1])

dax[:Δϕ1_hist] = Axis(fig[2, 4])
show_hist_Δϕ1!(dax[:Δϕ1_hist], PH, L"$\Phi - \varphi_1$\quad(\ast)")

dax[:λ_opt_2] = Axis(fig[3, 1],
    title=L"$\langle \chi^{2}\rangle$",
    xlabel=L"$\lambda$",
    xticklabelsize=8pt,
    yticklabelsize=8pt,
)

lines!(dax[:λ_opt_2], PH.λs_2, PH.χ2s_2 ./ PH.χ2s_2[1], color=:red)
scatter!(dax[:λ_opt_2], PH.λs_2, PH.χ2s_2 ./ PH.χ2s_2[1], color=:blue)
hideydecorations!(dax[:λ_opt_2])

dax[:ϕ1] = Axis(fig[3, 2])
show_map!(dax[:ϕ1], f_2π(ϕ_ML - ϕ1), L"$\Phi - \varphi_{1}$", S; cm=:roma, cr=(-π,π))

dax[:Φ_ϕ1] = Axis(fig[3, 3], title=L"$\Phi - \varphi_j$")
hist!(dax[:Φ_ϕ1], f_2π(ϕ_ML - ϕ1)[S_wo], bins=100, color=:blue, label=L"$j=1$", scale_to = 1)
stephist!(dax[:Φ_ϕ1], f_2π(ϕ_ML - ϕ0)[S_wo], bins=100, color= :red, label=L"$j=0$", scale_to = 1)
axislegend(dax[:Φ_ϕ1], merge=true, unique=true, labelsize=10pt, position = :lt)
#dax[:ϕt] = Axis(fig[3, 3])
#show_map!(dax[:ϕt], f_2π(ϕt), L"$\varphi_{true}$", Sp; cm=:romaO, cr=(-π,π))

dax[:δϕ] = Axis(fig[3, 4])
dax[:δϕ].title = L"$\varphi_j - \varphi_{true}$"

stephist!(dax[:δϕ], δϕ0[Sp], color=:red, bins=100, label=L"$j = 0$")
stephist!(dax[:δϕ], δϕ1[Sp], color=:blue, bins=100, label=L"$j = 1$")
hideydecorations!(dax[:δϕ])
axislegend(dax[:δϕ], merge=true, unique=true, labelsize=10pt, position = :ct)
#fig[3,4] = Legend(fig, dax[:δϕ])

display(fig);


#=
aumax = sim.PH.au_max[1]
vau = [sim.PH.au_hist[1].edges...;]
aus = 0.5(vau[1:end-1] + vau[2:end])
haus = sim.PH.au_hist[1].weights
iau = findlast(au -> au < aumax, aus)
lines!(dax[:au_hist], aus[1:iau], haus[1:iau], color = :blue)
scatter!(dax[:au_hist], aus[1:iau], haus[1:iau], color = :blue)
lines!(dax[:au_hist], aus[iau+1:end], haus[iau+1:end], color = :red)
scatter!(dax[:au_hist], aus[iau+1:end], haus[iau+1:end], color = :red)
hideydecorations!(dax[:au_hist])
=#


#=
Δϕ1_min = sim.PH.Δϕ1_min
Δϕ1_max = sim.PH.Δϕ1_max
vΔϕ1 = [sim.PH.Δϕ1_hist.edges...;]
Δϕ1s = 0.5(vΔϕ1[1:end-1] + vΔϕ1[2:end])
hΔϕ1s = sim.PH.Δϕ1_hist.weights
i1 = findfirst(x -> x > Δϕ1_min, Δϕ1s)
i2 = findlast(x -> x < Δϕ1_max, Δϕ1s[i1:end]) + i1
lines!(dax[:Δϕ1_hist], Δϕ1s[i1:i2], hΔϕ1s[i1:i2], color = :blue)
scatter!(dax[:Δϕ1_hist], Δϕ1s[i1:i2], hΔϕ1s[i1:i2], color = :blue)
lines!(dax[:Δϕ1_hist], Δϕ1s[1:i1-1], hΔϕ1s[1:i1-1], color = :red)
scatter!(dax[:Δϕ1_hist], Δϕ1s[1:i1-1], hΔϕ1s[1:i1-1], color = :red)
lines!(dax[:Δϕ1_hist], Δϕ1s[i2+1:end], hΔϕ1s[i2+1:end], color = :red)
scatter!(dax[:Δϕ1_hist], Δϕ1s[i2+1:end], hΔϕ1s[i2+1:end], color = :red)
hideydecorations!(dax[:Δϕ1_hist])
=#


#heatmap!(dax[:u_wo], sim.PH.Sj_wo[1], colormap=:grayC)
#hidedecorations!(dax[:u_wo])
#heatmap!(dax[:u], sim.PH.Sj[1], colormap=:grayC)
#hidedecorations!(dax[:u])

#dax[:δϕ].title = L"$\varphi_j - \varphi_t$"

#stephist!(dax[:δϕ], δϕ0[Sp], color=:red, label=L"$\varphi_0$")
#stephist!(dax[:δϕ], δϕ1[Sp], color=:blue, label=L"$\varphi_1$")
#dax[:Δϕ].title = L"$\Delta\varphi_j$"
#fig[2,4] = Legend(fig, dax[:δϕ])
#stephist!(dax[:Δϕ], Δϕ0[sim.PH.S], color=:red)
#stephist!(dax[:Δϕ], Δϕ1[sim.PH.S], color=:blue)
#hideydecorations!(dax[:Δϕ])
#hideydecorations!(dax[:δϕ])
#show_map!(dax[:Δϕ0], δϕ0, L"$\Delta\varphi_{0}$", Sp; cm=cm2π, cr=cr2π)
#show_map!(dax[:Δϕ1], δϕ1, L"$\Delta\varphi_{1}$", Sp; cm=cm2π, cr=cr2π)


##
#=
show_map!(dax[:ϕ0], sim.PH.ϕ0, L"$\varphi_{0}$", Sϕ; cm=:roma, cr=cr)
show_map!(dax[:ϕ1], sim.PH.ϕ1, L"$\varphi_{1}$", Sϕ; cm=:roma, cr=cr)
show_map!(dax[:ϕt], sim.phantom.ϕ, L"$\varphi_{t}$", Sϕ; cm=:roma, cr=cr)
show_map!(dax[:Δϕ0], sim.PH.Δϕ0, L"$\Delta\varphi_{0}$", Sϕ; cm=:roma, cr=cr)
show_map!(dax[:Δϕ1], sim.PH.Δϕ1, L"$\Delta\varphi_{1}$", Sϕ; cm=:roma, cr=cr)
=#


width = 800
height = 800
cm_phase = :roma
cm_fat = :imola

pt = 4 / 3
fig = Figure(size=(width, height), fontsize=12pt)


S = sim.fitpar.S
noS = (!).(S)
S_PH = sim.PH.S_wo
noS_PH = (!).(S_PH)

ϕ_ML = sim.PH.ϕ_ML
R2s_ML = sim.PH.R2s_ML
R2s_ML[noS_PH] .= NaN
ϕ0_PH = sim.PH.ϕ0
ϕ1_PH = sim.PH.ϕ1

bal_rng = (min(ϕ1_PH[S]..., -π), max(ϕ1_PH[S]..., π))

pdff = BM.fat_fraction_map(sim.fitpar, sim.fitopt)

fitpar_ML = deepcopy(sim.fitpar)
fo_ML = deepcopy(sim.fitopt)

locfit_0 = deepcopy(sim.fitpar)
fo_0 = deepcopy(sim.fitopt)

fitpar_ML.ϕ[:, :] = ϕ_ML
fitpar_ML.R2s[:, :] = R2s_ML
locfit_0.ϕ[:, :] = ϕ0_PH

BM.set_num_phase_intervals(locfit_0, fo_0, 0)
BM.local_fit(locfit_0, fo_0)

fitpar_ML.S[noS_PH] .= false
pdff_ML = BM.fat_fraction_map(fitpar_ML, fo_ML)
pdff_0 = BM.fat_fraction_map(locfit_0, fo_0)
pdff_ref = sim.phantom.f

pdff_ML[noS_PH] .= NaN
pdff_0[noS] .= NaN
pdff[noS] .= NaN
pdff_ref[noS] .= NaN

λs_1 = sim.PH.λs_1
λs_2 = sim.PH.λs_2
χ2s_1 = sim.PH.χ2s_1 ./ sim.PH.sumS_wo
χ2s_2 = sim.PH.χ2s_2 ./ sim.PH.sumS

dax = Dict()

ϕ0_PH_loc = locfit_0.ϕ
ϕ0_PH_loc[noS] .= NaN

ϕ1_PH_loc = sim.fitpar.ϕ
ϕ1_PH_loc[noS] .= NaN

# -------------------------------------------------

dax[:ϕ_ML] = Axis(fig[1, 1],
    title=L"$\Phi$",
)

heatmap!(dax[:ϕ_ML],
    ϕ_ML,
    colormap=cm_phase,
    colorrange=(-π, π),
    nan_color=:black
)

Label(fig[1, 1, TopLeft()], "A",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

dax[:ϕ0] = Axis(fig[1, 2],
    title=L"$\varphi^{(0)}$",
)

heatmap!(dax[:ϕ0],
    ϕ0_PH,
    colormap=cm_phase,
    colorrange=bal_rng,
    nan_color=:black
)

Label(fig[1, 2, TopLeft()], "B",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

dax[:ϕ1] = Axis(fig[1, 3],
    title=L"$\varphi^{(1)}$",
)

heatmap!(dax[:ϕ1],
    ϕ1_PH,
    colormap=cm_phase,
    colorrange=bal_rng,
    nan_color=:black
)

Label(fig[1, 3, TopLeft()], "C",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

dax[:λ_opt] = Axis(fig[2, 1],
    title=L"$\langle \chi^{2}\rangle$",
    xlabel=L"$\lambda$",
    xticklabelsize=8pt,
    yticklabelsize=8pt,
)

lines!(dax[:λ_opt], λs_2, χ2s_2 ./ χ2s_2[1], color=:red)
scatter!(dax[:λ_opt], λs_2, χ2s_2 ./ χ2s_2[1], color=:blue)

Label(fig[2, 1, TopLeft()], "D",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

dax[:ϕ0_loc] = Axis(fig[2, 2],
    title=L"$\Phi\left(\varphi^{(0)}\right)$",
)

heatmap!(dax[:ϕ0_loc],
    ϕ0_PH_loc,
    colormap=cm_phase,
    colorrange=bal_rng,
    nan_color=:black
)

Label(fig[2, 2, TopLeft()], "E",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

dax[:ϕ1_loc] = Axis(fig[2, 3],
    title=L"$\Phi\left(\varphi^{(1)}\right)$",
)

heatmap!(dax[:ϕ1_loc],
    ϕ1_PH_loc,
    colormap=cm_phase,
    colorrange=bal_rng,
    nan_color=:black
)

Label(fig[2, 3, TopLeft()], "F",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

dax[:pdff_ML] = Axis(fig[3, 1],
    title=L"PDFF: $\Phi$",
)

heatmap!(dax[:pdff_ML],
    pdff_ML,
    colormap=cm_fat,
    colorrange=(0, 1),
    nan_color=:black
)

Label(fig[3, 1, TopLeft()], "G",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

dax[:pdff_0] = Axis(fig[3, 2],
    title=L"PDFF: $\Phi\left(φ^{(0)}\right)$",
)

heatmap!(dax[:pdff_0],
    pdff_0,
    colormap=cm_fat,
    colorrange=(0, 1),
    nan_color=:black
)

Label(fig[3, 2, TopLeft()], "H",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

dax[:pdff] = Axis(fig[3, 3],
    title=L"PDFF: $\Phi\left(φ^{(1)}\right)$",
)

heatmap!(dax[:pdff],
    pdff,
    colormap=cm_fat,
    colorrange=(0, 1),
    nan_color=:black
)

Label(fig[3, 3, TopLeft()], "I",
    font=:bold,
    padding=(0, -20, 5, 0),
    halign=:right)

# -------------------------------------------------

Colorbar(fig[1, 4],
    colorrange=bal_rng,
    colormap=cm_phase,
    ticklabelsize=8pt,
)

Colorbar(fig[2, 4],
    colorrange=bal_rng,
    colormap=cm_phase,
    ticklabelsize=8pt,
)

Colorbar(fig[3, 4],
    colorrange=(0, 1),
    colormap=cm_fat,
    ticklabelsize=8pt,
)

for a in (:ϕ_ML, :ϕ0, :ϕ1, :ϕ0_loc, :ϕ1_loc, :pdff_ML, :pdff_0, :pdff)
    hidedecorations!(dax[a])
end
hideydecorations!(dax[:λ_opt])

display(fig)

## ================================================================================


function mod_phase(ϕ)
    while ϕ > π
        ϕ -= 2π
    end

    while ϕ <= -π
        ϕ += 2π
    end

    ϕ
end

sl = ceil(Int, 0.5 * size(sim.phantom.S, 3))

S_sl = sim.PH.S[:, :, sl]
ciS_sl = CartesianIndices(S_sl)[S_sl]

S_sl_wo = sim_wo.PH.S[:, :, sl]
ciS_sl_wo = CartesianIndices(S_sl_wo)[S_sl_wo]

width, height = 1000, 600

fig = Figure(size=(width, height))

# -------------------------------------------------

mima = (min(sim.phantom.ϕ[sim.PH.S]...), max(sim.phantom.ϕ[sim.PH.S]...))

ax = Axis(fig[1, 1],
    title=L"$\varphi$ (true)",
)
hidedecorations!(ax)

heatmap!(ax,
    sim.phantom.ϕ[:, :, sl],
    #mod_phase.(sim.phantom.ϕ[:, :, sl]),
    colormap=:batlow,
    colorrange=mima,
    #colorrange=(-π, π),
    nan_color=:black,
)

Colorbar(fig[1, 2],
    #colorrange=(-π, π),
    colorrange=mima,
    colormap=:batlow,
    #ticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

#=
Colorbar(fig[1, 2],
    colorrange=(-π, π),
    colormap=:batlow,
)
=#

# -------------------------------------------------

ax = Axis(fig[1, 3],
    title=L"$\varphi$ (ML)",
)
hidedecorations!(ax)

heatmap!(ax,
    sim.PH.ϕ_ML[:, :, sl],
    colormap=:batlow,
    colorrange=(-π, π),
    nan_color=:black,
)

Colorbar(fig[1, 4],
    colorrange=(-π, π),
    colormap=:batlow,
    ticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

# -------------------------------------------------

ax = Axis(fig[1, 5],
    title=L"$\varphi$ (ML) - $\varphi$ (true)",
)


bins = ceil(Int, (2.0 * sum(S_sl))^(1 / 3))

hist!(ax,
    mod_phase.(sim.PH.ϕ_ML[S_sl] .- sim.phantom.ϕ[S_sl]),
    bins=100,
)

stephist!(ax, sim_wo.PH.ϕ[ciS_sl_wo, sl] .- sim_wo.phantom.ϕ[ciS_sl_wo, sl],
    bins=20,
    color=:red,
)

stephist!(ax, sim.PH.ϕ[ciS_sl, sl] .- sim.phantom.ϕ[ciS_sl, sl],
    bins=20,
    color=:blue,
)

# -------------------------------------------------

ax = Axis(fig[2, 1],
    title=L"$\varphi$ (PHASER)",
)
hidedecorations!(ax)

heatmap!(ax,
    sim.PH.ϕ[:, :, sl],
    #mod_phase.(sim.PH.ϕ[:, :, sl]),
    colormap=:batlow,
    #colorrange=(-π, π),
    colorrange=mima,
    nan_color=:black,
)


Colorbar(fig[2, 2],
    #colorrange=(-π, π),
    colorrange=mima,
    colormap=:batlow,
    #ticks=([-π, 0.0, π], ["-π", "0", "π"]),
)

#=
Colorbar(fig[2, 2],
    colorrange=mima,
    colormap=:batlow,
)
=#

# -------------------------------------------------

ax = Axis(fig[2, 3],
    title=L"$Δ\varphi$ (PHASER - true)",
)

#=
heatmap!(ax, sim.PH.ϕ[:, :, end÷2+1] .- sim.phantom.ϕ[:, :, end÷2+1],
    colormap=:batlow,
    colorrange=(-0.5π, 0.5π),
    nan_color=:black,
)
=#

bins = ceil(Int, (2sum(S_sl))^(1 / 3))

stephist!(ax, sim.PH.ϕ[ciS_sl, sl] .- sim.phantom.ϕ[ciS_sl, sl],
    color=:blue,
    bins=bins ÷ 2,
)

stephist!(ax, sim_wo.PH.ϕ[ciS_sl_wo, sl] .- sim.phantom.ϕ[ciS_sl_wo, sl],
    color=:red,
    bins=bins,
)

# -------------------------------------------------

ax = Axis(fig[2, 5],
    title=L"$u_1$",
)

#=
heatmap!(ax, sim.PH.ϕ[:, :, end÷2+1] .- sim.phantom.ϕ[:, :, end÷2+1],
    colormap=:batlow,
    colorrange=(-0.5π, 0.5π),
    nan_color=:black,
)
=#

bins = ceil(Int, (2sum(sim.PH.Sj_wo[1]))^(1 / 3))

stephist!(ax, abs.(sim.PH.u_wo[1][sim.PH.Sj_wo[1]]),
    bins=bins,
    color=:red,
)

bins = ceil(Int, (2sum(sim.PH.Sj[1]))^(1 / 3))

hist!(ax, abs.(sim.PH.u[1][sim.PH.Sj[1]]),
    bins=bins ÷ 4,
    color=:blue,
)

display(fig)

##

fig = Figure()
ax = Axis(fig[1, 1])
n = ceil(Int, (2 * sum(sim.PH.Sj[1]))^(1 / 3))
aily = abs.(imag.(sim.PH.ly[1][sim.PH.Sj[1]]))
aily_0 = abs.(imag.(sim.PH.ly_0[1][sim.PH.Sj[1]]))
stephist!(ax, aily, bins=n, color=:red)
stephist!(ax, aily_0, bins=n, color=:blue)

med = median(aily)
med_0 = median(aily_0)
edges = range(0.0, max(aily...), n + 1)
edges_0 = range(0.0, max(aily_0...), n + 1)
iemin = findfirst(e -> e > med, edges)
iemin_0 = findfirst(e -> e > med_0, edges_0)
h = @views fit(Histogram, aily, edges)
h_0 = @views fit(Histogram, aily_0, edges_0)
fifi = @views findfirst(x -> x > 0, h.weights[iemin+1:end] - h.weights[iemin:end-1])
fifi_0 = @views findfirst(x -> x > 0, h_0.weights[iemin_0+1:end] - h_0.weights[iemin_0:end-1])
aily_max = edges[fifi+iemin-1]
aily_max_0 = edges[fifi_0+iemin_0-1]
println("n = ", n)
println("med = ", med)
println("med_0 = ", med_0)
println("aily_max = ", aily_max)
println("aily_max_0 = ", aily_max_0)

display(fig)

#hist!(ax, sim.PH.ϕ[ciS,sl] .- sim.phantom.ϕ[ciS,sl], bins=100)

#=

        # setting n^(1/3) for the number bins in the histogram was motivated in https://en.wikipedia.org/wiki/Histogram
        nbins = ceil(Int, sum(Sj_)^(1/3)) 
        # boundaries of bin intervals
        edges = @views range(0.0, max(abs.(Δy_[Sj_])...), nbins+1)
        # median of |Δy| over Sj
        med = @views median(abs.(Δy_[Sj_]))
        # due to dimensional arguments, the median of |Δy| should be in the first peak (starting at zero)
        # and should therefore not correspond to phase jumps at region boundaries
        # the index iemin can therefore be used as a starting point
        iemin = findfirst(e -> e > med, edges)
        # generate the histogram curve based upon the bins defined above
        h = @views fit(Histogram, abs.(Δy_[Sj_]), edges)
        # ideally, we want to fully include the first peak (related to actual ϕ variations) and 
        # ignore any other peak(s) (associated with phase jumps)
        # to do so, we search for the first index, when the |Δy| histogram starts to rise again
        # (ideally, this should more or less lie well between the first two peaks)
        fifi = @views findfirst(x -> x > 0, h.weights[iemin+1:end] - h.weights[iemin:end-1])
        # we choose the corresponding |Δy| as a value for Δy_max
        # (left edge of the bin, where the histogram has its first minimum)
        push!(Δy_max, edges[fifi+iemin-1])


# -------------------------------------------------

ax = Axis(fig[1, 2],
    title=L"$R_2^\ast$",
)
hidedecorations!(ax)

heatmap!(ax,
    res.R2s[:,:,sl],
    colormap=:roma,
    nan_color=:black,
)

# -------------------------------------------------

ax = Axis(fig[2, 2],
    title=L"$f$",
)
hidedecorations!(ax)

heatmap!(ax,
    res.f[:,:,sl],
    colormap=:roma,
    nan_color=:black,
)

# -------------------------------------------------

ax = Axis(fig[1, 3],
    title=L"$$abs(coil 1)",
)
hidedecorations!(ax)

heatmap!(ax,
    abs.(res.coils[:,:,1]),
    colormap=:roma,
    nan_color=:black,
)

# -------------------------------------------------

ax = Axis(fig[1, 4],
    title=L"$$abs(coil 2)",
)
hidedecorations!(ax)

heatmap!(ax,
    abs.(res.coils[:,:,2]),
    colormap=:roma,
    nan_color=:black,
)

# -------------------------------------------------

ax = Axis(fig[2, 3],
    title=L"$$angle(coil 1)",
)
hidedecorations!(ax)

heatmap!(ax,
    angle.(res.coils[:,:,1]),
    colormap=:romaO,
    nan_color=:black,
)

# -------------------------------------------------

ax = Axis(fig[2, 4],
    title=L"$$angle(coil 2)",
)
hidedecorations!(ax)

heatmap!(ax,
    angle.(res.coils[:,:,2]),
    colormap=:romaO,
    nan_color=:black,
)
=#

##

#gp = res.PH.fitpar.grePar
gp = sim.PH.fitpar.grePar
#gp = sim.PH.fitpar.grePar

gp2 = VP.modpar(BM.GREMultiEchoWF;
    ts=gp.ts,
    B0=gp.B0,
    ppm_fat=gp.ppm_fat,
    ampl_fat=gp.ampl_fat,
    precession=gp.precession,
    mode=:manual_fat,
    x_sym=[:ϕ, :R2s, :f],
)

gre = VP.create_model(gp2)

VP.x!(gre, [0.0, 0.0, 0.0])
VP.set_data!(gre, VP.A(gre) * [1.0;])

ϕs = [range(-π, π, 100);]

χ2s_w = [(VP.x!(gre, [ϕ, 0.0, 0.0]); VP.χ2(gre)) for ϕ in ϕs]
χ2s_f = [(VP.x!(gre, [ϕ, 0.0, 1.0]); VP.χ2(gre)) for ϕ in ϕs]
χ2s_wf = [(VP.x!(gre, [ϕ, 0.0, 0.5]); VP.χ2(gre)) for ϕ in ϕs]

cs_w = [(VP.x!(gre, [:ϕ, :f], [ϕ, 0.0]); abs(VP.c(gre)[1])) for ϕ in ϕs]
cs_f = [(VP.x!(gre, [:ϕ, :f], [ϕ, 1.0]); abs(VP.c(gre)[1])) for ϕ in ϕs]
cs_wf = [(VP.x!(gre, [:ϕ, :f], [ϕ, 0.5]); abs(VP.c(gre)[1])) for ϕ in ϕs]

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1])

lines!(ax, ϕs, χ2s_w, color=:blue)
scatter!(ax, ϕs, χ2s_f, color=:red)
scatter!(ax, ϕs, χ2s_wf, color=:green)

ax = Axis(fig[1, 2])

lines!(ax, ϕs, cs_w, color=:blue)
scatter!(ax, ϕs, cs_f, color=:red)
scatter!(ax, ϕs, cs_wf, color=:green)

display(fig)

#VP.x!(gre, [0.0, 0.0, 0.0])
#data_w = VP.A(gre) * [1.0;]

##

using StatsBase, Peaks, CairoMakie

n = 10000
x = [randn(2n)..., (0.5randn(n) .+ 2)...]
nb = ceil(Int, (2length(x))^(1 / 3))
edges = [range(min(x...), max(x...), nb + 1);]
x_hist = fit(Histogram, x, edges)
ip = argmax(x_hist.weights)

ip_min = ip_max = ip

#ip_min = ip > 1 ? ip - 1 : 1
#ip_max = ip < nb ? ip + 1 : nb

ip_max_test = ip_max + 1
while ip_max_test <= nb && x_hist.weights[ip_max_test] < x_hist.weights[ip_max]
    ip_max = ip_max_test
    ip_max_test += 1
end

ip_min_test = ip_min - 1
while ip_min_test >= 1 && x_hist.weights[ip_min_test] < x_hist.weights[ip_min]
    ip_min = ip_min_test
    ip_min_test -= 1
end

##

ip_min = max(floor(Int, pws.edges[ip][1]), 1)
ip_max = min(ceil(Int, pws.edges[ip][2]) + 1, length(x_hist.edges[1]))

##

A = rand(10, 10)

S = 0.1 .< A .< 0.9
S1 = deepcopy(S)

@. S1[S1] = 0.4 < A[S1] < 0.6

println("sum(S) = ", sum(S))
println("sum(S1) = ", sum(S1))

## BOGA

using Random
rng = MersenneTwister()

σ = 1e-6
ρ = 0.1 + rand(rng)
β = 0.1 + rand(rng)
g1 = randn(rng, ComplexF64)
g2 = randn(rng, ComplexF64)
TE2, TE1 = 1.0, 5.0
T2s = 50
E1 = exp(-TE1 / T2s)
E2 = exp(-TE2 / T2s)
ΔTE = TE1 - TE2
EΔ = exp(-ΔTE / T2s)
δ = rand(rng, range(-π, π, 100))
eiδ = exp(im * δ)

S1 = ρ * E1 * (g1 + g2) * β * eiδ + σ * randn(rng, ComplexF64)
S2 = ρ * E1 * (-g1 + g2) * β * eiδ + σ * randn(rng, ComplexF64)
S3_ = ρ * E2 * (g1 + g2) * β + σ * randn(rng, ComplexF64)
S4_ = ρ * E2 * (-g1 + g2) * β + σ * randn(rng, ComplexF64)
S3 = 0.5(S3_ - S4_)
S4 = 0.5(S3_ + S4_)

C1 = S3' * S1 + S4 * S2'
C2 = S4' * S1 - S3 * S2'
D1 = S3' * S1 - S4 * S2'
D2 = S4' * S1 + S3 * S2'

TIAMO = sqrt(0.5(abs2(C1) + abs2(C2))) / (abs2(S3) + abs2(S4))
I = 0.25(C1 + C2 + D1 + D2 + C1' - D1' + C2' - D2') / (abs2(S3) + abs2(S4))

#@assert TIAMO ≈ EΔ
#@assert I ≈ EΔ * eiδ

println("exp(- ΔTE / T2*) = ", EΔ)
println("exp(i * δ) = ", eiδ)
println()
println("abs(1 - TIAMO / exp(- ΔTE / T2*)) = ", abs(1 - TIAMO / EΔ))
println("abs(1 - |I| / (exp(- ΔTE / T2*))) = ", abs(1 - abs(I) / EΔ))
println("abs(1 - I / (exp(- ΔTE / T2*) * exp(i * δ))) = ", abs(1 - I / (EΔ * eiδ)))
