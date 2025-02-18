using StaticArrays, Random

import VP4Optim as VP
import B0Map as BM

# GRE parameters
nTE = 3
t0 = 0.5
ΔTE = 1.0
TEs = collect(range(t0, t0 + (nTE - 1) * ΔTE, nTE))
B0 = 3.0
precession = :clockwise
num_coils = 1

# fat model as defined by ISMRM challenge
ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

# invariant arguments for the constructor
args = (TEs, B0, ppm_fat, ampl_fat, precession)

# set up GRE sequence
greFW = BM.greMultiEchoWFFW(args...)
greRW = BM.greMultiEchoWFRW(args...)
gre = BM.greMultiEchoWF(args..., num_coils)

# ranges for ϕ, R2s and fat fraction to be tested
nϕ, nR2s, nf = 100, 20, 11
ϕs = collect(range(-π, π, nϕ))
R2ss = collect(range(0.0, 0.2, nR2s))
fs = collect(range(0.0, 1.0, nf))
cs = collect(1:num_coils)

# defined random numbers?
rng = MersenneTwister(42)

# setup data
dats = zeros(ComplexF64, nϕ, nR2s, nTE, num_coils, nf)

ϕmat = repeat(ϕs, 1, nR2s)
R2smat = repeat(R2ss', nϕ, 1)

c = randn(rng, ComplexF64, num_coils)

for (ϕidx, ϕ) in enumerate(ϕs)
    for (R2sidx, R2s) in enumerate(R2ss)
        VP.x!(greFW, [ϕ, R2s])
        for (fidx, f) in enumerate(fs)
            for (cidx, nc) in enumerate(cs)
                dats[ϕidx, R2sidx, :, cidx, fidx] = VP.A(greFW) * [(1 - f) * c[cidx], f * c[cidx]]
                #dats[ϕidx, R2sidx, :, cidx, fidx] += 0.005 .* randn(rng, ComplexF64, nTE)
            end
        end
    end
end

# initialize fit parameters
data = @views dats[:, :, :, :, 2]
S = trues(nϕ, nR2s)

fitpar = BM.fitPar(gre, data, S)
fitparRW = BM.fitPar(greRW, data, S)
fitparFW = BM.fitPar(greFW, data, S)

# fit options
fitopt = BM.fitOpt(greRW)
fitopt.optim = false
#fitopt.R2s_rng = [0.0, 0.0]
fitopt.ϕ_acc = 1e-6
fitopt.R2s_acc = 1e-6
BM.set_num_phase_intervals(fitpar, fitopt, 3)

BM.local_fit(fitpar, fitopt);
f = zeros(size(S))
BM.calc_par(fitpar, fitopt, BM.fat_fraction, f)

BM.local_fit(fitparRW, fitopt);
fRW = zeros(size(S))
BM.calc_par(fitparRW, fitopt, BM.fat_fraction, fRW)

BM.local_fit(fitparFW, fitopt);
fFW = zeros(size(S))
BM.calc_par(fitparFW, fitopt, BM.fat_fraction, fFW)


##



# starting value for optimization
x0_mf = [0.0, 1e-5, 1e-5]

# optimization constraints
lx_mf = [-π, 0.0, 0.0]
ux_mf = [π, Inf, 1.0]

# relative scale of parameters
x_scale_mf = [2π, 1.0, 1.0]

# for fat_mode == :auto_fat, only ϕ and R2s are accessible
x_af = x_mf[1:2]
x0_af = x0_mf[1:2]
lx_af = lx_mf[1:2]
ux_af = ux_mf[1:2]
x_scale_af = x_scale_mf[1:2]

# what to test
what = (:consistency, :derivatives, :optimization)

# visual confirmation of derivatives
visual = false

# include Hessian in derivative test
Hessian = true

# where to store test result
res = Dict()

# do the tests
for nc in ncs
    # linear coefficient
    local c = SVector{nc}(randn(ComplexF64, nc))

    for precession in (:counterclockwise, :clockwise)
        res[precession] = Dict()
        # check, whether the fat fraction is calculated correctly
        gre_mf = BM.greMultiEchoWF(args..., precession, :manual_fat)
        gre_af = BM.greMultiEchoWF(args..., precession, :auto_fat)
        VP.x!(gre_mf, x_mf)
        y = vec(VP.A(gre_mf) * transpose(c))
        VP.y!(gre_mf, y)
        VP.y!(gre_af, VP.y(gre_mf))
        VP.x!(gre_af, x_af)
        res[precession][:calc_fat_fraction] = BM.fat_fraction(gre_af)
        res[precession][:true_fat_fraction] = x_mf[3]
        res[precession][:calc_c_af] = VP.c(gre_af)
        res[precession][:calc_c_mf] = VP.c(gre_mf)
        res[precession][:true_c] = c
        @test c ≈ VP.c(gre_af) ≈ VP.c(gre_mf)

        res[precession][:check_model] = VP.check_model(BM.greMultiEchoWF, (args..., precession, :manual_fat),
            x_mf, c, y, what=what, x0=x0_mf, lx=lx_mf, ux=ux_mf, x_scale=x_scale_mf,
            visual=visual, rng=rng, Hessian=Hessian)

        res[precession][:check_model] = VP.check_model(BM.greMultiEchoWF, (args..., precession, :auto_fat),
            x_af, c, y, what=what, x0=x0_af, lx=lx_af, ux=ux_af, x_scale=x_scale_af,
            visual=visual, rng=rng, Hessian=Hessian)
    end
end
