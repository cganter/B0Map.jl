using StaticArrays, Random

import VP4Optim as VP
import B0Map as BM

# GRE parameters
nTE = 6
t0 = 0.5
ΔTE = 1.0
TEs = collect(range(t0, t0 + (nTE - 1) * ΔTE, nTE))
B0 = 3.0

# number of coils to be tested
ncs = 1

# fat model as defined by ISMRM challenge
ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

# set up model constructor parameters
pars = VP.modpar(BM.GREMultiEchoWF;
    ts = TEs,
    B0 = B0,
    ppm_fat = ppm_fat,
    ampl_fat = ampl_fat)

# true values for :ϕ, :R2s, :f
x_mf = [0.5, 0.05, 0.5]

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

# defined random numbers
rng = MersenneTwister(42)

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
        pars_mf = VP.modpar(pars, precession = precession, mode = :manual_fat, x_sym = [:ϕ, :R2s, :f])
        gre_mf = VP.create_model(pars_mf)
        pars_af = VP.modpar(pars, precession = precession, mode = :auto_fat, x_sym = [:ϕ, :R2s])
        gre_af = VP.create_model(pars_af)
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

        res[precession][:check_model] = VP.check_model(pars_mf, 
            x_mf, c, y, what = what, x0 = x0_mf, lx = lx_mf, ux = ux_mf, x_scale = x_scale_mf, 
            visual = visual, rng = rng, Hessian = Hessian)

        res[precession][:check_model] = VP.check_model(pars_af, 
            x_af, c, y, what = what, x0 = x0_af, lx = lx_af, ux = ux_af, x_scale = x_scale_af,
            visual = visual, rng = rng, Hessian = Hessian)
    end
end
