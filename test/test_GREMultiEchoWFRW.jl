using StaticArrays, Random, Statistics

import VP4Optim as VP
import B0Map as BM

# GRE parameters
nTE = 6
t0 = 0.5
ΔTE = 1.0
TEs = collect(range(t0, t0 + (nTE - 1) * ΔTE, nTE))
B0 = 3.0

# fat model as defined by ISMRM challenge
ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

# set up model constructor parameters
pars = VP.modpar(BM.GREMultiEchoWFRW;
    ts = TEs,
    B0 = B0,
    ppm_fat = ppm_fat,
    ampl_fat = ampl_fat)

# true values for :ϕ, :R2s
x = [0.5, 0.05]

# starting value for optimization
x0 = [0.0, 0.01]

# optimization constraints
lx = [-π, 0.0]
ux = [π, Inf]

# relative scale of parameters
x_scale = [2π, 1.0]

# fat fraction
f = 0.5

# defined random numbers
rng = MersenneTwister(42)

# linear coefficients
c = SVector{2, Float64}([1 - f, f])
eiϕ = exp(im * rand(rng, range(-0.5π, 0.5π, 1000)))

# what to test
what = (:consistency,)

# visual confirmation of derivatives
visual = false

# include Hessian in derivative test
Hessian = false

# where to store test results
res = Dict()

# do the tests
for precession in (:counterclockwise, :clockwise)
    # generate ideal data
    pars_ = VP.modpar(pars, precession = precession)
    gre_fw = VP.create_model(pars_)
    
    iΔTE = 1im / ΔTE
    local fac = im * 2π * 0.042577 * B0
    precession == :clockwise && (iΔTE = - iΔTE; fac = - fac)
    w = sum(ampl_fat' .* exp.(fac * ppm_fat' .* TEs), dims=2)
    e = exp.((iΔTE * x[1] - x[2]) .* TEs)
    ew = e .* w
    
    A = [j == 1 ? e[i] : ew[i] for i in 1:nTE, j in 1:2]
    data = eiϕ * A * c

    res[precession] = VP.check_model(pars_, x, c, data, what = what, x0 = x0, lx = lx, ux = ux, 
        x_scale = x_scale, visual = visual, rng = rng, Hessian = Hessian)
end
