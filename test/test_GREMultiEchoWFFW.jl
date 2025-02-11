using StaticArrays, Random

import VP4Optim as VP
import B0Map

# GRE parameters
nTE = 6
t0 = 0.5
ΔTE = 1.0
TEs = collect(range(t0, t0 + (nTE - 1) * ΔTE, nTE))
B0 = 3.0

# fat model as defined by ISMRM challenge
ppm_fat = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]
ampl_fat = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]

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

# linear coefficients
c = SVector{2, ComplexF64}(rand(ComplexF64) .* [1 - f, f])

# what to test
what = (:consistency, :derivatives, :optimization)

# visual confirmation of derivatives
visual = false

# defined random numbers
rng = MersenneTwister(42)

# include Hessian in derivative test
Hessian = true

# where to store test results
res = Dict()

# do the tests
for precession in (:counterclockwise, :clockwise)
    local args = (TEs, B0, ppm_fat, ampl_fat, precession)
    # generate ideal data
    gre_fw = B0Map.greMultiEchoWFFW(args...)
    VP.x!(gre_fw, x)
    y = VP.A(gre_fw) * c

    res[precession] = VP.check_model(B0Map.greMultiEchoWFFW, args, x, c, y, what = what, x0 = x0, lx = lx, ux = ux, 
        x_scale = x_scale, visual = visual, rng = rng, Hessian = Hessian)
end
