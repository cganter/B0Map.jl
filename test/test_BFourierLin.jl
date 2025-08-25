using Random, FFTW

import VP4Optim as VP
import B0Map as BM

include("test_tools.jl")

Nρ_orig = (20, 20, 20)
nd = length(Nρ_orig)
K = (2, 2, 2)
os_fac = 1.5
bs = BM.fourier_lin(Nρ_orig, K; os_fac=os_fac)
Nρ = BM.Nρ(bs)

@test BM.Nρ_orig(bs) == Nρ_orig
@test BM.Nν(bs) == nd

# create mask S

S_nSinc = 3
S_zc = 4.0
S_rng = (-1.0, 1.0)
S_holes = 0.5
S_io = :in
n_chunks = Threads.nthreads()
rng = MersenneTwister(42)

S_data = create_sinc_map(trues(Nρ_orig...), S_nSinc, S_zc, S_rng;
    n_chunks=n_chunks, rng=rng)

S = create_msk(S_data, S_holes, S_io)
noS = (!).(S)

ciN = CartesianIndices(S)
ciS = ciN[S]
cinoS = ciN[noS]

# construct Fourier matrix

ρ_rngs = [reshape(Float64[0:N-1;], ones(Int, j - 1)..., :) for (j, N) in enumerate(Nρ)]
κs = map(bs.ciκ) do ci
    collect(Tuple(ci)) .- 1.0
end

vκ = vec(κs)
Nκ = length(vκ)

F_nd = zeros(ComplexF64, Nρ_orig..., Nκ)

fac = 2π ./ [Nρ...;]

for ci in ciS
    fac_ρ = fac .* (collect(Tuple(ci)) .- 1.0)
    for (k, κ) in enumerate(κs)
        F_nd[ci, k] = exp(im * fac_ρ' * κ)
    end
end

F = F_nd[ciS, :]

# check that it generates correct DFT

x_os_nd = zeros(Float64, Nρ)
x_os_nd[ciN] = randn(Float64, Nρ_orig)
x_nd = zeros(Float64, Nρ_orig)
x_nd[ciN] = x_os_nd[ciN]
x_os_nd[cinoS] .= 0.0
x_os = vec(x_os_nd)
x = x_os_nd[ciS]

fft_x = fft(x_os_nd)
@test F' * x ≈ vec(fft_x[bs.ciκ])

# linear part of B

L_nd = zeros(ComplexF64, Nρ_orig..., nd)

for (i, ρi) in enumerate(bs.ρνs)
    L_nd[ciS, i] .= ρi[ciS]
end

L = L_nd[ciS, :]

# combine Fourier and linear part
B = cat(F, L, dims=2)

@test F ≈ B[:, 1:end-nd]
@test L ≈ B[:, end-nd+1:end]

# run a few tests on B' * B

BtB = B' * B

@test BtB ≈ BtB'
@test BtB[1:end-nd, 1:end-nd] ≈ F' * F
@test BtB[1:end-nd, end-nd+1:end] ≈ F' * L
@test BtB[end-nd+1:end, 1:end-nd] ≈ L' * F
@test BtB[end-nd+1:end, end-nd+1:end] ≈ L' * L
sumS = sum(S)
@test all(x -> real(x) ≈ sumS, diag(BtB')[1:end-nd])

BtB_calc = BM.calc_BtB(bs, S)

@test BtB_calc ≈ BtB

# B' * x

Btx = B' * x
Btx_calc = BM.calc_Btx(bs, S, x_nd)

@test Btx_calc ≈ Btx

# Moore Penrose inverse

c_calc = BtB_calc \ Btx_calc
c = BtB \ Btx

@test c_calc ≈ c

# ML projection

Px_nd_calc = BM.calc_Bc(bs, c)
Px_calc = Px_nd_calc[ciS]
Px = B * c

@test Px ≈ Px_calc
@test norm(Px) <= norm(x)

BtPx = B' * Px
BtPx_calc = BM.calc_Btx(bs, S, Px_nd_calc)

@test BtPx_calc ≈ BtPx

cP = BtB \ BtPx

P2x_nd_calc = BM.calc_Bc(bs, cP)
P2x_calc = P2x_nd_calc[ciS]
P2x = B * cP

@test Px ≈ P2x

@test c' * Btx ≈
      c' * BtB * c ≈
      Px' * Px ≈
      x' * Px

@test real.(x' * x - c' * BtB * c) ≈
      real.(x' * x - 2c' * Btx + c' * BtB * c) ≈
      norm(x)^2 - norm(Px)^2 ≈
      norm(x - Px)^2

# -------------------------------------------------------------

B_nd = cat(F_nd, L_nd, dims=nd + 1)

(∇B_nd, Sj) = BM.∇j(B_nd, S)

ciSj = [CartesianIndices(Sj_)[Sj_] for Sj_ in Sj]

∇B = cat([∇B_nd_[ciSj_, :][:, 2:end] for (∇B_nd_, ciSj_) in zip(∇B_nd, ciSj)]..., dims=1)

∇Bt∇B = ∇B' * ∇B
@test ∇Bt∇B ≈ ∇Bt∇B'

∇Bt∇B_calc = BM.calc_∇Bt∇B(bs, Sj)
@test ∇Bt∇B ≈ ∇Bt∇B_calc

∇F_nd = BM.∇j_(F_nd, Sj)
∇L_nd = BM.∇j_(L_nd, Sj)

∇F = cat([∇F_nd_[ciSj_, :][:, 2:end] for (∇F_nd_, ciSj_) in zip(∇F_nd, ciSj)]..., dims=1)
∇L = cat([∇L_nd_[ciSj_, :] for (∇L_nd_, ciSj_) in zip(∇L_nd, ciSj)]..., dims=1)

@test ∇Bt∇B[1:end-nd, 1:end-nd] ≈ ∇F' * ∇F
@test ∇Bt∇B[1:end-nd, end-nd+1:end] ≈ ∇F' * ∇L
@test ∇Bt∇B[end-nd+1:end, 1:end-nd] ≈ ∇L' * ∇F
@test ∇Bt∇B[end-nd+1:end, end-nd+1:end] ≈ ∇L' * ∇L

# ∇B' * x

xj_nd = [randn(Float64, Nρ_orig) for _ in 1:nd]
xj = Float64[]
for (xj_, Sj_) in zip(xj_nd, Sj)
    global xj = [xj; xj_[Sj_]]
end

∇Btx = ∇B' * xj
∇Btx_calc = BM.calc_∇Btx(bs, Sj, xj_nd)

@test ∇Btx ≈ ∇Btx_calc

# Moore Penrose inverse

∇c_calc = ∇Bt∇B_calc \ ∇Btx_calc
∇c = ∇Bt∇B \ ∇Btx

@test ∇c_calc ≈ ∇c

# ML projection

Pxj_nd_calc = BM.calc_∇Bc(bs, ∇c)
Pxj_calc = Float64[]
for (Pxj_nd_calc_, Sj_) in zip(Pxj_nd_calc, Sj)
    global Pxj_calc = [Pxj_calc; Pxj_nd_calc_[Sj_]]
end

Pxj = ∇B * ∇c

@test Pxj ≈ Pxj_calc
@test norm(Pxj) <= norm(xj)

∇BtPxj = ∇B' * Pxj
∇BtPxj_calc = BM.calc_∇Btx(bs, Sj, Pxj_nd_calc)

@test ∇BtPxj_calc ≈ ∇BtPxj

∇cP = ∇Bt∇B \ ∇BtPxj

P2xj_nd_calc = BM.calc_∇Bc(bs, ∇cP)
P2xj_calc = Float64[]
for (P2xj_nd_calc_, Sj_) in zip(P2xj_nd_calc, Sj)
    global P2xj_calc = [P2xj_calc; P2xj_nd_calc_[Sj_]]
end
P2xj = ∇B * ∇cP

@test Pxj ≈ P2xj

@test ∇c' * ∇Btx ≈
    ∇c' * ∇Bt∇B * ∇c ≈
    Pxj' * Pxj ≈
    xj' * Pxj

@test real.(xj' * xj - ∇c' * ∇Bt∇B * ∇c) ≈ 
    real.(xj' * xj - 2∇c' * ∇Btx + ∇c' * ∇Bt∇B * ∇c) ≈ 
    norm(xj)^2 - norm(Pxj)^2 ≈
    norm(xj - Pxj)^2