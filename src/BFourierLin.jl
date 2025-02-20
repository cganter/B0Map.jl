using Compat
@compat public BFourierLin

#= 
==================================================================  
Combination of Fourier and linear basis [B L]

F_{ρκ} = exp(i * 2π * κ * ρ / N)
L_{ρν} = ρ_ν

(notation as in the article)

Periodicity and kernel size are specified by vectors of length d:

- N[j] = assumed periodicity in direction j (>= number of entries in z along the same direction)
- K[j] = (integer >= 0), specifies Fourier kernel size: |κ[j]| <= K[j]
==================================================================  
=#

struct BFourierLin{N} <: BSmooth{N}
    # dimensions and kernel size
    Nρ::NTuple{N,Int}
    Nκ::NTuple{N,Int}
    Nν::Int
    K::NTuple{N,Int}
    Nρ_orig::NTuple{N,Int}
    # auxiliary quantities
    κ::Vector{Array{Int64,N}}
    aκ::Vector{Array{ComplexF64,N}}
    ciκ::Array{CartesianIndex{N},N}
    ciκ_po::Vector{CartesianIndex{N}}
    ciκ_ne::Vector{CartesianIndex{N}}
    idx_po::Vector{Int}
    idx_ne::Vector{Int}
    ciκmκ0::Matrix{CartesianIndex{N}}
    ciκmκ::Matrix{CartesianIndex{N}}
    ciκmκ_unique::Vector{CartesianIndex{N}}
    ciκpκ::Matrix{CartesianIndex{N}}
    ciκpκ_unique::Vector{CartesianIndex{N}}
    # work space to avoid reallocations for Fourier transform
    ws_ρ::Array{Float64,N}
    ws_κ::Array{ComplexF64,N}
    ρνs::Vector{Array{Float64,N}}
end

"""
    BFourierLin(Nρ_orig, K; os_fac=1.0)

Constructor of combined Fourier & linear basis

# Arguments

- `Nρ_orig`: Number of `ρ` locations in each direction of the original (not oversampled) FOV. (iterable)
- `K`: max. Fourier component in each direction. (iterable)
- `os_fac`: Oversampling factor (default == `1.0`), either global (scalar) or direction dependent (scalar or iterable)
"""
function BFourierLin(Nρ_orig, K; os_fac=1.0)
    @assert all(K .>= 0)
    @assert all(os_fac .>= 1.0)

    Nκ = tuple((2 .* K .+ 1)...)
    Nν = length(Nκ)
    Nρ = collect(Nρ_orig)
    any(os_fac .!= 1.0) && (Nρ = round.(Int, Nρ .* os_fac))
    # old[
    # @assert all(Nρ .> Nκ)
    # ]old
    # new[
    K = collect(K)
    for (ν, k, nρ, nκ) in zip(1:Nν, K, Nρ, Nκ)
        while nρ <= nκ
            k -= 1
            nκ -= 2
        end
        K[ν] = k
    end
    Nκ = tuple((2 .* K .+ 1)...)
    # ]new

    # to avoid distiction of cases, associated with real-valued Fourier transforms
    Nρ = [N_ <= 2K_ ? 2K_ + 1 : N_ for (N_, K_) in zip(Nρ, K)]

    # we prefer Nρ to be even
    # furthermore, FFTW works optimally, if each element of Nρ can be factorized in terms of the small primes (2, 3, 5, 7)
    Nρ = 2 * (x -> Base.nextprod((2, 3, 5, 7), x)).((Nρ .+ Nρ .% 2) .÷ 2)

    # to guarantee correct types
    Nρ = tuple(Nρ...)
    K = tuple(K...)
    Nρ_orig = tuple(Nρ_orig...)

    # auxiliary quantities
    κ = [repeat(
        reshape(
            circshift(-k:k, k + 1),
            ones(Int, j - 1)..., Nκ[j]),
        Nκ[1:j-1]..., 1, Nκ[j+1:end]...)
         for (j, k) in enumerate(K)]

    aκ = [repeat(
        reshape(
            circshift(exp.(-im .* 2π .* (-k:k) ./ Nρ[j]) .- 1.0, k + 1),
            ones(Int, j - 1)..., Nκ[j]),
        Nκ[1:j-1]..., 1, Nκ[j+1:end]...)
          for (j, k) in enumerate(K)]

    ciκ = [CartesianIndex(tuple(a...)) for a in zip(
        [(x -> mod(x, N_)).(κ_) .+ 1 for (κ_, N_) in zip(κ, Nρ)]...)]

    tκ = [tuple(a...) for a in zip(κ...)]

    tκ_po = filter(x -> first(filter(y -> y != 0, x)) > 0, tκ[2:end])
    tκ_ne = [map(x -> -x, t) for t in tκ_po]

    ciκ_po = [CartesianIndex(mod.(t, Nρ) .+ 1) for t in tκ_po]
    ciκ_ne = [CartesianIndex(mod.(t, Nρ) .+ 1) for t in tκ_ne]
    idx_po = convert(Vector{Int}, filter(x -> x !== nothing, indexin(ciκ_po, ciκ[2:end])))
    idx_ne = convert(Vector{Int}, filter(x -> x !== nothing, indexin(ciκ_ne, ciκ[2:end])))

    ciκmκ0 = [CartesianIndex(tuple(a...)) for a in zip(
        [(x -> mod(x, N_)).(κ_[:] .- κ_[:]') .+ 1 for (κ_, N_) in zip(κ, Nρ)]...)]

    ciκmκ = [CartesianIndex(tuple(a...)) for a in zip(
        [(x -> mod(x, N_)).(κ_[2:end] .- κ_[2:end]') .+ 1 for (κ_, N_) in zip(κ, Nρ)]...)]

    ciκmκ_unique = unique(ciκmκ)

    ciκpκ = [CartesianIndex(tuple(a...)) for a in zip(
        [(x -> mod(x, N_)).(κ_[2:end] .+ κ_[2:end]') .+ 1 for (κ_, N_) in zip(κ, Nρ)]...)]

    ciκpκ_unique = unique(ciκpκ)

    # work space
    ws_ρ = zeros(Float64, Nρ...)  # we only transform real functions from ρ → κ
    ws_κ = zeros(ComplexF64, Nρ...)

    ρνs = [repeat(
        reshape(collect(range(-N / 2 + 0.5, N / 2 - 0.5, N)), ones(Int, j - 1)..., :),
        Nρ_orig[1:j-1]..., 1, Nρ_orig[j+1:end]...) for
           (j, N) in enumerate(Nρ_orig)]

    BFourierLin(Nρ, Nκ, Nν, K, Nρ_orig, κ, aκ, ciκ, ciκ_po, ciκ_ne, idx_po, idx_ne, ciκmκ0, ciκmκ, ciκmκ_unique, ciκpκ, ciκpκ_unique, ws_ρ, ws_κ, ρνs)
end

"""
    Nρ(bf::BFourierLin)

Number of `ρ` locations in each direction of the (possibly extended resp. oversampled) FOV.
"""
function Nρ(bf::BFourierLin)
    bf.Nρ
end

"""
    Nρ_orig(bf::BFourierLin)

Number of `ρ` locations in each direction of the original (not oversampled) FOV.
"""
function Nρ_orig(bf::BFourierLin)
    bf.Nρ_orig
end

"""
    Nκ(bf::BFourierLin)

Specifies the size of the kernel. In the Fourier case, we have `length(Nκ(bs)) == length(Nρ(bs))`
and the entry specifies the largest Fourier component. For `BFree`, `prod(Nκ(bs))` simply denotes the total number 
of smooth basis functions.
"""
function Nκ(bf::BFourierLin)
    bf.Nκ
end

"""
    Nν(::BFourierLin{N}) where N <: Integer

Returns number of linear degrees of freedom, which is equal to the dimensionality `d`
of the problem and also the same as `N`.
"""
function Nν(bf::BFourierLin)
    bf.Nν
end

"""
    oBtoB_oBtϕ(bf::BFourierLin, S::AbstractArray, ϕ::AbstractArray)

Returns matrix and vector for direct (= not related to derivatives) Moore Penrose inversion (MPI)

# Arguments

- `bf`: Definition of smooth Fourier basis
- `S`: Boolean mask (points of interest)
- `ϕ`: Phase map to be projected

# Return values

The function returns a tuple `(oBtoB, oBtϕ)`, required for the MPI:

- `oBtoB`: Matrix `[1_S F L]* [1_S F L]`, as described in the supplementary information of the article
- `oBtϕ`: Vector `[1_S F L]* ϕ` (dito)
"""
#=
function oBtoB_oBtϕ(bf::BFourierLin, S::AbstractArray, ϕ::AbstractArray)
    nκ = prod(Nκ(bf))
    bf.ws_ρ .= 0.0
    bf.ws_ρ[CartesianIndices(S)[S]] .= @views ϕ[CartesianIndices(S)[S]]

    oBtϕ = zeros(ComplexF64, nκ + Nν(bf))
    oBtϕ[1:nκ] .= @views vec(fft(bf.ws_ρ)[bf.ciκ])
    for j in 1:Nν(bf)
        oBtϕ[nκ+j] = @views sum(bf.ρνs[j][S] .* ϕ[S])
    end

    bf.ws_ρ[filter(x -> S[x], CartesianIndices(S))] .= 1.0
    bf.ws_ρ[filter(x -> !S[x], CartesianIndices(S))] .= 0.0
    oBtoB = zeros(ComplexF64, nκ + Nν(bf), nκ + Nν(bf))
    oBtoB[1:nκ, 1:nκ] = @views fft(bf.ws_ρ)[bf.ciκmκ0]

    bf.ws_ρ .= 0.0
    for j in 1:Nν(bf)
        bf.ws_ρ[CartesianIndices(S)[S]] .= @views bf.ρνs[j][S]
        oBtoB[1:nκ, nκ+j] = @views vec(fft(bf.ws_ρ)[bf.ciκ])
        oBtoB[nκ+j, 1:nκ] = @views conj.(oBtoB[1:nκ, nκ+j])
    end
    for i in 1:Nν(bf)
        for j in 1:i
            oBtoB[nκ+i, nκ+j] = @views sum(bf.ρνs[i][S] .* bf.ρνs[j][S])
            i != j && (oBtoB[nκ+j, nκ+i] = oBtoB[nκ+i, nκ+j])
        end
    end

    (oBtoB, oBtϕ)
end
=#

function calc_BtB(bf::BFourierLin, S::AbstractArray)
    nκ = prod(Nκ(bf))

    cis = @views CartesianIndices(S)[S]

    bf.ws_ρ .= 0.0
    bf.ws_ρ[cis] .= 1.0

    BtB = zeros(ComplexF64, nκ + Nν(bf), nκ + Nν(bf))
    BtB[1:nκ, 1:nκ] = @views fft(bf.ws_ρ)[bf.ciκmκ0]

    bf.ws_ρ .= 0.0
    for j in 1:Nν(bf)
        bf.ws_ρ[cis] .= @views bf.ρνs[j][S]
        BtB[1:nκ, nκ+j] = @views vec(fft(bf.ws_ρ)[bf.ciκ])
        BtB[nκ+j, 1:nκ] = @views conj.(BtB[1:nκ, nκ+j])
    end

    for i in 1:Nν(bf)
        for j in 1:i
            BtB[nκ+i, nκ+j] = @views sum(bf.ρνs[i][S] .* bf.ρνs[j][S])
            i != j && (BtB[nκ+j, nκ+i] = BtB[nκ+i, nκ+j])
        end
    end

    return BtB
end

function calc_Btx(bf::BFourierLin, S::AbstractArray, x::AbstractArray)
    nκ = prod(Nκ(bf))

    cis = @views CartesianIndices(S)[S]

    bf.ws_ρ .= 0.0
    bf.ws_ρ[cis] .= @views x[cis]

    Btx = zeros(ComplexF64, nκ + Nν(bf))
    Btx[1:nκ] .= @views vec(fft(bf.ws_ρ)[bf.ciκ])
    for j in 1:Nν(bf)
        Btx[nκ+j] = @views sum(bf.ρνs[j][S] .* x[S])
    end

    return Btx
end

"""
    ∇Bt∇B_∇Bty(bf::BFourierLin, Sj::AbstractVector, y::AbstractVector)

Returns matrix and vector for derivative based Moore Penrose inversion (MPI)

# Arguments

- `bf`: Definition of smooth Fourier basis
- `Sj`: Vector of Boolean masks (points of interest) in each direction `j`
- `y`: Vector of difference maps to be approximated

# Return values

The function returns a tuple `(∇Bt∇B, ∇Bty)`, required for the MPI:

- `∇Bt∇B`: Matrix `∇B* ∇B`, as described in the supplementary information of the article
- `∇Bty`: Vector `∇B* y` (dito)
"""
#=
function ∇Bt∇B_∇Bty(bf::BFourierLin, Sj::AbstractVector, y::AbstractVector)
    nκ1 = prod(Nκ(bf)) - 1

    ∇Bt∇B = zeros(ComplexF64, nκ1 + Nν(bf), nκ1 + Nν(bf))
    ∇Bty = zeros(ComplexF64, nκ1 + Nν(bf))
    bf.ws_ρ .= 0.0

    for (ν, aκ_, Sj_, y_) in zip(1:Nν(bf), bf.aκ, Sj, y)
        copyto!(bf.ws_ρ, CartesianIndices(y_), y_, CartesianIndices(y_))

        ∇Bty[1:nκ1] += @views (aκ_.*fft(bf.ws_ρ)[bf.ciκ])[2:end]
        ∇Bty[nκ1+ν] = @views sum(y_[Sj_])

        # the following construct (instead of simple logical indexing)
        # is used, since ws_ρ and Sj do not have the same size, iff Nρ != Nρ_orig
        bf.ws_ρ[filter(x -> Sj_[x], CartesianIndices(Sj_))] .= 1.0
        bf.ws_ρ[filter(x -> !Sj_[x], CartesianIndices(Sj_))] .= 0.0
        FT_Sj = fft(bf.ws_ρ)

        ∇Bt∇B[1:nκ1, 1:nκ1] += @views aκ_[2:end] .* FT_Sj[bf.ciκmκ] .* aκ_[2:end]'
        ∇Bt∇B[1:nκ1, nκ1+ν] = @views aκ_[2:end] .* FT_Sj[bf.ciκ[2:end]]
        ∇Bt∇B[nκ1+ν, 1:nκ1] = @views conj.(∇Bt∇B[1:nκ1, nκ1+ν])
        ∇Bt∇B[nκ1+ν, nκ1+ν] = sum(Sj_)
    end

    (∇Bt∇B, ∇Bty)
end
=#

"""
    calc_∇Bt∇B(bf::BFourierLin, Sj::AbstractVector)

TBW
"""
function calc_∇Bt∇B(bf::BFourierLin, Sj::AbstractVector)
    nκ1 = prod(Nκ(bf)) - 1

    ∇Bt∇B = zeros(ComplexF64, nκ1 + Nν(bf), nκ1 + Nν(bf))

    for (ν, aκ_, Sj_) in zip(1:Nν(bf), bf.aκ, Sj)
        bf.ws_ρ .= 0.0
        bf.ws_ρ[CartesianIndices(Sj_)] .= 1.0

        FT_Sj = fft(bf.ws_ρ)

        ∇Bt∇B[1:nκ1, 1:nκ1] += @views aκ_[2:end] .* FT_Sj[bf.ciκmκ] .* aκ_[2:end]'
        ∇Bt∇B[1:nκ1, nκ1+ν] = @views aκ_[2:end] .* FT_Sj[bf.ciκ[2:end]]
        ∇Bt∇B[nκ1+ν, 1:nκ1] = @views conj.(∇Bt∇B[1:nκ1, nκ1+ν])
        ∇Bt∇B[nκ1+ν, nκ1+ν] = sum(Sj_)
    end

    return ∇Bt∇B
end

"""
    calc_∇Btx(bf::BFourierLin, Sj::AbstractVector, x::AbstractVector)

TBW
"""
function calc_∇Btx(bf::BFourierLin, Sj::AbstractVector, x::AbstractVector)
    nκ1 = prod(Nκ(bf)) - 1

    ∇Btx = zeros(ComplexF64, nκ1 + Nν(bf))
    bf.ws_ρ .= 0.0

    for (ν, aκ_, Sj_, x_) in zip(1:Nν(bf), bf.aκ, Sj, x)
        copyto!(bf.ws_ρ, CartesianIndices(x_), x_, CartesianIndices(x_))

        ∇Btx[1:nκ1] += @views (aκ_.*fft(bf.ws_ρ)[bf.ciκ])[2:end]
        ∇Btx[nκ1+ν] = @views sum(x_[Sj_])
    end

    return ∇Btx
end

"""
    phase_map(bf::BFourierLin, c::Vector)

Calculates and returns the regularized and real valued phase map `Bρκ * c` 
for the supplied coefficients `c`, assuming the global offset `b` to be zero.

The dimensions of the phase map correspond to the original dimensions `Nρ_orig`, not `Nρ`.

# Arguments

- `bf`: Definition of smooth Fourier basis
- `c`: Complex coefficient vector, including the redundancy imposed by the real valued phase map

# Return values

- Phase map [rad] as an array of dimension `Nρ_orig`. Also returns estimates for `ρ ∉ S`
"""
function phase_map(bf::BFourierLin, c::AbstractVector)
    bf.ws_κ .= zero(ComplexF64)
    nκ1 = prod(Nκ(bf)) - 1
    bf.ws_κ[bf.ciκ] = [0; c[1:nκ1]]

    ϕ = @views real.(bfft(bf.ws_κ)[(1:n for n in bf.Nρ_orig)...])
    for ν in 1:Nν(bf)
        ϕ += bf.ρνs[ν] .* real.(c[nκ1+ν])
    end

    return ϕ
end

"""
    refine_c!(c, y, Sj, bf::BFourierLin, optim, precon, precon_update)

Wrapper around the generic `refine` function: 

Converts complex Fourier coefficients `c` into real ones, as expected by `refine`. 
After execution, the complex vector `c` is updated with the result of the opimization.

# Arguments

See the doc for the `refine` function.

# Return value

Equals return value of `refine`.
"""
function refine_c!(c, y, Sj, bf::BFourierLin, optim, precon, precon_update)
    # extract independent real parameters
    nκ1 = prod(Nκ(bf)) - 1
    rc = @views [collect(reinterpret(Float64, c[bf.idx_po])); real.(c[nκ1+1:end])]

    # generic part
    res = refine(rc, y, Sj, bf, optim, precon, precon_update)

    # restore redundant complex coefficients
    c[bf.idx_po] = @views reinterpret(ComplexF64, res.minimizer[1:end-Nν(bf)])
    c[bf.idx_ne] = @views conj.(c[bf.idx_po])
    c[nκ1+1:end] = @views res.minimizer[nκ1+1:end]

    res
end

"""
    fgh!(F, G, H, rc, bf::BFourierLin, y, Sj)

Calculates (on demand) value, gradient and Hessian of the cost function `L`, which corrects for
phase factor linearization errors (cf. article and supplementary information for more details)

# Arguments

- `F`: If not equal to `nothing`, the value of the cost funcion is calculated and returned
- `G`: Work space for gradient. If `G === nothing`, no gradient is calculated.
- `H`: Work space for Hessian. If `H === nothing`, no Hessian is calculated.
- `rc`: Vector of real, non-redundant coefficients `cκ`
- `bf`: Definition of smooth Fourier basis.
- `y`: Vector of complex difference maps.
- `Sj`: Corresponding vector of boolean maps.

# Return value

- Value of cost function, if `F !== nothing`. Otherwise `nothing`.
"""
function fgh!(F, G, H, rc, bf::BFourierLin, y, Sj)
    nκ1 = prod(Nκ(bf)) - 1
    c = zeros(ComplexF64, nκ1 + Nν(bf))

    c[bf.idx_po] = reinterpret(ComplexF64, rc[1:nκ1])
    c[bf.idx_ne] = @views conj.(c[bf.idx_po])
    c[nκ1+1:end] = rc[nκ1+1:end]


    ∇ϕ = ∇j_(phase_map(bf, c), Sj)
    fjρ = [(-conj.(y_) .- one(ComplexF64)) .* exp.(im * ∇ϕ_) for (y_, ∇ϕ_) in zip(y, ∇ϕ)]

    if G !== nothing || H !== nothing
        re_rng = 1:2:nκ1
        im_rng = 2:2:nκ1

        # reset gradient and Hessian, if required
        G !== nothing && (G .= 0.0)
        H !== nothing && (H .= 0.0)

        for (ν, aκ_, Sj_, fjρ_) in zip(1:Nν(bf), bf.aκ, Sj, fjρ)
            # calculate gradient
            if G !== nothing
                bf.ws_ρ[filter(x -> !Sj_[x], CartesianIndices(Sj_))] .= 0.0
                bf.ws_ρ[filter(x -> Sj_[x], CartesianIndices(Sj_))] .= @views imag.(fjρ_[Sj_])

                tmp = @views (2aκ_.*fft(bf.ws_ρ)[bf.ciκ])[2:end]

                G[re_rng] -= @views real.(tmp[bf.idx_po])
                G[im_rng] -= @views imag.(tmp[bf.idx_po])
                G[nκ1+ν] = @views -sum(imag.(fjρ_[Sj_]))
            end

            # calculate Hessian
            if H !== nothing
                bf.ws_ρ[filter(x -> !Sj_[x], CartesianIndices(Sj_))] .= 0.0
                bf.ws_ρ[filter(x -> Sj_[x], CartesianIndices(Sj_))] .= real.(fjρ_)[Sj_]

                copy!(bf.ws_κ, fft(bf.ws_ρ))
                tmp_p = @views (aκ_[2:end].*bf.ws_κ[bf.ciκpκ].*conj.(aκ_[2:end]'))[bf.idx_po, bf.idx_po]
                tmp_m = @views (aκ_[2:end].*bf.ws_κ[bf.ciκmκ].*aκ_[2:end]')[bf.idx_po, bf.idx_po]

                H[re_rng, re_rng] -= 2real.(tmp_p + tmp_m)
                H[re_rng, im_rng] += 2imag.(tmp_m - tmp_p)
                H[im_rng, im_rng] += 2real.(tmp_p - tmp_m)
                H[re_rng, nκ1+ν] = @views -2real.(aκ_[2:end] .* bf.ws_κ[bf.ciκ[2:end]])[bf.idx_po]
                H[nκ1+ν, re_rng] = @views H[re_rng, nκ1+ν]
                H[im_rng, nκ1+ν] = @views -2imag.(aκ_[2:end] .* bf.ws_κ[bf.ciκ[2:end]])[bf.idx_po]
                H[nκ1+ν, im_rng] = @views H[im_rng, nκ1+ν]
                H[nκ1+ν, nκ1+ν] = @views -sum(real.(fjρ_)[Sj_])
            end
        end

        H !== nothing && (H[im_rng, re_rng] = @views H[re_rng, im_rng]')
    end

    # calculate cost function
    return F !== nothing ? sum(sum(real.(fjρ))) : nothing
end