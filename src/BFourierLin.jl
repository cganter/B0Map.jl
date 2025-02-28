using Compat
@compat public BFourierLin, fourier_lin, Nρ, Nρ_orig, Nκ, Nν

"""
    BFourierLin{N}

Combination of Fourier and linear basis [B L]

## Remarks
- Fourier component: ``F_{\\rho\\kappa} = \\exp(i\\,2\\pi\\kappa\\rho / N)``
- Linear component: ``L_{\\rho\\nu} = \\rho_\\nu``
- See article for more details
- Periodicity and kernel size are specified by vectors of length `d`:
- `N[j]`: assumed periodicity in direction `j`.
- `K[j]`: (integer >= 0), specifies Fourier kernel size: `|κ[j]| <= K[j]`
"""
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
    ciκpκ::Matrix{CartesianIndex{N}}
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
function fourier_lin(Nρ_orig, K; os_fac=1.0)
    @assert all(K .>= 0)
    @assert all(os_fac .>= 1.0)

    Nκ = tuple((2 .* K .+ 1)...)
    Nν = length(Nκ)
    Nρ = collect(Nρ_orig)

    any(os_fac .!= 1.0) && (Nρ = round.(Int, Nρ .* os_fac))
    K = collect(K)
    for (ν, k, nρ, nκ) in zip(1:Nν, K, Nρ, Nκ)
        while nρ <= nκ
            k -= 1
            nκ -= 2
        end
        K[ν] = k
    end
    Nκ = tuple((2 .* K .+ 1)...)

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

    ciκpκ = [CartesianIndex(tuple(a...)) for a in zip(
        [(x -> mod(x, N_)).(κ_[2:end] .+ κ_[2:end]') .+ 1 for (κ_, N_) in zip(κ, Nρ)]...)]

    # work space
    ws_ρ = zeros(Float64, Nρ...)  # we only transform real functions from ρ → κ
    ws_κ = zeros(ComplexF64, Nρ...)

    ρνs = [repeat(
        reshape(collect(range(-N / 2 + 0.5, N / 2 - 0.5, N)), ones(Int, j - 1)..., :),
        Nρ_orig[1:j-1]..., 1, Nρ_orig[j+1:end]...) for
           (j, N) in enumerate(Nρ_orig)]

    BFourierLin(Nρ, Nκ, Nν, K, Nρ_orig, κ, aκ, ciκ, ciκ_po, ciκ_ne, idx_po, idx_ne, ciκmκ0, ciκmκ, ciκpκ, ws_ρ, ws_κ, ρνs)
end

"""
    Nρ(bf::BFourierLin)

Return number of `ρ` locations in each direction of the (possibly extended resp. oversampled) FOV.
"""
function Nρ(bf::BFourierLin)
    bf.Nρ
end

"""
    Nρ_orig(bf::BFourierLin)

Return number of `ρ` locations in each direction of the original (not oversampled) FOV.
"""
function Nρ_orig(bf::BFourierLin)
    bf.Nρ_orig
end

"""
    Nκ(bf::BFourierLin)

Return size of the Fourier kernel. 

## Remarks
- `length(Nκ(bs)) == length(Nρ(bs))`
- `Nκ(bs)[j]` specifies the largest (absolute) Fourier component in direction `j`.
"""
function Nκ(bf::BFourierLin)
    bf.Nκ
end

"""
    Nν(::BFourierLin{N}) where N <: Integer

Returns number of *linear* degrees of freedom.

## Remarks
- Equals the dimensionality `d == N`, since we allow independent linear variation in each direction.
"""
function Nν(bf::BFourierLin)
    bf.Nν
end

"""
    calc_BtB(bf::BFourierLin, S::AbstractArray)

Return matrix for Moore Penrose inversion (MPI).

# Arguments

- `bf`: Definition of smooth LinFourier basis
- `S`: Boolean mask (points of interest)

# Return value

- The function returns the matrix ``\\mathrm{B}^\\ast\\mathrm{P}_S\\mathrm{B}``.
- ``\\mathrm{P}_S`` projects on the subspace ``\\mathrm{\\rho}\\in S``.
"""
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

"""
    calc_Btx(bf::BFourierLin, S::AbstractArray, x::AbstractArray)

Return vector for Moore Penrose inversion (MPI).

# Arguments

- `bf`: Definition of smooth LinFourier basis
- `S`: Boolean mask (points of interest)
- `x`: Vector, the matrix `B'` is multiplied to

# Return value

- The function returns the vector ``\\mathrm{B}^\\ast\\mathrm{P}_S\\mathrm{x}``.
- ``\\mathrm{P}_S`` projects on the subspace ``\\mathrm{\\rho}\\in S``.
"""
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
    calc_∇Bt∇B(bf::BFourierLin, Sj::AbstractVector)

Return matrix for derivative-based Moore Penrose inversion (MPI).

# Arguments

- `bf`: Definition of smooth LinFourier basis
- `Sj ⊂ S`: Boolean mask (points of interest) in direction `j`

# Return value

- The function returns the matrix ``\\mathrm{\\nabla B}^\\ast\\mathrm{P}_S\\mathrm{\\nabla B}``.
- ``\\mathrm{P}_S`` projects on the subspace ``\\mathrm{\\rho}\\in S``.
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

Return vector for derivative-based Moore Penrose inversion (MPI).

# Arguments

- `bf`: Definition of smooth LinFourier basis
- `Sj ⊂ S`: Boolean mask (points of interest) in direction `j`
- `x`: Vector, the matrix `∇B'` is multiplied to

# Return value

- The function returns the vector ``\\mathrm{\\nabla B}^\\ast\\mathrm{P}_S\\mathrm{x}``.
- ``\\mathrm{P}_S`` projects on the subspace ``\\mathrm{\\rho}\\in S``.
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