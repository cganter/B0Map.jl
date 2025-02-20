#=
==================================================================  
We consider a d-dimensional grid and corresponding arrays

- z = (complex) phase factor z = exp(i * (ϕ + η + ξ))
- S = mask (ROI) of valid z values z[S]

ϕ refers to a supposedly smooth phase map, which we approximate by some suitable basis 

ϕ = b + B * c,

as described in the article.
==================================================================  
=#

using LinearAlgebra, ChunkSplitters, FFTW, StatsBase, Optim, Compat
using Plots: plot, scatter!
import VP4Optim as VP
@compat public phase_map, phaser

#= 
==================================================================  
Generic implementations

- common methods for smooth basis
- phaser workflow (works for any basis)
==================================================================  
=#

abstract type BSmooth{N} end

"""
    ∇Bt∇B_∇Bty(::BSmooth, ::AbstractVector, ::AbstractVector)

Calculates and returns the tuple `(∇B' * ∇B, ∇B' * y)`.

Since efficient evaluation depends on the actual subtype of `BSmooth`,
no generic implementation is provided.
"""
function ∇Bt∇B_∇Bty(::BSmooth, ::AbstractVector, ::AbstractVector) end

"""
    phase_map(::BSmooth, ::AbstractArray)

Returns the phase map `φ = b + B' * c`.
"""
function phase_map(bs::BSmooth, b::Float64, c::AbstractVector)
    b .+ phase_map(bs, c)
end

"""
    phase_map(::BSmooth, ::AbstractVector)

Returns the phase map `φ = B' * c` assuming `b == 0`.

Since efficient evaluation depends on the actual subtype of `BSmooth`,
no generic implementation is provided.
"""
function phase_map(::BSmooth, ::AbstractVector) end

struct PurePhaserResults
    ϕ_ML::Array
    R2s_ML::Array
    χ2_ML::Array
    ϕ_0::Array
    Δϕ_0::Array
    ϕ_1::Array
    Δϕ_1::Array
    ϕ_2::Array
    Δϕ_2::Array
    ϕ_phaser::Array
    λs::Array
    χ2s::Array
    lz::Array
    ly::Array
end

function phaser(gre_con::Function, args, data, S, bs::BSmooth;
    λ_tikh=1e-6,
    n_ϕ=4,
    R2s_rng=(0.0, 1.0),
    ϕ_acc=1e-4,
    R2s_acc=1e-4,
    n_chunks=8Threads.nthreads(),
    local_fit=false)

    pure_phaser_res = pure_phaser(gre_con, args, data, S, bs,
        λ_tikh, n_ϕ, R2s_rng, ϕ_acc, R2s_acc, n_chunks)

    pure_phaser_res
end

"""
    phaser(z::AbstractArray, S::AbstractArray, bs::BSmooth;
    λ_tikh=1e-3,
    optim=:LBFGS,
    precon=true,
    precon_update=false)
    )

TBW
"""
function pure_phaser(gre_con::Function, args, data, S, bs::BSmooth,
    λ_tikh,
    n_ϕ,
    R2s_rng,
    ϕ_acc,
    R2s_acc,
    n_chunks)

    # size of data block
    sz = size(S)

    # Cartesian indices of valid data (defined by the mask S)
    cis = CartesianIndices(S)[S]
    cis_chunks = [view(cis, index_chunks(cis, n=n_chunks)[i]) for i in 1:n_chunks]

    # starting values for search
    gre_ = gre_con(args...)

    Δϕ2 = π / n_ϕ
    ϕs = range(-π + Δϕ2, π - Δϕ2, n_ϕ)
    ϕ_rngs = [[ϕ_ - Δϕ2, ϕ_ + Δϕ2] for ϕ_ in ϕs]

    # channel to prevent data races in case of multi-threaded execution
    ch_gre = Channel{typeof(gre_)}(Threads.nthreads())

    for _ in 1:Threads.nthreads()
        put!(ch_gre, gre_con(args...))
    end

    # ======================================================================
    # GSS search for optimal ϕ and R2s
    # ======================================================================

    @time begin
        print("Initial guess of ϕ and R2s ... ")

        # allocate space for local maximum likelihood (ML) estimates
        ϕ_ML = zeros(sz)
        R2s_ML = zeros(sz)
        χ2_ML = zeros(sz)
        z_ML = ones(ComplexF64, sz)

        # The actual work can be accelerated with multi-threading
        Threads.@threads for cis_chunk in cis_chunks
            # take free models
            gre = take!(ch_gre)

            # work on actual chunk
            MR.fit_chunk_GSS(gre, cis_chunk, data, ϕ_rngs, R2s_rng, ϕ_acc, R2s_acc, ϕ_ML, R2s_ML, χ2_ML)

            # put the model back
            put!(ch_gre, gre)
        end

        # phase factor
        z_ML[S] = @views exp.(im .* ϕ_ML[S])

        println("done.")
    end

    # ======================================================================
    # Smooth phase map (linearized gradients)
    # ======================================================================

    @time begin
        print("Smooth phase map (linearized gradients) ... ")

        # calculate local difference map ∇z and associated masks
        (y_ML, Sj) = calc_y(z_ML, S)

        # MPI estimate
        ∇Bt∇B = calc_∇Bt∇B(bs, Sj)
        ∇Bty = calc_∇Btx(bs, Sj, imag.(y_ML))
        c = (∇Bt∇B + (λ_tikh * mean(real.(diag(∇Bt∇B)))) * I) \ ∇Bty

        # first guess of smooth phase map ...
        ϕ_0 = zeros(sz)
        ϕ_0[S] = @views phase_map(bs, c)[S]

        # calculate generic global offset (for best match with phase factor z)
        calc_generic_offset!(ϕ_0, z_ML, S)

        # remaining deviations
        # since we now also look at local phase factors as such (instead on their gradients only)
        # we must be careful not to generate artificial jumps at phase wraps in the scaled map
        z_0 = ones(ComplexF64, sz)
        z_0[S] = @views exp.(im * ϕ_0[S])
        Δz_0 = ones(ComplexF64, sz)
        Δz_0[S] = @views z_ML[S] ./ z_0[S]   # smooth at 2π phase wraps
        Δϕ_0 = zeros(sz)
        Δϕ_0[S] = angle.(Δz_0[S])
        
        Δy_0 = calc_y_(Δz_0, Sj)

        println("done")
    end

    # ======================================================================
    # Improved smooth phase map (correct gradients)
    # ======================================================================

    @time begin
        print("Smooth phase map (correct gradients, no outliers) ... ")

        # to avoid linearization errors, we now use the logarithm of phase factor difference
        # (assuming that the magnitude of the latter is < π/2!)
        ly = [zeros(ComplexF64, sz) for _ in 1:length(Δy_0)]
        for (ly_, Sj_, y_) in zip(ly, Sj, Δy_0)
            ly_[Sj_] = @views log.(y_[Sj_] .+ 1)
        end

        # MPI estimate
        ∇Btly = calc_∇Btx(bs, Sj, imag.(ly))
        c = (∇Bt∇B + (λ_tikh * mean(real.(diag(∇Bt∇B)))) * I) \ ∇Btly
        ϕ_1 = zeros(sz)
        ϕ_1[S] = @views ϕ_0[S] + phase_map(bs, c)[S]

        # calculate generic global offset (for best match with phase factor z_ML)
        calc_generic_offset!(ϕ_1, z_ML, S)

        # remaining deviations
        # since we now also look at local phase factors as such (instead on their gradients only)
        # we must be careful not to generate artificial jumps at phase wraps in the scaled map
        z_1 = ones(ComplexF64, sz)
        z_1[S] = @views exp.(im * ϕ_1[S])
        Δz_1 = ones(ComplexF64, sz)
        Δz_1[S] = @views z_ML[S] ./ z_1[S]   # smooth at 2π phase wraps
        Δϕ_1 = zeros(sz)
        Δϕ_1[S] = angle.(Δz_1[S])

        Δy_1 = calc_y_(Δz_1, Sj)

        println("done.")
    end

    # ======================================================================
    # Balancing of agreement with phase factor and derivative
    # ======================================================================

    @time begin
        print("Search for best cost function to describe data ... ")

        BtB = calc_BtB(bs, S)

        lz = zeros(ComplexF64, sz)
        lz[S] = @views log.(Δz_1[S])
        Btlz = calc_Btx(bs, S, imag.(lz))

        for (ly_, Sj_, y_) in zip(ly, Sj, Δy_1)
            ly_[Sj_] = @views log.(y_[Sj_] .+ 1)
        end
        ∇Btly = calc_∇Btx(bs, Sj, imag.(ly))

        ϕ_λ = zeros(sz)
        R2s_λ = zeros(sz)
        χ2_λ = zeros(sz)
        A, a, B, b = BtB, Btlz, ∇Bt∇B, ∇Btly

        χ2_λ_fun = let ch_gre = ch_gre,
            data = data,
            z_ML = z_ML,
            A = A, a = a, B = B, b = b,
            S = S,
            ϕ_1 = ϕ_1,
            ϕ_λ = ϕ_λ,
            R2s_λ = R2s_λ,
            χ2_λ = χ2_λ,
            cis_chunks = cis_chunks

            λ -> begin
                if λ == 0
                    println("This should not happen.")
                    c = (B + λ_tikh * I) \ b
                    ϕ_λ[S] = @views phase_map(bs, c)[S]
                    calc_generic_offset!(ϕ_λ, z_ML, S)
                else
                    mat_λ = λ * A
                    mat_λ[2:end, 2:end] += (1 - λ) * B
                    vec_λ = λ * a
                    vec_λ[2:end] += (1 - λ) * b
                    c = (mat_λ + λ_tikh * I) \ vec_λ
                    ϕ_λ[S] = @views phase_map(bs, real(c[1]), c[2:end])[S] + ϕ_1[S]
                end

                χ2_λ[:] .= 0.0

                Threads.@threads for cis_chunk in cis_chunks
                    # take free models
                    gre = take!(ch_gre)

                    # work on actual chunk
                    MR.fit_chunk_GSS(gre, cis_chunk, data, R2s_rng, R2s_acc, ϕ_λ, R2s_λ, χ2_λ)

                    # put the model back
                    put!(ch_gre, gre)
                end

                sum(χ2_λ[S])
            end
        end

        λ_phaser, χ2_phaser, λs, χ2s = MR.GSS(χ2_λ_fun, (0.0, 1.0), 1e-4; show_all=true)

        ϕ_2 = zeros(sz)
        R2s_2 = zeros(sz)
        χ2_2 = zeros(sz)

        if λ_phaser == 0
            println("This should not happen.")
            c = (B + λ_tikh * I) \ b
            ϕ_2[S] = @views phase_map(bs, c)[S] + ϕ_1[S]
            calc_generic_offset!(ϕ_2, z_ML, S)
        else
            mat_λ = λ_phaser * A
            mat_λ[2:end, 2:end] += (1 - λ_phaser) * B
            vec_λ = λ_phaser * a
            vec_λ[2:end] += (1 - λ_phaser) * b
            c = (mat_λ + λ_tikh * I) \ vec_λ
            ϕ_2[S] = @views phase_map(bs, real(c[1]), c[2:end])[S] + ϕ_1[S]
        end
        
        z_2 = ones(ComplexF64, sz)
        z_2[S] = @views exp.(im * ϕ_2[S])
        Δz_2 = ones(ComplexF64, sz)
        Δz_2[S] = @views z_ML[S] ./ z_2[S]
        Δϕ_2 = zeros(sz)
        Δϕ_2[S] = @views angle.(Δz_2[S])

        println("done.")
    end

    # ======================================================================
    # Based upon this estimate, finalize the optimization
    # ======================================================================

    @time begin
        print("Final optimization ... ")

        # We do this for every point in S

        ϕ_phaser = deepcopy(ϕ_2)
        R2s_phaser = zeros(sz) #deepcopy(R2s_3)
        f_phaser = zeros(sz)
        c_phaser = zeros(ComplexF64, sz)
        χ2_phaser = zeros(sz)

        #=
        Threads.@threads for cis_chunk in cis_chunks
            # take free models
            gre = take!(ch_gre)

            # work on actual chunk
            MR.fit_chunk_GSS(gre, cis_chunk, data, R2s_rng, R2s_acc, ϕ_phaser, R2s_phaser, χ2_phaser)
            MR.fit_chunk_optim(gre, cis_chunk, data, R2s_rng, ϕ_phaser, R2s_phaser, χ2_phaser)
            MR.calc_f_c_chunk(gre, cis_chunk, data, ϕ_phaser, R2s_phaser, f_phaser, c_phaser)

            # put the model back
            put!(ch_gre, gre)
        end
        =#

        println("done.")
    end

    close(ch_gre)

    # return everything
    PurePhaserResults(
        ϕ_ML,
        R2s_ML,
        χ2_ML,
        ϕ_0,
        Δϕ_0,
        ϕ_1,
        Δϕ_1,
        ϕ_2,
        Δϕ_2,
        ϕ_phaser,
        λs,
        χ2s,
        lz,
        ly)
end

"""
    calc_generic_offset!(ϕ, z, S)

TBW
"""
function calc_generic_offset!(ϕ, z, S)
    # coefficient b
    b = @views angle(sum(z[S] .* exp.(-im .* ϕ[S])))

    # calculate the median of ϕ over S
    ϕ_med = median(ϕ[S]) + b

    # limit the median phase to [-π,π]
    while ϕ_med > π
        ϕ_med -= 2π
        b -= 2π
    end

    while ϕ_med < -π
        ϕ_med += 2π
        b += 2π
    end

    # add offset to ϕ
    ϕ[S] .+= b

    # return offset
    b
end

"""
    ϕ_subspace(ϕ::AbstractArray, S::AbstractArray, bs::BSmooth)

Return projection of `ϕ` on smooth subspace, defined by `bs`.
The required agreement is restricted to the mask `S`.
"""
function ϕ_subspace(ϕ::AbstractArray, S::AbstractArray, T::AbstractArray, bs::BSmooth, λ_tikh)
    # prepare Moore-Penrose pseudoinverse
    (oBtoB, oBtϕ) = oBtoB_oBtϕ(bs, T, ϕ)

    # Calculate the Moore-Penrose inverse.
    # To address potential ill-posedness of the matrix ∇Bt∇B, we make use of Tikhonov regularization.
    # The resulting bias, as long as not extremely large, should not affect the subsequent refinement step.
    c_mpi = (oBtoB + (λ_tikh * mean(real.(diag(oBtoB)))) * I) \ oBtϕ

    # calculate phase maps for b == 0
    ϕ_mpi = zeros(size(S))
    ϕ_mpi[S] = @views phase_map(bs, real(c_mpi[1]), c_mpi[2:end])[S]

    # return result
    ϕ_mpi
end

#= 
==================================================================  
Auxiliary routines
==================================================================  
=#

"""
    calc_y(z::AbstractArray, S::AbstractArray)

TBW
"""
function calc_y(z::AbstractArray, S::AbstractArray)
    # check for consistency of arguments
    (ndS = ndims(S)) != ndims(z) && throw(ArgumentError("z and S not compatible"))

    # compute difference map ∇z
    (∇z, Sj) = ∇j(z, S)

    # to compute y, we multiply ∇z with conj(z)
    y = [zeros(ComplexF64, size(z)) for _ in 1:length(∇z)]

    # ciS = CartesianIndices(S)
    for (∇z_, Sj_, y_) in zip(∇z, Sj, y)
        y_[Sj_] .= @views conj.(z[Sj_]) .* ∇z_[Sj_]
    end

    # result
    return (y, Sj)
end

"""
    calc_y_(z::AbstractArray, Sj::AbstractVector)

TBW
"""
function calc_y_(z::AbstractArray, Sj::AbstractVector)
    # compute difference map ∇z
    ∇z = ∇j_(z, Sj)

    # to compute y, we multiply ∇z with conj(z)
    y = [zeros(ComplexF64, size(z)) for _ in 1:length(∇z)]

    # ciS = CartesianIndices(S)
    for (∇z_, Sj_, y_) in zip(∇z, Sj, y)
        y_[Sj_] .= @views conj.(z[Sj_]) .* ∇z_[Sj_]
    end

    # result
    return y
end

"""
    ∇j(A::AbstractArray, S::AbstractArray)

Compute the local finite difference of array `A` along all dimensions of `S`.
as defined in the article.

# Arguments

- `A::AbstractArray`: arbitrary array, for which the difference shall be computed. `eltype(A)` is only restricted in the sense that it must support subtraction.
- `S::AbstractArray`: boolean mask, where values of `A` are meaningful. Can be a conventional Array or a BitArray.

# Boundary conditions

- The array `A` can have more dimensions than `S`, but the condition `size(A)[1:ndims(S)] == size(S)` must always be satisfied.

# Return values

The function returns a tuple `(∇A, Sj)`:

- `∇A`: difference array
- `Sj`: corresponding mask array

with dimensions:

- `size(∇A) == (size(S), ndims(S), size(A)[ndims(S)+1:end])`
- `size(Sj) == (size(S), ndims(S))`

# Example

```jldoctext
A = rand(5, 6, 7, 8)
S = A[:,:,:,1] .> 0.1   # ndims(S) < ndims(A) is allowed

# calculate differences along all directions
(∇A, Sj) = ∇j(A, S)
```
"""
function ∇j(A::AbstractArray, S::AbstractArray)
    # check for consistency of arguments
    ndA, ndS, szA, szS = ndims(A), ndims(S), size(A), size(S)
    ndA < ndS && throw(ArgumentError("ndims(A) < ndims(S)"))
    szA[1:ndS] != szS && throw(ArgumentError("A and S not compatible"))

    # set index ranges and allocate space
    ciS = CartesianIndices(S)
    fiS, liS = first(ciS), last(ciS)

    # compute Sj
    Sj = [falses(szS...) for _ in 1:ndS]

    for (j, Sj_) in zip(1:ndS, Sj)
        ej = CartesianIndex(ntuple(k -> k == j ? 1 : 0, ndS))

        for iS in fiS:liS-ej
            S[iS] && S[iS+ej] && (Sj_[iS] = true)
        end
    end

    # result
    return (∇j_(A, Sj), Sj)
end

"""
    ∇j_(A::AbstractArray, Sj::AbstractVector)

Helper function of `∇j`.

# Arguments

- `A::AbstractArray`: defined as in `∇j`
- `Sj::AbstractVector`: boolean mask vector (format as in `∇j`, but possibly with different content)

# Return values

- `∇A`: vector of difference arrays
"""
function ∇j_(A::AbstractArray, Sj::AbstractVector)
    # check for consistency of arguments
    ndA, ndS, szA, szS = ndims(A), ndims(Sj[1]), size(A), size(Sj[1])
    ndA < ndS && throw(ArgumentError("ndims(A) < ndims(S)"))
    szA[1:ndS] != szS && throw(ArgumentError("A and S not compatible"))
    szE = szA[ndS+1:ndA]   # possible extra dimensions, not affected by the gradient

    # set index ranges and allocate space
    ciS, ciE = CartesianIndices(Sj[1]), CartesianIndices(szE)
    fiS, liS = first(ciS), last(ciS)
    ∇A = [zeros(eltype(A), szA...) for _ in 1:ndS]

    # compute the local difference, where possible
    for (j, ∇A_, Sj_) in zip(1:ndS, ∇A, Sj)
        ej = CartesianIndex(ntuple(k -> k == j ? 1 : 0, ndS))

        for iE in ciE
            for iS in fiS:liS-ej
                Sj_[iS] && (∇A_[iS, iE] = A[iS+ej, iE] - A[iS, iE])
            end
        end
    end

    # result
    return ∇A
end
