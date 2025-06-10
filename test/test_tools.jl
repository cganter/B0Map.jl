using Random

function create_sinc_map(bs::BSmooth; nSincs, zc, rng=nothing, drift=nothing, offsets=nothing, noise=0.0, S=nothing, proj, rd=MersenneTwister())
    # map size
    Nρ = Nρ_orig(bs)
    sinc_map = zeros(Nρ...)
    cis = CartesianIndices(Nρ)
    
    # generate sinc map
    if rng !== nothing
        x0s = [(n -> rand(rd, 1:n)).(Nρ) for _ in 1:nSincs]
        ampls = randn(rd, nSincs)
    
        for ci in cis
            x = Tuple(ci)
            for (ampl, x0) in zip(ampls, x0s)
                sinc_map[ci] += ampl * (sinc ∘ norm)((x .- x0) ./ (Nρ .÷ zc))
            end
        end
    
        # calculate projection on smooth subspace (defined by bs)
        if proj
            sinc_map .= @views Phaser.ϕ_subspace(sinc_map, S, S, bs, 1e-6)
        end
        
        # adjust to the desired range
        min_sm = min(sinc_map...)
        max_sm = max(sinc_map...)
    
        sinc_map = ((rng[2] - rng[1]) * sinc_map .+ rng[1] * max_sm .- rng[2] * min_sm) / (max_sm - min_sm)
    end
    
    # add drift, if selected
    if drift !== nothing
        @assert length(drift) == ndims(sinc_map)
        for (j,d) in enumerate(drift)
            sinc_map .+= reshape(collect(range(-0.5d, 0.5d, size(sinc_map,j))), ones(Int, j-1)..., :)
        end
    end

    # ϕ offsets (restricted to predefined range on S)
    sinc_map_off = deepcopy(sinc_map)
    if offsets !== nothing
        n_off = length(offsets)
    
        for ci in cis
            sinc_map_off[ci] += offsets[mod(round(Int, sinc_map[ci]), n_off) + 1]
        end
    end
    
    # add noise, if needed
    if noise != 0.0
        noi_map = noise .* randn(rd, Nρ...)
        sinc_map .+= noi_map
        sinc_map_off .+= noi_map
    end

    # restrict to mask
    if S !== nothing
        sinc_map[(!).(S)] .= 0.0
        sinc_map_off[(!).(S)] .= 0.0
    end

    # return map
    (sinc_map, sinc_map_off)
end

function create_msk(ϕ; holes)
    qtl = quantile(ϕ, (0.5holes, 1-0.5holes))
    (ϕ .>= qtl[1]) .& (ϕ .<= qtl[2])
end